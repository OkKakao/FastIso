"""Prototype server surface for FastIso windowed profile simulation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import log, sqrt
from time import perf_counter
from typing import Any

import numpy as np

from .cache import TableCacheKey, make_table_cache_key
from .isotopes import (
    IsotopePattern,
    IsotopeRegistry,
    load_isotope_registry,
    split_formula_isotope_components,
)
from .log_table import CenteredLogPhaseTable, has_cython_backend


_TABLE_CACHE: dict[tuple[TableCacheKey, str, str], CenteredLogPhaseTable] = {}


def simulate_window(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Run a windowed CZT profile simulation from a JSON-like payload."""

    formula = _required_str(payload, "formula")
    preset = str(payload.get("preset") or "common")
    explicit_elements = payload.get("elements")
    elements = tuple(explicit_elements) if explicit_elements is not None else None
    table_dm = float(payload.get("table_dm", payload.get("dm", 0.002)))
    min_fft_len = int(payload.get("min_fft_len", 32768))
    default_resolving_power = None if "gaussian_sigma" in payload else 100_000.0
    resolving_power = _optional_float(payload.get("resolving_power", default_resolving_power))
    gaussian_sigma = _optional_float(payload.get("gaussian_sigma"))
    if resolving_power is not None and gaussian_sigma is not None:
        raise ValueError("Pass either resolving_power or gaussian_sigma, not both")
    output_dm = float(payload.get("output_dm", table_dm))
    method = _resolve_method(str(payload.get("method") or "cython_auto"))
    resource = str(payload.get("isotope_resource") or "common")
    safety_sigma = float(payload.get("safety_sigma", 6.0))
    storage_mode = _storage_mode_for_method(method)
    workers = _optional_int(payload.get("workers"))

    window = payload.get("window")
    if not isinstance(window, Mapping):
        raise ValueError("payload must include a window object")
    window_mode = str(window.get("mode") or "residual")
    window_start = float(window["start"])
    window_stop = float(window["stop"])

    started = perf_counter()
    registry = load_isotope_registry(resource)
    selected_elements = _resolve_elements(registry, preset=preset, elements=elements)
    components = split_formula_isotope_components(
        formula,
        registry.patterns,
        elements=selected_elements,
    )
    spectral_elements = components.spectral_elements
    patterns = {
        element: registry.patterns[element]
        for element in spectral_elements
    }
    counts = _counts_array_for_elements(components.spectral_counts, spectral_elements)
    spectral_mean_mass = _mean_mass_from_counts(components.spectral_counts, patterns)
    total_mean_mass = spectral_mean_mass + components.mass_shift
    effective_gaussian_sigma = gaussian_sigma
    if resolving_power is not None:
        effective_gaussian_sigma = _sigma_from_resolving_power(
            total_mean_mass,
            resolving_power,
        )

    table = _cached_table_for_counts(
        counts,
        elements=spectral_elements,
        patterns=patterns,
        isotope_data_version=registry.version,
        dm=table_dm,
        min_fft_len=min_fft_len,
        safety_sigma=safety_sigma,
        gaussian_sigma=effective_gaussian_sigma,
        storage_mode=storage_mode,
    )
    kwargs: dict[str, Any] = {
        "output_dm": output_dm,
        "method": method,
        "gaussian_sigma": effective_gaussian_sigma,
        "workers": workers,
    }
    if window_mode == "residual":
        kwargs.update({"residual_start": window_start, "residual_stop": window_stop})
    elif window_mode in {"mass", "mz", "absolute"}:
        kwargs.update({
            "mass_start": window_start - components.mass_shift,
            "mass_stop": window_stop - components.mass_shift,
        })
    else:
        raise ValueError("window.mode must be residual, mass, mz, or absolute")

    mass_axis, profile, info = table.mass_profile_window_many_counts(counts, **kwargs)
    if components.mass_shift != 0.0:
        mass_axis = mass_axis + components.mass_shift
        info["mean_masses"] = np.asarray(info["mean_masses"]) + components.mass_shift
    active_fraction = float(table.active_frequency_fraction(
        counts,
        gaussian_sigma=effective_gaussian_sigma,
    )[0])
    runtime_s = perf_counter() - started
    mass = mass_axis[0]
    intensity = profile[0]
    total = float(np.sum(intensity))
    centroid = float(np.sum(mass * intensity) / total) if total != 0.0 else None
    apex_idx = int(np.argmax(intensity))
    return {
        "formula": formula,
        "elements": list(table.elements),
        "selected_elements": list(selected_elements),
        "preset": preset,
        "mass_axis": mass.tolist(),
        "intensity": intensity.tolist(),
        "summary": {
            "apex_mass": float(mass[apex_idx]),
            "apex_intensity": float(intensity[apex_idx]),
            "centroid_mass": centroid,
            "area": total,
        },
        "metadata": {
            "method": method,
            "selected_method": info["method"],
            "transform": info["transform"],
            "runtime_s": runtime_s,
            "table_dm": table.dm,
            "output_dm": float(info["output_dm"]),
            "n_fft": table.n_fft,
            "n_points": int(info["n_points"]),
            "workers": workers,
            "table_storage": table.storage_mode,
            "attenuation_dtype": str(table.attenuation.dtype),
            "table_nbytes": table.table_nbytes,
            "active_fraction": active_fraction,
            "mass_shift": components.mass_shift,
            "mass_only_counts": dict(components.mass_only_counts),
            "spectral_elements": list(spectral_elements),
            "total_mean_mass": float(total_mean_mass),
            "gaussian_sigma": (
                None if effective_gaussian_sigma is None else float(effective_gaussian_sigma)
            ),
            "resolving_power": resolving_power,
            "isotope_data_version": table.isotope_data_version,
            "cache_key": table.cache_key.as_dict(),
        },
    }


def create_app():
    """Create a FastAPI app exposing ``/simulate/window``."""

    try:
        from fastapi import Body, FastAPI
    except ImportError as exc:
        raise RuntimeError(
            "FastIso server requires the server extra: pip install -e .[server]"
        ) from exc

    app = FastAPI(title="FastIso prototype server", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "ok": True,
            "cython_backend": has_cython_backend(),
            "cached_tables": len(_TABLE_CACHE),
        }

    @app.post("/simulate/window")
    def simulate_window_endpoint(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return simulate_window(payload)

    @app.post("/api/simulate/window")
    def api_simulate_window_endpoint(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return simulate_window(payload)

    return app


try:
    app = create_app()
except RuntimeError:
    app = None


def _cached_table_for_counts(
    counts: np.ndarray,
    *,
    elements: Sequence[str],
    patterns: Mapping[str, IsotopePattern],
    isotope_data_version: str,
    dm: float,
    min_fft_len: int,
    safety_sigma: float,
    gaussian_sigma: float | None,
    storage_mode: str,
) -> CenteredLogPhaseTable:
    attenuation_dtype = np.dtype(np.float32 if storage_mode == "production" else np.float64)
    sizing_table = CenteredLogPhaseTable.build(
        elements=elements,
        dm=dm,
        min_fft_len=min_fft_len,
        attenuation_dtype=attenuation_dtype,
        storage_mode=storage_mode,
        isotope_patterns=patterns,
        isotope_data_version=isotope_data_version,
    )
    n_fft = sizing_table.suggest_n_fft_for_counts(
        counts,
        min_fft_len=min_fft_len,
        safety_sigma=safety_sigma,
        gaussian_sigma=gaussian_sigma,
    )
    key = make_table_cache_key(
        elements=elements,
        dm=dm,
        n_fft=n_fft,
        isotope_data_version=isotope_data_version,
    )
    cache_lookup_key = (key, storage_mode, attenuation_dtype.str)
    cached = _TABLE_CACHE.get(cache_lookup_key)
    if cached is not None:
        return cached
    table = sizing_table if n_fft == sizing_table.n_fft else CenteredLogPhaseTable.build(
        elements=elements,
        dm=dm,
        n_fft=n_fft,
        attenuation_dtype=attenuation_dtype,
        storage_mode=storage_mode,
        isotope_patterns=patterns,
        isotope_data_version=isotope_data_version,
    )
    _TABLE_CACHE[cache_lookup_key] = table
    return table


def _counts_array_for_elements(
    counts: Mapping[str, int],
    elements: Sequence[str],
) -> np.ndarray:
    return np.array([[int(counts[element]) for element in elements]], dtype=np.int64)


def _mean_mass_from_counts(
    counts: Mapping[str, int],
    patterns: Mapping[str, IsotopePattern],
) -> float:
    return float(sum(count * patterns[element].mean_mass for element, count in counts.items()))


def _sigma_from_resolving_power(mean_mass: float, resolving_power: float) -> float:
    if resolving_power <= 0.0:
        raise ValueError("resolving_power must be positive")
    fwhm = float(mean_mass) / float(resolving_power)
    return fwhm / (2.0 * sqrt(2.0 * log(2.0)))


def _resolve_elements(
    registry: IsotopeRegistry,
    *,
    preset: str,
    elements: Sequence[str] | None,
) -> tuple[str, ...]:
    if elements is not None:
        return tuple(elements)
    return registry.elements_for_preset(preset)


def _resolve_method(method: str) -> str:
    if (
        method in {
            "cython_log_pruned",
            "cython_log_pruned_modphase",
            "cython_log_pruned_cyclephase",
            "cython_log_pruned_uintphase",
            "cython_log_pruned_uintphase_threshold",
            "cython_log_pruned_attn32",
            "cython_log_pruned_attn32_uintphase",
            "cython_log_pruned_attn32_uintphase_threshold",
            "cython_auto",
            "auto",
        }
        and not has_cython_backend()
    ):
        return "log_pruned"
    return method


def _storage_mode_for_method(method: str) -> str:
    if not has_cython_backend():
        return "research"
    if method in {
        "auto",
        "cython_auto",
        "cython_log_pruned_uintphase",
        "cython_log_pruned_uintphase_threshold",
        "cython_log_pruned_attn32_uintphase",
        "cython_log_pruned_attn32_uintphase_threshold",
    }:
        return "production"
    return "research"


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"payload must include a non-empty {key!r}")
    return value


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    result = int(value)
    if result < 1:
        raise ValueError("workers must be positive")
    return result
