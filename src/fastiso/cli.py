"""Command-line interface for FastIso."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from math import log, sqrt
from typing import Any, TextIO

import numpy as np
from scipy.special import erf

from .isotopes import (
    IsotopePattern,
    load_isotope_registry,
    split_formula_isotope_components,
)
from .log_table import CenteredLogPhaseTable, has_cython_backend


# Exact state enumeration is a small-formula display path; large formulas should
# stay on the FT/CZT backend even when exact support would be possible.
_EXACT_PROFILE_MAX_ATOMS = 64
_EXACT_PROFILE_MAX_STATE_POINT_PRODUCT = 8_000_000


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"fastiso: error: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fastiso",
        description="Fast isotope profile simulation from the command line.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser(
        "simulate",
        help="simulate a full dense mass profile",
    )
    _add_profile_arguments(simulate)
    simulate.set_defaults(func=_run_simulate)

    window = subparsers.add_parser(
        "window",
        help="simulate a local CZT mass window",
    )
    _add_profile_arguments(window)
    _add_window_arguments(window)
    window.set_defaults(func=_run_window)

    isotopes = subparsers.add_parser("isotopes", help="inspect isotope datasets")
    isotope_subparsers = isotopes.add_subparsers(dest="isotope_command", required=True)
    list_parser = isotope_subparsers.add_parser("list", help="list elements in a preset")
    list_parser.add_argument("--preset", default="common")
    list_parser.add_argument("--resource", default=None)
    list_parser.add_argument("--format", choices=("text", "json"), default="text")
    list_parser.set_defaults(func=_run_isotopes_list)

    inspect_parser = isotope_subparsers.add_parser("inspect", help="inspect one element")
    inspect_parser.add_argument("element")
    inspect_parser.add_argument("--preset", default="common")
    inspect_parser.add_argument("--resource", default=None)
    inspect_parser.add_argument("--format", choices=("text", "json"), default="text")
    inspect_parser.set_defaults(func=_run_isotopes_inspect)

    return parser


def _add_profile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "formula",
        nargs="+",
        help="chemical formula, e.g. C6H12O6 or '(CH3OH)2(HCl)2'",
    )
    parser.add_argument("--preset", default="common", help="element preset")
    parser.add_argument("--resource", default=None, help="isotope data resource")
    parser.add_argument(
        "--elements",
        nargs="+",
        default=None,
        help="explicit spectral/preset elements; mass-only elements may be omitted",
    )
    parser.add_argument("--dm", type=float, default=0.002, help="table mass spacing")
    parser.add_argument(
        "--auto-grid",
        action="store_true",
        help="choose table/output dm from resolving power or Gaussian width",
    )
    parser.add_argument("--samples-per-fwhm", type=float, default=8.0)
    parser.add_argument("--min-fft-len", type=int, default=32768)
    parser.add_argument("--safety-sigma", type=float, default=6.0)
    broadening = parser.add_mutually_exclusive_group()
    broadening.add_argument(
        "--rp",
        "--resolving-power",
        dest="resolving_power",
        type=float,
        default=None,
        help="instrument resolving power; default 100000 when no broadening option is passed",
    )
    broadening.add_argument("--gaussian-sigma", type=float, default=None)
    parser.add_argument("--method", default="cython_auto")
    parser.add_argument("--storage-mode", choices=("research", "production", "minimal", "auto"), default="auto")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--format", choices=("csv", "json"), default="csv")
    parser.add_argument("--output", "-o", default=None, help="output path; default stdout")


def _add_window_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start", type=float, default=None, help="window start")
    parser.add_argument("--stop", type=float, default=None, help="window stop")
    parser.add_argument(
        "--window-mode",
        choices=("auto", "adaptive", "residual", "mass"),
        default="auto",
        help=(
            "auto uses start/stop as a residual window when provided, otherwise "
            "chooses an adaptive residual window; residual and mass are retained "
            "for explicit advanced use"
        ),
    )
    parser.add_argument("--output-dm", type=float, default=None)
    parser.add_argument("--points", type=int, default=None)
    parser.add_argument("--auto-window-sigma", type=float, default=6.0)
    parser.add_argument("--auto-window-min-half-width", type=float, default=0.1)
    parser.add_argument("--auto-window-cutoff", type=float, default=1e-8)
    parser.add_argument("--auto-window-max-states", type=int, default=200_000)


def _run_simulate(args: argparse.Namespace) -> int:
    result = simulate_profiles(
        args.formula,
        preset=args.preset,
        resource=args.resource,
        elements=args.elements,
        dm=args.dm,
        auto_grid=args.auto_grid,
        samples_per_fwhm=args.samples_per_fwhm,
        min_fft_len=args.min_fft_len,
        safety_sigma=args.safety_sigma,
        resolving_power=args.resolving_power,
        gaussian_sigma=args.gaussian_sigma,
        method=args.method,
        storage_mode=args.storage_mode,
        workers=args.workers,
    )
    _write_profile_result(result, output_format=args.format, output_path=args.output)
    return 0


def _run_window(args: argparse.Namespace) -> int:
    result = simulate_profiles(
        args.formula,
        preset=args.preset,
        resource=args.resource,
        elements=args.elements,
        dm=args.dm,
        auto_grid=args.auto_grid,
        samples_per_fwhm=args.samples_per_fwhm,
        min_fft_len=args.min_fft_len,
        safety_sigma=args.safety_sigma,
        resolving_power=args.resolving_power,
        gaussian_sigma=args.gaussian_sigma,
        method=args.method,
        storage_mode=args.storage_mode,
        workers=args.workers,
        window_mode=args.window_mode,
        start=args.start,
        stop=args.stop,
        output_dm=args.output_dm,
        points=args.points,
        auto_window_sigma=args.auto_window_sigma,
        auto_window_min_half_width=args.auto_window_min_half_width,
        auto_window_cutoff=args.auto_window_cutoff,
        auto_window_max_states=args.auto_window_max_states,
    )
    _write_profile_result(result, output_format=args.format, output_path=args.output)
    return 0


def _run_isotopes_list(args: argparse.Namespace) -> int:
    registry = load_isotope_registry(_resource_for_request(args.preset, args.resource))
    elements = registry.elements_for_preset(args.preset)
    patterns = registry.isotope_patterns(elements=elements)
    rows = [
        {
            "element": element,
            "isotopes": int(patterns[element].masses.size),
            "mass_only": bool(patterns[element].is_mass_only),
            "mean_mass": patterns[element].mean_mass,
        }
        for element in elements
    ]
    payload = {
        "preset": args.preset,
        "resource": _resource_for_request(args.preset, args.resource),
        "version": registry.version,
        "elements": rows,
    }
    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        for row in rows:
            marker = "mass-only" if row["mass_only"] else "spectral"
            print(f"{row['element']}\t{row['isotopes']}\t{marker}\t{row['mean_mass']:.10f}")
    return 0


def _run_isotopes_inspect(args: argparse.Namespace) -> int:
    registry = load_isotope_registry(_resource_for_request(args.preset, args.resource))
    if args.element not in registry.patterns:
        raise ValueError(
            f"element {args.element!r} is not in isotope resource "
            f"{_resource_for_request(args.preset, args.resource)!r}"
        )
    pattern = registry.patterns[args.element]
    rows = [
        {"mass": float(mass), "abundance": float(abundance)}
        for mass, abundance in zip(pattern.masses, pattern.abundances)
    ]
    payload = {
        "element": args.element,
        "resource": _resource_for_request(args.preset, args.resource),
        "version": registry.version,
        "mean_mass": pattern.mean_mass,
        "variance": pattern.variance,
        "mass_only": pattern.is_mass_only,
        "isotopes": rows,
    }
    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(f"element\t{args.element}")
        print(f"mean_mass\t{pattern.mean_mass:.12f}")
        print(f"variance\t{pattern.variance:.12g}")
        print(f"mass_only\t{pattern.is_mass_only}")
        print("mass\tabundance")
        for row in rows:
            print(f"{row['mass']:.12f}\t{row['abundance']:.16g}")
    return 0


def simulate_profiles(
    formulas: Sequence[str],
    *,
    preset: str = "common",
    resource: str | None = None,
    elements: Sequence[str] | None = None,
    dm: float = 0.002,
    auto_grid: bool = False,
    samples_per_fwhm: float = 8.0,
    min_fft_len: int = 32768,
    safety_sigma: float = 6.0,
    resolving_power: float | None = None,
    gaussian_sigma: float | None = None,
    method: str = "cython_auto",
    storage_mode: str = "auto",
    workers: int | None = None,
    window_mode: str | None = None,
    start: float | None = None,
    stop: float | None = None,
    output_dm: float | None = None,
    points: int | None = None,
    auto_window_sigma: float = 6.0,
    auto_window_min_half_width: float = 0.1,
    auto_window_cutoff: float = 1e-8,
    auto_window_max_states: int = 200_000,
) -> dict[str, Any]:
    requested_window_mode = window_mode
    if window_mode == "auto":
        window_mode = "residual" if start is not None or stop is not None else "adaptive"
    if resolving_power is None and gaussian_sigma is None:
        resolving_power = 100_000.0
    if resolving_power is not None and gaussian_sigma is not None:
        raise ValueError("Pass either resolving_power or gaussian_sigma, not both")
    if window_mode in {"residual", "mass"} and (start is None or stop is None):
        raise ValueError("window start/stop are required")
    if window_mode == "adaptive" and (start is not None or stop is not None):
        raise ValueError("adaptive window mode does not accept start/stop")
    registry = load_isotope_registry(_resource_for_request(preset, resource))
    selected_elements = tuple(elements) if elements is not None else registry.elements_for_preset(preset)
    components = [
        split_formula_isotope_components(
            formula,
            registry.patterns,
            elements=selected_elements,
        )
        for formula in formulas
    ]
    spectral_elements = tuple(
        element
        for element in selected_elements
        if any(element in component.spectral_counts for component in components)
    )
    patterns = {element: registry.patterns[element] for element in spectral_elements}
    counts = _counts_array_for_components(components, spectral_elements)
    mass_shifts = np.array([component.mass_shift for component in components], dtype=np.float64)
    total_mean_masses = np.array([
        _mean_mass_from_counts(component.spectral_counts, patterns) + component.mass_shift
        for component in components
    ])
    effective_sigma = gaussian_sigma
    if resolving_power is not None:
        effective_sigma = _sigma_from_resolving_power(total_mean_masses, resolving_power)
    requested_dm = float(dm)
    if auto_grid:
        dm = _auto_grid_dm(
            effective_sigma,
            samples_per_fwhm=samples_per_fwhm,
        )
    table_storage = _resolve_storage_mode(storage_mode)

    table = CenteredLogPhaseTable.build_for_counts(
        counts,
        elements=spectral_elements,
        dm=dm,
        min_fft_len=min_fft_len,
        safety_sigma=safety_sigma,
        gaussian_sigma=effective_sigma,
        attenuation_dtype=np.float32 if table_storage != "research" else np.float64,
        storage_mode=table_storage,
        isotope_patterns=patterns,
        isotope_data_version=registry.version,
    )
    selected_method = _resolve_method(method)
    actual_window_start = start
    actual_window_stop = stop
    adaptive_window_method = None
    window_output_dm = output_dm
    if window_mode is not None and window_output_dm is None and points is None:
        window_output_dm = dm
    if window_mode is None:
        mass_axis, profiles, info = table.mass_profile_many_counts(
            counts,
            method=selected_method,
            gaussian_sigma=effective_sigma,
            workers=workers,
        )
    elif window_mode == "residual":
        exact_result = _exact_gaussian_window_many_counts(
            table,
            counts,
            residual_start=float(start),
            residual_stop=float(stop),
            output_dm=window_output_dm,
            n_points=points,
            gaussian_sigma=effective_sigma,
            probability_cutoff=auto_window_cutoff,
            max_states=auto_window_max_states,
        )
        if exact_result is None:
            mass_axis, profiles, info = table.mass_profile_window_many_counts(
                counts,
                residual_start=float(start),
                residual_stop=float(stop),
                output_dm=window_output_dm,
                n_points=points,
                method=selected_method,
                gaussian_sigma=effective_sigma,
                workers=workers,
            )
        else:
            mass_axis, profiles, info = exact_result
    elif window_mode == "adaptive":
        actual_window_start, actual_window_stop, adaptive_window_method = _adaptive_residual_window(
            table,
            counts,
            gaussian_sigma=effective_sigma,
            auto_window_sigma=auto_window_sigma,
            min_half_width=auto_window_min_half_width,
            probability_cutoff=auto_window_cutoff,
            max_states=auto_window_max_states,
        )
        exact_result = _exact_gaussian_window_many_counts(
            table,
            counts,
            residual_start=actual_window_start,
            residual_stop=actual_window_stop,
            output_dm=window_output_dm,
            n_points=points,
            gaussian_sigma=effective_sigma,
            probability_cutoff=auto_window_cutoff,
            max_states=auto_window_max_states,
        )
        if exact_result is None:
            mass_axis, profiles, info = table.mass_profile_window_many_counts(
                counts,
                residual_start=actual_window_start,
                residual_stop=actual_window_stop,
                output_dm=window_output_dm,
                n_points=points,
                method=selected_method,
                gaussian_sigma=effective_sigma,
                workers=workers,
            )
        else:
            mass_axis, profiles, info = exact_result
        info["auto_window_method"] = adaptive_window_method
    elif window_mode == "mass":
        shifted_start = float(start) - mass_shifts
        shifted_stop = float(stop) - mass_shifts
        rows_mass = []
        rows_profile = []
        infos = []
        for row_idx in range(counts.shape[0]):
            row_mass, row_profile, row_info = table.mass_profile_window_many_counts(
                counts[row_idx : row_idx + 1],
                mass_start=float(shifted_start[row_idx]),
                mass_stop=float(shifted_stop[row_idx]),
                output_dm=window_output_dm,
                n_points=points,
                method=selected_method,
                gaussian_sigma=_row_gaussian_sigma(effective_sigma, row_idx),
                workers=workers,
            )
            rows_mass.append(row_mass[0])
            rows_profile.append(row_profile[0])
            infos.append(row_info)
        mass_axis = np.stack(rows_mass)
        profiles = np.stack(rows_profile)
        info = dict(infos[0])
        info["mean_masses"] = table.mean_mass_many_counts(counts)
    else:
        raise ValueError("window_mode must be auto, residual, mass, or adaptive")

    mass_axis = mass_axis + mass_shifts[:, None]
    info_mean = np.asarray(info.get("mean_masses", table.mean_mass_many_counts(counts)))
    info["mean_masses"] = info_mean + mass_shifts
    metadata = {
        "preset": preset,
        "resource": _resource_for_request(preset, resource),
        "isotope_data_version": registry.version,
        "selected_elements": list(selected_elements),
        "spectral_elements": list(spectral_elements),
        "mass_shifts": mass_shifts.tolist(),
        "total_mean_masses": total_mean_masses.tolist(),
        "method": info["method"],
        "requested_method": method,
        "transform": info.get("transform", "fast_odd_irfft"),
        "profile_backend": info.get("profile_backend", "ft"),
        "dm": table.dm,
        "requested_dm": requested_dm,
        "auto_grid": bool(auto_grid),
        "samples_per_fwhm": float(samples_per_fwhm),
        "output_dm": info.get("output_dm", table.dm),
        "n_fft": table.n_fft,
        "n_points": int(mass_axis.shape[1]),
        "storage_mode": table.storage_mode,
        "table_nbytes": table.table_nbytes,
        "workers": workers,
        "resolving_power": resolving_power,
        "gaussian_sigma": (
            None if effective_sigma is None else np.asarray(effective_sigma).tolist()
        ),
    }
    if "exact_state_counts" in info:
        metadata["exact_state_counts"] = np.asarray(info["exact_state_counts"]).tolist()
        metadata["exact_probability_sums"] = np.asarray(
            info["exact_probability_sums"]
        ).tolist()
    if window_mode is not None:
        metadata.update(
            {
                "window_mode": window_mode,
                "window_start": float(actual_window_start),
                "window_stop": float(actual_window_stop),
            }
        )
        if requested_window_mode is not None and requested_window_mode != window_mode:
            metadata["requested_window_mode"] = requested_window_mode
        if window_mode == "adaptive":
            metadata["auto_window_sigma"] = float(auto_window_sigma)
            metadata["auto_window_min_half_width"] = float(auto_window_min_half_width)
            metadata["auto_window_cutoff"] = float(auto_window_cutoff)
            metadata["auto_window_max_states"] = int(auto_window_max_states)
            metadata["auto_window_method"] = info.get("auto_window_method", "unknown")
    return {
        "formulas": list(formulas),
        "mass_axis": mass_axis,
        "intensity": profiles,
        "metadata": metadata,
    }


def _write_profile_result(
    result: Mapping[str, Any],
    *,
    output_format: str,
    output_path: str | None,
) -> None:
    with _open_output(output_path) as out:
        if output_format == "json":
            payload = {
                "formulas": result["formulas"],
                "metadata": result["metadata"],
                "mass_axis": np.asarray(result["mass_axis"]).tolist(),
                "intensity": np.asarray(result["intensity"]).tolist(),
            }
            json.dump(payload, out, indent=2)
            out.write("\n")
            return
        writer = csv.writer(out)
        writer.writerow(["formula", "mass", "intensity"])
        mass_axis = np.asarray(result["mass_axis"])
        intensity = np.asarray(result["intensity"])
        for row_idx, formula in enumerate(result["formulas"]):
            for mass, value in zip(mass_axis[row_idx], intensity[row_idx]):
                writer.writerow([formula, f"{float(mass):.12g}", f"{float(value):.12g}"])


class _StdoutContext:
    def __enter__(self) -> TextIO:
        return sys.stdout

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


def _open_output(path: str | None):
    if path is None:
        return _StdoutContext()
    return open(path, "w", encoding="utf-8", newline="")


def _counts_array_for_components(
    components: Sequence[Any],
    elements: Sequence[str],
) -> np.ndarray:
    return np.array(
        [
            [int(component.spectral_counts.get(element, 0)) for element in elements]
            for component in components
        ],
        dtype=np.int64,
    )


def _mean_mass_from_counts(
    counts: Mapping[str, int],
    patterns: Mapping[str, IsotopePattern],
) -> float:
    return float(sum(count * patterns[element].mean_mass for element, count in counts.items()))


def _sigma_from_resolving_power(mean_mass: np.ndarray, resolving_power: float) -> np.ndarray:
    if resolving_power <= 0.0:
        raise ValueError("resolving_power must be positive")
    fwhm = np.asarray(mean_mass, dtype=np.float64) / float(resolving_power)
    return fwhm / (2.0 * sqrt(2.0 * log(2.0)))


def _auto_grid_dm(
    gaussian_sigma: float | np.ndarray | None,
    *,
    samples_per_fwhm: float,
) -> float:
    if samples_per_fwhm <= 0.0:
        raise ValueError("samples_per_fwhm must be positive")
    if gaussian_sigma is None:
        raise ValueError("auto_grid requires resolving_power or gaussian_sigma")
    sigma = np.asarray(gaussian_sigma, dtype=np.float64)
    if sigma.ndim == 0:
        sigma = sigma[None]
    finite_positive = sigma[np.isfinite(sigma) & (sigma > 0.0)]
    if finite_positive.size == 0:
        raise ValueError("auto_grid requires a positive Gaussian width")
    min_fwhm = float(np.min(finite_positive) * (2.0 * sqrt(2.0 * log(2.0))))
    return min_fwhm / float(samples_per_fwhm)


def _exact_gaussian_window_many_counts(
    table: CenteredLogPhaseTable,
    counts: np.ndarray,
    *,
    residual_start: float,
    residual_stop: float,
    output_dm: float | None,
    n_points: int | None,
    gaussian_sigma: float | np.ndarray | None,
    probability_cutoff: float,
    max_states: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    if not _can_try_exact_states(counts):
        return None
    sigma = _sigma_vector(gaussian_sigma, counts.shape[0])
    if sigma is None or np.any(sigma <= 0.0):
        return None
    output_axis = _window_output_axis(
        residual_start,
        residual_stop,
        output_dm=output_dm,
        n_points=n_points,
    )
    if output_axis.shape[0] == 1:
        axis_step = table.dm
    else:
        axis_step = float(output_axis[1] - output_axis[0])
    states_by_row: list[dict[float, float]] = []
    state_counts: list[int] = []
    probability_sums: list[float] = []
    for row_counts in counts:
        states = _exact_states_for_count_row(
            table,
            row_counts,
            probability_cutoff=probability_cutoff,
            max_states=max_states,
        )
        if states is None:
            return None
        states_by_row.append(states)
        state_counts.append(len(states))
        probability_sums.append(float(sum(states.values())))
    if sum(state_counts) * output_axis.shape[0] > _EXACT_PROFILE_MAX_STATE_POINT_PRODUCT:
        return None

    profiles = np.zeros((counts.shape[0], output_axis.shape[0]), dtype=np.float64)
    lower_edges = output_axis - 0.5 * axis_step
    upper_edges = output_axis + 0.5 * axis_step
    root_two = sqrt(2.0)
    for row_idx, states in enumerate(states_by_row):
        centers = np.fromiter(states.keys(), dtype=np.float64)
        probabilities = np.fromiter(states.values(), dtype=np.float64)
        z_upper = (upper_edges[None, :] - centers[:, None]) / (root_two * sigma[row_idx])
        z_lower = (lower_edges[None, :] - centers[:, None]) / (root_two * sigma[row_idx])
        bin_probabilities = 0.5 * (erf(z_upper) - erf(z_lower))
        profiles[row_idx] = probabilities @ bin_probabilities

    mean_masses = table.mean_mass_many_counts(counts)
    mass_axis = mean_masses[:, None] + output_axis[None, :]
    info: dict[str, Any] = {
        "method": "exact_gaussian",
        "transform": "direct_gaussian_bins",
        "profile_backend": "exact_gaussian",
        "dm": table.dm,
        "output_dm": axis_step,
        "n_fft": table.n_fft,
        "n_points": output_axis.shape[0],
        "mean_masses": mean_masses,
        "residual_axis": output_axis,
        "exact_state_counts": np.array(state_counts, dtype=np.int64),
        "exact_probability_sums": np.array(probability_sums, dtype=np.float64),
    }
    return mass_axis, profiles, info


def _window_output_axis(
    start: float,
    stop: float,
    *,
    output_dm: float | None,
    n_points: int | None,
) -> np.ndarray:
    start = float(start)
    stop = float(stop)
    if stop < start:
        raise ValueError("window stop must be greater than or equal to start")
    if n_points is not None:
        n_points = int(n_points)
        if n_points < 1:
            raise ValueError("points must be positive")
        return np.linspace(start, stop, n_points, dtype=np.float64)
    if output_dm is None:
        raise ValueError("Pass output_dm or n_points")
    output_dm = float(output_dm)
    if output_dm <= 0.0:
        raise ValueError("output_dm must be positive")
    count = int(np.floor((stop - start) / output_dm + 0.5)) + 1
    axis = start + output_dm * np.arange(count, dtype=np.float64)
    return axis[axis <= stop + 0.5 * output_dm]


def _adaptive_residual_window(
    table: CenteredLogPhaseTable,
    counts: np.ndarray,
    *,
    gaussian_sigma: float | np.ndarray | None,
    auto_window_sigma: float,
    min_half_width: float,
    probability_cutoff: float,
    max_states: int,
) -> tuple[float, float, str]:
    if auto_window_sigma <= 0.0:
        raise ValueError("auto_window_sigma must be positive")
    if min_half_width < 0.0:
        raise ValueError("auto_window_min_half_width must be non-negative")
    if not (0.0 < probability_cutoff < 1.0):
        raise ValueError("auto_window_cutoff must be between 0 and 1")
    if max_states < 1:
        raise ValueError("auto_window_max_states must be positive")

    sigma = _sigma_vector(gaussian_sigma, counts.shape[0])
    exact_window = _exact_support_residual_window(
        table,
        counts,
        sigma=sigma,
        auto_window_sigma=auto_window_sigma,
        min_margin=min_half_width,
        probability_cutoff=probability_cutoff,
        max_states=max_states,
    )
    if exact_window is not None:
        return exact_window[0], exact_window[1], "exact_support"

    profile_sigma = table.profile_sigma_many_counts(
        counts,
        gaussian_sigma=gaussian_sigma,
    )
    half_width = float(np.max(profile_sigma) * float(auto_window_sigma))
    half_width = max(half_width, float(min_half_width), table.dm)
    return -half_width, half_width, "sigma"


def _exact_support_residual_window(
    table: CenteredLogPhaseTable,
    counts: np.ndarray,
    *,
    sigma: np.ndarray | None,
    auto_window_sigma: float,
    min_margin: float,
    probability_cutoff: float,
    max_states: int,
) -> tuple[float, float] | None:
    if not _can_try_exact_states(counts):
        return None
    starts: list[float] = []
    stops: list[float] = []
    for row_idx, row_counts in enumerate(counts):
        support = _exact_support_for_count_row(
            table,
            row_counts,
            probability_cutoff=probability_cutoff,
            max_states=max_states,
        )
        if support is None:
            return None
        row_sigma = 0.0 if sigma is None else float(sigma[row_idx])
        margin = max(float(min_margin), float(auto_window_sigma) * row_sigma, table.dm)
        starts.append(support[0] - margin)
        stops.append(support[1] + margin)
    return float(min(starts)), float(max(stops))


def _exact_support_for_count_row(
    table: CenteredLogPhaseTable,
    counts: np.ndarray,
    *,
    probability_cutoff: float,
    max_states: int,
) -> tuple[float, float] | None:
    states = _exact_states_for_count_row(
        table,
        counts,
        probability_cutoff=probability_cutoff,
        max_states=max_states,
    )
    if states is None:
        return None
    masses = np.fromiter(states.keys(), dtype=np.float64)
    return float(np.min(masses)), float(np.max(masses))


def _exact_states_for_count_row(
    table: CenteredLogPhaseTable,
    counts: np.ndarray,
    *,
    probability_cutoff: float,
    max_states: int,
) -> dict[float, float] | None:
    if not _can_try_exact_states(counts):
        return None
    states = {0.0: 1.0}
    for element_idx, count in enumerate(counts):
        atom_count = int(count)
        if atom_count == 0:
            continue
        pattern = table.isotope_patterns[element_idx]
        residual_masses = pattern.masses - pattern.mean_mass
        abundances = pattern.abundances
        for _ in range(atom_count):
            if len(states) * residual_masses.size > max_states:
                return None
            next_states: dict[float, float] = {}
            for mass, probability in states.items():
                for delta, abundance in zip(residual_masses, abundances):
                    next_probability = probability * float(abundance)
                    next_mass = round(mass + float(delta), 12)
                    next_states[next_mass] = next_states.get(next_mass, 0.0) + next_probability
            next_states = {
                mass: probability
                for mass, probability in next_states.items()
                if probability >= probability_cutoff
            }
            if not next_states:
                return None
            states = next_states
            if len(states) > max_states:
                return None
    return states


def _can_try_exact_states(counts: np.ndarray) -> bool:
    counts_array = np.asarray(counts, dtype=np.int64)
    if counts_array.ndim == 1:
        return int(np.sum(counts_array)) <= _EXACT_PROFILE_MAX_ATOMS
    if counts_array.size == 0:
        return True
    row_atom_counts = np.sum(counts_array, axis=1)
    return bool(np.max(row_atom_counts) <= _EXACT_PROFILE_MAX_ATOMS)


def _sigma_vector(
    gaussian_sigma: float | np.ndarray | None,
    n_rows: int,
) -> np.ndarray | None:
    if gaussian_sigma is None:
        return None
    sigma = np.asarray(gaussian_sigma, dtype=np.float64)
    if sigma.ndim == 0:
        return np.full(n_rows, float(sigma), dtype=np.float64)
    if sigma.shape != (n_rows,):
        raise ValueError("gaussian_sigma must be a scalar or one value per formula")
    return sigma


def _row_gaussian_sigma(
    gaussian_sigma: float | np.ndarray | None,
    row_idx: int,
) -> float | np.ndarray | None:
    if gaussian_sigma is None:
        return None
    sigma = np.asarray(gaussian_sigma)
    if sigma.ndim == 0:
        return float(sigma)
    return sigma[row_idx : row_idx + 1]


def _resource_for_request(preset: str, resource: str | None) -> str:
    if resource is not None:
        return resource
    if preset == "full":
        return "full"
    return "common"


def _resolve_method(method: str) -> str:
    if method in {
        "auto",
        "cython_auto",
        "cython_log_pruned",
        "cython_log_pruned_attn32",
        "cython_log_pruned_attn32_uintphase",
        "cython_log_pruned_attn32_uintphase_threshold",
    } and not has_cython_backend():
        return "log_pruned"
    return method


def _resolve_storage_mode(storage_mode: str) -> str:
    if storage_mode != "auto":
        return storage_mode
    return "production" if has_cython_backend() else "research"


if __name__ == "__main__":
    raise SystemExit(main())
