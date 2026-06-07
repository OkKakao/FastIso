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

from .isotopes import (
    IsotopePattern,
    load_isotope_registry,
    split_formula_isotope_components,
)
from .log_table import CenteredLogPhaseTable, has_cython_backend


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
    parser.add_argument("formula", nargs="+", help="chemical formula, e.g. C6H12O6")
    parser.add_argument("--preset", default="common", help="element preset")
    parser.add_argument("--resource", default=None, help="isotope data resource")
    parser.add_argument(
        "--elements",
        nargs="+",
        default=None,
        help="explicit spectral/preset elements; mass-only elements may be omitted",
    )
    parser.add_argument("--dm", type=float, default=0.002, help="table mass spacing")
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
    parser.add_argument("--start", type=float, required=True, help="window start")
    parser.add_argument("--stop", type=float, required=True, help="window stop")
    parser.add_argument(
        "--window-mode",
        choices=("residual", "mass"),
        default="residual",
        help="residual window is relative to formula mean mass; mass is absolute",
    )
    parser.add_argument("--output-dm", type=float, default=None)
    parser.add_argument("--points", type=int, default=None)


def _run_simulate(args: argparse.Namespace) -> int:
    result = simulate_profiles(
        args.formula,
        preset=args.preset,
        resource=args.resource,
        elements=args.elements,
        dm=args.dm,
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
) -> dict[str, Any]:
    if resolving_power is None and gaussian_sigma is None:
        resolving_power = 100_000.0
    if resolving_power is not None and gaussian_sigma is not None:
        raise ValueError("Pass either resolving_power or gaussian_sigma, not both")
    if window_mode is not None and (start is None or stop is None):
        raise ValueError("window start/stop are required")
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
    if window_mode is None:
        mass_axis, profiles, info = table.mass_profile_many_counts(
            counts,
            method=selected_method,
            gaussian_sigma=effective_sigma,
            workers=workers,
        )
    elif window_mode == "residual":
        mass_axis, profiles, info = table.mass_profile_window_many_counts(
            counts,
            residual_start=float(start),
            residual_stop=float(stop),
            output_dm=output_dm,
            n_points=points,
            method=selected_method,
            gaussian_sigma=effective_sigma,
            workers=workers,
        )
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
                output_dm=output_dm,
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
        raise ValueError("window_mode must be residual or mass")

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
        "dm": table.dm,
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
    if window_mode is not None:
        metadata.update(
            {
                "window_mode": window_mode,
                "window_start": float(start),
                "window_stop": float(stop),
            }
        )
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
