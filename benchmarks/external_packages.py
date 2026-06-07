"""Compare fastiso against external isotope packages.

This benchmark compares full instrument-broadened profiles on the same mass grid.
IsoSpecPy produces fine-structure peak lists, so the script records both peak-list
generation time and peak-list plus Gaussian-convolution profile time.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fastiso import CenteredLogPhaseTable, IsotopePattern, has_cython_backend

try:
    import IsoSpecPy
except ImportError:
    IsoSpecPy = None

try:
    import brainpy
except ImportError:
    brainpy = None


DEFAULT_OUTPUT_DIR = Path("benchmark_results")


@dataclass(frozen=True)
class FormulaFamily:
    name: str
    base_counts: dict[str, int]
    step_counts: dict[str, int]


FAMILIES = (
    FormulaFamily(
        name="small",
        base_counts={"C": 6, "H": 12, "N": 0, "O": 6, "S": 0},
        step_counts={"C": 1, "H": 2, "N": 1, "O": 1, "S": 1},
    ),
    FormulaFamily(
        name="medium",
        base_counts={"C": 100, "H": 160, "N": 25, "O": 40, "S": 2},
        step_counts={"C": 2, "H": 3, "N": 1, "O": 1, "S": 1},
    ),
    FormulaFamily(
        name="large",
        base_counts={"C": 500, "H": 800, "N": 125, "O": 200, "S": 10},
        step_counts={"C": 3, "H": 5, "N": 1, "O": 1, "S": 1},
    ),
)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_table = CenteredLogPhaseTable.build(
        elements=("C", "H", "N", "O", "S"),
        dm=args.dm,
        min_fft_len=args.min_fft_len,
        isotope_patterns=isospec_isotope_patterns(("C", "H", "N", "O", "S")),
    )
    rows: list[dict[str, str | float | int]] = []
    for family in FAMILIES:
        for batch_size in args.batch_size:
            formulas = make_formulas(family, batch_size)
            if args.auto_window:
                table = CenteredLogPhaseTable.build_for_formulas(
                    formulas,
                    elements=("C", "H", "N", "O", "S"),
                    dm=args.dm,
                    min_fft_len=args.min_fft_len,
                    safety_sigma=args.safety_sigma,
                    resolving_power=args.resolving_power,
                    isotope_patterns=isospec_isotope_patterns(("C", "H", "N", "O", "S")),
                )
            else:
                table = base_table
            counts = table.counts_from_formulas(formulas)
            active_fraction = float(np.mean(
                table.active_frequency_fraction(
                    counts,
                    resolving_power=args.resolving_power,
                )
            ))

            fastiso_timing, fastiso_profiles, fastiso_axes = time_fastiso_profiles(
                table,
                counts,
                resolving_power=args.resolving_power,
                repeats=args.repeats,
                warmups=args.warmups,
            )

            isospec_result = None
            if IsoSpecPy is not None:
                isospec_result = time_external_profiles(
                    formulas,
                    fastiso_axes,
                    resolving_power=args.resolving_power,
                    repeats=args.external_repeats,
                    warmups=args.warmups,
                    peak_fn=lambda formula: isospec_peaks(formula, args.isospec_coverage),
                )
            reference_profiles = (
                isospec_result["profiles"] if isospec_result is not None else fastiso_profiles
            )
            rows.append({
                "package": "fastiso",
                "method": "cython_log_pruned_profile",
                "family": family.name,
                "batch_size": batch_size,
                "auto_window": args.auto_window,
                "mean_mass_da": float(np.mean(table.mean_mass_many_counts(counts))),
                "active_fraction": active_fraction,
                "resolving_power": args.resolving_power,
                "dm": table.dm,
                "n_fft": table.n_fft,
                "n_positive": table.n_positive,
                "median_s": fastiso_timing["median_s"],
                "min_s": fastiso_timing["min_s"],
                "max_s": fastiso_timing["max_s"],
                "peak_count_mean": 0.0,
                "coverage_mean": 1.0,
                "rel_l2_vs_isospec": relative_l2(fastiso_profiles, reference_profiles),
            })

            if isospec_result is not None:
                rows.append(result_row(
                    package="IsoSpecPy",
                    method=f"IsoTotalProb_{args.isospec_coverage:g}_convolved",
                    family=family.name,
                    batch_size=batch_size,
                    auto_window=args.auto_window,
                    table=table,
                    counts=counts,
                    active_fraction=active_fraction,
                    resolving_power=args.resolving_power,
                    timing=isospec_result["timing"],
                    peak_counts=isospec_result["peak_counts"],
                    coverages=isospec_result["coverages"],
                    rel_l2=0.0,
                ))
                rows.append(result_row(
                    package="IsoSpecPy",
                    method=f"IsoTotalProb_{args.isospec_coverage:g}_peaks_only",
                    family=family.name,
                    batch_size=batch_size,
                    auto_window=args.auto_window,
                    table=table,
                    counts=counts,
                    active_fraction=active_fraction,
                    resolving_power=args.resolving_power,
                    timing=isospec_result["peak_timing"],
                    peak_counts=isospec_result["peak_counts"],
                    coverages=isospec_result["coverages"],
                    rel_l2=float("nan"),
                ))

            if brainpy is not None:
                brain_result = time_external_profiles(
                    formulas,
                    fastiso_axes,
                    resolving_power=args.resolving_power,
                    repeats=args.external_repeats,
                    warmups=args.warmups,
                    peak_fn=lambda formula: brainpy_peaks(formula, args.brainpy_peaks),
                )
                rows.append(result_row(
                    package="brainpy",
                    method=f"isotopic_variants_{args.brainpy_peaks}_convolved",
                    family=family.name,
                    batch_size=batch_size,
                    auto_window=args.auto_window,
                    table=table,
                    counts=counts,
                    active_fraction=active_fraction,
                    resolving_power=args.resolving_power,
                    timing=brain_result["timing"],
                    peak_counts=brain_result["peak_counts"],
                    coverages=brain_result["coverages"],
                    rel_l2=relative_l2(brain_result["profiles"], reference_profiles),
                ))

    csv_path = output_dir / "external_package_benchmark.csv"
    md_path = output_dir / "external_package_benchmark.md"
    write_csv(csv_path, rows)
    write_summary(md_path, rows)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dm", type=float, default=0.002)
    parser.add_argument("--min-fft-len", type=int, default=32768)
    parser.add_argument("--resolving-power", type=float, default=100_000.0)
    parser.add_argument("--isospec-coverage", type=float, default=0.999)
    parser.add_argument("--brainpy-peaks", type=int, default=200)
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--external-repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--auto-window", action="store_true")
    parser.add_argument("--safety-sigma", type=float, default=6.0)
    return parser.parse_args()


def isospec_isotope_patterns(elements: tuple[str, ...]) -> dict[str, IsotopePattern]:
    if IsoSpecPy is None:
        raise RuntimeError("IsoSpecPy is required to build matched isotope patterns")
    patterns: dict[str, IsotopePattern] = {}
    periodic_table = IsoSpecPy.PeriodicTbl
    for element in elements:
        patterns[element] = IsotopePattern(
            element=element,
            masses=np.array(periodic_table.symbol_to_masses[element], dtype=np.float64),
            abundances=np.array(periodic_table.symbol_to_probs[element], dtype=np.float64),
        )
    return patterns


def time_fastiso_profiles(
    table: CenteredLogPhaseTable,
    counts: np.ndarray,
    *,
    resolving_power: float,
    repeats: int,
    warmups: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    method = "cython_log_pruned" if has_cython_backend() else "log_pruned"
    last_result: tuple[np.ndarray, np.ndarray, dict[str, object]] | None = None

    def run() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        return table.mass_profile_many_counts(
            counts,
            method=method,
            resolving_power=resolving_power,
        )

    timing = time_call(lambda: _capture(run), repeats=repeats, warmups=warmups)
    last_result = run()
    axes, profiles, _ = last_result
    return timing, normalize_profiles(profiles), axes


def _capture(fn: Callable[[], object]) -> None:
    fn()


def time_external_profiles(
    formulas: list[str],
    axes: np.ndarray,
    *,
    resolving_power: float,
    repeats: int,
    warmups: int,
    peak_fn: Callable[[str], tuple[np.ndarray, np.ndarray, float]],
) -> dict[str, object]:
    peak_cache: list[tuple[np.ndarray, np.ndarray, float]] = []

    def generate_peaks() -> list[tuple[np.ndarray, np.ndarray, float]]:
        return [peak_fn(formula) for formula in formulas]

    peak_timing = time_call(lambda: _capture(generate_peaks), repeats=repeats, warmups=warmups)
    peak_cache = generate_peaks()

    def generate_profiles() -> np.ndarray:
        profiles = []
        for row_axis, (masses, probs, _) in zip(axes, peak_cache, strict=True):
            mean_mass = float(np.sum(masses * probs) / np.sum(probs))
            sigma = gaussian_sigma(mean_mass, resolving_power)
            profiles.append(convolve_peaks_sparse(row_axis, masses, probs, sigma))
        return normalize_profiles(np.stack(profiles))

    timing = time_call(lambda: _capture(generate_profiles), repeats=repeats, warmups=warmups)
    profiles = generate_profiles()
    return {
        "timing": add_timing(timing, peak_timing),
        "peak_timing": peak_timing,
        "profiles": profiles,
        "peak_counts": [len(peaks[0]) for peaks in peak_cache],
        "coverages": [peaks[2] for peaks in peak_cache],
    }


def isospec_peaks(formula: str, coverage: float) -> tuple[np.ndarray, np.ndarray, float]:
    distribution = IsoSpecPy.IsoTotalProb(coverage, formula=formula)
    masses = np.array(distribution.np_masses(), dtype=np.float64)
    probs = np.array(distribution.np_probs(), dtype=np.float64)
    order = np.argsort(masses)
    masses = masses[order]
    probs = probs[order]
    return masses, probs, float(np.sum(probs))


def brainpy_peaks(formula: str, npeaks: int) -> tuple[np.ndarray, np.ndarray, float]:
    composition = brainpy.parse_formula(formula)
    peaks = brainpy.isotopic_variants(composition, npeaks=npeaks)
    masses = np.array([peak.mz for peak in peaks], dtype=np.float64)
    probs = np.array([peak.intensity for peak in peaks], dtype=np.float64)
    order = np.argsort(masses)
    masses = masses[order]
    probs = probs[order]
    return masses, probs, float(np.sum(probs))


def convolve_peaks_sparse(
    axis: np.ndarray,
    masses: np.ndarray,
    probs: np.ndarray,
    sigma: float,
    *,
    sigma_radius: float = 7.0,
) -> np.ndarray:
    profile = np.zeros_like(axis, dtype=np.float64)
    dm = float(axis[1] - axis[0])
    start_axis = float(axis[0])
    window = max(1, int(np.ceil(sigma_radius * sigma / dm)))
    coefficient = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    for mass, prob in zip(masses, probs, strict=True):
        center = int(round((float(mass) - start_axis) / dm))
        lo = max(0, center - window)
        hi = min(len(axis), center + window + 1)
        if lo >= hi:
            continue
        x = axis[lo:hi]
        profile[lo:hi] += prob * coefficient * np.exp(-0.5 * ((x - mass) / sigma) ** 2)
    profile *= dm
    return profile


def gaussian_sigma(mean_mass: float, resolving_power: float) -> float:
    return mean_mass / resolving_power / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def normalize_profiles(profiles: np.ndarray) -> np.ndarray:
    totals = profiles.sum(axis=-1, keepdims=True)
    return np.divide(profiles, totals, out=np.zeros_like(profiles), where=totals > 0)


def result_row(
    *,
    package: str,
    method: str,
    family: str,
    batch_size: int,
    auto_window: bool,
    table: CenteredLogPhaseTable,
    counts: np.ndarray,
    active_fraction: float,
    resolving_power: float,
    timing: dict[str, float],
    peak_counts: list[int],
    coverages: list[float],
    rel_l2: float,
) -> dict[str, str | float | int]:
    return {
        "package": package,
        "method": method,
        "family": family,
        "batch_size": batch_size,
        "auto_window": auto_window,
        "mean_mass_da": float(np.mean(table.mean_mass_many_counts(counts))),
        "active_fraction": active_fraction,
        "resolving_power": resolving_power,
        "dm": table.dm,
        "n_fft": table.n_fft,
        "n_positive": table.n_positive,
        "median_s": timing["median_s"],
        "min_s": timing["min_s"],
        "max_s": timing["max_s"],
        "peak_count_mean": float(np.mean(peak_counts)) if peak_counts else 0.0,
        "coverage_mean": float(np.mean(coverages)) if coverages else 1.0,
        "rel_l2_vs_isospec": rel_l2,
    }


def make_formulas(family: FormulaFamily, n: int) -> list[str]:
    formulas: list[str] = []
    for i in range(n):
        counts = {
            element: family.base_counts[element] + i * family.step_counts[element]
            for element in family.base_counts
        }
        formulas.append(format_formula(counts))
    return formulas


def format_formula(counts: dict[str, int]) -> str:
    return "".join(
        f"{element}{count}"
        for element, count in counts.items()
        if count > 0
    )


def time_call(
    fn: Callable[[], object],
    *,
    repeats: int,
    warmups: int,
) -> dict[str, float]:
    for _ in range(warmups):
        fn()
    elapsed: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        elapsed.append(time.perf_counter() - start)
    return {
        "median_s": float(statistics.median(elapsed)),
        "min_s": float(min(elapsed)),
        "max_s": float(max(elapsed)),
    }


def add_timing(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {
        "median_s": a["median_s"] + b["median_s"],
        "min_s": a["min_s"] + b["min_s"],
        "max_s": a["max_s"] + b["max_s"],
    }


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def write_csv(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    if not rows:
        raise ValueError("no benchmark rows to write")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    lines = [
        "# External Package Benchmark",
        "",
        f"- IsoSpecPy available: {IsoSpecPy is not None}",
        f"- brainpy available: {brainpy is not None}",
        f"- Cython backend: {has_cython_backend()}",
        "- FastIso uses isotope masses/abundances copied from IsoSpecPy for this benchmark.",
        "- Runtime includes profile generation on the same mass grid; IsoSpecPy peaks-only rows exclude Gaussian convolution.",
        "- With dm=0.002 and R=100000, small-mass profiles are under-sampled; medium/large rows are the relevant comparison.",
        "",
        "| family | batch | auto | n_fft | package | method | median s | speedup vs fastiso | peaks | coverage | rel L2 vs IsoSpec |",
        "| --- | ---: | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    fastiso_by_case = {
        (row["family"], row["batch_size"]): float(row["median_s"])
        for row in rows
        if row["package"] == "fastiso"
    }
    for row in rows:
        key = (row["family"], row["batch_size"])
        fastiso_s = fastiso_by_case.get(key, float(row["median_s"]))
        speedup = float(row["median_s"]) / fastiso_s
        rel_l2 = row["rel_l2_vs_isospec"]
        rel_l2_text = "NA" if isinstance(rel_l2, float) and np.isnan(rel_l2) else f"{float(rel_l2):.2e}"
        lines.append(
            "| "
            f"{row['family']} | {row['batch_size']} | {bool(row['auto_window'])} | "
            f"{row['n_fft']} | {row['package']} | {row['method']} | "
            f"{float(row['median_s']):.6f} | {speedup:.2f}x | "
            f"{float(row['peak_count_mean']):.1f} | {float(row['coverage_mean']):.6f} | "
            f"{rel_l2_text} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
