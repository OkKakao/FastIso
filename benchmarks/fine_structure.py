"""Fine-structure comparison against IsoSpecPy, OpenMS, and brainpy.

IsoSpecPy is used as the fine-structure reference. brainpy is treated as an
aggregated isotope-distribution baseline, so the benchmark reports both runtime
and how much cluster-level fine structure is collapsed.
"""

from __future__ import annotations

import argparse
import csv
import io
import shutil
import statistics
import subprocess
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

try:
    import pyopenms as oms
except ImportError:
    oms = None


DEFAULT_OUTPUT_DIR = Path("benchmark_results")


@dataclass(frozen=True)
class FormulaCase:
    name: str
    formula: str


CASES = (
    FormulaCase("small_glucose", "C6H12O6"),
    FormulaCase("medium_2p4k", "C100H160N25O40S2"),
    FormulaCase("large_12k", "C500H800N125O200S10"),
)


def main() -> None:
    if IsoSpecPy is None:
        raise RuntimeError("IsoSpecPy is required as the fine-structure reference")

    args = parse_args()
    rscript = find_rscript(args.rscript)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, str | float | int]] = []
    cluster_rows: list[dict[str, str | float | int]] = []
    isotope_patterns = isospec_isotope_patterns(("C", "H", "N", "O", "S"))
    for case in CASES:
        table = CenteredLogPhaseTable.build_for_formulas(
            [case.formula],
            elements=("C", "H", "N", "O", "S"),
            dm=args.dm,
            min_fft_len=args.min_fft_len,
            safety_sigma=args.safety_sigma,
            resolving_power=args.resolving_power,
            isotope_patterns=isotope_patterns,
        )
        counts = table.counts_from_formulas([case.formula])

        fastiso_timing, axis, fastiso_profile = time_fastiso(
            table,
            counts,
            resolving_power=args.resolving_power,
            repeats=args.repeats,
            warmups=args.warmups,
        )
        isospec_timing, isospec_masses, isospec_probs, isospec_profile = time_peak_profile(
            case.formula,
            axis,
            resolving_power=args.resolving_power,
            repeats=args.external_repeats,
            warmups=args.warmups,
            peak_fn=lambda formula: isospec_peaks(formula, args.isospec_coverage),
        )
        comparison_results: dict[str, tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]] = {}
        if brainpy is not None:
            comparison_results["brainpy"] = time_peak_profile(
                case.formula,
                axis,
                resolving_power=args.resolving_power,
                repeats=args.external_repeats,
                warmups=args.warmups,
                peak_fn=lambda formula: brainpy_peaks(formula, args.brainpy_peaks),
            )
        if oms is not None:
            comparison_results["pyopenms"] = time_peak_profile(
                case.formula,
                axis,
                resolving_power=args.resolving_power,
                repeats=args.external_repeats,
                warmups=args.warmups,
                peak_fn=lambda formula: pyopenms_peaks(
                    formula,
                    threshold=args.pyopenms_threshold,
                    absolute=args.pyopenms_absolute,
                ),
            )
        if rscript is not None:
            comparison_results["envipat"] = time_peak_profile(
                case.formula,
                axis,
                resolving_power=args.resolving_power,
                repeats=args.envipat_repeats,
                warmups=args.warmups,
                peak_fn=lambda formula: envipat_peaks(
                    formula,
                    rscript=rscript,
                    threshold=args.envipat_threshold,
                    rel_to=args.envipat_rel_to,
                    algo=args.envipat_algo,
                ),
            )

        mean_mass = float(table.mean_mass_many_counts(counts)[0])
        sigma = gaussian_sigma(mean_mass, args.resolving_power)
        mono_mass = float(np.min(isospec_masses))
        row: dict[str, str | float | int] = {
            "case": case.name,
            "formula": case.formula,
            "mean_mass_da": mean_mass,
            "n_fft": table.n_fft,
            "dm": table.dm,
            "resolving_power": args.resolving_power,
            "instrument_sigma_da": sigma,
            "isospec_coverage": args.isospec_coverage,
            "isospec_peak_count": len(isospec_masses),
            "fastiso_profile_s": fastiso_timing["median_s"],
            "isospec_profile_s": isospec_timing["median_s"],
            "isospec_peak_s": isospec_timing["peak_median_s"],
            "isospec_convolve_s": isospec_timing["convolve_median_s"],
            "fastiso_rel_l2_vs_isospec": relative_l2(fastiso_profile, isospec_profile),
            "fastiso_local_maxima": count_local_maxima(fastiso_profile, args.peak_rel_height),
            "isospec_local_maxima": count_local_maxima(isospec_profile, args.peak_rel_height),
            "fastiso_apex_shift_da": apex_shift(axis, fastiso_profile, isospec_profile),
        }
        for backend, (timing, masses, _probs, profile) in comparison_results.items():
            row[f"{backend}_peak_count"] = len(masses)
            row[f"{backend}_profile_s"] = timing["median_s"]
            row[f"{backend}_peak_s"] = timing["peak_median_s"]
            row[f"{backend}_convolve_s"] = timing["convolve_median_s"]
            row[f"{backend}_rel_l2_vs_isospec"] = relative_l2(profile, isospec_profile)
            row[f"{backend}_local_maxima"] = count_local_maxima(profile, args.peak_rel_height)
            row[f"{backend}_apex_shift_da"] = apex_shift(axis, profile, isospec_profile)
            if backend == "brainpy":
                row["brainpy_requested_peaks"] = args.brainpy_peaks
            if backend == "pyopenms":
                row["pyopenms_threshold"] = args.pyopenms_threshold
                row["pyopenms_absolute"] = args.pyopenms_absolute
            if backend == "envipat":
                row["envipat_threshold"] = args.envipat_threshold
                row["envipat_rel_to"] = args.envipat_rel_to
                row["envipat_algo"] = args.envipat_algo
                row["envipat_rscript"] = str(rscript)
        summary_rows.append(row)

        for backend, (_timing, masses, probs, _profile) in comparison_results.items():
            cluster_rows.extend(cluster_metrics(
                case_name=case.name,
                formula=case.formula,
                backend=backend,
                mono_mass=mono_mass,
                isospec_masses=isospec_masses,
                isospec_probs=isospec_probs,
                comparison_masses=masses,
                comparison_probs=probs,
                max_clusters=args.max_clusters,
            ))

    summary_csv = output_dir / "fine_structure_benchmark.csv"
    clusters_csv = output_dir / "fine_structure_clusters.csv"
    summary_md = output_dir / "fine_structure_benchmark.md"
    write_csv(summary_csv, summary_rows)
    write_csv(clusters_csv, cluster_rows)
    write_summary(summary_md, summary_rows, cluster_rows)
    print(f"wrote {summary_csv}")
    print(f"wrote {clusters_csv}")
    print(f"wrote {summary_md}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dm", type=float, default=0.0002)
    parser.add_argument("--min-fft-len", type=int, default=32768)
    parser.add_argument("--resolving-power", type=float, default=240_000.0)
    parser.add_argument("--safety-sigma", type=float, default=6.0)
    parser.add_argument("--isospec-coverage", type=float, default=0.999)
    parser.add_argument("--brainpy-peaks", type=int, default=200)
    parser.add_argument("--pyopenms-threshold", type=float, default=1e-6)
    parser.add_argument("--pyopenms-absolute", action="store_true")
    parser.add_argument("--rscript", default=None)
    parser.add_argument("--envipat-threshold", type=float, default=1e-4)
    parser.add_argument("--envipat-rel-to", type=int, default=0)
    parser.add_argument("--envipat-algo", type=int, default=1)
    parser.add_argument("--envipat-repeats", type=int, default=1)
    parser.add_argument("--peak-rel-height", type=float, default=0.005)
    parser.add_argument("--max-clusters", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--external-repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
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


def time_fastiso(
    table: CenteredLogPhaseTable,
    counts: np.ndarray,
    *,
    resolving_power: float,
    repeats: int,
    warmups: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    method = "cython_log_pruned" if has_cython_backend() else "log_pruned"

    def run() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        return table.mass_profile_many_counts(
            counts,
            method=method,
            resolving_power=resolving_power,
        )

    timing = time_call(lambda: capture(run), repeats=repeats, warmups=warmups)
    axis, profiles, _ = run()
    return timing, axis[0], normalize(profiles[0])


def time_peak_profile(
    formula: str,
    axis: np.ndarray,
    *,
    resolving_power: float,
    repeats: int,
    warmups: int,
    peak_fn: Callable[[str], tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    def generate_peaks() -> tuple[np.ndarray, np.ndarray]:
        return peak_fn(formula)

    peak_timing = time_call(lambda: capture(generate_peaks), repeats=repeats, warmups=warmups)
    masses, probs = generate_peaks()
    mean_mass = float(np.sum(masses * probs) / np.sum(probs))
    sigma = gaussian_sigma(mean_mass, resolving_power)

    def run() -> np.ndarray:
        return normalize(convolve_peaks_sparse(axis, masses, probs, sigma))

    convolve_timing = time_call(lambda: capture(run), repeats=repeats, warmups=warmups)
    timing = add_timing(convolve_timing, peak_timing)
    timing["peak_median_s"] = peak_timing["median_s"]
    timing["convolve_median_s"] = convolve_timing["median_s"]
    return timing, masses, probs, run()


def capture(fn: Callable[[], object]) -> None:
    fn()


def isospec_peaks(formula: str, coverage: float) -> tuple[np.ndarray, np.ndarray]:
    distribution = IsoSpecPy.IsoTotalProb(coverage, formula=formula)
    masses = np.array(distribution.np_masses(), dtype=np.float64)
    probs = np.array(distribution.np_probs(), dtype=np.float64)
    order = np.argsort(masses)
    return masses[order], probs[order]


def brainpy_peaks(formula: str, npeaks: int) -> tuple[np.ndarray, np.ndarray]:
    if brainpy is None:
        raise RuntimeError("brainpy is not installed")
    peaks = brainpy.isotopic_variants(brainpy.parse_formula(formula), npeaks=npeaks)
    masses = np.array([peak.mz for peak in peaks], dtype=np.float64)
    probs = np.array([peak.intensity for peak in peaks], dtype=np.float64)
    order = np.argsort(masses)
    return masses[order], probs[order]


def pyopenms_peaks(
    formula: str,
    *,
    threshold: float,
    absolute: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if oms is None:
        raise RuntimeError("pyOpenMS is not installed")
    generator = oms.FineIsotopePatternGenerator(threshold, False, absolute)
    distribution = generator.run(oms.EmpiricalFormula(formula))
    peaks = distribution.getContainer()
    masses = np.array([peak.getMZ() for peak in peaks], dtype=np.float64)
    probs = np.array([peak.getIntensity() for peak in peaks], dtype=np.float64)
    order = np.argsort(masses)
    return masses[order], probs[order]


def envipat_peaks(
    formula: str,
    *,
    rscript: Path,
    threshold: float,
    rel_to: int,
    algo: int,
) -> tuple[np.ndarray, np.ndarray]:
    script = Path(__file__).with_name("envipat_peaks.R")
    completed = subprocess.run(
        [
            str(rscript),
            str(script),
            formula,
            str(threshold),
            str(rel_to),
            str(algo),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    reader = csv.DictReader(io.StringIO(completed.stdout))
    rows = list(reader)
    masses = np.array([float(row["mass"]) for row in rows], dtype=np.float64)
    probs = np.array([float(row["prob"]) for row in rows], dtype=np.float64)
    order = np.argsort(masses)
    return masses[order], probs[order]


def find_rscript(explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    discovered = shutil.which("Rscript")
    if discovered:
        return Path(discovered)
    for candidate in (
        Path(r"C:\Program Files\R\R-4.6.0\bin\Rscript.exe"),
        Path(r"C:\Program Files\R\R-4.6.0\bin\x64\Rscript.exe"),
    ):
        if candidate.exists():
            return candidate
    return None


def cluster_metrics(
    *,
    case_name: str,
    formula: str,
    backend: str,
    mono_mass: float,
    isospec_masses: np.ndarray,
    isospec_probs: np.ndarray,
    comparison_masses: np.ndarray,
    comparison_probs: np.ndarray,
    max_clusters: int,
) -> list[dict[str, str | float | int]]:
    nominal = np.rint(isospec_masses - mono_mass).astype(int)
    comparison_nominal = np.rint(comparison_masses - mono_mass).astype(int)
    cluster_probs = {
        int(cluster): float(isospec_probs[nominal == cluster].sum())
        for cluster in np.unique(nominal)
    }
    top_clusters = sorted(cluster_probs, key=cluster_probs.get, reverse=True)[:max_clusters]
    rows: list[dict[str, str | float | int]] = []
    for cluster in top_clusters:
        iso_mask = nominal == cluster
        comparison_mask = comparison_nominal == cluster
        iso_masses = isospec_masses[iso_mask]
        iso_probs = isospec_probs[iso_mask]
        comparison_masses_cluster = comparison_masses[comparison_mask]
        comparison_probs_cluster = comparison_probs[comparison_mask]
        iso_probs_norm = iso_probs / iso_probs.sum()
        rows.append({
            "case": case_name,
            "formula": formula,
            "backend": backend,
            "nominal_cluster": cluster,
            "isospec_cluster_prob": float(iso_probs.sum()),
            "isospec_fine_peak_count": len(iso_masses),
            "isospec_mass_span_da": float(iso_masses.max() - iso_masses.min()) if len(iso_masses) else 0.0,
            "isospec_effective_peaks": float(1.0 / np.sum(iso_probs_norm * iso_probs_norm)),
            "isospec_top_peak_share": float(iso_probs_norm.max()),
            "comparison_peak_count": len(comparison_masses_cluster),
            "comparison_cluster_prob": float(comparison_probs_cluster.sum()) if len(comparison_probs_cluster) else 0.0,
            "comparison_centroid_da": weighted_mean(comparison_masses_cluster, comparison_probs_cluster),
            "isospec_centroid_da": weighted_mean(iso_masses, iso_probs),
            "centroid_shift_da": weighted_mean(comparison_masses_cluster, comparison_probs_cluster) - weighted_mean(iso_masses, iso_probs),
        })
    return rows


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0 or float(np.sum(weights)) == 0.0:
        return float("nan")
    return float(np.sum(values * weights) / np.sum(weights))


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


def normalize(profile: np.ndarray) -> np.ndarray:
    profile = np.maximum(profile, 0.0)
    total = profile.sum()
    if total <= 0.0:
        return profile
    return profile / total


def count_local_maxima(profile: np.ndarray, rel_height: float) -> int:
    threshold = float(profile.max()) * rel_height
    if len(profile) < 3:
        return 0
    maxima = (
        (profile[1:-1] > profile[:-2])
        & (profile[1:-1] >= profile[2:])
        & (profile[1:-1] >= threshold)
    )
    return int(maxima.sum())


def apex_shift(axis: np.ndarray, profile: np.ndarray, reference: np.ndarray) -> float:
    return float(axis[int(np.argmax(profile))] - axis[int(np.argmax(reference))])


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
        raise ValueError("no rows")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    summary_rows: list[dict[str, str | float | int]],
    cluster_rows: list[dict[str, str | float | int]],
) -> None:
    backends = sorted({
        key[: -len("_profile_s")]
        for row in summary_rows
        for key in row
        if key.endswith("_profile_s") and key not in {"fastiso_profile_s", "isospec_profile_s"}
    })
    lines = [
        "# Fine-Structure Benchmark",
        "",
        "- IsoSpecPy is the fine-structure reference.",
        "- brainpy is an aggregated isotope-distribution baseline.",
        "- pyOpenMS uses OpenMS FineIsotopePatternGenerator when pyopenms is installed.",
        "- enviPat uses the R package through Rscript when available.",
        "- Runtime includes dense Gaussian profile generation on the same grid.",
        "- External backend timing is split into peak-list generation and Gaussian convolution.",
        "- Fine-structure comparisons need dm smaller than the instrument sigma.",
        "- pyOpenMS uses individual-peak threshold mode, not total-coverage mode.",
        f"- Cython backend: {has_cython_backend()}.",
        f"- brainpy available: {brainpy is not None}.",
        f"- pyOpenMS available: {oms is not None}.",
        f"- enviPat available: {any('envipat_profile_s' in row for row in summary_rows)}.",
    ]
    if summary_rows:
        first_row = summary_rows[0]
        lines.extend([
            f"- IsoSpecPy coverage: {first_row.get('isospec_coverage', 'NA')}.",
            f"- brainpy requested peaks: {first_row.get('brainpy_requested_peaks', 'NA')}.",
            f"- pyOpenMS threshold: {first_row.get('pyopenms_threshold', 'NA')}; absolute: {first_row.get('pyopenms_absolute', 'NA')}.",
            f"- enviPat threshold: {first_row.get('envipat_threshold', 'NA')}; rel_to: {first_row.get('envipat_rel_to', 'NA')}; algo: {first_row.get('envipat_algo', 'NA')}.",
        ])
    lines.extend([
        "",
        "| case | formula | backend | n_fft | total s | peak s | convolve s | speed vs FastIso | peaks | rel L2 vs IsoSpec | local maxima | apex shift Da |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in summary_rows:
        fastiso_s = float(row["fastiso_profile_s"])
        backend_rows: list[tuple[str, float, str, str, float, int | str, float, int | str, float]] = [
            (
                "fastiso",
                fastiso_s,
                "NA",
                "NA",
                1.0,
                "NA",
                float(row["fastiso_rel_l2_vs_isospec"]),
                int(row["fastiso_local_maxima"]),
                float(row["fastiso_apex_shift_da"]),
            ),
            (
                "IsoSpecPy",
                float(row["isospec_profile_s"]),
                f"{float(row['isospec_peak_s']):.6f}",
                f"{float(row['isospec_convolve_s']):.6f}",
                float(row["isospec_profile_s"]) / fastiso_s,
                int(row["isospec_peak_count"]),
                0.0,
                int(row["isospec_local_maxima"]),
                0.0,
            ),
        ]
        for backend in backends:
            backend_rows.append((
                backend,
                float(row[f"{backend}_profile_s"]),
                f"{float(row[f'{backend}_peak_s']):.6f}",
                f"{float(row[f'{backend}_convolve_s']):.6f}",
                float(row[f"{backend}_profile_s"]) / fastiso_s,
                int(row[f"{backend}_peak_count"]),
                float(row[f"{backend}_rel_l2_vs_isospec"]),
                int(row[f"{backend}_local_maxima"]),
                float(row[f"{backend}_apex_shift_da"]),
            ))
        for backend, median_s, peak_s, convolve_s, speed_ratio, peaks, rel_l2, maxima, shift in backend_rows:
            peaks_text = str(peaks)
            lines.append(
                "| "
                f"{row['case']} | {row['formula']} | {backend} | {row['n_fft']} | "
                f"{median_s:.6f} | {peak_s} | {convolve_s} | {speed_ratio:.2f}x | {peaks_text} | "
                f"{rel_l2:.2e} | {maxima} | {shift:.4e} |"
            )
    lines.extend([
        "",
        "## Top Cluster Fine Structure",
        "",
        "| case | backend | cluster | Iso fine peaks | Iso span Da | effective peaks | top peak share | backend peaks | centroid shift Da |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in cluster_rows:
        lines.append(
            "| "
            f"{row['case']} | {row['backend']} | {row['nominal_cluster']} | "
            f"{row['isospec_fine_peak_count']} | {float(row['isospec_mass_span_da']):.4f} | "
            f"{float(row['isospec_effective_peaks']):.2f} | "
            f"{float(row['isospec_top_peak_share']):.3f} | "
            f"{row['comparison_peak_count']} | {float(row['centroid_shift_da']):.4e} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
