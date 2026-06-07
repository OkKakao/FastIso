"""Benchmark phase modulo reduction in the Cython log-pruned kernel."""

from __future__ import annotations

import argparse
import csv
import math
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from fastiso import CenteredLogPhaseTable, has_cython_backend, parse_formula


CASES = (
    ("large_12k", "C500H800N125O200S10"),
    ("xlarge_29k", "C1200H1900N320O450S25"),
    ("huge_60k", "C2500H4000N650O900S50"),
    ("huge_120k", "C5000H8000N1300O1800S100"),
    ("huge_240k", "C10000H16000N2600O3600S200"),
)
METHODS = (
    "cython_log_pruned",
    "cython_log_pruned_modphase",
    "cython_log_pruned_cyclephase",
    "cython_log_pruned_uintphase",
    "cython_log_pruned_uintphase_threshold",
    "cython_auto",
)


def main() -> None:
    args = parse_args()
    if not has_cython_backend():
        raise RuntimeError("phase modulo benchmark requires the Cython backend")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_benchmark(args)
    csv_path = output_dir / "phase_modulo_benchmark.csv"
    md_path = output_dir / "phase_modulo_benchmark.md"
    write_csv(csv_path, rows)
    write_summary(md_path, rows)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--dm", type=float, default=0.002)
    parser.add_argument("--min-fft-len", type=int, default=32768)
    parser.add_argument("--safety-sigma", type=float, default=6.0)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1, 10, 50])
    parser.add_argument(
        "--resolving-power",
        type=float,
        nargs="+",
        default=[0.0, 100_000.0],
        help="Use 0 for no Gaussian resolving-power damping.",
    )
    return parser.parse_args()


def run_benchmark(args: argparse.Namespace) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for case_name, formula in CASES:
        for resolving_power_arg in args.resolving_power:
            for batch_size in args.batch_size:
                resolving_power = None if resolving_power_arg == 0.0 else resolving_power_arg
                formulas = make_formulas(formula, batch_size)
                table = CenteredLogPhaseTable.build_for_formulas(
                    formulas,
                    elements=("C", "H", "N", "O", "S"),
                    dm=args.dm,
                    min_fft_len=args.min_fft_len,
                    safety_sigma=args.safety_sigma,
                    resolving_power=resolving_power,
                )
                counts = table.counts_from_formulas(formulas)
                mean_mass = float(np.mean(table.mean_mass_many_counts(counts)))
                active_fraction = float(np.mean(table.active_frequency_fraction(
                    counts,
                    resolving_power=resolving_power,
                )))
                phase_stats = active_phase_stats(
                    table,
                    counts,
                    resolving_power=resolving_power,
                )
                baseline = table.residual_spectrum_many_counts(
                    counts,
                    method="cython_log_pruned",
                    resolving_power=resolving_power,
                )

                timing_by_method: dict[str, dict[str, float]] = {}
                spectra_by_method: dict[str, np.ndarray] = {}
                for method in METHODS:
                    timing_by_method[method] = time_call(
                        lambda method=method: table.residual_spectrum_many_counts(
                            counts,
                            method=method,
                            resolving_power=resolving_power,
                        ),
                        repeats=args.repeats,
                        warmups=args.warmups,
                    )
                    spectra_by_method[method] = table.residual_spectrum_many_counts(
                        counts,
                        method=method,
                        resolving_power=resolving_power,
                    )

                baseline_s = timing_by_method["cython_log_pruned"]["median_s"]
                for method in METHODS:
                    spectrum = spectra_by_method[method]
                    selected_method = table.select_spectrum_method(
                        counts,
                        method=method,
                        resolving_power=resolving_power,
                    )
                    rows.append({
                        "case": case_name,
                        "formula": formula,
                        "batch_size": batch_size,
                        "method": method,
                        "selected_method": selected_method,
                        "resolving_power": 0.0 if resolving_power is None else float(resolving_power),
                        "dm": table.dm,
                        "n_fft": table.n_fft,
                        "n_positive": table.n_positive,
                        "mean_mass_da": mean_mass,
                        "active_fraction": active_fraction,
                        "active_phase_max_abs_rad": phase_stats["max_abs_rad"],
                        "active_phase_median_abs_rad": phase_stats["median_abs_rad"],
                        "median_s": timing_by_method[method]["median_s"],
                        "min_s": timing_by_method[method]["min_s"],
                        "max_s": timing_by_method[method]["max_s"],
                        "baseline_over_method": baseline_s / timing_by_method[method]["median_s"],
                        "rel_l2_vs_baseline": relative_l2(spectrum, baseline),
                        "max_abs_vs_baseline": float(np.max(np.abs(spectrum - baseline))),
                        "repeats": args.repeats,
                    })
    return rows


def make_formulas(formula: str, n: int) -> list[str]:
    base = parse_formula(formula)
    steps = {"C": 3, "H": 5, "N": 1, "O": 1, "S": 1}
    formulas: list[str] = []
    for i in range(n):
        counts = {
            element: count + i * steps.get(element, 0)
            for element, count in base.items()
        }
        formulas.append(format_formula(counts))
    return formulas


def format_formula(counts: dict[str, int]) -> str:
    return "".join(
        f"{element}{count}"
        for element, count in counts.items()
        if count > 0
    )


def active_phase_stats(
    table: CenteredLogPhaseTable,
    counts: np.ndarray,
    *,
    resolving_power: float | None,
    prune_cutoff: float = 1e-12,
) -> dict[str, float]:
    counts_float = counts.astype(np.float64, copy=False)
    attenuation = counts_float @ table.attenuation
    sigma = table._resolve_gaussian_sigma(  # noqa: SLF001 - benchmark diagnostic.
        counts,
        gaussian_sigma=None,
        resolving_power=resolving_power,
    )
    if sigma is not None:
        attenuation = attenuation + 0.5 * (sigma[:, None] * table.omega[None, :]) ** 2
    active = attenuation <= -math.log(prune_cutoff)
    if not np.any(active):
        return {"max_abs_rad": 0.0, "median_abs_rad": 0.0}
    phase = counts_float @ table.phase
    abs_phase = np.abs(phase[active])
    return {
        "max_abs_rad": float(np.max(abs_phase)),
        "median_abs_rad": float(np.median(abs_phase)),
    }


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


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(a - b) / denom)


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        raise ValueError("no benchmark rows to write")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    index = {
        (row["case"], row["batch_size"], row["resolving_power"], row["method"]): row
        for row in rows
    }
    lines = [
        "# Phase Modulo Benchmark",
        "",
        f"- Python: {platform.python_version()}",
        f"- Platform: {platform.platform()}",
        "- Measured kernel: residual spectrum generation only.",
        "- Ratio columns above 1 mean the alternative phase reduction is faster.",
        "",
        "| case | batch | R | n_fft | active | max active phase rad | baseline s | fmod s | cycle s | uint s | threshold s | auto s | auto selected | baseline/fmod | baseline/cycle | baseline/uint | baseline/threshold | baseline/auto | auto rel L2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    keys = sorted(
        {(row["case"], row["batch_size"], row["resolving_power"]) for row in rows},
        key=lambda item: (str(item[0]), int(item[1]), float(item[2])),
    )
    for case_name, batch_size, resolving_power in keys:
        baseline = index[(case_name, batch_size, resolving_power, "cython_log_pruned")]
        modphase = index[(case_name, batch_size, resolving_power, "cython_log_pruned_modphase")]
        cyclephase = index[(case_name, batch_size, resolving_power, "cython_log_pruned_cyclephase")]
        uintphase = index[(case_name, batch_size, resolving_power, "cython_log_pruned_uintphase")]
        threshold = index[(case_name, batch_size, resolving_power, "cython_log_pruned_uintphase_threshold")]
        auto = index[(case_name, batch_size, resolving_power, "cython_auto")]
        baseline_s = float(baseline["median_s"])
        modphase_s = float(modphase["median_s"])
        cyclephase_s = float(cyclephase["median_s"])
        uintphase_s = float(uintphase["median_s"])
        threshold_s = float(threshold["median_s"])
        auto_s = float(auto["median_s"])
        lines.append(
            "| "
            f"{case_name} | {int(batch_size)} | {float(resolving_power):.0f} | "
            f"{baseline['n_fft']} | "
            f"{float(baseline['active_fraction']):.4g} | "
            f"{float(baseline['active_phase_max_abs_rad']):.3g} | "
            f"{baseline_s:.6f} | {modphase_s:.6f} | {cyclephase_s:.6f} | "
            f"{uintphase_s:.6f} | {threshold_s:.6f} | {auto_s:.6f} | "
            f"{auto['selected_method']} | "
            f"{baseline_s / modphase_s:.3f}x | "
            f"{baseline_s / cyclephase_s:.3f}x | "
            f"{baseline_s / uintphase_s:.3f}x | "
            f"{baseline_s / threshold_s:.3f}x | "
            f"{baseline_s / auto_s:.3f}x | "
            f"{float(auto['rel_l2_vs_baseline']):.2e} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
