"""Benchmark Cython residual-spectrum worker parallelism."""

from __future__ import annotations

import argparse
import csv
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from fastiso import CenteredLogPhaseTable, has_cython_backend, parse_formula


CASES = (
    ("large_12k", "C500H800N125O200S10"),
    ("huge_60k", "C2500H4000N650O900S50"),
    ("huge_240k", "C10000H16000N2600O3600S200"),
)


def main() -> None:
    args = parse_args()
    if not has_cython_backend():
        raise RuntimeError("parallel worker benchmark requires the Cython backend")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_benchmark(args)
    csv_path = output_dir / "parallel_workers_benchmark.csv"
    md_path = output_dir / "parallel_workers_benchmark.md"
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
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1, 10, 50, 100])
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
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
            resolving_power = None if resolving_power_arg == 0.0 else resolving_power_arg
            for batch_size in args.batch_size:
                formulas = make_formulas(formula, batch_size)
                table = CenteredLogPhaseTable.build_for_formulas(
                    formulas,
                    elements=("C", "H", "N", "O", "S"),
                    dm=args.dm,
                    min_fft_len=args.min_fft_len,
                    safety_sigma=args.safety_sigma,
                    resolving_power=resolving_power,
                    storage_mode="production",
                )
                counts = table.counts_from_formulas(formulas)
                selected_method = table.select_spectrum_method(
                    counts,
                    method="cython_auto",
                    resolving_power=resolving_power,
                )
                reference = table.residual_spectrum_many_counts(
                    counts,
                    method=selected_method,
                    resolving_power=resolving_power,
                    workers=1,
                )
                serial_timing = time_call(
                    lambda: table.residual_spectrum_many_counts(
                        counts,
                        method=selected_method,
                        resolving_power=resolving_power,
                        workers=1,
                    ),
                    repeats=args.repeats,
                    warmups=args.warmups,
                )
                active_fraction = float(np.mean(table.active_frequency_fraction(
                    counts,
                    resolving_power=resolving_power,
                )))
                mean_mass = float(np.mean(table.mean_mass_many_counts(counts)))
                for workers in args.workers:
                    timing = serial_timing if workers == 1 else time_call(
                        lambda workers=workers: table.residual_spectrum_many_counts(
                            counts,
                            method=selected_method,
                            resolving_power=resolving_power,
                            workers=workers,
                        ),
                        repeats=args.repeats,
                        warmups=args.warmups,
                    )
                    spectrum = reference if workers == 1 else table.residual_spectrum_many_counts(
                        counts,
                        method=selected_method,
                        resolving_power=resolving_power,
                        workers=workers,
                    )
                    rows.append({
                        "case": case_name,
                        "formula": formula,
                        "batch_size": batch_size,
                        "resolving_power": 0.0 if resolving_power is None else float(resolving_power),
                        "workers": workers,
                        "selected_method": selected_method,
                        "dm": table.dm,
                        "n_fft": table.n_fft,
                        "n_positive": table.n_positive,
                        "mean_mass_da": mean_mass,
                        "active_fraction": active_fraction,
                        "median_s": timing["median_s"],
                        "min_s": timing["min_s"],
                        "max_s": timing["max_s"],
                        "speedup_vs_1_worker": serial_timing["median_s"] / timing["median_s"],
                        "rel_l2_vs_1_worker": relative_l2(spectrum, reference),
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
        (row["case"], row["batch_size"], row["resolving_power"], row["workers"]): row
        for row in rows
    }
    groups = sorted(
        {(row["case"], row["batch_size"], row["resolving_power"]) for row in rows},
        key=lambda item: (str(item[0]), int(item[1]), float(item[2])),
    )
    worker_values = sorted({int(row["workers"]) for row in rows})
    lines = [
        "# Parallel Workers Benchmark",
        "",
        f"- Python: {platform.python_version()}",
        f"- Platform: {platform.platform()}",
        "- Measured kernel: residual spectrum generation only.",
        "- Parallelization is over formula rows; single-formula requests usually do not benefit.",
        "- Speedup is relative to the same selected kernel with `workers=1`.",
        "",
        "| case | batch | R | n_fft | active | selected | "
        + " | ".join(f"{workers} worker s" for workers in worker_values)
        + " | "
        + " | ".join(f"{workers} worker speedup" for workers in worker_values)
        + " | max rel L2 |",
        "| --- | ---: | ---: | ---: | ---: | --- | "
        + " | ".join("---:" for _ in worker_values)
        + " | "
        + " | ".join("---:" for _ in worker_values)
        + " | ---: |",
    ]
    for case_name, batch_size, resolving_power in groups:
        records = [index[(case_name, batch_size, resolving_power, workers)] for workers in worker_values]
        base = records[0]
        max_rel_l2 = max(float(row["rel_l2_vs_1_worker"]) for row in records)
        times = " | ".join(f"{float(row['median_s']):.6f}" for row in records)
        speedups = " | ".join(f"{float(row['speedup_vs_1_worker']):.3f}x" for row in records)
        lines.append(
            "| "
            f"{case_name} | {int(batch_size)} | {float(resolving_power):.0f} | "
            f"{base['n_fft']} | {float(base['active_fraction']):.4g} | "
            f"{base['selected_method']} | {times} | {speedups} | {max_rel_l2:.2e} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
