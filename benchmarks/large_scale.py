"""Large-molecule scaling benchmark for FastIso.

This script focuses on the FastIso internal methods rather than external package
comparisons. It is intended to test whether active-frequency pruning continues
to help for very large molecules.
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

from fastiso import CenteredLogPhaseTable, has_cython_backend


DEFAULT_OUTPUT_DIR = Path("benchmark_results")


@dataclass(frozen=True)
class LargeCase:
    name: str
    counts: dict[str, int]


CASES = (
    LargeCase("large_12k", {"C": 500, "H": 800, "N": 125, "O": 200, "S": 10}),
    LargeCase("xlarge_29k", {"C": 1200, "H": 1900, "N": 320, "O": 450, "S": 25}),
    LargeCase("huge_60k", {"C": 2500, "H": 4000, "N": 650, "O": 900, "S": 50}),
    LargeCase("huge_120k", {"C": 5000, "H": 8000, "N": 1300, "O": 1800, "S": 100}),
    LargeCase("huge_240k", {"C": 10000, "H": 16000, "N": 2600, "O": 3600, "S": 200}),
)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | float | int]] = []
    for case in CASES:
        formulas = [format_formula(case.counts) for _ in range(args.batch_size)]
        if args.auto_window:
            table = CenteredLogPhaseTable.build_for_formulas(
                formulas,
                elements=("C", "H", "N", "O", "S"),
                dm=args.dm,
                min_fft_len=args.min_fft_len,
                safety_sigma=args.safety_sigma,
                resolving_power=args.resolving_power,
            )
        else:
            table = CenteredLogPhaseTable.build(
                elements=("C", "H", "N", "O", "S"),
                dm=args.dm,
                min_fft_len=args.min_fft_len,
            )
        counts = table.counts_from_formulas(formulas)
        mean_mass = float(np.mean(table.mean_mass_many_counts(counts)))
        active_fraction = float(np.mean(
            table.active_frequency_fraction(
                counts,
                resolving_power=args.resolving_power,
            )
        ))
        methods = ["log_table", "log_pruned"]
        if has_cython_backend():
            methods.append("cython_log_pruned")
        if args.include_direct:
            methods.append("direct_rebuild")
        reference = table.residual_spectrum_many_counts(
            counts[:1],
            method="log_table",
            resolving_power=args.resolving_power,
        )
        for method in methods:
            repeats = args.direct_repeats if method == "direct_rebuild" else args.repeats
            timing = time_call(
                lambda method=method: table.residual_spectrum_many_counts(
                    counts,
                    method=method,
                    resolving_power=args.resolving_power,
                ),
                repeats=repeats,
                warmups=args.warmups,
            )
            sample = table.residual_spectrum_many_counts(
                counts[:1],
                method=method,
                resolving_power=args.resolving_power,
            )
            rows.append({
                "case": case.name,
                "method": method,
                "batch_size": args.batch_size,
                "auto_window": args.auto_window,
                "mean_mass_da": mean_mass,
                "active_fraction": active_fraction,
                "resolving_power": args.resolving_power,
                "dm": table.dm,
                "n_fft": table.n_fft,
                "n_positive": table.n_positive,
                "median_s": timing["median_s"],
                "min_s": timing["min_s"],
                "max_s": timing["max_s"],
                "repeats": repeats,
                "rel_l2_vs_full": relative_l2(sample, reference),
            })

    csv_path = output_dir / "large_scale_benchmark.csv"
    md_path = output_dir / "large_scale_benchmark.md"
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
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--direct-repeats", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--include-direct", action="store_true")
    parser.add_argument("--auto-window", action="store_true")
    parser.add_argument("--safety-sigma", type=float, default=6.0)
    return parser.parse_args()


def format_formula(counts: dict[str, int]) -> str:
    return "".join(f"{element}{count}" for element, count in counts.items() if count > 0)


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
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def write_csv(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    if not rows:
        raise ValueError("no benchmark rows")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    index = {
        (row["case"], row["method"]): row
        for row in rows
    }
    lines = [
        "# Large-Scale Benchmark",
        "",
        f"- Cython backend: {has_cython_backend()}",
        "- Table construction time is excluded.",
        "- Runtime is residual spectrum generation, not final irFFT profile generation.",
        "",
        "| case | auto | n_fft | window Da | mean mass Da | active | log_full s | cython_pruned s | cython/full | rel L2 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    cases = sorted({row["case"] for row in rows}, key=lambda case: float(index[(case, "log_table")]["mean_mass_da"]))
    for case in cases:
        full = index[(case, "log_table")]
        fast = index.get((case, "cython_log_pruned")) or index[(case, "log_pruned")]
        full_s = float(full["median_s"])
        fast_s = float(fast["median_s"])
        lines.append(
            "| "
            f"{case} | {bool(full['auto_window'])} | {full['n_fft']} | "
            f"{float(full['n_fft']) * float(full['dm']):.1f} | "
            f"{float(full['mean_mass_da']):.1f} | "
            f"{float(full['active_fraction']):.4f} | "
            f"{full_s:.6f} | {fast_s:.6f} | "
            f"{full_s / fast_s:.2f}x | {float(fast['rel_l2_vs_full']):.2e} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
