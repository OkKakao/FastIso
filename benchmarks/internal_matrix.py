"""Internal benchmark matrix for log-table isotope spectra.

The benchmark measures runtime after a reusable table has already been built.
It is intended to locate the crossover between full log-table evaluation and
active-frequency pruning before comparing against external isotope tools.
"""

from __future__ import annotations

import argparse
import csv
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fastiso import CenteredLogPhaseTable, has_cython_backend


DEFAULT_OUTPUT_DIR = Path("benchmark_results")
METHODS = ("log_table", "log_pruned", "cython_log_pruned", "direct_rebuild")


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
    FormulaFamily(
        name="xlarge",
        base_counts={"C": 1200, "H": 1900, "N": 320, "O": 450, "S": 25},
        step_counts={"C": 4, "H": 6, "N": 2, "O": 2, "S": 1},
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
    )

    rows: list[dict[str, str | float | int]] = []
    for resolving_power in args.resolving_power:
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
                        resolving_power=resolving_power,
                    )
                else:
                    table = base_table
                counts = table.counts_from_formulas(formulas)
                direct_reference = table.residual_spectrum_many_counts(
                    counts[: min(3, batch_size)],
                    method="direct_rebuild",
                    resolving_power=resolving_power,
                )
                full_reference = table.residual_spectrum_many_counts(
                    counts[: min(3, batch_size)],
                    method="log_table",
                    resolving_power=resolving_power,
                )

                for method in METHODS:
                    if method == "cython_log_pruned" and not has_cython_backend():
                        continue
                    repeats = args.direct_repeats if method == "direct_rebuild" else args.repeats
                    timing = time_call(
                        lambda method=method: table.residual_spectrum_many_counts(
                            counts,
                            method=method,
                            resolving_power=resolving_power,
                        ),
                        repeats=repeats,
                        warmups=args.warmups,
                    )

                    sample_spectrum = table.residual_spectrum_many_counts(
                        counts[: min(3, batch_size)],
                        method=method,
                        resolving_power=resolving_power,
                    )
                    rel_l2_vs_direct = relative_l2(sample_spectrum, direct_reference)
                    rel_l2_vs_full = relative_l2(sample_spectrum, full_reference)

                    mean_mass = float(np.mean(table.mean_mass_many_counts(counts)))
                    active_fraction = float(np.mean(
                        table.active_frequency_fraction(
                            counts,
                            resolving_power=resolving_power,
                        )
                    ))
                    rows.append(
                        {
                            "method": method,
                            "family": family.name,
                            "batch_size": batch_size,
                            "auto_window": args.auto_window,
                            "mean_mass_da": mean_mass,
                            "active_fraction": active_fraction,
                            "resolving_power": resolving_power,
                            "dm": table.dm,
                            "n_fft": table.n_fft,
                            "n_positive": table.n_positive,
                            "median_s": timing["median_s"],
                            "min_s": timing["min_s"],
                            "max_s": timing["max_s"],
                            "repeats": repeats,
                            "rel_l2_vs_direct": rel_l2_vs_direct,
                            "rel_l2_vs_full": rel_l2_vs_full,
                        }
                    )

    csv_path = output_dir / "internal_log_table_benchmark.csv"
    write_csv(csv_path, rows)
    summary_path = output_dir / "internal_log_table_benchmark.md"
    write_summary(summary_path, rows)
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dm", type=float, default=0.002)
    parser.add_argument("--min-fft-len", type=int, default=32768)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--direct-repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--auto-window", action="store_true")
    parser.add_argument("--safety-sigma", type=float, default=6.0)
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1, 10, 50])
    parser.add_argument(
        "--resolving-power",
        type=float,
        nargs="+",
        default=[50_000.0, 100_000.0, 200_000.0],
    )
    return parser.parse_args()


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


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def write_csv(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    if not rows:
        raise ValueError("no benchmark rows to write")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    rows: list[dict[str, str | float | int]],
) -> None:
    index = {
        (
            row["family"],
            row["batch_size"],
            row["resolving_power"],
            row["method"],
        ): row
        for row in rows
    }
    lines = [
        "# Internal Log-Table Benchmark",
        "",
        f"- Python: {platform.python_version()}",
        f"- Platform: {platform.platform()}",
        f"- Cython backend: {has_cython_backend()}",
        "- Table construction time is excluded.",
        "",
        "| family | batch | R | auto | n_fft | active | log_full s | cython_pruned s | direct s | cython/full | direct/cython |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    combinations = sorted(
        {
            (row["family"], row["batch_size"], row["resolving_power"])
            for row in rows
        },
        key=lambda item: (str(item[0]), int(item[1]), float(item[2])),
    )
    for family, batch_size, resolving_power in combinations:
        full = index.get((family, batch_size, resolving_power, "log_table"))
        cython = index.get((family, batch_size, resolving_power, "cython_log_pruned"))
        pruned = index.get((family, batch_size, resolving_power, "log_pruned"))
        direct = index.get((family, batch_size, resolving_power, "direct_rebuild"))
        fast = cython or pruned
        if full is None or fast is None or direct is None:
            continue
        full_s = float(full["median_s"])
        fast_s = float(fast["median_s"])
        direct_s = float(direct["median_s"])
        lines.append(
            "| "
            f"{family} | {batch_size} | {float(resolving_power):.0f} | "
            f"{bool(full['auto_window'])} | {full['n_fft']} | "
            f"{float(full['active_fraction']):.3f} | "
            f"{full_s:.6f} | {fast_s:.6f} | {direct_s:.6f} | "
            f"{full_s / fast_s:.2f}x | {direct_s / fast_s:.2f}x |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
