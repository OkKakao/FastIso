"""Push FastIso to ultra-large CHNOS formulas.

This benchmark is intended to answer how far the current FT/log-table/CZT path
can be pushed when the table is built in production storage. It records table
construction time, table memory, residual generation, local CZT window profile,
and single-formula full dense profile time.
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


BASE_240K = {"C": 10000, "H": 16000, "N": 2600, "O": 3600, "S": 200}


@dataclass(frozen=True)
class UltraCase:
    name: str
    scale: int
    counts: dict[str, int]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_benchmark(args)
    csv_path = output_dir / "ultra_scale_benchmark.csv"
    md_path = output_dir / "ultra_scale_benchmark.md"
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
    parser.add_argument("--resolving-power", type=float, default=100_000.0)
    parser.add_argument("--scales", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--window-width-da", type=float, default=0.2)
    parser.add_argument("--output-dm", type=float, default=0.002)
    parser.add_argument("--skip-full-profile", action="store_true")
    return parser.parse_args()


def run_benchmark(args: argparse.Namespace) -> list[dict[str, str | int | float | bool]]:
    rows: list[dict[str, str | int | float | bool]] = []
    for case in make_cases(args.scales):
        formula = format_formula(case.counts)
        formulas = make_batch_formulas(case.counts, args.batch_size)

        build_started = time.perf_counter()
        table = CenteredLogPhaseTable.build_for_formulas(
            formulas,
            elements=("C", "H", "N", "O", "S"),
            dm=args.dm,
            min_fft_len=args.min_fft_len,
            safety_sigma=args.safety_sigma,
            resolving_power=args.resolving_power,
            storage_mode="production",
        )
        build_s = time.perf_counter() - build_started

        counts = table.counts_from_formulas(formulas)
        selected_method = table.select_spectrum_method(
            counts,
            method="cython_auto",
            resolving_power=args.resolving_power,
        )
        mean_mass = float(table.mean_mass_many_counts(counts[:1])[0])
        profile_sigma = float(table.profile_sigma_many_counts(
            counts[:1],
            resolving_power=args.resolving_power,
        )[0])
        active_fraction = float(np.mean(table.active_frequency_fraction(
            counts,
            resolving_power=args.resolving_power,
        )))

        residual_1w = time_call(
            lambda: table.residual_spectrum_many_counts(
                counts,
                method=selected_method,
                resolving_power=args.resolving_power,
                workers=1,
            ),
            repeats=args.repeats,
            warmups=args.warmups,
        )
        residual_nw = time_call(
            lambda: table.residual_spectrum_many_counts(
                counts,
                method=selected_method,
                resolving_power=args.resolving_power,
                workers=args.workers,
            ),
            repeats=args.repeats,
            warmups=args.warmups,
        )
        residual_ref = table.residual_spectrum_many_counts(
            counts,
            method=selected_method,
            resolving_power=args.resolving_power,
            workers=1,
        )
        residual_parallel = table.residual_spectrum_many_counts(
            counts,
            method=selected_method,
            resolving_power=args.resolving_power,
            workers=args.workers,
        )

        half_window = 0.5 * args.window_width_da
        czt_time = time_call(
            lambda: table.mass_profile_window_many_counts(
                counts,
                residual_start=-half_window,
                residual_stop=half_window,
                output_dm=args.output_dm,
                method=selected_method,
                resolving_power=args.resolving_power,
                workers=args.workers,
            ),
            repeats=args.repeats,
            warmups=args.warmups,
        )

        full_profile_s = float("nan")
        if not args.skip_full_profile:
            full_profile_s = time_call(
                lambda: table.mass_profile_many_counts(
                    counts[:1],
                    method=selected_method,
                    resolving_power=args.resolving_power,
                    workers=args.workers,
                ),
                repeats=args.repeats,
                warmups=args.warmups,
            )["median_s"]

        rows.append({
            "case": case.name,
            "scale_vs_240k": case.scale,
            "formula": formula,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "selected_method": selected_method,
            "mean_mass_da": mean_mass,
            "mass_mda": mean_mass / 1_000_000.0,
            "profile_sigma_da": profile_sigma,
            "dm": table.dm,
            "resolving_power": args.resolving_power,
            "n_fft": table.n_fft,
            "n_positive": table.n_positive,
            "window_width_da": table.n_fft * table.dm,
            "active_fraction": active_fraction,
            "table_nbytes": table.table_nbytes,
            "table_mib": table.table_nbytes / 1024.0 / 1024.0,
            "build_s": build_s,
            "residual_1_worker_s": residual_1w["median_s"],
            "residual_parallel_s": residual_nw["median_s"],
            "parallel_speedup": residual_1w["median_s"] / residual_nw["median_s"],
            "parallel_rel_l2": relative_l2(residual_parallel, residual_ref),
            "czt_window_s": czt_time["median_s"],
            "full_profile_single_s": full_profile_s,
            "repeats": args.repeats,
        })
    return rows


def make_cases(scales: list[int]) -> list[UltraCase]:
    cases: list[UltraCase] = []
    for scale in scales:
        counts = {element: count * scale for element, count in BASE_240K.items()}
        cases.append(UltraCase(f"ultra_{scale}x_240k", scale, counts))
    return cases


def make_batch_formulas(base_counts: dict[str, int], batch_size: int) -> list[str]:
    steps = {"C": 3, "H": 5, "N": 1, "O": 1, "S": 1}
    formulas: list[str] = []
    for i in range(batch_size):
        counts = {
            element: count + i * steps.get(element, 0)
            for element, count in base_counts.items()
        }
        formulas.append(format_formula(counts))
    return formulas


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
    denom = np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(a - b) / denom)


def write_csv(path: Path, rows: list[dict[str, str | int | float | bool]]) -> None:
    if not rows:
        raise ValueError("no benchmark rows")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, str | int | float | bool]]) -> None:
    lines = [
        "# Ultra-Scale Push Benchmark",
        "",
        f"- Python: {platform.python_version()}",
        f"- Platform: {platform.platform()}",
        f"- Cython backend: {has_cython_backend()}",
        "- Production storage is used for all tables.",
        "- Residual timings use the selected Cython auto kernel.",
        "- CZT window timing includes residual spectrum generation plus local profile transform.",
        "",
        "| case | mass MDa | n_fft | table MiB | build s | active | residual 1w s | residual parallel s | speedup | CZT window s | full profile 1 formula s | rel L2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['case']} | "
            f"{float(row['mass_mda']):.3f} | "
            f"{row['n_fft']} | "
            f"{float(row['table_mib']):.1f} | "
            f"{float(row['build_s']):.3f} | "
            f"{float(row['active_fraction']):.3g} | "
            f"{float(row['residual_1_worker_s']):.6f} | "
            f"{float(row['residual_parallel_s']):.6f} | "
            f"{float(row['parallel_speedup']):.2f}x | "
            f"{float(row['czt_window_s']):.6f} | "
            f"{format_float(row['full_profile_single_s'])} | "
            f"{float(row['parallel_rel_l2']):.2e} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_float(value: object) -> str:
    value = float(value)
    if np.isnan(value):
        return "NA"
    return f"{value:.6f}"


if __name__ == "__main__":
    main()
