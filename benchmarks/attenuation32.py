"""Benchmark float32 attenuation tables and production storage."""

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
    ("xlarge_29k", "C1200H1900N320O450S25"),
    ("huge_60k", "C2500H4000N650O900S50"),
    ("huge_120k", "C5000H8000N1300O1800S100"),
    ("huge_240k", "C10000H16000N2600O3600S200"),
)


def main() -> None:
    args = parse_args()
    if not has_cython_backend():
        raise RuntimeError("attenuation32 benchmark requires the Cython backend")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_benchmark(args)
    csv_path = output_dir / "attenuation32_benchmark.csv"
    md_path = output_dir / "attenuation32_benchmark.md"
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


def run_benchmark(args: argparse.Namespace) -> list[dict[str, str | int | float | bool]]:
    rows: list[dict[str, str | int | float | bool]] = []
    for case_name, formula in CASES:
        for resolving_power_arg in args.resolving_power:
            resolving_power = None if resolving_power_arg == 0.0 else resolving_power_arg
            for batch_size in args.batch_size:
                formulas = make_formulas(formula, batch_size)
                research64 = build_table(args, formulas, resolving_power=resolving_power)
                research32 = build_table(
                    args,
                    formulas,
                    resolving_power=resolving_power,
                    attenuation_dtype="float32",
                )
                production32 = build_table(
                    args,
                    formulas,
                    resolving_power=resolving_power,
                    storage_mode="production",
                )
                minimal32 = build_table(
                    args,
                    formulas,
                    resolving_power=resolving_power,
                    storage_mode="minimal",
                )
                counts = research64.counts_from_formulas(formulas)
                baseline = research64.residual_spectrum_many_counts(
                    counts,
                    method="cython_log_pruned",
                    resolving_power=resolving_power,
                )
                baseline_timing = time_call(
                    lambda: research64.residual_spectrum_many_counts(
                        counts,
                        method="cython_log_pruned",
                        resolving_power=resolving_power,
                    ),
                    repeats=args.repeats,
                    warmups=args.warmups,
                )
                specs = (
                    ("research64", research64, "cython_log_pruned", baseline_timing),
                    ("research32_phase64", research32, "cython_log_pruned_attn32", None),
                    (
                        "research32_uintphase",
                        research32,
                        "cython_log_pruned_attn32_uintphase",
                        None,
                    ),
                    ("production32_auto", production32, "cython_auto", None),
                    ("minimal32_auto", minimal32, "cython_auto", None),
                )
                mean_mass = float(np.mean(research64.mean_mass_many_counts(counts)))
                active_fraction = float(np.mean(research64.active_frequency_fraction(
                    counts,
                    resolving_power=resolving_power,
                )))
                for variant, table, method, precomputed_timing in specs:
                    selected_method = table.select_spectrum_method(
                        counts,
                        method=method,
                        resolving_power=resolving_power,
                    )
                    timed_method = selected_method if method in {"auto", "cython_auto"} else method
                    timing = precomputed_timing
                    if timing is None:
                        timing = time_call(
                            lambda table=table, timed_method=timed_method: table.residual_spectrum_many_counts(
                                counts,
                                method=timed_method,
                                resolving_power=resolving_power,
                            ),
                            repeats=args.repeats,
                            warmups=args.warmups,
                        )
                    spectrum = baseline if variant == "research64" else (
                        table.residual_spectrum_many_counts(
                            counts,
                            method=timed_method,
                            resolving_power=resolving_power,
                        )
                    )
                    rows.append({
                        "case": case_name,
                        "formula": formula,
                        "batch_size": batch_size,
                        "resolving_power": 0.0 if resolving_power is None else float(resolving_power),
                        "variant": variant,
                        "method": method,
                        "selected_method": selected_method,
                        "timed_method": timed_method,
                        "dm": table.dm,
                        "n_fft": table.n_fft,
                        "n_positive": table.n_positive,
                        "mean_mass_da": mean_mass,
                        "active_fraction": active_fraction,
                        "storage_mode": table.storage_mode,
                        "attenuation_dtype": str(table.attenuation.dtype),
                        "has_phase_table": table.has_phase_table,
                        "table_nbytes": table.table_nbytes,
                        "memory_vs_research64": table.table_nbytes / research64.table_nbytes,
                        "median_s": timing["median_s"],
                        "min_s": timing["min_s"],
                        "max_s": timing["max_s"],
                        "research64_over_variant": baseline_timing["median_s"] / timing["median_s"],
                        "rel_l2_vs_research64": relative_l2(spectrum, baseline),
                        "max_abs_vs_research64": float(np.max(np.abs(spectrum - baseline))),
                        "repeats": args.repeats,
                    })
    return rows


def build_table(
    args: argparse.Namespace,
    formulas: list[str],
    *,
    resolving_power: float | None,
    attenuation_dtype: str | None = None,
    storage_mode: str = "research",
) -> CenteredLogPhaseTable:
    return CenteredLogPhaseTable.build_for_formulas(
        formulas,
        elements=("C", "H", "N", "O", "S"),
        dm=args.dm,
        min_fft_len=args.min_fft_len,
        safety_sigma=args.safety_sigma,
        resolving_power=resolving_power,
        attenuation_dtype=attenuation_dtype,
        storage_mode=storage_mode,
    )


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


def write_csv(path: Path, rows: list[dict[str, str | int | float | bool]]) -> None:
    if not rows:
        raise ValueError("no benchmark rows to write")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, str | int | float | bool]]) -> None:
    index = {
        (row["case"], row["batch_size"], row["resolving_power"], row["variant"]): row
        for row in rows
    }
    groups = sorted(
        {(row["case"], row["batch_size"], row["resolving_power"]) for row in rows},
        key=lambda item: (str(item[0]), int(item[1]), float(item[2])),
    )
    lines = [
        "# Float32 Attenuation And Production Storage Benchmark",
        "",
        f"- Python: {platform.python_version()}",
        f"- Platform: {platform.platform()}",
        "- Baseline is research storage with float64 attenuation and double phase.",
        "- `research32_phase64` isolates the float32 attenuation effect while keeping double phase.",
        "- `production32_auto` stores float32 attenuation, double phase, uint64 phase, and thresholds; it discards cyclephase-only tables.",
        "- `minimal32_auto` stores float32 attenuation plus uint64 phase and thresholds; it discards double phase tables.",
        "- Auto variants are timed after resolving the selected kernel, matching the table/profile call path where selection happens once.",
        "- Speed ratios above 1 mean the variant kernel is faster than the research64 baseline.",
        "",
        "| case | batch | R | n_fft | active | research64 s | attn32 s | attn32+uint s | production s | minimal s | prod selected | min selected | prod memory | min memory | prod rel L2 | prod speed | min speed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case_name, batch_size, resolving_power in groups:
        baseline = index[(case_name, batch_size, resolving_power, "research64")]
        attn32 = index[(case_name, batch_size, resolving_power, "research32_phase64")]
        uintphase = index[(case_name, batch_size, resolving_power, "research32_uintphase")]
        production = index[(case_name, batch_size, resolving_power, "production32_auto")]
        minimal = index[(case_name, batch_size, resolving_power, "minimal32_auto")]
        lines.append(
            "| "
            f"{case_name} | {int(batch_size)} | {float(resolving_power):.0f} | "
            f"{baseline['n_fft']} | {float(baseline['active_fraction']):.4g} | "
            f"{float(baseline['median_s']):.6f} | "
            f"{float(attn32['median_s']):.6f} | "
            f"{float(uintphase['median_s']):.6f} | "
            f"{float(production['median_s']):.6f} | "
            f"{float(minimal['median_s']):.6f} | "
            f"{production['selected_method']} | "
            f"{minimal['selected_method']} | "
            f"{float(production['memory_vs_research64']):.3f}x | "
            f"{float(minimal['memory_vs_research64']):.3f}x | "
            f"{float(production['rel_l2_vs_research64']):.2e} | "
            f"{float(production['research64_over_variant']):.3f}x |"
            f" {float(minimal['research64_over_variant']):.3f}x |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
