"""Compare log-table runtime with direct element FT rebuild.

This benchmark excludes table construction time. That matches the intended
workflow: build or load a reusable element resource once, then evaluate many
candidate formulas against the same mass grid.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from fastiso import CenteredLogPhaseTable, has_cython_backend


def main() -> None:
    resolving_power = 100_000.0
    table = CenteredLogPhaseTable.build(
        elements=("C", "H", "N", "O", "S"),
        dm=0.002,
        min_fft_len=32768,
    )
    print(f"n_fft={table.n_fft} n_positive={table.n_positive} dm={table.dm}")
    print(f"gaussian resolving_power={resolving_power:g}")
    print(f"cython_backend={has_cython_backend()}")
    print("table construction is excluded from timings")
    print()
    print(
        "case\tbatch\tmean_mass_da\tactive_frac\t"
        "log_full_s\tlog_pruned_s\tcython_pruned_s\tdirect_s\t"
        "pruned_vs_full\tfull_speedup\tpruned_speedup\trel_l2"
    )

    for case_name, formulas in [
        ("small_b1", make_small_formulas(1)),
        ("small_b50", make_small_formulas(50)),
        ("large_b1", make_large_formulas(1)),
        ("large_b50", make_large_formulas(50)),
    ]:
        counts = table.counts_from_formulas(formulas)
        mean_mass = float(np.mean(table.mean_mass_many_counts(counts)))
        active_fraction = float(np.mean(
            table.active_frequency_fraction(
                counts,
                resolving_power=resolving_power,
            )
        ))

        log_time = median_time(
            lambda: table.residual_spectrum_many_counts(
                counts,
                resolving_power=resolving_power,
            )
        )
        pruned_time = median_time(
            lambda: table.residual_spectrum_many_counts(
                counts,
                method="log_pruned",
                resolving_power=resolving_power,
            )
        )
        cython_pruned_time = float("nan")
        if has_cython_backend():
            cython_pruned_time = median_time(
                lambda: table.residual_spectrum_many_counts(
                    counts,
                    method="cython_log_pruned",
                    resolving_power=resolving_power,
                )
            )
        direct_time = median_time(
            lambda: table.residual_spectrum_many_counts(
                counts,
                method="direct_rebuild",
                resolving_power=resolving_power,
            ),
            repeats=3,
        )

        log_spectrum = table.residual_spectrum_many_counts(
            counts[:3],
            resolving_power=resolving_power,
        )
        direct_spectrum = table.residual_spectrum_many_counts(
            counts[:3],
            method="direct_rebuild",
            resolving_power=resolving_power,
        )
        pruned_spectrum = table.residual_spectrum_many_counts(
            counts[:3],
            method="log_pruned",
            resolving_power=resolving_power,
        )
        rel_l2 = relative_l2(log_spectrum, direct_spectrum)
        pruned_rel_l2 = relative_l2(pruned_spectrum, log_spectrum)
        pruned_vs_full = log_time / pruned_time
        full_speedup = direct_time / log_time
        pruned_speedup = direct_time / pruned_time
        print(
            f"{case_name}\t{len(formulas)}\t{mean_mass:.1f}\t"
            f"{active_fraction:.3f}\t"
            f"{log_time:.6f}\t{pruned_time:.6f}\t"
            f"{_format_seconds(cython_pruned_time)}\t{direct_time:.6f}\t"
            f"{pruned_vs_full:.2f}x\t{full_speedup:.2f}x\t"
            f"{pruned_speedup:.2f}x\t{max(rel_l2, pruned_rel_l2):.2e}"
        )


def make_small_formulas(n: int) -> list[str]:
    formulas: list[str] = []
    for i in range(n):
        formulas.append(
            format_formula(
                {
                    "C": 6 + i % 12,
                    "H": 10 + (3 * i) % 24,
                    "N": i % 3,
                    "O": 1 + i % 5,
                    "S": i % 2,
                }
            )
        )
    return formulas


def make_large_formulas(n: int) -> list[str]:
    formulas: list[str] = []
    for i in range(n):
        formulas.append(
            "C"
            f"{500 + 3 * i}"
            "H"
            f"{800 + 5 * i}"
            "N"
            f"{125 + i % 11}"
            "O"
            f"{200 + i % 17}"
            "S"
            f"{10 + i % 5}"
        )
    return formulas


def format_formula(counts: dict[str, int]) -> str:
    return "".join(
        f"{element}{count}"
        for element, count in counts.items()
        if count > 0
    )


def median_time(fn: Callable[[], object], *, repeats: int = 7) -> float:
    times: list[float] = []
    for _ in range(2):
        fn()
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _format_seconds(value: float) -> str:
    if np.isnan(value):
        return "NA"
    return f"{value:.6f}"


if __name__ == "__main__":
    main()
