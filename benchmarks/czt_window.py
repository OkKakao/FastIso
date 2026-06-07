from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

import numpy as np

from fastiso import CenteredLogPhaseTable, czt_irfft_window, fast_odd_irfft, has_cython_backend


CASES = (
    ("large_12k", "C500H800N125O200S10"),
    ("xlarge_29k", "C1200H1900N320O450S25"),
    ("huge_60k", "C2500H4000N650O900S50"),
    ("huge_120k", "C5000H8000N1300O1800S100"),
    ("huge_240k", "C10000H16000N2600O3600S200"),
)
WINDOW_WIDTHS = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
OUTPUT_DMS = (0.002, 0.001, 0.0005)


def main() -> None:
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)
    rows = run_benchmark()
    csv_path = output_dir / "czt_window_benchmark.csv"
    md_path = output_dir / "czt_window_benchmark.md"
    write_csv(csv_path, rows)
    write_summary(md_path, rows)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


def run_benchmark() -> list[dict[str, str | int | float]]:
    method = "cython_auto" if has_cython_backend() else "log_pruned"
    rows: list[dict[str, str | int | float]] = []
    for case_name, formula in CASES:
        table = CenteredLogPhaseTable.build_for_formulas(
            [formula],
            elements=("C", "H", "N", "O", "S"),
            dm=0.002,
            min_fft_len=32768,
            safety_sigma=6.0,
            resolving_power=100_000,
        )
        counts = table.counts_from_formulas([formula])
        mean_mass = float(table.mean_mass_many_counts(counts)[0])
        active_fraction = float(table.active_frequency_fraction(
            counts,
            resolving_power=100_000,
        )[0])

        full_time = median_time(
            lambda: table.mass_profile_many_counts(
                counts,
                method=method,
                resolving_power=100_000,
            ),
        )
        spectrum_time = median_time(
            lambda: table.residual_spectrum_many_counts(
                counts,
                method=method,
                resolving_power=100_000,
            ),
        )
        spectrum = table.residual_spectrum_many_counts(
            counts,
            method=method,
            resolving_power=100_000,
        )
        full_transform_time = median_time(lambda: full_profile_transform(spectrum, table.n_fft))
        full_mass, full_profile, full_info = table.mass_profile_many_counts(
            counts,
            method=method,
            resolving_power=100_000,
        )

        for width in WINDOW_WIDTHS:
            residual_start = -0.5 * width
            residual_stop = 0.5 * width
            for output_dm in OUTPUT_DMS:
                czt_time = median_time(
                    lambda: table.mass_profile_window_many_counts(
                        counts,
                        residual_start=residual_start,
                        residual_stop=residual_stop,
                        output_dm=output_dm,
                        method=method,
                        resolving_power=100_000,
                    ),
                )
                czt_transform_time = median_time(
                    lambda: czt_window_transform(
                        spectrum,
                        n_fft=table.n_fft,
                        table_dm=table.dm,
                        residual_start=residual_start,
                        output_dm=output_dm,
                        points=window_point_count(width, output_dm),
                    )
                )
                mass_axis, profile, window_info = table.mass_profile_window_many_counts(
                    counts,
                    residual_start=residual_start,
                    residual_stop=residual_stop,
                    output_dm=output_dm,
                    method=method,
                    resolving_power=100_000,
                )
                rel_l2 = np.nan
                if np.isclose(output_dm, table.dm):
                    indices = np.searchsorted(full_mass[0], mass_axis[0])
                    rel_l2 = relative_l2(profile[0], full_profile[0, indices])
                rows.append(
                    {
                        "case": case_name,
                        "formula": formula,
                        "requested_method": method,
                        "method": str(window_info["method"]),
                        "spectrum_method": str(window_info["method"]),
                        "full_profile_method": str(full_info["method"]),
                        "n_fft": table.n_fft,
                        "table_dm": table.dm,
                        "mean_mass_da": mean_mass,
                        "mass_kda": mean_mass / 1000.0,
                        "active_fraction": active_fraction,
                        "window_width_da": width,
                        "output_dm": output_dm,
                        "points": profile.shape[-1],
                        "spectrum_s": spectrum_time,
                        "full_transform_s": full_transform_time,
                        "full_profile_s": full_time,
                        "czt_transform_s": czt_transform_time,
                        "czt_window_s": czt_time,
                        "full_over_czt": full_time / czt_time,
                        "full_transform_over_czt_transform": full_transform_time / czt_transform_time,
                        "rel_l2_vs_full_grid": rel_l2,
                    }
                )
    return rows


def median_time(fn, *, repeats: int = 15, warmups: int = 3) -> float:
    for _ in range(warmups):
        fn()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    return statistics.median(timings)


def full_profile_transform(spectrum: np.ndarray, n_fft: int) -> np.ndarray:
    profile = fast_odd_irfft(spectrum, n=n_fft, axis=-1)
    profile = np.fft.fftshift(profile.real, axes=-1)
    return profile[:, ::-1]


def czt_window_transform(
    spectrum: np.ndarray,
    *,
    n_fft: int,
    table_dm: float,
    residual_start: float,
    output_dm: float,
    points: int,
) -> np.ndarray:
    return czt_irfft_window(
        spectrum,
        n=n_fft,
        start=-residual_start / table_dm,
        step=-output_dm / table_dm,
        m=points,
        axis=-1,
        trim_zeros=True,
    )


def window_point_count(width: float, output_dm: float) -> int:
    return int(np.floor(width / output_dm + 0.5)) + 1


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    lines = [
        "# CZT Window Benchmark",
        "",
        "- Runtime includes residual spectrum generation plus profile transform.",
        "- `requested_method` is the public method passed to FastIso; `spectrum_method` is the selected internal spectrum kernel.",
        "- Transform-only columns assume the residual spectrum is already computed.",
        "- `full_profile_s` computes the complete dense profile.",
        "- `czt_window_s` computes only the requested residual mass window.",
        "- Grid-matched rows report relative L2 against the same full-profile samples.",
        "",
        "| case | method | n_fft | active | window Da | output dm | points | full profile s | czt window s | full/czt | xform full/czt | rel L2 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            " | ".join(
                [
                    f"{row['case']}",
                    f"{row['spectrum_method']}",
                    f"{row['n_fft']}",
                    f"{row['active_fraction']:.4g}",
                    f"{row['window_width_da']:.3g}",
                    f"{row['output_dm']:.4g}",
                    f"{row['points']}",
                    f"{row['full_profile_s']:.6f}",
                    f"{row['czt_window_s']:.6f}",
                    f"{row['full_over_czt']:.2f}x",
                    f"{row['full_transform_over_czt_transform']:.2f}x",
                    format_metric(row["rel_l2_vs_full_grid"]),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n")


def format_metric(value: str | int | float) -> str:
    if isinstance(value, float) and np.isnan(value):
        return "NA"
    return f"{float(value):.2e}"


if __name__ == "__main__":
    main()
