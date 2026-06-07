"""Audit whether FFT mass windows contain large-molecule isotope envelopes."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fastiso import CenteredLogPhaseTable


DEFAULT_OUTPUT_DIR = Path("benchmark_results")


@dataclass(frozen=True)
class WindowCase:
    name: str
    counts: dict[str, int]


CASES = (
    WindowCase("large_12k", {"C": 500, "H": 800, "N": 125, "O": 200, "S": 10}),
    WindowCase("xlarge_29k", {"C": 1200, "H": 1900, "N": 320, "O": 450, "S": 25}),
    WindowCase("huge_60k", {"C": 2500, "H": 4000, "N": 650, "O": 900, "S": 50}),
    WindowCase("huge_120k", {"C": 5000, "H": 8000, "N": 1300, "O": 1800, "S": 100}),
    WindowCase("huge_240k", {"C": 10000, "H": 16000, "N": 2600, "O": 3600, "S": 200}),
)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | float | int | bool]] = []
    for min_fft_len in args.min_fft_len:
        for case in CASES:
            formula = format_formula(case.counts)
            if args.auto_window:
                table = CenteredLogPhaseTable.build_for_formulas(
                    [formula],
                    elements=("C", "H", "N", "O", "S"),
                    dm=args.dm,
                    min_fft_len=min_fft_len,
                    safety_sigma=args.sigma_radius,
                    resolving_power=args.resolving_power,
                )
            else:
                table = CenteredLogPhaseTable.build(
                    elements=("C", "H", "N", "O", "S"),
                    dm=args.dm,
                    min_fft_len=min_fft_len,
                )
            half_window = 0.5 * table.n_fft * table.dm
            edge_bins = max(1, int(np.ceil(args.edge_width_da / table.dm)))
            counts = table.counts_from_formulas([formula])
            mean_mass = float(table.mean_mass_many_counts(counts)[0])
            isotope_sigma = float(np.sqrt(table.isotope_variance_many_counts(counts)[0]))
            profile_sigma = float(table.profile_sigma_many_counts(
                counts,
                resolving_power=args.resolving_power,
            )[0])
            required_half_width = args.sigma_radius * profile_sigma
            _, profiles, _ = table.mass_profile_many_counts(
                counts,
                method=args.method,
                resolving_power=args.resolving_power,
            )
            profile = profiles[0]
            profile = np.maximum(profile, 0.0)
            total = float(profile.sum())
            left_edge_fraction = float(profile[:edge_bins].sum() / total)
            right_edge_fraction = float(profile[-edge_bins:].sum() / total)
            edge_fraction = left_edge_fraction + right_edge_fraction
            rows.append({
                "case": case.name,
                "formula": formula,
                "min_fft_len": min_fft_len,
                "auto_window": args.auto_window,
                "n_fft": table.n_fft,
                "dm": table.dm,
                "window_width_da": table.n_fft * table.dm,
                "half_window_da": half_window,
                "mean_mass_da": mean_mass,
                "isotope_sigma_da": isotope_sigma,
                "profile_sigma_da": profile_sigma,
                "sigma_radius": args.sigma_radius,
                "required_half_width_da": required_half_width,
                "sigma_radius_fits": required_half_width < half_window,
                "edge_width_da": args.edge_width_da,
                "edge_fraction": edge_fraction,
                "left_edge_fraction": left_edge_fraction,
                "right_edge_fraction": right_edge_fraction,
                "edge_ok": edge_fraction < args.edge_threshold,
            })

    csv_path = output_dir / "window_audit.csv"
    md_path = output_dir / "window_audit.md"
    write_csv(csv_path, rows)
    write_summary(md_path, rows, args.edge_threshold)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dm", type=float, default=0.002)
    parser.add_argument("--min-fft-len", type=int, nargs="+", default=[32768, 65536, 131072])
    parser.add_argument("--resolving-power", type=float, default=100_000.0)
    parser.add_argument("--method", default="cython_log_pruned")
    parser.add_argument("--sigma-radius", type=float, default=6.0)
    parser.add_argument("--edge-width-da", type=float, default=1.0)
    parser.add_argument("--edge-threshold", type=float, default=1e-6)
    parser.add_argument("--auto-window", action="store_true")
    return parser.parse_args()


def format_formula(counts: dict[str, int]) -> str:
    return "".join(f"{element}{count}" for element, count in counts.items() if count > 0)


def write_csv(path: Path, rows: list[dict[str, str | float | int | bool]]) -> None:
    if not rows:
        raise ValueError("no rows")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, str | float | int | bool]], edge_threshold: float) -> None:
    lines = [
        "# FFT Window Audit",
        "",
        f"- Edge threshold: {edge_threshold:g}",
        "- `sigma_radius_fits` checks whether +/- sigma_radius * profile_sigma fits within half the FFT mass window.",
        "- `edge_fraction` is the fraction of profile area in the first plus last edge-width bins.",
        "",
        "| case | min fft | auto | n_fft | window Da | mean mass Da | profile sigma Da | 6sigma halfwidth Da | fits | edge fraction | edge ok |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['case']} | {row['min_fft_len']} | {bool(row['auto_window'])} | "
            f"{row['n_fft']} | {float(row['window_width_da']):.1f} | "
            f"{float(row['mean_mass_da']):.1f} | {float(row['profile_sigma_da']):.3f} | "
            f"{float(row['required_half_width_da']):.2f} | "
            f"{bool(row['sigma_radius_fits'])} | {float(row['edge_fraction']):.2e} | {bool(row['edge_ok'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
