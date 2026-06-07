"""Generate publication-feasibility plots from benchmark CSV files."""

from __future__ import annotations

import math
import os
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_results"
PLOTS = RESULTS / "plots"
os.environ.setdefault("MPLCONFIGDIR", str(RESULTS / ".mplconfig"))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns



FONT_FAMILY = ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
MONO_FONT_FAMILY = ["Consolas", "DejaVu Sans Mono", "monospace"]

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLORS = {
    "fastiso": "#A3BEFA",
    "IsoSpecPy": "#C5CAD3",
    "brainpy": "#F0986E",
    "pyopenms": "#A3D576",
    "envipat": "#F390CA",
}

CASE_COLORS = {
    "large_12k": "#A3BEFA",
    "xlarge_29k": "#FFE15B",
    "huge_60k": "#F0986E",
    "huge_120k": "#A3D576",
    "huge_240k": "#F390CA",
}

EDGES = {
    "fastiso": "#2E4780",
    "IsoSpecPy": "#464C55",
    "brainpy": "#804126",
    "pyopenms": "#386411",
    "envipat": "#8A3A6F",
}


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    use_chart_theme()

    fine = pd.read_csv(RESULTS / "fine_structure_benchmark.csv")
    clusters = pd.read_csv(RESULTS / "fine_structure_clusters.csv")
    external = pd.read_csv(RESULTS / "external_package_benchmark.csv")
    large_scale = pd.read_csv(RESULTS / "large_scale_benchmark.csv")
    internal = pd.read_csv(RESULTS / "internal_log_table_benchmark.csv")
    czt = pd.read_csv(RESULTS / "czt_window_benchmark.csv")

    outputs = [
        plot_speed_accuracy(fine),
        plot_fine_time_breakdown(fine),
        plot_cluster_preservation(clusters),
        plot_large_scale(large_scale),
        plot_czt_window_crossover(czt),
        plot_czt_grid_matched_heatmap(czt),
        plot_external_batch10(external),
        plot_internal_speedup(internal),
    ]
    write_summary(fine, clusters, large_scale, external, czt, outputs)
    print("wrote:")
    for output in outputs:
        print(output)
    print(PLOTS / "publication_feasibility_plots.md")


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "figure.edgecolor": "none",
            "savefig.facecolor": TOKENS["surface"],
            "savefig.edgecolor": "none",
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.monospace": MONO_FONT_FAMILY,
            "patch.linewidth": 1.0,
        },
    )


def add_chart_header(
    fig: plt.Figure,
    ax: plt.Axes,
    title: str,
    subtitle: str,
    *,
    title_width: int = 82,
    subtitle_width: int = 118,
) -> None:
    title = textwrap.fill(title.strip(), width=title_width, break_long_words=False)
    subtitle = textwrap.fill(subtitle.strip(), width=subtitle_width, break_long_words=False)
    title_lines = title.count("\n") + 1
    subtitle_lines = subtitle.count("\n") + 1
    ax.set_title("")
    fig.subplots_adjust(
        top=max(0.62, 0.86 - 0.045 * (title_lines - 1) - 0.032 * (subtitle_lines - 1))
    )
    left = ax.get_position().x0
    fig.text(
        left,
        0.985,
        title,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color=TOKENS["ink"],
        linespacing=1.08,
    )
    fig.text(
        left,
        0.93 - 0.045 * (title_lines - 1),
        subtitle,
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
        linespacing=1.18,
    )
    sns.despine(ax=ax)


def save(fig: plt.Figure, name: str) -> Path:
    path = PLOTS / name
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def fine_long(fine: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in fine.iterrows():
        rows.append(
            {
                "case": row["case"],
                "backend": "fastiso",
                "time_s": row["fastiso_profile_s"],
                "rel_l2": row["fastiso_rel_l2_vs_isospec"],
                "peaks": np.nan,
                "local_maxima": row["fastiso_local_maxima"],
            }
        )
        rows.append(
            {
                "case": row["case"],
                "backend": "IsoSpecPy",
                "time_s": row["isospec_profile_s"],
                "rel_l2": 1e-6,
                "peaks": row["isospec_peak_count"],
                "local_maxima": row["isospec_local_maxima"],
            }
        )
        for backend in ("brainpy", "pyopenms", "envipat"):
            profile_key = f"{backend}_profile_s"
            if profile_key not in row or pd.isna(row[profile_key]):
                continue
            rows.append(
                {
                    "case": row["case"],
                    "backend": backend,
                    "time_s": row[profile_key],
                    "rel_l2": max(float(row[f"{backend}_rel_l2_vs_isospec"]), 1e-6),
                    "peaks": row[f"{backend}_peak_count"],
                    "local_maxima": row[f"{backend}_local_maxima"],
                }
            )
    return pd.DataFrame(rows)


def plot_speed_accuracy(fine: pd.DataFrame) -> Path:
    data = fine_long(fine)
    cases = ["small_glucose", "medium_2p4k", "large_12k"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8), sharex=True, sharey=True)
    for ax, case in zip(axes, cases, strict=True):
        part = data[data["case"] == case]
        for _, row in part.iterrows():
            backend = str(row["backend"])
            ax.scatter(
                row["rel_l2"],
                row["time_s"],
                s=92,
                color=COLORS[backend],
                edgecolor=EDGES[backend],
                linewidth=1.0,
                zorder=3,
            )
            ax.text(
                row["rel_l2"] * 1.08,
                row["time_s"] * 1.03,
                backend,
                fontsize=7.6,
                color=TOKENS["ink"],
            )
        ax.set_title(case.replace("_", " "), fontsize=9.5, color=TOKENS["ink"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", axis="both")
        ax.set_xlabel("Relative L2 vs IsoSpecPy")
    axes[0].set_ylabel("Dense profile time (s)")
    add_chart_header(
        fig,
        axes[0],
        "FastIso occupies the useful speed-accuracy corner for fine-structure profiles",
        "Same m/z grid, R=240,000, dm=0.0002; IsoSpecPy is plotted at 1e-6 L2 as the zero-error reference floor.",
    )
    return save(fig, "fine_structure_speed_accuracy.png")


def plot_fine_time_breakdown(fine: pd.DataFrame) -> Path:
    row = fine.loc[fine["case"] == "large_12k"].iloc[0]
    records = [
        {"backend": "fastiso", "component": "profile", "time_s": row["fastiso_profile_s"]},
        {"backend": "IsoSpecPy", "component": "peak", "time_s": row["isospec_peak_s"]},
        {"backend": "IsoSpecPy", "component": "convolve", "time_s": row["isospec_convolve_s"]},
        {"backend": "brainpy", "component": "peak", "time_s": row["brainpy_peak_s"]},
        {"backend": "brainpy", "component": "convolve", "time_s": row["brainpy_convolve_s"]},
        {"backend": "pyopenms", "component": "peak", "time_s": row["pyopenms_peak_s"]},
        {"backend": "pyopenms", "component": "convolve", "time_s": row["pyopenms_convolve_s"]},
        {"backend": "envipat", "component": "Rscript wrapper + peak", "time_s": row["envipat_peak_s"]},
        {"backend": "envipat", "component": "convolve", "time_s": row["envipat_convolve_s"]},
    ]
    data = pd.DataFrame(records)
    order = ["fastiso", "brainpy", "IsoSpecPy", "pyopenms", "envipat"]
    totals = data.groupby("backend")["time_s"].sum().reindex(order)
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    left = np.zeros(len(order))
    component_colors = {
        "profile": "#A3BEFA",
        "peak": "#FFE15B",
        "Rscript wrapper + peak": "#F390CA",
        "convolve": "#F0986E",
    }
    for component in ["profile", "peak", "Rscript wrapper + peak", "convolve"]:
        values = (
            data[data["component"] == component]
            .set_index("backend")["time_s"]
            .reindex(order)
            .fillna(0.0)
            .to_numpy()
        )
        bars = ax.barh(
            y,
            values,
            left=left,
            label=component,
            color=component_colors[component],
            edgecolor=TOKENS["ink"],
            linewidth=0.8,
        )
        for bar in bars:
            bar.set_alpha(0.95)
        left += values
    ax.set_yticks(y, order)
    ax.set_xscale("log")
    ax.set_xlabel("Total dense-profile time (s, log scale)")
    ax.invert_yaxis()
    for i, backend in enumerate(order):
        ax.text(
            totals[backend] * 1.08,
            i,
            f"{totals[backend]:.3g}s",
            va="center",
            fontsize=8,
            color=TOKENS["ink"],
            family="monospace",
        )
    ax.legend(loc="upper right", frameon=True, framealpha=0.94, ncol=2, fontsize=8)
    add_chart_header(
        fig,
        ax,
        "Per-peak convolution, not peak generation, dominates fine-structure profile baselines",
        "Large 12 kDa case; enviPat peak segment includes Rscript startup in the end-to-end Python wrapper measurement.",
    )
    return save(fig, "fine_structure_time_breakdown_large12k.png")


def plot_cluster_preservation(clusters: pd.DataFrame) -> Path:
    part = clusters[
        (clusters["case"] == "large_12k")
        & (clusters["nominal_cluster"].isin([7, 8, 9, 10]))
    ].copy()
    iso = (
        part[["nominal_cluster", "isospec_fine_peak_count"]]
        .drop_duplicates()
        .rename(columns={"isospec_fine_peak_count": "comparison_peak_count"})
    )
    iso["backend"] = "IsoSpecPy"
    data = pd.concat(
        [
            part[["nominal_cluster", "backend", "comparison_peak_count"]],
            iso[["nominal_cluster", "backend", "comparison_peak_count"]],
        ],
        ignore_index=True,
    )
    order = ["brainpy", "IsoSpecPy", "pyopenms", "envipat"]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    x = np.arange(len(sorted(data["nominal_cluster"].unique())))
    clusters_order = sorted(data["nominal_cluster"].unique())
    width = 0.18
    for i, backend in enumerate(order):
        values = (
            data[data["backend"] == backend]
            .set_index("nominal_cluster")["comparison_peak_count"]
            .reindex(clusters_order)
            .fillna(0.0)
            .to_numpy()
        )
        ax.bar(
            x + (i - 1.5) * width,
            values,
            width=width,
            label=backend,
            color=COLORS[backend],
            edgecolor=EDGES[backend],
            linewidth=1.0,
        )
    ax.set_yscale("log")
    ax.set_xticks(x, [f"+{int(c)}" for c in clusters_order])
    ax.set_xlabel("Nominal isotope cluster")
    ax.set_ylabel("Fine peaks in cluster (log scale)")
    ax.legend(loc="upper left", frameon=True, framealpha=0.94, ncol=2, fontsize=8)
    add_chart_header(
        fig,
        ax,
        "Aggregated methods erase the fine structure that fine generators preserve",
        "Large 12 kDa case; bars count backend peaks assigned to the same nominal isotope clusters as the IsoSpecPy reference.",
    )
    return save(fig, "fine_cluster_preservation_large12k.png")


def plot_large_scale(large_scale: pd.DataFrame) -> Path:
    data = (
        large_scale[large_scale["method"] == "cython_log_pruned"]
        .sort_values("mean_mass_da")
        .copy()
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.plot(
        data["mean_mass_da"] / 1000.0,
        data["median_s"],
        marker="o",
        color=COLORS["fastiso"],
        markeredgecolor=EDGES["fastiso"],
        linewidth=1.2,
        label="FastIso pruned residual",
    )
    ax.set_xlabel("Mean molecular mass (kDa)")
    ax.set_ylabel("Residual generation time (s)")
    ax.set_yscale("log")
    for _, row in data.iterrows():
        ax.text(
            row["mean_mass_da"] / 1000.0,
            row["median_s"] * 1.15,
            f"n={int(row['n_fft'])}",
            fontsize=7,
            ha="center",
            color=TOKENS["muted"],
        )
    ax2 = ax.twinx()
    ax2.plot(
        data["mean_mass_da"] / 1000.0,
        data["active_fraction"],
        marker="s",
        color="#F0986E",
        markeredgecolor="#804126",
        linewidth=1.0,
        linestyle=":",
        label="Active frequency fraction",
    )
    ax2.set_yscale("log")
    ax2.set_ylabel("Active frequency fraction")
    ax2.grid(False)
    lines = ax.lines + ax2.lines
    ax.legend(
        lines,
        [line.get_label() for line in lines],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        frameon=True,
        framealpha=0.94,
        ncol=2,
        fontsize=8,
    )
    add_chart_header(
        fig,
        ax,
        "Frequency pruning keeps large-molecule residual generation nearly flat",
        "Auto-windowed CHNOS cases from 12 kDa to 240 kDa; labels show selected odd FFT length.",
    )
    return save(fig, "large_scale_fastiso_pruned_runtime.png")


def plot_czt_window_crossover(czt: pd.DataFrame) -> Path:
    data = czt.copy()
    data["output_dm_label"] = data["output_dm"].map(lambda value: f"output dm={value:g}")
    panels = sorted(data["output_dm"].unique())
    fig, axes = plt.subplots(1, len(panels), figsize=(13.6, 4.9), sharex=True, sharey=True)
    for ax, output_dm in zip(axes, panels, strict=True):
        panel = data[data["output_dm"] == output_dm]
        for case, part in panel.groupby("case", sort=False):
            part = part.sort_values("points")
            ax.plot(
                part["points"],
                part["full_over_czt"],
                marker="o",
                linewidth=1.0,
                markersize=4.8,
                color=CASE_COLORS[case],
                markeredgecolor=TOKENS["ink"],
                markeredgewidth=0.5,
                label=case.replace("_", " "),
            )
        ax.axhline(1.0, color=TOKENS["ink"], linestyle=":", linewidth=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Window points (log scale)")
        ax.set_title(f"output dm={output_dm:g}", fontsize=9.5, color=TOKENS["ink"])
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        ax.grid(True, which="both", axis="both")
    axes[0].set_ylabel("Full dense profile / CZT window time (x)")
    add_chart_header(
        fig,
        axes[0],
        "CZT windowing crosses over once the full FFT grid grows but the inspected window stays local",
        "End-to-end timing includes residual spectrum generation plus profile transform; dotted line marks parity with full dense profile.",
        title_width=86,
        subtitle_width=124,
    )
    axes[0].set_title(f"output dm={panels[0]:g}", fontsize=9.5, color=TOKENS["ink"])
    fig.subplots_adjust(bottom=0.22, wspace=0.20)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.54, 0.025),
        frameon=True,
        framealpha=0.94,
        ncol=5,
        fontsize=7.6,
    )
    return save(fig, "czt_window_speedup_crossover.png")


def plot_czt_grid_matched_heatmap(czt: pd.DataFrame) -> Path:
    data = czt[np.isclose(czt["output_dm"], czt["table_dm"])].copy()
    case_order = (
        data[["case", "mass_kda"]]
        .drop_duplicates()
        .sort_values("mass_kda")["case"]
        .to_list()
    )
    columns = sorted(data["window_width_da"].unique())
    matrix = (
        data.pivot_table(
            index="case",
            columns="window_width_da",
            values="full_over_czt",
            aggfunc="first",
        )
        .reindex(index=case_order, columns=columns)
    )
    labels = matrix.map(lambda value: f"{value:.2g}x")
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    cmap = sns.blend_palette(
        [TOKENS["panel"], "#E2E5EA", "#A3BEFA", "#5477C4", "#2E4780"],
        as_cmap=True,
    )
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        vmin=0.0,
        vmax=max(1.0, float(np.nanmax(matrix.to_numpy()))),
        annot=labels,
        fmt="",
        linewidths=1.0,
        linecolor=TOKENS["panel"],
        cbar_kws={"label": "Full / CZT speedup"},
    )
    ax.set_xlabel("Residual window width (Da), output dm matches table dm")
    ax.set_ylabel("")
    ax.set_yticklabels([label.replace("_", " ") for label in matrix.index], rotation=0)
    ax.set_xticklabels([f"{value:g}" for value in matrix.columns], rotation=0)
    for _, spine in ax.spines.items():
        spine.set_visible(False)
    add_chart_header(
        fig,
        ax,
        "Grid-matched CZT is strongest for large molecules and narrow inspection windows",
        "Cells show end-to-end speedup over full dense profile; values above 1x are faster while preserving full-grid samples.",
    )
    return save(fig, "czt_window_grid_matched_heatmap.png")


def plot_external_batch10(external: pd.DataFrame) -> Path:
    data = external[
        (external["batch_size"] == 10)
        & (external["family"].isin(["medium", "large"]))
        & (
            (external["package"] == "fastiso")
            | (
                (external["package"] == "IsoSpecPy")
                & external["method"].str.contains("convolved", regex=False)
            )
            | (external["package"] == "brainpy")
        )
    ].copy()
    label_map = {"fastiso": "fastiso", "IsoSpecPy": "IsoSpecPy", "brainpy": "brainpy"}
    data["backend"] = data["package"].map(label_map)
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    families = ["medium", "large"]
    backends = ["fastiso", "brainpy", "IsoSpecPy"]
    x = np.arange(len(families))
    width = 0.22
    for i, backend in enumerate(backends):
        values = (
            data[data["backend"] == backend]
            .set_index("family")["median_s"]
            .reindex(families)
            .to_numpy()
        )
        ax.bar(
            x + (i - 1) * width,
            values,
            width=width,
            label=backend,
            color=COLORS[backend],
            edgecolor=EDGES[backend],
            linewidth=1.0,
        )
    ax.set_yscale("log")
    ax.set_xticks(x, ["Medium batch 10", "Large batch 10"])
    ax.set_ylabel("Dense profile time (s, log scale)")
    ax.legend(loc="upper left", frameon=True, framealpha=0.94, ncol=3, fontsize=8)
    add_chart_header(
        fig,
        ax,
        "FastIso separates from peak-list baselines when dense profiles are required",
        "External benchmark at R=100,000 and dm=0.002; small molecules are omitted because this grid undersamples them.",
    )
    return save(fig, "external_profile_batch10.png")


def plot_internal_speedup(internal: pd.DataFrame) -> Path:
    filtered = internal[(internal["resolving_power"] == 100000) & (internal["batch_size"] == 50)].copy()
    pivot = filtered.pivot_table(
        index="family",
        columns="method",
        values="median_s",
        aggfunc="first",
    )
    data = pivot.reset_index()
    data["direct_over_cython"] = data["direct_rebuild"] / data["cython_log_pruned"]
    data["family"] = pd.Categorical(data["family"], ["small", "medium", "large", "xlarge"], ordered=True)
    data = data.sort_values("family")
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    ax.bar(
        data["family"].astype(str),
        data["direct_over_cython"],
        color="#A3D576",
        edgecolor="#386411",
        linewidth=1.0,
        label="Direct rebuild / Cython pruned",
    )
    ax.set_yscale("log")
    ax.set_ylabel("Speedup (x, log scale)")
    for i, row in enumerate(data.itertuples(index=False)):
        ax.text(
            i,
            row.direct_over_cython * 1.08,
            f"{row.direct_over_cython:.0f}x",
            ha="center",
            va="bottom",
            fontsize=8,
            color=TOKENS["ink"],
            family="monospace",
        )
    add_chart_header(
        fig,
        ax,
        "Log-table reuse gives the largest payoff for large batches and larger formulas",
        "Internal benchmark at R=100,000 and batch=50; table construction time is excluded.",
    )
    return save(fig, "internal_direct_vs_pruned_speedup.png")


def write_summary(
    fine: pd.DataFrame,
    clusters: pd.DataFrame,
    large_scale: pd.DataFrame,
    external: pd.DataFrame,
    czt: pd.DataFrame,
    outputs: list[Path],
) -> None:
    large = fine.loc[fine["case"] == "large_12k"].iloc[0]
    medium_external = external[
        (external["family"] == "medium")
        & (external["batch_size"] == 10)
        & (external["package"].isin(["fastiso", "IsoSpecPy", "brainpy"]))
    ]
    large_external = external[
        (external["family"] == "large")
        & (external["batch_size"] == 10)
        & (external["package"].isin(["fastiso", "IsoSpecPy", "brainpy"]))
    ]
    cluster10 = clusters[
        (clusters["case"] == "large_12k") & (clusters["nominal_cluster"] == 10)
    ][["backend", "comparison_peak_count"]]
    czt_best = czt.loc[czt["full_over_czt"].idxmax()]
    czt_grid = czt[np.isclose(czt["output_dm"], czt["table_dm"])]
    czt_grid_best = czt_grid.loc[czt_grid["full_over_czt"].idxmax()]

    lines = [
        "# Publication Feasibility Plots",
        "",
        "## Chart Contract",
        "",
        "- Question: does FastIso occupy a publishable speed/fidelity niche for instrument-broadened fine-isotopic profiles?",
        "- Takeaway: FastIso is not a faster fine peak enumerator; it is a faster dense-profile simulator once fine structure and peak broadening are required.",
        "- Surface: static PNG figures for quick manuscript feasibility review.",
        "- Palette: explicit blue/gold/orange/olive/pink roots plus neutral reference marks.",
        "",
        "## Key Readout",
        "",
        f"- Large 12 kDa FastIso profile: {large['fastiso_profile_s']:.4g} s with L2 {large['fastiso_rel_l2_vs_isospec']:.2e} vs IsoSpecPy.",
        f"- Large 12 kDa IsoSpecPy dense profile: {large['isospec_profile_s']:.4g} s.",
        f"- Large 12 kDa pyOpenMS dense profile: {large['pyopenms_profile_s']:.4g} s with L2 {large['pyopenms_rel_l2_vs_isospec']:.2e}.",
        f"- Large 12 kDa enviPat dense profile through the current wrapper: {large['envipat_profile_s']:.4g} s with L2 {large['envipat_rel_l2_vs_isospec']:.2e}.",
        f"- Large 12 kDa Brainpy dense profile: {large['brainpy_profile_s']:.4g} s with L2 {large['brainpy_rel_l2_vs_isospec']:.2e}; it is an aggregated baseline, not a fine-structure baseline.",
        "- Large 12 kDa cluster +10 peak counts: "
        + ", ".join(f"{row.backend}={int(row.comparison_peak_count)}" for row in cluster10.itertuples())
        + ".",
        f"- Auto-windowed largest case: {large_scale['mean_mass_da'].max() / 1000:.1f} kDa with residual generation "
        f"{large_scale[(large_scale['method'] == 'cython_log_pruned')].loc[large_scale[(large_scale['method'] == 'cython_log_pruned')]['mean_mass_da'].idxmax(), 'median_s']:.4g} s.",
        f"- Best CZT window speedup: {czt_best['full_over_czt']:.2f}x for {czt_best['case']} at {czt_best['window_width_da']:.3g} Da and output dm={czt_best['output_dm']:.4g}.",
        f"- Best grid-matched CZT speedup: {czt_grid_best['full_over_czt']:.2f}x for {czt_grid_best['case']} at {czt_grid_best['window_width_da']:.3g} Da.",
        "",
        "## Plots",
        "",
    ]
    for output in outputs:
        lines.append(f"- {output.name}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The current evidence supports a plausible methods/software paper if the claim is framed around dense, instrument-broadened profile simulation rather than peak-list enumeration.",
            "The strongest figure is the speed-accuracy plot: FastIso remains near IsoSpecPy-level profile fidelity while avoiding the per-peak convolution cost that dominates IsoSpecPy, pyOpenMS, and enviPat dense-profile workflows.",
            "Brainpy is useful as an aggregated baseline, but the cluster preservation plot shows why it should not be presented as a fine-structure competitor.",
        ]
    )
    (PLOTS / "publication_feasibility_plots.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
