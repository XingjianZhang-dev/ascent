#!/usr/bin/env python3
"""Plot the panel-level Qwen evidence used in the TNNLS manuscript."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


STUDIES = (
    ("Official 16K", "Official 16K confirmation", "ASCENT $-$ Foundation", "#0072B2", "o"),
    ("Semantic holdout", "Zero-overlap validation", "ASCENT $-$ Foundation", "#009E73", "s"),
    ("Direct peer", "Direct peer comparison", "ASCENT $-$ Q-RAG", "#D55E00", "D"),
)
SCALES = ("0.5B", "1.5B", "3B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-basename", type=Path, required=True)
    parser.add_argument("--skill-scripts", type=Path, required=True)
    return parser.parse_args()


def validate_data(frame: pd.DataFrame) -> None:
    required = {"study", "metric", "scale_b", "scale_label", "panel", "value"}
    if set(frame.columns) != required:
        raise RuntimeError(f"unexpected columns: {list(frame.columns)}")
    if len(frame) != 90 or frame.isna().any().any():
        raise RuntimeError("expected 90 complete panel-level observations")
    expected = {(study, scale, panel) for study, *_ in STUDIES for scale in SCALES for panel in range(1, 11)}
    observed = set(frame[["study", "scale_label", "panel"]].itertuples(index=False, name=None))
    if observed != expected:
        raise RuntimeError("study/scale/panel grid is incomplete or duplicated")
    if not frame["value"].between(-1, 1).all():
        raise RuntimeError("accuracy differences must lie in [-1, 1]")


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.skill_scripts))
    from export_figure import export_figure
    from layout_tools import add_panel_labels, finalize_figure
    from setup_style import setup_style
    from visual_qa import audit_layout, print_report, render_preview

    frame = pd.read_csv(args.data)
    validate_data(frame)
    setup_style(journal="ieee", lang="en", constrained_layout=True)
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55), sharey=True, constrained_layout=True)
    x = np.arange(len(SCALES), dtype=float)
    for ax, (study, title, metric, color, marker) in zip(axes, STUDIES):
        subset = frame[frame["study"] == study]
        for panel in range(1, 11):
            panel_values = (
                subset[subset["panel"] == panel]
                .set_index("scale_label")
                .loc[list(SCALES), "value"]
                .to_numpy(dtype=float)
                * 100
            )
            ax.plot(x, panel_values, color="#A7ADB2", linewidth=0.55, alpha=0.70, zorder=1)
            ax.scatter(x, panel_values, s=8, facecolor="white", edgecolor="#8B9298", linewidth=0.45, zorder=2)

        grouped = subset.groupby("scale_label", sort=False)["value"]
        means = np.array([grouped.get_group(scale).mean() for scale in SCALES]) * 100
        sems = np.array([stats.sem(grouped.get_group(scale)) for scale in SCALES]) * 100
        half_widths = stats.t.ppf(0.975, 9) * sems
        ax.errorbar(
            x,
            means,
            yerr=half_widths,
            color=color,
            marker=marker,
            markerfacecolor="white",
            markeredgewidth=1.2,
            markersize=5.2,
            linewidth=1.7,
            elinewidth=1.0,
            capsize=2.4,
            zorder=4,
        )
        for index, mean in enumerate(means):
            y_offset = -12 if mean > 75 else 7
            ax.annotate(
                f"{mean:.1f}",
                (x[index], mean),
                xytext=(0, y_offset),
                textcoords="offset points",
                ha="center",
                va="center",
                color=color,
                fontsize=7.5,
                fontweight="bold",
                zorder=5,
            )

        ax.axhline(0, color="#333333", linewidth=0.7, linestyle=(0, (3, 2)), zorder=0)
        ax.set_title(f"{title}\n{metric}", pad=5)
        ax.set_xticks(x, SCALES)
        ax.set_xlim(-0.28, 2.28)
        ax.set_ylim(-100, 100)
        ax.set_yticks((-100, -50, 0, 50, 100))
        ax.set_xlabel("Qwen2.5 reader size")
        ax.tick_params(direction="out", length=2.7, width=0.7)
        ax.grid(axis="y", color="#E1E4E6", linewidth=0.45, zorder=-1)
    axes[0].set_ylabel("Accuracy difference (percentage points)")

    finalize_figure(fig)
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(w_pad=0.10, h_pad=0.08, wspace=0.04)
        fig.canvas.draw()
    add_panel_labels(fig, axes=axes, style="ieee", x_offset_pt=-13, y_offset_pt=4)
    verdict = print_report(audit_layout(fig))
    if verdict == "FAIL":
        raise RuntimeError("figure layout audit failed")

    preview = args.output_basename.with_name(args.output_basename.name + "_preview.png")
    render_preview(fig, str(preview), dpi=180)
    export_figure(
        fig,
        str(args.output_basename),
        formats=("pdf", "svg", "png"),
        size_inches=(7.16, 2.55),
        dpi=600,
        grayscale_preview=True,
        tight=False,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
