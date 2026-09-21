#!/usr/bin/env python3
"""Render the latest frozen SmolLM2 factorials for the manuscript appendix.

The figure deliberately separates positive co-scaled diagonal contrasts from
fixed-refinement interactions.  Every number is loaded from an immutable
analysis JSON; no result is entered manually.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


NAVY = "#173B57"
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
GRAY = "#687178"
BLACK = "#20262B"
GRID = "#DCE3E8"
GAIN_CMAP = LinearSegmentedColormap.from_list(
    "ascent_gain", ("#F7FBFD", "#A6CEE3", "#0072B2", "#173B57")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generative", type=Path, required=True)
    parser.add_argument("--certified", type=Path, required=True)
    parser.add_argument("--output-basename", type=Path, required=True)
    parser.add_argument("--skill-scripts", type=Path, required=True)
    return parser.parse_args()


def validate(analysis: dict, *, endpoints: int, panels: int) -> None:
    if len(analysis["endpoints"]) != endpoints or analysis["panels"] != panels:
        raise RuntimeError("unexpected factorial dimensions")
    if len(analysis["config_hashes"]) != 1:
        raise RuntimeError("factorial must have one frozen configuration hash")
    if not analysis["foundation_cache_equal_across_state_sizes"]:
        raise RuntimeError("foundation cache must be identical across state sizes")
    if not analysis["gates"]["provenance_and_alignment"]:
        raise RuntimeError("provenance/alignment gate failed")
    if not analysis["gates"]["diagonal_gain_adjacent_lcbs_positive"]:
        raise RuntimeError("co-scaled diagonal gate failed")


def matrix(analysis: dict) -> np.ndarray:
    return np.asarray([
        [analysis["cells"][model][str(slot)]["absolute_gain"]["mean"] * 100
         for slot in analysis["state_slots"]]
        for model in analysis["endpoints"]
    ])


def heatmap(ax: plt.Axes, analysis: dict, title: str,
            model_labels: list[str], slot_labels: list[str]) -> None:
    values = matrix(analysis)
    image = ax.imshow(values, cmap=GAIN_CMAP, vmin=0, vmax=100, aspect="auto")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            color = "white" if values[row, col] >= 56 else BLACK
            ax.text(col, row, f"{values[row, col]:.1f}", ha="center", va="center",
                    color=color, fontsize=7.2, fontweight="bold")
    for index in range(3):
        ax.add_patch(Rectangle((index - 0.48, index - 0.48), 0.96, 0.96,
                               fill=False, edgecolor=ORANGE, linewidth=1.8))
    ax.set_xticks(np.arange(3), slot_labels)
    ax.set_yticks(np.arange(3), model_labels)
    ax.set_xlabel("State budget")
    ax.set_ylabel("Frozen reader")
    ax.set_title(title, color=NAVY, fontweight="bold")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def interval_rows(generative: dict, certified: dict) -> list[tuple[str, dict, str, str]]:
    gen_adj = list(generative["diagonal_absolute_gain_adjacent"].values())
    gen_int = list(generative["registered_foundation_by_state_interactions"].values())
    cert_adj = list(certified["diagonal_absolute_gain_adjacent"].values())
    cert_ints = certified["foundation_by_state_interactions"]
    cert_int = [
        cert_ints["smollm2-135m_to_smollm2-360m-node2"]["slots_1_to_2"],
        cert_ints["smollm2-360m-node2_to_smollm2-1p7b"]["slots_2_to_8"],
    ]
    return [
        ("Generation diagonal 135M-360M", gen_adj[0], BLUE, "o"),
        ("Generation diagonal 360M-1.7B", gen_adj[1], BLUE, "o"),
        ("Generation interaction 1", gen_int[0], ORANGE, "s"),
        ("Generation interaction 2", gen_int[1], ORANGE, "s"),
        ("QA7/8 diagonal 135M-360M", cert_adj[0], GREEN, "D"),
        ("QA7/8 diagonal 360M-1.7B", cert_adj[1], GREEN, "D"),
        ("QA7/8 interaction 1", cert_int[0], GRAY, "^"),
        ("QA7/8 interaction 2", cert_int[1], GRAY, "^"),
    ]


def forest(ax: plt.Axes, rows: list[tuple[str, dict, str, str]]) -> None:
    y = np.arange(len(rows))[::-1]
    for yy, (label, item, color, marker) in zip(y, rows):
        mean = item["mean"] * 100
        low = item["ci95_low"] * 100
        high = item["ci95_high"] * 100
        ax.errorbar(mean, yy, xerr=[[mean - low], [high - mean]], fmt=marker,
                    color=color, markerfacecolor="white", markeredgewidth=1.0,
                    markersize=4.8, elinewidth=1.1, capsize=2.2, zorder=3)
        ax.annotate(f"{mean:.1f}", (high, yy), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=6.4, color=color, fontweight="bold")
    ax.axvline(0, color=BLACK, linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_yticks(y, [row[0] for row in rows])
    ax.set_xlim(-15, 40)
    ax.set_xticks((-10, 0, 10, 20, 30, 40))
    ax.set_xlabel("Accuracy-gain contrast (points; 95% CI)")
    ax.set_title("Diagonal scaling and interaction structure",
                 color=NAVY, fontweight="bold")
    ax.grid(axis="x", color=GRID, linewidth=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=2.5, width=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#87939B")


def main() -> None:
    args = parse_args()
    generative = json.loads(args.generative.read_text())
    certified = json.loads(args.certified.read_text())
    validate(generative, endpoints=3, panels=3)
    validate(certified, endpoints=3, panels=5)

    sys.path.insert(0, str(args.skill_scripts))
    from export_figure import export_figure
    from layout_tools import add_panel_labels, finalize_figure
    from setup_style import setup_style
    from visual_qa import audit_layout, print_report, render_preview

    setup_style(journal="general", lang="en", constrained_layout=True,
                use_sciplots=False)
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                         "axes.titlesize": 8.5, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none"})

    fig = plt.figure(figsize=(7.48, 3.35), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 1.7))
    ax_gen = fig.add_subplot(grid[0, 0])
    ax_cert = fig.add_subplot(grid[0, 1])
    ax_forest = fig.add_subplot(grid[0, 2])
    image = heatmap(ax_gen, generative, "Free generation\n(QA1-QA3)",
                    ["135M", "360M", "1.7B"], ["1", "2", "3"])
    heatmap(ax_cert, certified, "Certified readout\n(QA7-QA8)",
            ["135M", "360M", "1.7B"], ["1", "2", "8"])
    forest(ax_forest, interval_rows(generative, certified))
    colorbar = fig.colorbar(image, ax=(ax_gen, ax_cert), location="bottom",
                            fraction=0.07, pad=0.10, aspect=24)
    colorbar.set_label("ASCENT $-$ Foundation accuracy (points)")
    colorbar.set_ticks((0, 25, 50, 75, 100))

    finalize_figure(fig)
    fig.canvas.draw()
    add_panel_labels(fig, axes=(ax_gen, ax_cert, ax_forest), style="upper_paren",
                     x_offset_pt=-20, y_offset_pt=4)
    verdict = print_report(audit_layout(fig))
    if verdict == "FAIL":
        raise RuntimeError("latest-factorial figure layout audit failed")
    args.output_basename.parent.mkdir(parents=True, exist_ok=True)
    render_preview(fig, str(args.output_basename) + "_preview.png", dpi=180)
    export_figure(fig, str(args.output_basename), formats=("pdf", "svg", "png"),
                  size_inches=(7.48, 3.35), dpi=600, grayscale_preview=True,
                  tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
