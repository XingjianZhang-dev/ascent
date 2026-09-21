#!/usr/bin/env python3
"""Render the data-backed Array graphical abstract.

All displayed confirmation values are loaded from the immutable factorial
analysis.  The output follows Elsevier's landscape graphical-abstract ratio
and is exported as editable vector artwork plus a 600-dpi raster.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image


NAVY = "#173B57"
BLUE = "#0072B2"
LIGHT_BLUE = "#E8F3FA"
GREEN = "#009E73"
LIGHT_GREEN = "#E8F6F1"
ORANGE = "#E69F00"
LIGHT_ORANGE = "#FFF5DD"
INK = "#20262B"
MUTED = "#5D6972"
GRID = "#DCE3E8"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-basename", type=Path, required=True)
    parser.add_argument("--skill-scripts", type=Path, required=True)
    return parser.parse_args()


def rounded(ax: plt.Axes, xy: tuple[float, float], width: float, height: float,
            *, face: str, edge: str, radius: float = 0.018,
            linewidth: float = 1.0) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy, width, height,
        boxstyle=f"round,pad=0.009,rounding_size={radius}",
        transform=ax.transAxes, facecolor=face, edgecolor=edge,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float],
          *, color: str = NAVY) -> None:
    ax.add_patch(FancyArrowPatch(
        start, end, transform=ax.transAxes, arrowstyle="-|>",
        mutation_scale=9, linewidth=1.15, color=color,
        shrinkA=0, shrinkB=0,
    ))


def load_evidence(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    document = json.loads(path.read_text())
    if not document["registered_performance_pass"]:
        raise RuntimeError("registered performance gate did not pass")
    if not document["supplementary_bonferroni_familywise_performance_pass"]:
        raise RuntimeError("simultaneous familywise gate did not pass")
    estimands = document["supplementary_bonferroni_familywise_estimands"]
    keys = (
        "gain_qwen2p5-1p5b-instruct_k2",
        "gain_qwen2p5-3b-instruct_k3",
        "gain_qwen2p5-7b-instruct_k5",
    )
    means = np.asarray([estimands[key]["mean"] for key in keys])
    lows = np.asarray([estimands[key]["ci95_low"] for key in keys])
    highs = np.asarray([estimands[key]["ci95_high"] for key in keys])
    if not np.all(np.diff(means) > 0) or not np.all(lows > 0):
        raise RuntimeError("co-scaled gain evidence is inconsistent")
    interactions = np.asarray([
        estimands["first_model_by_state_interaction"]["ci95_low"],
        estimands["second_model_by_state_interaction"]["ci95_low"],
    ])
    if not np.all(interactions > 0):
        raise RuntimeError("simultaneous interaction lower bound is not positive")
    return means, lows, highs


def main() -> None:
    args = arguments()
    means, lows, highs = load_evidence(args.analysis)

    sys.path.insert(0, str(args.skill_scripts))
    from export_figure import export_figure
    from setup_style import setup_style
    from visual_qa import audit_layout, print_report, render_preview

    setup_style(journal="general", lang="en", constrained_layout=False,
                use_sciplots=False)
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig = plt.figure(figsize=(5.25, 2.10), facecolor="white")
    canvas = fig.add_axes((0, 0, 1, 1))
    canvas.set_axis_off()
    canvas.text(
        0.5, 0.925,
        "ASCENT: Scale-Complementary External State\n"
        "for Frozen Long-Context Language Models",
        transform=canvas.transAxes, ha="center", va="center",
        fontsize=9.0, linespacing=0.95, fontweight="bold", color=NAVY,
    )

    # Streaming evidence.
    rounded(canvas, (0.02, 0.20), 0.14, 0.60, face="#F4F6F7", edge=GRID)
    canvas.text(0.09, 0.74, "Long evidence", transform=canvas.transAxes,
                ha="center", va="center", fontsize=8.0, fontweight="bold",
                color=INK)
    for index, width in enumerate((0.096, 0.112, 0.085, 0.105, 0.093)):
        y = 0.63 - index * 0.078
        canvas.add_patch(Rectangle(
            (0.038, y), width, 0.032, transform=canvas.transAxes,
            facecolor=BLUE if index in (1, 4) else "#C9D2D8",
            edgecolor="none",
        ))
    canvas.text(0.09, 0.25, "query arrives last", transform=canvas.transAxes,
                ha="center", va="center", fontsize=6.7, color=MUTED)

    # Causal writer and nested state.
    arrow(canvas, (0.17, 0.50), (0.205, 0.50))
    rounded(canvas, (0.21, 0.20), 0.255, 0.60,
            face=LIGHT_GREEN, edge=GREEN)
    canvas.text(0.338, 0.735, "Target-blind causal writer",
                transform=canvas.transAxes, ha="center", va="center",
                fontsize=8.0, fontweight="bold", color=INK)
    state_specs = (
        (0.242, 0.30, 0.078, 0.27, r"$S_1$", "K=2"),
        (0.315, 0.30, 0.090, 0.34, r"$S_2$", "K=3"),
        (0.400, 0.30, 0.045, 0.41, r"$S_3$", "K=5"),
    )
    for x, y, width, height, label, budget in state_specs:
        canvas.add_patch(FancyBboxPatch(
            (x, y), width, height, transform=canvas.transAxes,
            boxstyle="round,pad=0.006,rounding_size=0.010",
            facecolor="white", edgecolor=GREEN, linewidth=0.9,
        ))
        canvas.text(x + width / 2, y + height - 0.055, label,
                    transform=canvas.transAxes, ha="center", va="center",
                    fontsize=7.3, fontweight="bold", color=GREEN)
        canvas.text(x + width / 2, y + 0.050, budget,
                    transform=canvas.transAxes, ha="center", va="center",
                    fontsize=6.5, color=MUTED)
    canvas.text(0.338, 0.245, "nested, bounded state",
                transform=canvas.transAxes, ha="center", va="center",
                fontsize=6.8, color=MUTED)

    # Frozen readers.
    arrow(canvas, (0.475, 0.50), (0.510, 0.50))
    rounded(canvas, (0.515, 0.20), 0.145, 0.60,
            face=LIGHT_ORANGE, edge=ORANGE)
    canvas.text(0.588, 0.735, "Frozen readers",
                transform=canvas.transAxes, ha="center", va="center",
                fontsize=8.0, fontweight="bold", color=INK)
    for index, (reader, budget) in enumerate(
        (("1.5B", r"$S_1$"), ("3B", r"$S_2$"), ("7B", r"$S_3$"))
    ):
        y = 0.61 - index * 0.145
        rounded(canvas, (0.535, y), 0.105, 0.085,
                face="white", edge=ORANGE, radius=0.012, linewidth=0.8)
        canvas.text(0.588, y + 0.044, f"{reader}  +  {budget}",
                    transform=canvas.transAxes, ha="center", va="center",
                    fontsize=7.0, fontweight="bold", color=INK)

    # Confirmatory evidence chart.
    # The reader-to-chart gap is reserved for the rotated y-axis title.  A
    # connector here would cross that title at publication size; the aligned
    # left-to-right composition already supplies the intended reading order.
    chart = fig.add_axes((0.725, 0.265, 0.255, 0.385))
    x = np.arange(3)
    chart.fill_between(x, lows, highs, color=LIGHT_BLUE, linewidth=0)
    chart.plot(x, means, color=BLUE, linewidth=1.7, marker="o",
               markersize=4.4, markerfacecolor="white", markeredgewidth=1.2)
    for xx, yy in zip(x, means, strict=True):
        chart.annotate(f"{yy:.3f}", (xx, yy), xytext=(0, 6),
                       textcoords="offset points", ha="center", va="bottom",
                       fontsize=7.0, fontweight="bold", color=NAVY)
    chart.set_xticks(x, ("1.5B", "3B", "7B"))
    chart.set_ylim(0, 8.4)
    chart.set_yticks((0, 4, 8))
    chart.set_ylabel("Gain (nats)", labelpad=0)
    chart.grid(axis="y", color=GRID, linewidth=0.55)
    chart.set_axisbelow(True)
    chart.spines[["top", "right"]].set_visible(False)
    chart.tick_params(length=2.5, width=0.7)
    canvas.text(0.852, 0.765, "State value rises",
                transform=canvas.transAxes, ha="center", va="center",
                fontsize=8.0, fontweight="bold", color=INK)
    canvas.text(0.852, 0.705, "with scale; both interactions > 0",
                transform=canvas.transAxes, ha="center", va="center",
                fontsize=6.8, fontweight="bold", color=GREEN)
    canvas.text(0.50, 0.075,
                "A compact external state becomes more valuable as reader capacity grows.",
                transform=canvas.transAxes, ha="center", va="center",
                fontsize=8.1, fontweight="bold", color=NAVY)

    fig.canvas.draw()
    verdict = print_report(audit_layout(fig))
    if verdict == "FAIL":
        raise RuntimeError("graphical-abstract layout audit failed")
    args.output_basename.parent.mkdir(parents=True, exist_ok=True)
    render_preview(fig, str(args.output_basename) + "_preview.png", dpi=180)
    export_figure(
        fig,
        str(args.output_basename),
        formats=("pdf", "svg", "png"),
        size_inches=(5.25, 2.10),
        dpi=600,
        grayscale_preview=False,
        tight=False,
    )
    png_path = Path(str(args.output_basename) + ".png")
    Image.open(png_path).convert("L").save(
        Path(str(args.output_basename) + "_grayscale.png")
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
