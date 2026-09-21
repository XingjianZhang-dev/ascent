#!/usr/bin/env python3
"""Render the selected theory--mechanism--evidence ASCENT schematic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


NAVY = "#173B57"
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"
GRAY = "#666666"
BLACK = "#20262B"


def box(ax, xy, wh, *, edge, face="white", radius=0.02, linewidth=1.15):
    patch = FancyBboxPatch(xy, *wh, boxstyle=f"round,pad=0.012,rounding_size={radius}",
                           transform=ax.transAxes, facecolor=face, edgecolor=edge,
                           linewidth=linewidth, clip_on=False)
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=BLUE):
    ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes,
                                arrowstyle="-|>", mutation_scale=10,
                                color=color, linewidth=1.15,
                                shrinkA=2, shrinkB=2, clip_on=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-basename", type=Path, required=True)
    parser.add_argument("--skill-scripts", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.skill_scripts))
    from export_figure import export_figure
    from setup_style import setup_style
    from visual_qa import audit_layout, print_report, render_preview

    setup_style(journal="general", lang="en", constrained_layout=False,
                use_sciplots=False)
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 8, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(7.48, 3.35))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.012, 0.965, "(A)", transform=ax.transAxes, fontweight="bold",
            fontsize=9, va="top")
    ax.text(0.050, 0.942, "MECHANISM", transform=ax.transAxes,
            fontweight="bold", fontsize=7.2, color=BLUE, va="top")
    cards = (
        (0.025, 0.54, 0.16, 0.34, GRAY, "#F3F5F6", "Long context", "target remains hidden"),
        (0.245, 0.54, 0.17, 0.34, BLUE, "#EAF4FA", "Target-blind writer", "causal typed events"),
        (0.485, 0.50, 0.22, 0.42, GREEN, "#EAF7F2", "ASCENT state", "nested, bounded slots"),
        (0.795, 0.54, 0.16, 0.34, ORANGE, "#FFF4D8", "Frozen reader", "query-conditioned read"),
    )
    for x, y, w, h, edge, face, title, subtitle in cards:
        box(ax, (x, y), (w, h), edge=edge, face=face)
        ax.text(x + w / 2, y + h - 0.07, title, transform=ax.transAxes,
                ha="center", va="center", fontweight="bold", color=edge, fontsize=8.3)
        ax.text(x + w / 2, y + 0.045, subtitle, transform=ax.transAxes,
                ha="center", va="center", color=GRAY, fontsize=6.8)
    for yy, width in zip((0.74, 0.70, 0.66, 0.62), (0.11, 0.09, 0.12, 0.075)):
        ax.plot([0.05, 0.05 + width], [yy, yy], transform=ax.transAxes,
                color="#9AA3A9", linewidth=1.25)
    for yy, color in zip((0.76, 0.70, 0.64), (BLUE, GREEN, ORANGE)):
        box(ax, (0.535, yy - 0.02), (0.12, 0.035), edge=color,
            face="white", radius=0.01, linewidth=0.9)
    arrow(ax, (0.19, 0.71), (0.24, 0.71))
    arrow(ax, (0.42, 0.71), (0.48, 0.71))
    arrow(ax, (0.71, 0.71), (0.79, 0.71), color=BLACK)

    ax.text(0.012, 0.415, "(B)", transform=ax.transAxes, fontweight="bold",
            fontsize=9, va="top")
    box(ax, (0.025, 0.08), (0.42, 0.29), edge=BLUE, face="white")
    ax.text(0.235, 0.305, "Information value of nested refinement", transform=ax.transAxes,
            ha="center", fontweight="bold", color=BLUE, fontsize=8.3)
    ax.text(0.235, 0.215, r"$L^*(Z)-L^*(Z')=I(Y;Z'\mid Q,Z)>0$",
            transform=ax.transAxes, ha="center", fontweight="bold", fontsize=9)
    ax.text(0.235, 0.125, "A nested refinement adds target-relevant conditional information",
            transform=ax.transAxes, ha="center", color=GRAY, fontsize=6.8)

    ax.text(0.535, 0.415, "(C)", transform=ax.transAxes, fontweight="bold",
            fontsize=9, va="top")
    box(ax, (0.55, 0.08), (0.40, 0.29), edge=GREEN, face="white")
    ax.text(0.75, 0.305, "Factorial identification", transform=ax.transAxes,
            ha="center", fontweight="bold", color=GREEN, fontsize=8.3)
    xs = (0.61, 0.75, 0.89)
    for i, (x, model, state, gain) in enumerate(zip(xs, ("1.5B", "3B", "7B"),
                                                   ("K2", "K3", "K5"),
                                                   ("0.39", "2.10", "7.41"))):
        ax.text(x, 0.235, f"{model} + {state}", transform=ax.transAxes,
                ha="center", color=BLUE if i < 2 else ORANGE,
                fontsize=7.4, fontweight="bold")
        ax.text(x, 0.155, gain, transform=ax.transAxes, ha="center",
                color=GREEN, fontsize=9, fontweight="bold")
    arrow(ax, (0.66, 0.205), (0.70, 0.205), color=BLACK)
    arrow(ax, (0.80, 0.205), (0.84, 0.205), color=BLACK)
    ax.text(0.75, 0.10, "NLL gain; both model-by-state interactions > 0",
            transform=ax.transAxes, ha="center", color=GRAY, fontsize=6.7)

    fig.subplots_adjust(left=0.02, right=0.99, top=0.99, bottom=0.03)
    verdict = print_report(audit_layout(fig))
    if verdict == "FAIL":
        raise RuntimeError("method figure layout audit failed")
    render_preview(fig, str(args.output_basename) + "_preview.png", dpi=180)
    export_figure(fig, str(args.output_basename), formats=("pdf", "svg", "png"),
                  size_inches=(7.48, 3.35), dpi=600, grayscale_preview=True,
                  tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
