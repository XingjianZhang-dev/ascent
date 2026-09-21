#!/usr/bin/env python3
"""Render the frozen ASCENT evidence as final-size Elsevier figures.

Every displayed estimate is recomputed from panel-level CSV exports.  The
figures expose the panel observations, use redundant encodings, and share a
single theory--mechanism--systems visual language.
"""

from __future__ import annotations

import argparse
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


NAVY = "#173B57"
BLUE = "#0072B2"
TEAL = "#009E73"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
INK = "#20262B"
MUTED = "#66727B"
POINT = "#8E9AA3"
GRID = "#DCE3E8"
PALE_BLUE = "#EAF4FA"


def fmt(value: float, places: int) -> str:
    """Manuscript rounding rule (round half up on the decimal value), as in the table renderers.

    Panel means are exact decimals, so ties occur at the one-decimal figure labels (77.25 -> 77.3,
    26.25 -> 26.3); ``f"{value:.1f}"`` rounds the binary double half to even and would print 77.2.
    """
    return str(Decimal(f"{value:.12f}").quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skill-scripts", type=Path, required=True)
    return parser.parse_args()


def ci95(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    half = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
    return mean, mean - half, mean + half


def validate(primary: pd.DataFrame, factorial: pd.DataFrame,
             intervals: pd.DataFrame, breadth: pd.DataFrame,
             systems: pd.DataFrame) -> None:
    expected = {
        "primary": ({"study", "contrast", "scale_b", "scale_label", "panel", "value"}, 90),
        "factorial": ({"model", "model_b", "slots", "panel", "gain_nll"}, 81),
        "intervals": ({"estimand", "mean", "ci_low", "ci_high"}, 5),
        "breadth": ({"model", "panel", "gain", "foundation", "ascent"}, 50),
        "systems": ({"model", "state_bytes", "state_model_ratio", "decode_ratio", "flop_reduction"}, 3),
    }
    for name, frame in (("primary", primary), ("factorial", factorial),
                        ("intervals", intervals), ("breadth", breadth),
                        ("systems", systems)):
        columns, rows = expected[name]
        if set(frame.columns) != columns or len(frame) != rows or frame.isna().any().any():
            raise RuntimeError(f"{name}: malformed evidence table")
    if not primary["value"].between(-1, 1).all():
        raise RuntimeError("primary accuracy contrasts must lie in [-1,1]")
    if not (intervals["ci_low"] > 0).all():
        raise RuntimeError("all simultaneous lower bounds must be positive")


def style_axes(ax: plt.Axes, *, grid: str | None = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#87939B")
    ax.spines[["left", "bottom"]].set_linewidth(0.7)
    ax.tick_params(direction="out", length=2.4, width=0.7, color="#87939B")
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)


def plot_primary(frame: pd.DataFrame, output: Path, helpers: dict) -> None:
    studies = (
        ("Official 16K", "Official 16K", "ASCENT - Foundation", BLUE, "o"),
        ("Semantic holdout", "Zero-overlap holdout", "ASCENT - Foundation", TEAL, "s"),
        ("Direct peer", "Direct Q-RAG peer", "ASCENT - Q-RAG", ORANGE, "D"),
    )
    scales = ("0.5B", "1.5B", "3B")
    x = np.log2(np.asarray((0.5, 1.5, 3.0)))
    fig, axes = plt.subplots(1, 3, figsize=(7.48, 2.72), sharey=True,
                             constrained_layout=True)
    offsets = np.linspace(-0.075, 0.075, 10)
    for ax, (study, title, contrast, color, marker) in zip(axes, studies):
        sub = frame[frame.study == study]
        ax.axhspan(0, 100, color=PALE_BLUE, alpha=0.34, zorder=0)
        for j, scale in enumerate(scales):
            vals = (sub[sub.scale_label == scale].sort_values("panel").value.to_numpy()
                    * 100)
            ax.scatter(np.full(10, x[j]) + offsets, vals, s=12,
                       facecolor="white", edgecolor=POINT, linewidth=0.55,
                       alpha=0.95, zorder=2)
        summaries = [ci95(sub[sub.scale_label == scale].value.to_numpy() * 100)
                     for scale in scales]
        means = np.asarray([v[0] for v in summaries])
        lows = np.asarray([v[1] for v in summaries])
        highs = np.asarray([v[2] for v in summaries])
        ax.errorbar(x, means, yerr=(means - lows, highs - means), color=color,
                    marker=marker, markerfacecolor="white", markeredgewidth=1.25,
                    markersize=5.4, linewidth=1.8, elinewidth=1.05, capsize=2.5,
                    zorder=4)
        for xx, mean in zip(x, means):
            dy = -14 if mean > 70 else (11 if mean >= 0 else 12)
            ax.annotate(fmt(mean, 1), (xx, mean), xytext=(0, dy),
                        textcoords="offset points", ha="center", va="center",
                        color=color, fontsize=7.2, fontweight="bold",
                        bbox={"boxstyle": "round,pad=0.12", "fc": "white",
                              "ec": "none", "alpha": 0.90})
        ax.axhline(0, color=INK, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
        ax.set_title(f"{title}\n{contrast}", pad=5, color=NAVY,
                     fontweight="bold", linespacing=1.1)
        ax.set_xticks(x, scales)
        ax.set_xlim(x[0] - 0.28, x[-1] + 0.28)
        ax.set_ylim(-100, 100)
        ax.set_yticks((-100, -50, 0, 50, 100))
        ax.set_xlabel("Reader parameters")
        ax.text(0.98, 0.03, "10 panels", transform=ax.transAxes,
                ha="right", va="bottom", color=MUTED, fontsize=6.4)
        style_axes(ax)
    axes[0].set_ylabel("Accuracy difference (points)")
    helpers["finish"](fig, axes, output, (7.48, 2.72), label_offset=(-16, 5))


def plot_factorial(frame: pd.DataFrame, intervals: pd.DataFrame,
                   output: Path, helpers: dict) -> None:
    models = ("1.5B", "3B", "7B")
    slots = (2, 3, 5)
    x = np.log2(np.asarray((1.5, 3.0, 7.0)))
    enc = {
        2: (BLUE, "o", "-", "$K=2$"),
        3: (TEAL, "s", "--", "$K=3$"),
        5: (ORANGE, "D", "-.", "$K=5$"),
    }
    fig, (ax, forest) = plt.subplots(
        1, 2, figsize=(7.48, 3.12), gridspec_kw={"width_ratios": [1.15, 1]},
        constrained_layout=True,
    )
    cell_means: dict[tuple[str, int], float] = {}
    # Panel trajectories expose all nine independent factorial panels.
    for slot in slots:
        for panel in sorted(frame.panel.unique()):
            values = [frame[(frame.model == model) & (frame.slots == slot)
                            & (frame.panel == panel)].gain_nll.iloc[0]
                      for model in models]
            ax.plot(x, values, color=POINT, alpha=0.18, linewidth=0.55, zorder=1)
    for slot in slots:
        summaries = []
        for model in models:
            values = frame[(frame.model == model) &
                           (frame.slots == slot)].gain_nll.to_numpy()
            summary = ci95(values)
            cell_means[(model, slot)] = summary[0]
            summaries.append(summary)
        means = np.asarray([v[0] for v in summaries])
        lows = np.asarray([v[1] for v in summaries])
        highs = np.asarray([v[2] for v in summaries])
        color, marker, linestyle, label = enc[slot]
        ax.errorbar(x, means, yerr=(means - lows, highs - means), color=color,
                    marker=marker, linestyle=linestyle, markerfacecolor="white",
                    markeredgewidth=1.1, markersize=5.0, linewidth=1.35,
                    elinewidth=0.9, capsize=2.2, label=label, zorder=3)
    diagonal = np.asarray([cell_means[("1.5B", 2)], cell_means[("3B", 3)],
                           cell_means[("7B", 5)]])
    ax.plot(x, diagonal, color=NAVY, linewidth=2.5, marker="*", markersize=8,
            markerfacecolor=ORANGE, markeredgecolor=NAVY,
            label="Co-scaled", zorder=5)
    for xx, value in zip(x, diagonal):
        ax.annotate(fmt(value, 2), (xx, value), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=7.2,
                    fontweight="bold", color=NAVY)
    ax.set_xticks(x, models)
    ax.set_xlabel("Qwen2.5 reader parameters")
    ax.set_ylabel("NLL gain over Foundation (nats)")
    ax.set_ylim(0, 8.35)
    ax.set_yticks((0, 2, 4, 6, 8))
    ax.set_title("Full 3 x 3 reader-state factorial", color=NAVY,
                 fontweight="bold")
    ax.legend(frameon=False, ncol=4, loc="upper left", columnspacing=0.8,
              handlelength=1.7, borderaxespad=0.15, fontsize=6.6)
    style_axes(ax)

    labels = ["Diagonal 1.5-3B", "Diagonal 3-7B",
              "Interaction 1.5-3B", "Interaction 3-7B",
              "7B refinement K3-K5"]
    y = np.arange(len(labels))[::-1]
    means = intervals["mean"].to_numpy()
    lows = intervals.ci_low.to_numpy()
    highs = intervals.ci_high.to_numpy()
    colors = [ORANGE, ORANGE, TEAL, TEAL, BLUE]
    markers = ["D", "D", "s", "s", "o"]
    for yy, mean, low, high, color, marker in zip(y, means, lows, highs,
                                                  colors, markers):
        forest.errorbar(mean, yy, xerr=[[mean - low], [high - mean]], fmt=marker,
                        color=color, markerfacecolor="white", markeredgewidth=1.1,
                        markersize=5.2, elinewidth=1.25, capsize=2.3, zorder=3)
        forest.annotate(fmt(mean, 2), (high, yy), xytext=(5, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=6.7, color=color, fontweight="bold")
    forest.axvline(0, color=INK, linewidth=0.8, linestyle=(0, (3, 2)))
    forest.set_yticks(y, labels)
    forest.set_xlabel("Gain contrast (simultaneous 95% CI)")
    forest.set_xlim(-0.25, 6.1)
    forest.set_xticks((0, 1, 2, 3, 4, 5, 6))
    forest.set_title("All mechanism contrasts are positive", color=NAVY,
                     fontweight="bold")
    style_axes(forest, grid="x")
    helpers["finish"](fig, (ax, forest), output, (7.48, 3.12),
                      label_offset=(-17, 5))


def plot_breadth(frame: pd.DataFrame, output: Path, helpers: dict) -> None:
    means_by_model = frame.groupby("model").gain.mean().sort_values()
    order = tuple(means_by_model.index)
    fig, ax = plt.subplots(figsize=(7.48, 2.60), constrained_layout=True)
    y = np.arange(len(order))
    offsets = np.linspace(-0.13, 0.13, 10)
    for yy, model in zip(y, order):
        values = frame[frame.model == model].sort_values("panel").gain.to_numpy() * 100
        ax.scatter(values, np.full(10, yy) + offsets, s=12, facecolor="white",
                   edgecolor=POINT, linewidth=0.55, zorder=2)
        mean, low, high = ci95(values)
        color = ORANGE if model == order[-1] else BLUE
        marker = "D" if model == order[-1] else "o"
        ax.errorbar(mean, yy, xerr=[[mean - low], [high - mean]], fmt=marker,
                    color=color, markerfacecolor="white", markeredgewidth=1.2,
                    markersize=5.8, elinewidth=1.4, capsize=2.6, zorder=4)
        ax.annotate(f"{fmt(mean, 1)}  [{fmt(low, 1)}, {fmt(high, 1)}]", (high, yy),
                    xytext=(6, 0), textcoords="offset points", ha="left",
                    va="center", fontsize=7.0, color=color, fontweight="bold")
    ax.axvline(0, color=INK, linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_yticks(y, order)
    ax.set_ylim(-0.55, len(order) - 0.45)
    ax.set_xlim(0, 100)
    ax.set_xticks((0, 20, 40, 60, 80, 100))
    ax.set_xlabel("ASCENT - Foundation accuracy (points)")
    ax.set_title("Strong gains across five public 7B-8B reader families",
                 color=NAVY, fontweight="bold")
    ax.text(0.995, 0.03, "dots: 10 task panels  |  symbols: mean ± 95% CI",
            transform=ax.transAxes, ha="right", va="bottom", color=MUTED,
            fontsize=6.5)
    style_axes(ax, grid="x")
    helpers["finish"](fig, (ax,), output, (7.48, 2.60), panel_labels=False)


def plot_systems(frame: pd.DataFrame, output: Path, helpers: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.48, 4.45),
                             constrained_layout=True)
    models = frame.model.tolist()
    x = np.arange(3)
    specs = (
        ("state_bytes", "External state", "Bytes", BLUE, "o", None, ".0f"),
        ("state_model_ratio", "Vanishing relative footprint", r"State/model ratio ($10^{-6}$)",
         TEAL, "s", None, ".2f"),
        ("decode_ratio", "Shorter isolated decoding", "ASCENT / Foundation time",
         ORANGE, "D", None, ".3f"),
        ("flop_reduction", "Supported-operator compute", "FLOP reduction (×)",
         PURPLE, "^", None, ".1f"),
    )
    for ax, (column, title, ylabel, color, marker, scale, value_fmt) in zip(
        axes.ravel(), specs
    ):
        values = frame[column].to_numpy()
        if column == "state_model_ratio":
            values = values * 1e6
        ax.plot(x, values, color=color, marker=marker, markerfacecolor="white",
                markeredgewidth=1.2, linewidth=1.8, markersize=5.6, zorder=3)
        for xx, value in zip(x, values):
            ax.annotate(format(value, value_fmt), (xx, value), xytext=(0, 8),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=6.8, fontweight="bold", color=color)
        ax.set_xticks(x, models)
        ax.set_ylabel(ylabel)
        if scale:
            ax.set_yscale(scale)
        ax.set_title(title, color=NAVY, fontweight="bold", pad=5)
        ax.margins(x=0.16, y=0.20)
        style_axes(ax)
    helpers["finish"](fig, axes.ravel(), output, (7.48, 4.45),
                      label_offset=(-17, 5))


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.skill_scripts))
    from export_figure import export_figure
    from layout_tools import add_panel_labels, finalize_figure
    from setup_style import setup_style
    from visual_qa import audit_layout, print_report, render_preview

    setup_style(journal="general", lang="en", constrained_layout=True,
                use_sciplots=False)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    def finish(fig, axes, output, size, label_offset=(-16, 5), panel_labels=True):
        finalize_figure(fig)
        fig.canvas.draw()
        if panel_labels:
            add_panel_labels(fig, axes=axes, style="upper_paren",
                             x_offset_pt=label_offset[0],
                             y_offset_pt=label_offset[1])
        verdict = print_report(audit_layout(fig))
        if verdict == "FAIL":
            raise RuntimeError(f"layout audit failed for {output}")
        render_preview(fig, str(output) + "_preview.png", dpi=180)
        export_figure(fig, str(output), formats=("pdf", "svg", "png"),
                      size_inches=size, dpi=600, grayscale_preview=True,
                      tight=False)
        plt.close(fig)

    data = args.data_dir
    primary = pd.read_csv(data / "primary_panel.csv")
    factorial = pd.read_csv(data / "factorial_panel.csv")
    intervals = pd.read_csv(data / "interaction_intervals.csv")
    breadth = pd.read_csv(data / "around7b_panel.csv")
    systems = pd.read_csv(data / "systems.csv")
    validate(primary, factorial, intervals, breadth, systems)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    helpers = {"finish": finish}
    plot_primary(primary, args.output_dir / "primary_scaling", helpers)
    plot_factorial(factorial, intervals, args.output_dir / "factorial_interaction", helpers)
    plot_breadth(breadth, args.output_dir / "large_model_breadth", helpers)
    plot_systems(systems, args.output_dir / "systems_efficiency", helpers)


if __name__ == "__main__":
    main()
