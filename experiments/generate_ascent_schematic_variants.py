#!/usr/bin/env python3
"""Generate three editable Elsevier-width SVG concepts for ASCENT."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from xml.sax.saxutils import escape


W = 190.0
FONT = "Helvetica, Arial, Liberation Sans, sans-serif"
NAVY, BLUE, TEAL, ORANGE = "#173B57", "#0072B2", "#009E73", "#E69F00"
INK, MUTED, RULE = "#20262B", "#66727B", "#CBD5DC"
PALE_BLUE, PALE_TEAL = "#EAF4FA", "#EAF7F2"
PALE_ORANGE, PALE_NEUTRAL = "#FFF5DD", "#F4F6F7"


def num(value: float) -> str:
    return f"{value:.1f}"


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")


def header(height: float, background: str = "#FFFFFF") -> str:
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
     width="{num(W)}mm" height="{num(height)}mm"
     viewBox="0 0 {num(W)} {num(height)}" version="1.1">
  <rect id="background" x="0" y="0" width="{num(W)}" height="{num(height)}"
        fill="{background}"/>
'''


def text(ident: str, x: float, y: float, value: str, *, size: float = 2.7,
         weight: str = "normal", fill: str = INK,
         anchor: str = "middle", italic: bool = False) -> str:
    italic_attr = ' font-style="italic"' if italic else ""
    return (
        f'    <text id="{slug(ident)}" x="{num(x)}" y="{num(y)}" '
        f'font-family="{FONT}" font-size="{num(size)}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}"{italic_attr}>{escape(value)}</text>\n'
    )


def multiline(ident: str, x: float, y: float, lines: tuple[str, ...], *,
              size: float = 2.5, gap: float = 3.6, weight: str = "normal",
              fill: str = INK, anchor: str = "middle") -> str:
    return "".join(
        text(f"{ident}-{i}", x, y + i * gap, line, size=size, weight=weight,
             fill=fill, anchor=anchor)
        for i, line in enumerate(lines)
    )


def rect(ident: str, x: float, y: float, w: float, h: float, *, fill: str,
         stroke: str = RULE, radius: float = 2.0, width: float = 0.4) -> str:
    return (
        f'    <rect id="{slug(ident)}" x="{num(x)}" y="{num(y)}" '
        f'width="{num(w)}" height="{num(h)}" rx="{num(radius)}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{num(width)}"/>\n'
    )


def line(ident: str, x1: float, y1: float, x2: float, y2: float, *,
         stroke: str = RULE, width: float = 0.4, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'    <line id="{slug(ident)}" x1="{num(x1)}" y1="{num(y1)}" '
        f'x2="{num(x2)}" y2="{num(y2)}" stroke="{stroke}" '
        f'stroke-width="{num(width)}"{dash_attr}/>\n'
    )


def arrow(ident: str, x1: float, y1: float, x2: float, y2: float, *,
          color: str = NAVY, width: float = 0.55) -> str:
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 10.0:
        raise ValueError(f"{ident}: {length:.1f} mm arrow is shorter than 10 mm")
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    px, py = -uy, ux
    bx, by = x2 - 2.2 * ux, y2 - 2.2 * uy
    p1 = (bx + 1.05 * px, by + 1.05 * py)
    p2 = (bx - 1.05 * px, by - 1.05 * py)
    return (
        line(ident, x1, y1, bx, by, stroke=color, width=width)
        + f'    <polygon id="{slug(ident)}-head" points="{num(x2)},{num(y2)} '
          f'{num(p1[0])},{num(p1[1])} {num(p2[0])},{num(p2[1])}" fill="{color}"/>\n'
    )


def panel_label(ident: str, x: float, content_y: float, value: str) -> str:
    return text(ident, x, content_y - 2.4, value, size=3.3, weight="bold",
                anchor="start")


def context_lines(x: float, y: float, widths: tuple[float, ...]) -> str:
    return "".join(
        line(f"context-line-{i}", x, y + i * 3.0, x + w, y + i * 3.0,
             stroke="#97A3AC", width=0.65)
        for i, w in enumerate(widths)
    )


def state_stack(x: float, y: float, w: float) -> str:
    parts: list[str] = []
    for i, (color, pale) in enumerate(((BLUE, PALE_BLUE), (TEAL, PALE_TEAL),
                                       (ORANGE, PALE_ORANGE))):
        parts.append(rect(f"state-layer-{i}", x, y + i * 5.2, w, 3.6,
                          fill=pale, stroke=color, radius=0.9, width=0.38))
    return "".join(parts)


def factorial_grid(x: float, y: float, cell: float = 8.3) -> str:
    # Only the three co-scaled values are printed here; the complete nine-cell
    # numeric matrix is displayed in the dedicated factorial result figure.
    values: tuple[tuple[float | None, ...], ...] = (
        (0.39, None, None), (None, 2.10, None), (None, None, 7.41)
    )
    parts: list[str] = []
    for row in range(3):
        for col in range(3):
            diagonal = row == col
            parts.append(rect(f"factorial-{row}-{col}", x + col * cell,
                              y + row * cell, cell - 0.6, cell - 0.6,
                              fill=PALE_ORANGE if diagonal else PALE_BLUE,
                              stroke=ORANGE if diagonal else RULE,
                              radius=0.7, width=0.7 if diagonal else 0.3))
            if values[row][col] is not None:
                parts.append(text(f"factorial-value-{row}-{col}",
                                  x + col * cell + (cell - 0.6) / 2,
                                  y + row * cell + 4.9,
                                  f"{values[row][col]:.2f}",
                                  size=2.15, weight="bold", fill=NAVY))
    return "".join(parts)


def variant_a() -> str:
    """Editorial triptych: interface above, theorem and identification below."""
    h = 105.0
    p = [header(h)]
    p.append('  <g id="panel-a" inkscape:groupmode="layer" inkscape:label="Panel A">\n')
    p.append(panel_label("label-a", 6.0, 13.0, "(A)"))
    p.append(text("mechanism-kicker", 15.0, 18.0, "MECHANISM", size=2.0,
                  weight="bold", fill=BLUE, anchor="start"))
    cards = (
        (8.0, 22.0, 31.0, 29.0, PALE_NEUTRAL, MUTED, "Long context", "query unseen"),
        (58.0, 22.0, 34.0, 29.0, PALE_BLUE, BLUE, "Causal writer", "typed evidence"),
        (111.0, 18.0, 37.0, 37.0, PALE_TEAL, TEAL, "Nested ASCENT state", "bounded refinement"),
        (167.0, 22.0, 17.0, 29.0, PALE_ORANGE, ORANGE, "Frozen", "reader"),
    )
    for i, (x, y, w, hh, fill, stroke, title, subtitle) in enumerate(cards):
        p.append(rect(f"mechanism-card-{i}", x, y, w, hh, fill=fill,
                      stroke=stroke, radius=2.3, width=0.5))
        p.append(text(f"mechanism-title-{i}", x + w / 2, y + 6.3, title,
                      size=2.65, weight="bold", fill=stroke))
        p.append(text(f"mechanism-subtitle-{i}", x + w / 2, y + hh - 3.3,
                      subtitle, size=2.0, fill=MUTED))
    p.append(context_lines(14.0, 33.0, (18.0, 15.0, 20.0, 13.0)))
    p.append(multiline("writer-detail", 75.0, 33.5,
                       ("target-blind parse", "deterministic write"), size=2.15,
                       gap=4.2, fill=INK))
    p.append(state_stack(118.0, 30.0, 23.0))
    p.append(multiline("reader-detail", 175.5, 33.6,
                       ("query", "readout"), size=2.15, gap=4.2,
                       weight="bold", fill=ORANGE))
    p.append(arrow("arrow-context-writer", 42.0, 39.0, 55.0, 39.0, color=BLUE))
    p.append(arrow("arrow-writer-state", 95.0, 39.0, 108.0, 39.0, color=BLUE))
    p.append(arrow("arrow-state-reader", 151.0, 39.0, 164.0, 39.0, color=NAVY))
    p.append('  </g>\n')

    p.append('  <g id="panel-b" inkscape:groupmode="layer" inkscape:label="Panel B">\n')
    p.append(panel_label("label-b", 6.0, 69.0, "(B)"))
    p.append(rect("theory-card", 8.0, 69.0, 82.0, 29.0, fill="#FFFFFF",
                  stroke=BLUE, radius=2.2, width=0.5))
    p.append(text("theory-kicker", 14.0, 75.0, "INFORMATION VALUE", size=2.0,
                  weight="bold", fill=BLUE, anchor="start"))
    p.append(text("theory-law", 49.0, 85.0,
                  "L*(Z) - L*(Z') = I(Y; Z' | Q, Z) > 0",
                  size=3.0, weight="bold", fill=NAVY))
    p.append(text("theory-meaning", 49.0, 92.7,
                  "Nested refinements add target-relevant conditional information",
                  size=2.2, fill=MUTED))
    p.append('  </g>\n')

    p.append('  <g id="panel-c" inkscape:groupmode="layer" inkscape:label="Panel C">\n')
    p.append(panel_label("label-c", 99.0, 69.0, "(C)"))
    p.append(rect("evidence-card", 101.0, 69.0, 83.0, 29.0, fill="#FFFFFF",
                  stroke=TEAL, radius=2.2, width=0.5))
    p.append(text("evidence-kicker", 107.0, 75.0, "FACTORIAL IDENTIFICATION",
                  size=2.0, weight="bold", fill=TEAL, anchor="start"))
    p.append(factorial_grid(108.0, 77.5, 6.2))
    p.append(text("evidence-gains", 153.5, 83.0, "0.39  <  2.10  <  7.41",
                  size=3.0, weight="bold", fill=NAVY))
    p.append(text("evidence-interactions", 153.5, 90.0,
                  "both model x state interactions > 0", size=2.25,
                  weight="bold", fill=TEAL))
    p.append(text("evidence-estimand", 153.5, 95.0,
                  "NLL gain over Foundation (nats)", size=2.0, fill=MUTED))
    p.append('  </g>\n</svg>\n')
    return "".join(p)


def variant_b() -> str:
    """Central-state composition with theory and evidence as balanced wings."""
    h = 99.0
    p = [header(h, "#FBFCFD")]
    p.append(panel_label("label-a", 6.0, 14.0, "(A)"))
    p.append(rect("left-theory", 8.0, 14.0, 48.0, 68.0, fill="#FFFFFF",
                  stroke=BLUE, radius=3.0, width=0.5))
    p.append(text("left-kicker", 14.0, 22.0, "THEORY", size=2.1,
                  weight="bold", fill=BLUE, anchor="start"))
    p.append(multiline("left-premise", 32.0, 34.0,
                       ("Nested state", "adds conditional", "information"),
                       size=3.0, gap=5.0, weight="bold", fill=NAVY))
    p.append(text("left-law", 32.0, 57.0, "I(Y; Z' | Q, Z) > 0", size=2.7,
                  weight="bold", fill=BLUE))
    p.append(text("left-result", 32.0, 73.0, "strict oracle loss reduction",
                  size=2.15, fill=MUTED))

    p.append(panel_label("label-b", 69.0, 14.0, "(B)"))
    p.append(rect("state-hero", 71.0, 14.0, 48.0, 68.0, fill=PALE_TEAL,
                  stroke=TEAL, radius=4.0, width=0.7))
    p.append(text("state-kicker", 77.0, 22.0, "ASCENT INTERFACE", size=2.1,
                  weight="bold", fill=TEAL, anchor="start"))
    p.append(text("state-title", 95.0, 32.0, "Target-blind state", size=3.2,
                  weight="bold", fill=NAVY))
    p.append(state_stack(81.0, 40.0, 28.0))
    p.append(multiline("state-description", 95.0, 61.5,
                       ("causal write", "query-conditioned read", "frozen reader"),
                       size=2.25, gap=4.2, fill=MUTED))
    p.append(arrow("theory-to-state", 58.5, 48.0, 68.5, 48.0, color=BLUE))

    p.append(panel_label("label-c", 132.0, 14.0, "(C)"))
    p.append(rect("right-evidence", 134.0, 14.0, 48.0, 68.0, fill="#FFFFFF",
                  stroke=ORANGE, radius=3.0, width=0.5))
    p.append(text("right-kicker", 140.0, 22.0, "MEASUREMENT", size=2.1,
                  weight="bold", fill=ORANGE, anchor="start"))
    p.append(factorial_grid(145.0, 29.0, 8.8))
    p.append(text("right-diagonal", 158.0, 60.0, "co-scaled diagonal",
                  size=2.25, weight="bold", fill=ORANGE))
    p.append(text("right-gains", 158.0, 68.0, "0.39 < 2.10 < 7.41",
                  size=3.0, weight="bold", fill=NAVY))
    p.append(text("right-interactions", 158.0, 75.0, "interactions > 0",
                  size=2.25, weight="bold", fill=TEAL))
    p.append(arrow("state-to-evidence", 121.5, 48.0, 131.5, 48.0, color=ORANGE))
    p.append(text("bottom-thesis", 95.0, 93.0,
                  "Reader scale amplifies the value of nested external state.",
                  size=3.0, weight="bold", fill=NAVY))
    p.append('</svg>\n')
    return "".join(p)


def variant_c() -> str:
    """Two-row narrative emphasizing co-scaling as a measurable law."""
    h = 111.0
    p = [header(h)]
    p.append(panel_label("label-a", 6.0, 14.0, "(A)"))
    p.append(text("top-title", 10.0, 20.0, "TARGET-BLIND STATE CONSTRUCTION",
                  size=2.1, weight="bold", fill=BLUE, anchor="start"))
    x_cards = (8.0, 61.0, 114.0, 167.0)
    widths = (28.0, 30.0, 31.0, 15.0)
    titles = ("Context", "Writer", "Nested state", "LM")
    fills = (PALE_NEUTRAL, PALE_BLUE, PALE_TEAL, PALE_ORANGE)
    strokes = (MUTED, BLUE, TEAL, ORANGE)
    for i, (x, w, title, fill, stroke) in enumerate(zip(x_cards, widths, titles,
                                                        fills, strokes)):
        p.append(rect(f"top-card-{i}", x, 25.0, w, 24.0, fill=fill,
                      stroke=stroke, radius=2.2, width=0.5))
        p.append(text(f"top-card-title-{i}", x + w / 2, 32.0, title,
                      size=2.7, weight="bold", fill=stroke))
    p.append(context_lines(14.0, 38.0, (16.0, 12.0, 17.0)))
    p.append(multiline("top-writer", 76.0, 39.0,
                       ("causal parse", "typed write"), size=2.1, gap=4.0))
    p.append(state_stack(120.0, 34.5, 19.0))
    p.append(text("top-lm-read", 174.5, 41.0, "read", size=2.1,
                  weight="bold", fill=ORANGE))
    p.append(arrow("top-arrow-1", 39.0, 38.0, 58.0, 38.0, color=BLUE))
    p.append(arrow("top-arrow-2", 94.0, 38.0, 111.0, 38.0, color=BLUE))
    p.append(arrow("top-arrow-3", 148.0, 38.0, 164.0, 38.0, color=NAVY))

    p.append(panel_label("label-b", 6.0, 67.0, "(B)"))
    p.append(rect("bottom-theory", 8.0, 67.0, 54.0, 34.0, fill="#FFFFFF",
                  stroke=BLUE, radius=2.2, width=0.5))
    p.append(text("bottom-theory-kicker", 14.0, 74.0, "PROPOSITION",
                  size=2.0, weight="bold", fill=BLUE, anchor="start"))
    p.append(text("bottom-theory-law", 35.0, 84.0,
                  "Delta L* = I(Y; Z' | Q, Z)", size=2.8,
                  weight="bold", fill=NAVY))
    p.append(text("bottom-theory-meaning", 35.0, 94.0,
                  "refinement value is conditional information",
                  size=2.0, fill=MUTED))

    p.append(panel_label("label-c", 74.0, 67.0, "(C)"))
    p.append(rect("bottom-scale", 76.0, 67.0, 106.0, 34.0, fill="#FFFFFF",
                  stroke=TEAL, radius=2.2, width=0.5))
    p.append(text("bottom-scale-kicker", 82.0, 74.0, "FINITE-READER TEST",
                  size=2.0, weight="bold", fill=TEAL, anchor="start"))
    p.append(factorial_grid(84.0, 77.0, 6.7))
    for i, (model, state, gain) in enumerate((("1.5B", "K2", "0.39"),
                                               ("3B", "K3", "2.10"),
                                               ("7B", "K5", "7.41"))):
        x = 119.0 + i * 20.0
        p.append(text(f"scale-combo-{i}", x, 82.0, f"{model} + {state}",
                      size=2.15, weight="bold",
                      fill=BLUE if i < 2 else ORANGE))
        p.append(text(f"scale-gain-{i}", x, 91.0, gain, size=3.2,
                      weight="bold", fill=NAVY))
    p.append(arrow("scale-arrow-1", 127.0, 87.0, 137.0, 87.0, color=ORANGE))
    p.append(arrow("scale-arrow-2", 147.0, 87.0, 157.0, 87.0, color=ORANGE))
    p.append(text("scale-interactions", 149.0, 97.0,
                  "positive model x state interactions", size=2.1,
                  weight="bold", fill=TEAL))
    p.append('</svg>\n')
    return "".join(p)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, svg in {"A": variant_a(), "B": variant_b(), "C": variant_c()}.items():
        (args.output_dir / f"ascent_method_variant_{name}.svg").write_text(svg)


if __name__ == "__main__":
    main()
