#!/usr/bin/env python3
"""Render the primary Qwen confirmation table from immutable analyses."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


ENDPOINTS = (
    ("qwen2p5-0p5b-instruct", "0.5B"),
    ("qwen2p5-1p5b-instruct", "1.5B"),
    ("qwen2p5-3b-instruct", "3B"),
)


def decimal(value: float) -> str:
    text = format(
        Decimal(f"{value:.12f}").quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        ),
        "f",
    )
    return text[1:] if text.startswith("0.") else text


def rows(label: str, analysis: dict) -> list[str]:
    rendered = []
    for index, (endpoint, reader) in enumerate(ENDPOINTS):
        primary = analysis["endpoint_results"][endpoint]["primary"]
        prefix = rf"\multirow{{3}}{{*}}{{{label}}}" if index == 0 else ""
        gain = primary["gain"]
        rendered.append(
            f"{prefix} & {reader} & {decimal(primary['foundation']['mean'])} & "
            f"{decimal(primary['ascent']['mean'])} & {decimal(gain['mean'])} "
            f"[{decimal(gain['ci95_low'])},{decimal(gain['ci95_high'])}] \\\\"
        )
    return rendered


def render(official: dict, semantic: dict) -> str:
    """Return the complete immutable primary-table source."""
    if not official["primary_gate_pass"] or not semantic["primary_gate_pass"]:
        raise RuntimeError("refusing to render a table from a failed primary gate")
    lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Split & Reader & Foundation & \ascent{} & Gain [95\% CI] \\",
        r"\midrule",
        *rows("Official 16K", official),
        r"\midrule",
        *rows("Semantic holdout", semantic),
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    official = json.loads(args.official.read_text())
    semantic = json.loads(args.semantic.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(official, semantic))


if __name__ == "__main__":
    main()
