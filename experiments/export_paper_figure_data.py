#!/usr/bin/env python3
"""Export panel-level primary evidence for reproducible manuscript figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ENDPOINTS = (
    ("qwen2p5-0p5b-instruct", 0.5, "0.5B"),
    ("qwen2p5-1p5b-instruct", 1.5, "1.5B"),
    ("qwen2p5-3b-instruct", 3.0, "3B"),
)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--qrag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    official = load(args.official)
    semantic = load(args.semantic)
    qrag = load(args.qrag)
    if not official["primary_gate_pass"] or not semantic["primary_gate_pass"]:
        raise RuntimeError("confirmation primary gate is not passed")

    rows: list[dict[str, object]] = []
    for endpoint, scale_b, scale_label in ENDPOINTS:
        sources = (
            (
                "Official 16K",
                "ASCENT - Foundation",
                official["endpoint_results"][endpoint]["primary"]["gain"],
            ),
            (
                "Semantic holdout",
                "ASCENT - Foundation",
                semantic["endpoint_results"][endpoint]["primary"]["gain"],
            ),
            (
                "Direct peer",
                "ASCENT - Q-RAG",
                qrag["endpoint_results"][endpoint]["overall"][
                    "ascent_minus_qrag"
                ],
            ),
        )
        for study, metric, result in sources:
            values = result["panel_values"]
            if len(values) != 10 or result["clusters"] != 10:
                raise RuntimeError(f"expected ten panel clusters for {study} {endpoint}")
            for panel, value in enumerate(values, start=1):
                rows.append(
                    {
                        "study": study,
                        "metric": metric,
                        "scale_b": scale_b,
                        "scale_label": scale_label,
                        "panel": panel,
                        "value": value,
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
