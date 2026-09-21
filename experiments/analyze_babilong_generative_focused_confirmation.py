#!/usr/bin/env python3
"""Analyze the preregistered ten-panel focused generative confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


CELLS = (
    ("smollm2-135m-instruct", "node1", 2, 134_515_008),
    ("smollm2-360m-instruct", "node2", 2, 361_821_120),
    ("smollm2-360m-instruct", "node2", 3, 361_821_120),
    ("smollm2-1p7b-instruct", "node1", 2, 1_711_376_384),
    ("smollm2-1p7b-instruct", "node1", 3, 1_711_376_384),
)
T95_DF9 = 2.2621571627409915


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_ci(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size != 10:
        raise ValueError("focused confirmation requires exactly ten panels")
    standard_error = float(array.std(ddof=1) / np.sqrt(array.size))
    mean = float(array.mean())
    return {
        "values": array.tolist(),
        "mean": mean,
        "standard_error": standard_error,
        "ci95_low": mean - T95_DF9 * standard_error,
        "ci95_high": mean + T95_DF9 * standard_error,
    }


def paired(left: list[float], right: list[float]) -> list[float]:
    return (np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)).tolist()


def values(
    results: dict[tuple[str, int], list[dict[str, Any]]],
    endpoint: str,
    slots: int,
    field: str,
) -> list[float]:
    return [float(row[field]["mean"]) for row in results[(endpoint, slots)]]


def focused_estimands(
    results: dict[tuple[str, int], list[dict[str, Any]]],
) -> dict[str, Any]:
    small = "smollm2-135m-instruct"
    middle = "smollm2-360m-instruct"
    large = "smollm2-1p7b-instruct"
    middle_state_effect = paired(
        values(results, middle, 3, "ascent"),
        values(results, middle, 2, "ascent"),
    )
    large_state_effect = paired(
        values(results, large, 3, "ascent"),
        values(results, large, 2, "ascent"),
    )
    return {
        "first_coscale_excess": mean_ci(
            paired(
                values(results, middle, 2, "gain"),
                values(results, small, 2, "gain"),
            )
        ),
        "second_registered_interaction": mean_ci(
            paired(large_state_effect, middle_state_effect)
        ),
        "upper_diagonal_gain_increment": mean_ci(
            paired(
                values(results, large, 3, "gain"),
                values(results, middle, 2, "gain"),
            )
        ),
        "upper_coscale_excess": mean_ci(
            paired(
                values(results, large, 3, "gain"),
                values(results, middle, 3, "gain"),
            )
        ),
    }


def collect(
    root: Path,
    config_path: Path,
    run_commit: str,
) -> tuple[dict[tuple[str, int], list[dict[str, Any]]], list[str], dict[str, Any]]:
    config = load(config_path)
    config_sha256 = sha256_file(config_path)
    panels = list(config["panels"])
    if len(panels) != 10:
        raise RuntimeError("expected ten panels")
    endpoints = {endpoint["name"]: endpoint for endpoint in config["endpoints"]}
    results: dict[tuple[str, int], list[dict[str, Any]]] = {}
    checks: list[bool] = []
    for endpoint, node, slots, _ in CELLS:
        rows = [
            load(
                root
                / node
                / f"generative_slots_{slots}_{panel}_{endpoint}.json"
            )
            for panel in panels
        ]
        results[(endpoint, slots)] = rows
        for panel, row in zip(panels, rows, strict=True):
            checks.extend(
                (
                    row["endpoint"] == endpoints[endpoint],
                    row["condition"]["name"] == f"generative_slots_{slots}",
                    row["condition"]["fact_slots"] == slots,
                    row["condition"]["readout_path"] == "foundation_generation",
                    row["condition"]["memory_source"] == "relevant",
                    row["environment"]["git_commit"] == run_commit,
                    not row["environment"]["git_dirty"],
                    row["config"]["sha256"] == config_sha256,
                    row["data"]["sha256"]
                    == config["panel_sha256_by_name"][panel],
                    row["model_files"] == [endpoints[endpoint]["model_artifact"]],
                    row["status"] == config["status"],
                    row["parser_accuracy"] == 1.0,
                    len(row["predictions"]) == 120,
                    row["data"]["path"].endswith(f"{panel}.jsonl"),
                )
            )

    def foundation_projection(row: dict[str, Any]) -> list[tuple[Any, ...]]:
        return [
            (
                prediction["row_id"],
                prediction["task"],
                prediction["target"],
                prediction["foundation_output"],
                prediction["foundation_answer"],
                prediction["foundation_score"],
                prediction["foundation_prompt_tokens"],
            )
            for prediction in row["predictions"]
        ]

    foundation_equal = True
    for endpoint in ("smollm2-360m-instruct", "smollm2-1p7b-instruct"):
        for panel_index in range(len(panels)):
            foundation_equal &= foundation_projection(
                results[(endpoint, 2)][panel_index]
            ) == foundation_projection(results[(endpoint, 3)][panel_index])

    panel_alignment = True
    for panel_index in range(len(panels)):
        identities = []
        for rows in results.values():
            identities.append(
                [
                    (prediction["row_id"], prediction["task"], prediction["target"])
                    for prediction in rows[panel_index]["predictions"]
                ]
            )
        panel_alignment &= all(identity == identities[0] for identity in identities[1:])

    details = {
        "config_sha256": config_sha256,
        "run_commit": run_commit,
        "all_cell_checks_pass": all(checks),
        "foundation_equal_across_repeated_state_sizes": foundation_equal,
        "all_cells_row_task_target_aligned": panel_alignment,
    }
    details["provenance_and_alignment_pass"] = (
        details["all_cell_checks_pass"]
        and foundation_equal
        and panel_alignment
    )
    return results, panels, details


def analyze(root: Path, config_path: Path, run_commit: str) -> dict[str, Any]:
    results, panels, provenance = collect(root, config_path, run_commit)
    cells = {
        f"{endpoint}__slots_{slots}": {
            "model_parameters": parameters,
            "state_slots": slots,
            "foundation_accuracy": mean_ci(values(results, endpoint, slots, "foundation")),
            "ascent_accuracy": mean_ci(values(results, endpoint, slots, "ascent")),
            "absolute_gain": mean_ci(values(results, endpoint, slots, "gain")),
        }
        for endpoint, _, slots, parameters in CELLS
    }
    estimands = focused_estimands(results)
    positive = {key: value["ci95_low"] > 0.0 for key, value in estimands.items()}
    gate = provenance["provenance_and_alignment_pass"] and all(positive.values())
    return {
        "schema_version": 1,
        "root": str(root),
        "panels": panels,
        "panel_count": len(panels),
        "interval": "two-sided Student-t 95% interval over panels (df=9)",
        "provenance": provenance,
        "cells": cells,
        "primary_estimands": estimands,
        "estimand_lcbs_positive": positive,
        "focused_confirmation_primary_pass": gate,
        "interpretation": "Prospective independent confirmation of marginal estimands; the earlier five-panel primary outcome remains unchanged.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.root, args.config, args.run_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
