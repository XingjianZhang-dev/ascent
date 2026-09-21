#!/usr/bin/env python3
"""Analyze frozen certified-path controls against the primary confirmation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_babilong_structured_aggregation import mean_ci


ENDPOINTS = (
    ("smollm2-135m", "node1"),
    ("smollm2-360m-node2", "node2"),
    ("smollm2-1p7b", "node1"),
)
CONDITIONS = (
    "fixed_one_slot",
    "irrelevant_structured",
    "matched_raw_event_fifo",
    "generic_bm25",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def analyze(primary_root: Path, control_root: Path) -> dict[str, Any]:
    primary: dict[str, list[dict[str, Any]]] = {}
    controls: dict[str, dict[str, list[dict[str, Any]]]] = {
        condition: {} for condition in CONDITIONS
    }
    provenance_checks: list[bool] = []
    for endpoint, node in ENDPOINTS:
        primary[endpoint] = [
            load_json(primary_root / node / f"panel{panel}_{endpoint}.json")
            for panel in range(1, 6)
        ]
        for condition in CONDITIONS:
            controls[condition][endpoint] = [
                load_json(
                    control_root / node / f"{condition}_panel{panel}_{endpoint}.json"
                )
                for panel in range(1, 6)
            ]
            for primary_result, control_result in zip(
                primary[endpoint], controls[condition][endpoint], strict=True
            ):
                provenance_checks.extend(
                    (
                        primary_result["environment"]["git_commit"]
                        == control_result["environment"]["git_commit"],
                        primary_result["config"]["sha256"]
                        == control_result["config"]["sha256"],
                        primary_result["data"]["sha256"]
                        == control_result["data"]["sha256"],
                        primary_result["condition"]["readout_path"]
                        == control_result["condition"]["readout_path"]
                        == "certified_evidence",
                        control_result["condition"]["name"] == condition,
                        [row["row_id"] for row in primary_result["predictions"]]
                        == [row["row_id"] for row in control_result["predictions"]],
                        [
                            row["foundation_score"]
                            for row in primary_result["predictions"]
                        ]
                        == [
                            row["foundation_score"]
                            for row in control_result["predictions"]
                        ],
                        not primary_result["environment"]["git_dirty"],
                        not control_result["environment"]["git_dirty"],
                    )
                )

    summaries: dict[str, Any] = {}
    for condition in CONDITIONS:
        endpoint_rows: dict[str, Any] = {}
        for endpoint, _ in ENDPOINTS:
            primary_accuracy = [
                result["ascent"]["mean"] for result in primary[endpoint]
            ]
            control_accuracy = [
                result["ascent"]["mean"] for result in controls[condition][endpoint]
            ]
            deltas = (
                np.asarray(primary_accuracy) - np.asarray(control_accuracy)
            ).tolist()
            endpoint_rows[endpoint] = {
                "primary_accuracy": mean_ci(primary_accuracy),
                "control_accuracy": mean_ci(control_accuracy),
                "primary_minus_control": mean_ci(deltas),
                "primary_noninferior_on_every_panel": all(
                    value >= 0 for value in deltas
                ),
            }
        summaries[condition] = endpoint_rows

    fixed_advantage = np.asarray(
        [
            summaries["fixed_one_slot"][endpoint]["primary_minus_control"]["values"]
            for endpoint, _ in ENDPOINTS
        ],
        dtype=np.float64,
    ).T
    fixed_adjacent = {
        f"{ENDPOINTS[index][0]}_to_{ENDPOINTS[index + 1][0]}": mean_ci(
            (fixed_advantage[:, index + 1] - fixed_advantage[:, index]).tolist()
        )
        for index in (0, 1)
    }
    findings = {
        "all_provenance_and_foundation_cache_checks": all(provenance_checks),
        "fixed_state_advantage_by_endpoint": {
            endpoint: summaries["fixed_one_slot"][endpoint]["primary_minus_control"]
            for endpoint, _ in ENDPOINTS
        },
        "fixed_state_adjacent_interactions": fixed_adjacent,
        "fixed_state_both_interaction_lcbs_positive": all(
            interval["ci95_low"] > 0 for interval in fixed_adjacent.values()
        ),
        "irrelevant_all_endpoint_lcbs_positive": all(
            summaries["irrelevant_structured"][endpoint]["primary_minus_control"][
                "ci95_low"
            ]
            > 0
            for endpoint, _ in ENDPOINTS
        ),
        "bm25_all_endpoint_lcbs_positive": all(
            summaries["generic_bm25"][endpoint]["primary_minus_control"]["ci95_low"] > 0
            for endpoint, _ in ENDPOINTS
        ),
        "raw_fifo_primary_noninferior_on_every_panel": all(
            summaries["matched_raw_event_fifo"][endpoint][
                "primary_noninferior_on_every_panel"
            ]
            for endpoint, _ in ENDPOINTS
        ),
        "raw_fifo_medium_endpoint_lcb_positive": summaries["matched_raw_event_fifo"][
            "smollm2-360m-node2"
        ]["primary_minus_control"]["ci95_low"]
        > 0,
        "raw_fifo_large_endpoint_mean_positive_lcb_borderline": summaries[
            "matched_raw_event_fifo"
        ]["smollm2-1p7b"]["primary_minus_control"],
    }
    return {
        "schema_version": 1,
        "primary_root": str(primary_root),
        "control_root": str(control_root),
        "endpoints": [endpoint for endpoint, _ in ENDPOINTS],
        "panels": 5,
        "conditions": summaries,
        "findings": findings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.primary_root, args.control_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
