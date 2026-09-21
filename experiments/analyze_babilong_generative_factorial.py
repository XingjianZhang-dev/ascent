#!/usr/bin/env python3
"""Analyze the frozen 3x3 generative BABILong model-by-state factorial."""

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

from experiments.analyze_babilong_scale_panel import PANEL_T_95, mean_ci as panel_mean_ci


ENDPOINTS = (
    ("smollm2-135m-instruct", "node1", 134_515_008),
    ("smollm2-360m-instruct", "node2", 361_821_120),
    ("smollm2-1p7b-instruct", "node1", 1_711_376_384),
)
STATE_SLOTS = (1, 2, 3)
PANELS = (1, 2, 3)
COMMIT = "693c41455d3a431a6da6880d64eb791ec56a6592"
RESULT_STATUS = "posthoc_code_audit_correction_rerun_no_independence_claim"
CONFIG_SHA256 = "5085ca195269ae38830d8967cd0e0ae3a308764b1a83cd03266229343fc546a6"
EXPECTED_DATA_SHA256 = {
    1: "dea69dbd409f4f1dc01068da99114fa3bfe5d6caf2d32e72c39e3a43b9752b85",
    2: "2902dbf7993438a249e5212a09d5eb9a02c3f17e18613319ee3375f7fa957bf5",
    3: "705cc7d28ff041719ea66b8d7e27f3182d05f60325da47a52d3dae12dfb05492",
}
EXPECTED_MODEL_FILES = {
    "smollm2-135m-instruct": [
        {
            "path": "model.safetensors",
            "bytes": 269060552,
            "sha256": "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c",
        }
    ],
    "smollm2-360m-instruct": [
        {
            "path": "model.safetensors",
            "bytes": 723674912,
            "sha256": "e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e",
        }
    ],
    "smollm2-1p7b-instruct": [
        {
            "path": "model.safetensors",
            "bytes": 3422777952,
            "sha256": "f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8",
        }
    ],
}
FOUNDATION_PREDICTION_FIELDS = (
    "row_id",
    "task",
    "target",
    "foundation_output",
    "foundation_answer",
    "foundation_score",
    "foundation_prompt_tokens",
)


def mean_ci(values: list[float]) -> dict[str, Any]:
    """Three-panel Student-t interval (df=2), not the five-panel df=4 interval."""
    return panel_mean_ci(values, critical=PANEL_T_95)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def paired(left: list[float], right: list[float]) -> list[float]:
    return (np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)).tolist()


def values(
    results: dict[tuple[str, int], list[dict[str, Any]]],
    endpoint: str,
    slots: int,
    field: str,
) -> list[float]:
    return [float(row[field]["mean"]) for row in results[(endpoint, slots)]]


def analyze(root: Path) -> dict[str, Any]:
    results: dict[tuple[str, int], list[dict[str, Any]]] = {}
    checks: list[bool] = []
    for endpoint, node, _ in ENDPOINTS:
        for slots in STATE_SLOTS:
            rows = [
                load(
                    root
                    / node
                    / f"generative_slots_{slots}_panel{panel}_{endpoint}.json"
                )
                for panel in PANELS
            ]
            results[(endpoint, slots)] = rows
            for panel, row in zip(PANELS, rows, strict=True):
                checks.extend(
                    (
                        row["endpoint"]["name"] == endpoint,
                        row["condition"]["name"] == f"generative_slots_{slots}",
                        row["condition"]["fact_slots"] == slots,
                        row["condition"]["readout_path"] == "foundation_generation",
                        row["condition"]["memory_source"] == "relevant",
                        row["environment"]["git_commit"] == COMMIT,
                        not row["environment"]["git_dirty"],
                        row["config"]["sha256"] == CONFIG_SHA256,
                        row["data"]["sha256"] == EXPECTED_DATA_SHA256[panel],
                        row["model_files"] == EXPECTED_MODEL_FILES[endpoint],
                        row["status"] == RESULT_STATUS,
                        row["parser_accuracy"] == 1.0,
                        len(row["predictions"]) == 120,
                        row["data"]["path"].endswith(
                            f"neural_control_panel_{panel}.jsonl"
                        ),
                    )
                )

    config_hashes = {
        row["config"]["sha256"]
        for rows in results.values()
        for row in rows
    }
    data_hashes_by_panel = {
        panel: {
            results[(endpoint, slots)][panel - 1]["data"]["sha256"]
            for endpoint, _, _ in ENDPOINTS
            for slots in STATE_SLOTS
        }
        for panel in PANELS
    }
    foundation_equal = True
    aligned = True
    panel_alignment = True
    for endpoint, _, _ in ENDPOINTS:
        for panel_index in range(len(PANELS)):
            rows = [results[(endpoint, slots)][panel_index] for slots in STATE_SLOTS]
            foundation_equal &= all(
                [
                    {field: p.get(field) for field in FOUNDATION_PREDICTION_FIELDS}
                    for p in row["predictions"]
                ]
                == [
                    {field: p.get(field) for field in FOUNDATION_PREDICTION_FIELDS}
                    for p in rows[0]["predictions"]
                ]
                for row in rows[1:]
            )
            aligned &= all(
                [p["row_id"] for p in row["predictions"]]
                == [p["row_id"] for p in rows[0]["predictions"]]
                for row in rows[1:]
            )
    for panel_index in range(len(PANELS)):
        reference_predictions = results[(ENDPOINTS[0][0], STATE_SLOTS[0])][
            panel_index
        ]["predictions"]
        reference_identity = [
            (p["row_id"], p["task"], p["target"]) for p in reference_predictions
        ]
        panel_alignment &= all(
            [(p["row_id"], p["task"], p["target"]) for p in row["predictions"]]
            == reference_identity
            for rows in results.values()
            for row in [rows[panel_index]]
        )

    cells: dict[str, dict[str, Any]] = {}
    for endpoint, _, parameters in ENDPOINTS:
        cells[endpoint] = {
            str(slots): {
                "model_parameters": parameters,
                "state_slots": slots,
                "foundation_accuracy": mean_ci(values(results, endpoint, slots, "foundation")),
                "ascent_accuracy": mean_ci(values(results, endpoint, slots, "ascent")),
                "absolute_gain": mean_ci(values(results, endpoint, slots, "gain")),
            }
            for slots in STATE_SLOTS
        }

    state_effects: dict[str, Any] = {}
    for endpoint, _, _ in ENDPOINTS:
        state_effects[endpoint] = {
            f"slots_{low}_to_{high}": mean_ci(
                paired(
                    values(results, endpoint, high, "ascent"),
                    values(results, endpoint, low, "ascent"),
                )
            )
            for low, high in zip(STATE_SLOTS[:-1], STATE_SLOTS[1:], strict=True)
        }

    model_effects: dict[str, Any] = {}
    adjacent_models = list(zip(ENDPOINTS[:-1], ENDPOINTS[1:], strict=True))
    for (low_endpoint, _, _), (high_endpoint, _, _) in adjacent_models:
        model_effects[f"{low_endpoint}_to_{high_endpoint}"] = {
            f"slots_{slots}": mean_ci(
                paired(
                    values(results, high_endpoint, slots, "ascent"),
                    values(results, low_endpoint, slots, "ascent"),
                )
            )
            for slots in STATE_SLOTS
        }

    registered_interactions: dict[str, Any] = {}
    for index, ((low_endpoint, _, _), (high_endpoint, _, _)) in enumerate(
        adjacent_models
    ):
        low_slots, high_slots = STATE_SLOTS[index], STATE_SLOTS[index + 1]
        high_model_state = paired(
            values(results, high_endpoint, high_slots, "ascent"),
            values(results, high_endpoint, low_slots, "ascent"),
        )
        low_model_state = paired(
            values(results, low_endpoint, high_slots, "ascent"),
            values(results, low_endpoint, low_slots, "ascent"),
        )
        registered_interactions[f"{low_endpoint}_to_{high_endpoint}__slots_{low_slots}_to_{high_slots}"] = mean_ci(
            paired(high_model_state, low_model_state)
        )

    diagonal_gain = [
        values(results, endpoint, slots, "gain")
        for (endpoint, _, _), slots in zip(ENDPOINTS, STATE_SLOTS, strict=True)
    ]
    diagonal_gain_summary = {
        endpoint: mean_ci(panel_values)
        for (endpoint, _, _), panel_values in zip(ENDPOINTS, diagonal_gain, strict=True)
    }
    diagonal_adjacent: dict[str, Any] = {}
    coscale_excess: dict[str, Any] = {}
    for index, ((low_endpoint, _, _), (high_endpoint, _, _)) in enumerate(
        adjacent_models
    ):
        key = f"{low_endpoint}_to_{high_endpoint}"
        coscale_increment = paired(diagonal_gain[index + 1], diagonal_gain[index])
        diagonal_adjacent[key] = mean_ci(coscale_increment)
        low_slots, high_slots = STATE_SLOTS[index], STATE_SLOTS[index + 1]
        memory_only = paired(
            values(results, low_endpoint, high_slots, "gain"),
            values(results, low_endpoint, low_slots, "gain"),
        )
        coscale_excess[key] = mean_ci(paired(coscale_increment, memory_only))

    provenance_pass = (
        all(checks)
        and len(config_hashes) == 1
        and all(len(hashes) == 1 for hashes in data_hashes_by_panel.values())
        and foundation_equal
        and aligned
        and panel_alignment
    )
    gates = {
        "provenance_and_alignment": provenance_pass,
        "diagonal_gain_adjacent_lcbs_positive": all(
            interval["ci95_low"] > 0 for interval in diagonal_adjacent.values()
        ),
        "registered_interaction_lcbs_positive": all(
            interval["ci95_low"] > 0 for interval in registered_interactions.values()
        ),
        "coscale_increment_exceeds_memory_only_lcbs_positive": all(
            interval["ci95_low"] > 0 for interval in coscale_excess.values()
        ),
    }
    gates["generative_factorial_scale_complementarity_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "root": str(root),
        "endpoints": [endpoint for endpoint, _, _ in ENDPOINTS],
        "state_slots": list(STATE_SLOTS),
        "panels": len(PANELS),
        "config_hashes": sorted(config_hashes),
        "foundation_cache_equal_across_state_sizes": foundation_equal,
        "all_cells_row_task_target_aligned": panel_alignment,
        "cells": cells,
        "state_main_effects": state_effects,
        "fixed_state_model_effects": model_effects,
        "registered_foundation_by_state_interactions": registered_interactions,
        "diagonal_absolute_gain": diagonal_gain_summary,
        "diagonal_absolute_gain_adjacent": diagonal_adjacent,
        "coscale_increment_minus_memory_only": coscale_excess,
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
