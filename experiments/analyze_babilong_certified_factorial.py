#!/usr/bin/env python3
"""Analyze the frozen 3x3 certified model-by-state factorial."""

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
    ("smollm2-135m", "node1", 134_515_008),
    ("smollm2-360m-node2", "node2", 361_821_120),
    ("smollm2-1p7b", "node1", 1_711_376_384),
)
STATE_SLOTS = (1, 2, 8)
PANELS = tuple(range(1, 6))


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def result_path(root: Path, node: str, endpoint: str, slots: int, panel: int) -> Path:
    return root / node / f"certified_slots_{slots}_panel{panel}_{endpoint}.json"


def panel_values(
    results: dict[tuple[str, int], list[dict[str, Any]]],
    endpoint: str,
    slots: int,
    field: str,
) -> list[float]:
    return [float(row[field]["mean"]) for row in results[(endpoint, slots)]]


def paired_difference(left: list[float], right: list[float]) -> list[float]:
    return (np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)).tolist()


def analyze(root: Path) -> dict[str, Any]:
    results: dict[tuple[str, int], list[dict[str, Any]]] = {}
    provenance_checks: list[bool] = []
    for endpoint, node, _ in ENDPOINTS:
        for slots in STATE_SLOTS:
            rows = [
                load(result_path(root, node, endpoint, slots, panel))
                for panel in PANELS
            ]
            results[(endpoint, slots)] = rows
            for panel, row in zip(PANELS, rows, strict=True):
                provenance_checks.extend(
                    (
                        row["endpoint"]["name"] == endpoint,
                        row["condition"]["name"] == f"certified_slots_{slots}",
                        row["condition"]["fact_slots"] == slots,
                        row["condition"]["readout_path"] == "certified_evidence",
                        row["environment"]["git_commit"]
                        == "ab4222a7a61413da5aacb9ad840122091278bc86",
                        not row["environment"]["git_dirty"],
                        row["parser_accuracy"] == 1.0,
                        row["data"]["path"].endswith(
                            f"structured_certified_confirmation_{panel}.jsonl"
                        ),
                    )
                )

    config_hashes = {
        row["config"]["sha256"]
        for rows in results.values()
        for row in rows
    }
    data_hashes_by_panel: dict[int, set[str]] = {panel: set() for panel in PANELS}
    for endpoint, _, _ in ENDPOINTS:
        for slots in STATE_SLOTS:
            for panel, row in zip(PANELS, results[(endpoint, slots)], strict=True):
                data_hashes_by_panel[panel].add(row["data"]["sha256"])
    foundation_cache_equal = True
    row_alignment_equal = True
    for endpoint, _, _ in ENDPOINTS:
        for panel_index in range(len(PANELS)):
            slot_rows = [results[(endpoint, slots)][panel_index] for slots in STATE_SLOTS]
            foundation_cache_equal &= all(
                [p["foundation_score"] for p in candidate["predictions"]]
                == [p["foundation_score"] for p in slot_rows[0]["predictions"]]
                for candidate in slot_rows[1:]
            )
            row_alignment_equal &= all(
                [p["row_id"] for p in candidate["predictions"]]
                == [p["row_id"] for p in slot_rows[0]["predictions"]]
                for candidate in slot_rows[1:]
            )

    cells: dict[str, dict[str, Any]] = {}
    for endpoint, _, parameters in ENDPOINTS:
        cells[endpoint] = {}
        for slots in STATE_SLOTS:
            cells[endpoint][str(slots)] = {
                "model_parameters": parameters,
                "state_slots": slots,
                "foundation_accuracy": mean_ci(
                    panel_values(results, endpoint, slots, "foundation")
                ),
                "ascent_accuracy": mean_ci(
                    panel_values(results, endpoint, slots, "ascent")
                ),
                "absolute_gain": mean_ci(
                    panel_values(results, endpoint, slots, "gain")
                ),
            }

    state_effects: dict[str, dict[str, Any]] = {}
    for endpoint, _, _ in ENDPOINTS:
        state_effects[endpoint] = {}
        for low_slots, high_slots in zip(STATE_SLOTS[:-1], STATE_SLOTS[1:], strict=True):
            values = paired_difference(
                panel_values(results, endpoint, high_slots, "ascent"),
                panel_values(results, endpoint, low_slots, "ascent"),
            )
            state_effects[endpoint][f"slots_{low_slots}_to_{high_slots}"] = mean_ci(values)

    model_effects: dict[str, dict[str, Any]] = {}
    adjacent_endpoints = list(zip(ENDPOINTS[:-1], ENDPOINTS[1:], strict=True))
    for (low_endpoint, _, _), (high_endpoint, _, _) in adjacent_endpoints:
        key = f"{low_endpoint}_to_{high_endpoint}"
        model_effects[key] = {}
        for slots in STATE_SLOTS:
            model_effects[key][f"slots_{slots}"] = mean_ci(
                paired_difference(
                    panel_values(results, high_endpoint, slots, "ascent"),
                    panel_values(results, low_endpoint, slots, "ascent"),
                )
            )

    interactions: dict[str, dict[str, Any]] = {}
    for (low_endpoint, _, _), (high_endpoint, _, _) in adjacent_endpoints:
        model_key = f"{low_endpoint}_to_{high_endpoint}"
        interactions[model_key] = {}
        for low_slots, high_slots in zip(STATE_SLOTS[:-1], STATE_SLOTS[1:], strict=True):
            high_model_state_effect = paired_difference(
                panel_values(results, high_endpoint, high_slots, "ascent"),
                panel_values(results, high_endpoint, low_slots, "ascent"),
            )
            low_model_state_effect = paired_difference(
                panel_values(results, low_endpoint, high_slots, "ascent"),
                panel_values(results, low_endpoint, low_slots, "ascent"),
            )
            interactions[model_key][f"slots_{low_slots}_to_{high_slots}"] = mean_ci(
                paired_difference(high_model_state_effect, low_model_state_effect)
            )

    diagonal = [
        panel_values(results, endpoint, slots, "gain")
        for (endpoint, _, _), slots in zip(ENDPOINTS, STATE_SLOTS, strict=True)
    ]
    diagonal_gain = {
        endpoint: mean_ci(values)
        for (endpoint, _, _), values in zip(ENDPOINTS, diagonal, strict=True)
    }
    diagonal_adjacent = {
        f"{ENDPOINTS[index][0]}_to_{ENDPOINTS[index + 1][0]}": mean_ci(
            paired_difference(diagonal[index + 1], diagonal[index])
        )
        for index in (0, 1)
    }

    coscale_excess: dict[str, Any] = {}
    for index, ((low_endpoint, _, _), (high_endpoint, _, _)) in enumerate(
        adjacent_endpoints
    ):
        high_slots = STATE_SLOTS[index + 1]
        coscale_increment = paired_difference(diagonal[index + 1], diagonal[index])
        memory_only_increment = paired_difference(
            panel_values(results, low_endpoint, high_slots, "gain"),
            panel_values(results, low_endpoint, STATE_SLOTS[index], "gain"),
        )
        coscale_excess[f"{low_endpoint}_to_{high_endpoint}"] = mean_ci(
            paired_difference(coscale_increment, memory_only_increment)
        )

    all_interaction_intervals = [
        interval
        for model_pair in interactions.values()
        for interval in model_pair.values()
    ]
    gates = {
        "provenance_and_alignment": (
            all(provenance_checks)
            and len(config_hashes) == 1
            and all(len(values) == 1 for values in data_hashes_by_panel.values())
            and foundation_cache_equal
            and row_alignment_equal
        ),
        "diagonal_gain_adjacent_lcbs_positive": all(
            interval["ci95_low"] > 0 for interval in diagonal_adjacent.values()
        ),
        "state_main_effect_all_lcbs_positive": all(
            interval["ci95_low"] > 0
            for endpoint_rows in state_effects.values()
            for interval in endpoint_rows.values()
        ),
        "larger_model_extracts_more_same_state": all(
            interval["ci95_low"] > 0
            for model_rows in model_effects.values()
            for interval in model_rows.values()
        ),
        "foundation_by_state_interaction_lcbs_positive": all(
            interval["ci95_low"] > 0 for interval in all_interaction_intervals
        ),
        "coscale_increment_exceeds_memory_only_lcbs_positive": all(
            interval["ci95_low"] > 0 for interval in coscale_excess.values()
        ),
    }
    gates["factorial_scale_complementarity_pass"] = all(
        gates[key]
        for key in (
            "provenance_and_alignment",
            "diagonal_gain_adjacent_lcbs_positive",
            "state_main_effect_all_lcbs_positive",
            "larger_model_extracts_more_same_state",
            "foundation_by_state_interaction_lcbs_positive",
            "coscale_increment_exceeds_memory_only_lcbs_positive",
        )
    )
    return {
        "schema_version": 1,
        "root": str(root),
        "endpoints": [endpoint for endpoint, _, _ in ENDPOINTS],
        "state_slots": list(STATE_SLOTS),
        "panels": len(PANELS),
        "config_hashes": sorted(config_hashes),
        "foundation_cache_equal_across_state_sizes": foundation_cache_equal,
        "cells": cells,
        "state_main_effects": state_effects,
        "fixed_state_model_effects": model_effects,
        "foundation_by_state_interactions": interactions,
        "diagonal_absolute_gain": diagonal_gain,
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
