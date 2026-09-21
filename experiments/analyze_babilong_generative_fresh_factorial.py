#!/usr/bin/env python3
"""Analyze the preregistered five-panel generative model-by-state factorial."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ENDPOINTS = (
    ("smollm2-135m-instruct", "node1", 134_515_008),
    ("smollm2-360m-instruct", "node2", 361_821_120),
    ("smollm2-1p7b-instruct", "node1", 1_711_376_384),
)
STATE_SLOTS = (1, 2, 3)
T95_DF4 = 2.7764451051977987
T95_DF7 = 2.3646242510102993


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_ci(values: list[float], critical: float) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    standard_error = float(array.std(ddof=1) / np.sqrt(array.size))
    mean = float(array.mean())
    return {
        "values": array.tolist(),
        "mean": mean,
        "standard_error": standard_error,
        "ci95_low": mean - critical * standard_error,
        "ci95_high": mean + critical * standard_error,
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


def expected_model_files(endpoint: dict[str, Any]) -> list[dict[str, Any]]:
    return [endpoint["model_artifact"]]


def collect(
    root: Path,
    config_path: Path,
    run_commit: str,
) -> tuple[
    dict[tuple[str, int], list[dict[str, Any]]],
    list[str],
    bool,
    dict[str, Any],
]:
    config = load(config_path)
    config_sha256 = sha256_file(config_path)
    panels = list(config["panels"])
    endpoints = {endpoint["name"]: endpoint for endpoint in config["endpoints"]}
    results: dict[tuple[str, int], list[dict[str, Any]]] = {}
    checks: list[bool] = []
    for endpoint, node, _ in ENDPOINTS:
        for slots in STATE_SLOTS:
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
                        row["model_files"] == expected_model_files(endpoints[endpoint]),
                        row["status"] == config["status"],
                        row["parser_accuracy"] == 1.0,
                        len(row["predictions"]) == 120,
                        row["data"]["path"].endswith(f"{panel}.jsonl"),
                    )
                )

    foundation_equal = True
    panel_alignment = True
    for endpoint, _, _ in ENDPOINTS:
        for panel_index in range(len(panels)):
            rows = [results[(endpoint, slots)][panel_index] for slots in STATE_SLOTS]
            projection = lambda row: [
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
            foundation_equal &= all(
                projection(row) == projection(rows[0]) for row in rows[1:]
            )
    for panel_index in range(len(panels)):
        reference = results[(ENDPOINTS[0][0], STATE_SLOTS[0])][panel_index]
        identity = [
            (prediction["row_id"], prediction["task"], prediction["target"])
            for prediction in reference["predictions"]
        ]
        panel_alignment &= all(
            [
                (prediction["row_id"], prediction["task"], prediction["target"])
                for prediction in rows[panel_index]["predictions"]
            ]
            == identity
            for rows in results.values()
        )
    provenance = all(checks) and foundation_equal and panel_alignment
    details = {
        "config_sha256": config_sha256,
        "run_commit": run_commit,
        "foundation_equal_across_state_sizes": foundation_equal,
        "all_cells_row_task_target_aligned": panel_alignment,
        "all_cell_checks_pass": all(checks),
    }
    return results, panels, provenance, details


def summarize(
    results: dict[tuple[str, int], list[dict[str, Any]]],
    critical: float,
) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for endpoint, _, parameters in ENDPOINTS:
        cells[endpoint] = {
            str(slots): {
                "model_parameters": parameters,
                "state_slots": slots,
                "foundation_accuracy": mean_ci(
                    values(results, endpoint, slots, "foundation"), critical
                ),
                "ascent_accuracy": mean_ci(
                    values(results, endpoint, slots, "ascent"), critical
                ),
                "absolute_gain": mean_ci(
                    values(results, endpoint, slots, "gain"), critical
                ),
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
                ),
                critical,
            )
            for low, high in zip(STATE_SLOTS[:-1], STATE_SLOTS[1:], strict=True)
        }

    adjacent_models = list(zip(ENDPOINTS[:-1], ENDPOINTS[1:], strict=True))
    model_effects: dict[str, Any] = {}
    interactions: dict[str, Any] = {}
    for index, ((low_endpoint, _, _), (high_endpoint, _, _)) in enumerate(
        adjacent_models
    ):
        model_effects[f"{low_endpoint}_to_{high_endpoint}"] = {
            f"slots_{slots}": mean_ci(
                paired(
                    values(results, high_endpoint, slots, "ascent"),
                    values(results, low_endpoint, slots, "ascent"),
                ),
                critical,
            )
            for slots in STATE_SLOTS
        }
        low_slots, high_slots = STATE_SLOTS[index], STATE_SLOTS[index + 1]
        interaction_values = paired(
            paired(
                values(results, high_endpoint, high_slots, "ascent"),
                values(results, high_endpoint, low_slots, "ascent"),
            ),
            paired(
                values(results, low_endpoint, high_slots, "ascent"),
                values(results, low_endpoint, low_slots, "ascent"),
            ),
        )
        interactions[
            f"{low_endpoint}_to_{high_endpoint}__slots_{low_slots}_to_{high_slots}"
        ] = mean_ci(interaction_values, critical)

    diagonal_gain = [
        values(results, endpoint, slots, "gain")
        for (endpoint, _, _), slots in zip(ENDPOINTS, STATE_SLOTS, strict=True)
    ]
    diagonal_summary = {
        endpoint: mean_ci(panel_values, critical)
        for (endpoint, _, _), panel_values in zip(ENDPOINTS, diagonal_gain, strict=True)
    }
    diagonal_adjacent: dict[str, Any] = {}
    coscale_excess: dict[str, Any] = {}
    for index, ((low_endpoint, _, _), (high_endpoint, _, _)) in enumerate(
        adjacent_models
    ):
        key = f"{low_endpoint}_to_{high_endpoint}"
        coscale_increment = paired(diagonal_gain[index + 1], diagonal_gain[index])
        diagonal_adjacent[key] = mean_ci(coscale_increment, critical)
        low_slots, high_slots = STATE_SLOTS[index], STATE_SLOTS[index + 1]
        memory_only = paired(
            values(results, low_endpoint, high_slots, "gain"),
            values(results, low_endpoint, low_slots, "gain"),
        )
        coscale_excess[key] = mean_ci(
            paired(coscale_increment, memory_only), critical
        )
    return {
        "cells": cells,
        "state_main_effects": state_effects,
        "fixed_state_model_effects": model_effects,
        "registered_foundation_by_state_interactions": interactions,
        "diagonal_absolute_gain": diagonal_summary,
        "diagonal_absolute_gain_adjacent": diagonal_adjacent,
        "coscale_increment_minus_memory_only": coscale_excess,
    }


def combined_intervals(
    previous: dict[str, Any], fresh: dict[str, Any], field: str
) -> dict[str, Any]:
    return {
        key: mean_ci(
            previous[field][key]["values"] + fresh[field][key]["values"], T95_DF7
        )
        for key in fresh[field]
    }


def positive_lcbs(intervals: dict[str, Any]) -> bool:
    return all(interval["ci95_low"] > 0 for interval in intervals.values())


def analyze(
    root: Path,
    config_path: Path,
    run_commit: str,
    previous_analysis: dict[str, Any],
) -> dict[str, Any]:
    results, panels, provenance, provenance_details = collect(
        root, config_path, run_commit
    )
    fresh = summarize(results, T95_DF4)
    fresh_gates = {
        "provenance_and_alignment": provenance,
        "diagonal_gain_adjacent_lcbs_positive": positive_lcbs(
            fresh["diagonal_absolute_gain_adjacent"]
        ),
        "registered_interaction_lcbs_positive": positive_lcbs(
            fresh["registered_foundation_by_state_interactions"]
        ),
        "coscale_increment_exceeds_memory_only_lcbs_positive": positive_lcbs(
            fresh["coscale_increment_minus_memory_only"]
        ),
    }
    fresh_gates["fresh_five_panel_primary_pass"] = all(fresh_gates.values())

    combined = {
        field: combined_intervals(previous_analysis, fresh, field)
        for field in (
            "registered_foundation_by_state_interactions",
            "diagonal_absolute_gain_adjacent",
            "coscale_increment_minus_memory_only",
        )
    }
    combined_gates = {
        "previous_provenance_gate": previous_analysis["gates"][
            "provenance_and_alignment"
        ],
        "fresh_provenance_gate": provenance,
        "diagonal_gain_adjacent_lcbs_positive": positive_lcbs(
            combined["diagonal_absolute_gain_adjacent"]
        ),
        "registered_interaction_lcbs_positive": positive_lcbs(
            combined["registered_foundation_by_state_interactions"]
        ),
        "coscale_increment_exceeds_memory_only_lcbs_positive": positive_lcbs(
            combined["coscale_increment_minus_memory_only"]
        ),
    }
    combined_gates["combined_eight_panel_supporting_pass"] = all(
        combined_gates.values()
    )
    return {
        "schema_version": 1,
        "root": str(root),
        "panels": panels,
        "fresh_panel_count": len(panels),
        "fresh_interval": "Student-t df=4",
        "combined_panel_count": 8,
        "combined_interval": "Student-t df=7",
        "provenance": provenance_details,
        "fresh": fresh,
        "fresh_gates": fresh_gates,
        "combined": combined,
        "combined_gates": combined_gates,
        "candidate_generational_scale_complementarity_pass": fresh_gates[
            "fresh_five_panel_primary_pass"
        ]
        and combined_gates["combined_eight_panel_supporting_pass"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-commit", required=True)
    parser.add_argument("--previous-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.root,
        args.config,
        args.run_commit,
        load(args.previous_analysis),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
