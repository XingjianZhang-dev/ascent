#!/usr/bin/env python3
"""Analyze the frozen Qwen2.5 3B/7B by State-4/State-8 QA7/QA8 factorial."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ENDPOINTS = ("qwen2p5-3b-instruct", "qwen2p5-7b-instruct")
CONDITIONS = ("state4", "state8")
T95_DF4 = 2.7764451051977987


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_ci(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    standard_error = float(array.std(ddof=1) / math.sqrt(array.size))
    return {
        "clusters": int(array.size),
        "degrees_of_freedom": int(array.size - 1),
        "mean": mean,
        "standard_error": standard_error,
        "ci95_low": mean - T95_DF4 * standard_error,
        "ci95_high": mean + T95_DF4 * standard_error,
        "panel_values": array.tolist(),
    }


def result_path(root: Path, phase: str, panel: int, endpoint: str, condition: str) -> Path:
    phase_root = root / phase
    if phase == "confirmation":
        phase_root = phase_root / f"panel{panel}"
    return phase_root / f"{endpoint}_{condition}.json"


def _foundation_signature(prediction: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        prediction[key]
        for key in (
            "row_id",
            "target",
            "foundation_output",
            "foundation_answer",
            "foundation_score",
        )
    )


def _load_panel(
    root: Path,
    phase: str,
    panel: int,
    config: dict[str, Any],
    config_sha256: str,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    results = {
        (endpoint, condition): json.loads(
            result_path(root, phase, panel, endpoint, condition).read_text()
        )
        for endpoint in ENDPOINTS
        for condition in CONDITIONS
    }
    expected_data_name = (
        "structured_development"
        if phase == "development"
        else f"structured_fresh_confirmation_{panel}"
    )
    expected_data_hash = config["panel_sha256_by_name"][expected_data_name]
    commits = {result["environment"]["git_commit"] for result in results.values()}
    row_ids = {
        key: [prediction["row_id"] for prediction in result["predictions"]]
        for key, result in results.items()
    }
    reference_ids = next(iter(row_ids.values()))
    foundation_invariance: dict[str, bool] = {}
    nested_state: dict[str, bool] = {}
    nonredundant_projection_counts: dict[str, int] = {}
    for endpoint in ENDPOINTS:
        state4 = results[(endpoint, "state4")]
        state8 = results[(endpoint, "state8")]
        foundation_invariance[endpoint] = all(
            _foundation_signature(left) == _foundation_signature(right)
            for left, right in zip(
                state4["predictions"], state8["predictions"], strict=True
            )
        )
        nested_state[endpoint] = all(
            tuple(large["retained_facts"][-len(small["retained_facts"]) :])
            == tuple(small["retained_facts"])
            for small, large in zip(
                state4["predictions"], state8["predictions"], strict=True
            )
        )
        nonredundant_projection_counts[endpoint] = sum(
            small["structured_inventory"] != large["structured_inventory"]
            or small["structured_count"] != large["structured_count"]
            for small, large in zip(
                state4["predictions"], state8["predictions"], strict=True
            )
        )
    checks = {
        "one_clean_commit": len(commits) == 1
        and all(not result["environment"]["git_dirty"] for result in results.values()),
        "exact_config_hash": all(
            result["config"]["sha256"] == config_sha256
            for result in results.values()
        ),
        "exact_data_hash": all(
            result["data"]["sha256"] == expected_data_hash
            for result in results.values()
        ),
        "row_alignment": all(ids == reference_ids for ids in row_ids.values()),
        "exact_rows": len(reference_ids) == int(config["evaluation_samples"]),
        "task_balance": all(
            result["data"]["task_counts"]
            == {task: int(config["rows_per_task"]) for task in config["tasks"]}
            for result in results.values()
        ),
        "parser_accuracy_one": all(
            result["parser_accuracy"] == 1.0 for result in results.values()
        ),
        "endpoint_identity": all(
            result["endpoint"]["name"] == endpoint
            for (endpoint, _), result in results.items()
        ),
        "condition_identity": all(
            result["condition"]["name"] == condition
            and result["condition"]["memory_representation"]
            == "structured_inventory"
            and result["condition"]["readout_path"] == "foundation_generation"
            and result["condition"]["fact_slots"] == int(condition.removeprefix("state"))
            and not result["condition"]["diagnostic_no_promotion"]
            for (_, condition), result in results.items()
        ),
        "foundation_invariance": all(foundation_invariance.values()),
        "nested_state": all(nested_state.values()),
        "nonredundant_projection": all(
            count > 0 for count in nonredundant_projection_counts.values()
        ),
    }
    cells: dict[str, Any] = {}
    for (endpoint, condition), result in results.items():
        key = f"{endpoint}@{condition}"
        cells[key] = {
            "foundation": result["foundation"]["mean"],
            "ascent": result["ascent"]["mean"],
            "gain": result["gain"]["mean"],
            "remaining_error_elimination": result["remaining_error_elimination"],
            "wins": result["wins"],
            "regressions": result["regressions"],
            "foundation_invalid_outputs": sum(
                prediction["foundation_answer"] is None
                for prediction in result["predictions"]
            ),
            "ascent_invalid_outputs": sum(
                prediction["ascent_answer"] is None
                for prediction in result["predictions"]
            ),
        }
    g3s4 = cells["qwen2p5-3b-instruct@state4"]["gain"]
    g3s8 = cells["qwen2p5-3b-instruct@state8"]["gain"]
    g7s4 = cells["qwen2p5-7b-instruct@state4"]["gain"]
    g7s8 = cells["qwen2p5-7b-instruct@state8"]["gain"]
    r3s4 = cells["qwen2p5-3b-instruct@state4"]["remaining_error_elimination"]
    r7s8 = cells["qwen2p5-7b-instruct@state8"]["remaining_error_elimination"]
    contrasts = {
        "co_scaled_gain_increment": g7s8 - g3s4,
        "co_scaled_remaining_error_increment": r7s8 - r3s4,
        "model_by_state_interaction": (g7s8 - g7s4) - (g3s8 - g3s4),
        "large_model_state_expansion": g7s8 - g7s4,
    }
    return results, {
        "panel": panel,
        "data_sha256": expected_data_hash,
        "cells": cells,
        "contrasts": contrasts,
        "checks": checks,
        "nonredundant_projection_counts": nonredundant_projection_counts,
        "commits": sorted(commits),
    }


def analyze(config_path: Path, root: Path, phase: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    config_sha256 = sha256_file(config_path)
    if config["status"] != (
        "prospective_qwen_qa78_headroom_factorial_frozen_before_any_qwen_qa78_decoder_score"
    ):
        raise RuntimeError("unexpected frozen status")
    panel_numbers = (0,) if phase == "development" else (1, 2, 3, 4, 5)
    panels = [
        _load_panel(root, phase, panel, config, config_sha256)[1]
        for panel in panel_numbers
    ]
    provenance = {
        "config_sha256": config_sha256,
        "all_panel_checks_pass": all(
            all(panel["checks"].values()) for panel in panels
        ),
        "single_commit_across_panels": len(
            {commit for panel in panels for commit in panel["commits"]}
        )
        == 1,
        "all_models_states_tasks_panels_retained": True,
        "invalid_outputs_and_regressions_retained": True,
    }
    cell_keys = [f"{endpoint}@{condition}" for endpoint in ENDPOINTS for condition in CONDITIONS]
    contrast_keys = list(panels[0]["contrasts"])
    if phase == "development":
        cells = panels[0]["cells"]
        contrasts = panels[0]["contrasts"]
        gates = {
            "provenance": all(
                value
                for key, value in provenance.items()
                if key != "config_sha256"
            ),
            "all_cell_gains_positive": all(cells[key]["gain"] > 0 for key in cell_keys),
            "co_scaled_gain_increment_positive": contrasts["co_scaled_gain_increment"] > 0,
            "co_scaled_remaining_error_increment_positive": contrasts[
                "co_scaled_remaining_error_increment"
            ]
            > 0,
            "model_by_state_interaction_positive": contrasts[
                "model_by_state_interaction"
            ]
            > 0,
            "large_model_state_expansion_positive": contrasts[
                "large_model_state_expansion"
            ]
            > 0,
        }
        gates["development_pass"] = all(gates.values())
        return {
            "schema_version": 1,
            "phase": phase,
            "panels": panels,
            "provenance": provenance,
            "gates": gates,
        }
    cell_intervals = {
        key: mean_ci([panel["cells"][key]["gain"] for panel in panels])
        for key in cell_keys
    }
    contrast_intervals = {
        key: mean_ci([panel["contrasts"][key] for panel in panels])
        for key in contrast_keys
    }
    gates = {
        "provenance": all(
            value for key, value in provenance.items() if key != "config_sha256"
        ),
        "all_cell_gain_lcbs_positive": all(
            interval["ci95_low"] > 0 for interval in cell_intervals.values()
        ),
        "co_scaled_gain_increment_lcb_positive": contrast_intervals[
            "co_scaled_gain_increment"
        ]["ci95_low"]
        > 0,
        "co_scaled_remaining_error_increment_lcb_positive": contrast_intervals[
            "co_scaled_remaining_error_increment"
        ]["ci95_low"]
        > 0,
        "model_by_state_interaction_lcb_positive": contrast_intervals[
            "model_by_state_interaction"
        ]["ci95_low"]
        > 0,
        "large_model_state_expansion_lcb_positive": contrast_intervals[
            "large_model_state_expansion"
        ]["ci95_low"]
        > 0,
    }
    gates["confirmation_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "phase": phase,
        "panels": panels,
        "cell_gain_intervals": cell_intervals,
        "contrast_intervals": contrast_intervals,
        "provenance": provenance,
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.config, args.root, args.phase)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
