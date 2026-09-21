#!/usr/bin/env python3
"""Analyze the frozen three-scale official RULER HotpotQA development panel."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_stage_b_endpoint import array_sha256


def paired_interval(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("paired interval requires a nontrivial vector")
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / math.sqrt(values.size))
    return {
        "samples": int(values.size),
        "mean": mean,
        "standard_error": standard_error,
        "ci95_low": mean - 1.96 * standard_error,
        "ci95_high": mean + 1.96 * standard_error,
    }


def per_example_slopes(gains: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """Return one log-parameter slope for each shared test example."""

    if gains.ndim != 2 or gains.shape[0] != parameters.size:
        raise ValueError("gain matrix must be endpoints by examples")
    x = np.log(parameters.astype(np.float64))
    centered = x - x.mean()
    return (centered[:, None] * gains).sum(axis=0) / float(np.square(centered).sum())


def load_cell(
    result_path: Path, arrays_path: Path, expected_endpoint: str
) -> dict[str, Any]:
    result = json.loads(result_path.read_text())
    if result["endpoint"]["name"] != expected_endpoint:
        raise ValueError(f"endpoint mismatch in {result_path}")
    with np.load(arrays_path) as archive:
        arrays = {name: archive[name].astype(np.float64) for name in archive.files}
    if set(arrays) != set(result["arrays_sha256"]):
        raise ValueError(f"array manifest mismatch in {arrays_path}")
    for name, values in arrays.items():
        if array_sha256(values) != result["arrays_sha256"][name]:
            raise ValueError(f"array hash mismatch for {name} in {arrays_path}")
    return {"result": result, "arrays": arrays}


def analyze(
    config_path: Path, results_root: Path, arrays_root: Path
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    endpoints = [row["name"] for row in config["endpoints"]]
    if len(endpoints) != 3:
        raise ValueError("the scale analyzer requires exactly three endpoints")
    cells = [
        load_cell(
            results_root / f"{endpoint}.json",
            arrays_root / f"{endpoint}.npz",
            endpoint,
        )
        for endpoint in endpoints
    ]
    results = [cell["result"] for cell in cells]
    arrays = [cell["arrays"] for cell in cells]
    expected_examples = int(config["test_samples"])
    for endpoint_arrays in arrays:
        if any(values.shape != (expected_examples,) for values in endpoint_arrays.values()):
            raise ValueError("all frozen result arrays must align on the test panel")

    scale_gains = np.stack(
        [cell["base_nll"] - cell["scale_nll"] for cell in arrays]
    )
    endpoint_scale_intervals = {
        endpoint: paired_interval(scale_gains[index])
        for index, endpoint in enumerate(endpoints)
    }
    adjacent_intervals = [
        paired_interval(scale_gains[index + 1] - scale_gains[index])
        for index in range(2)
    ]
    parameters = np.asarray(
        [row["model_parameters"] for row in config["endpoints"]], dtype=np.float64
    )
    slope_interval = paired_interval(per_example_slopes(scale_gains, parameters))

    fixed_advantage = [
        paired_interval(cell["fixed_nll"] - cell["scale_nll"])
        for cell in arrays
    ]
    query_graph_advantage = [
        paired_interval(cell["query_only_nll"] - cell["scale_nll"])
        for cell in arrays
    ]
    irrelevant_advantage = [
        paired_interval(cell["irrelevant_nll"] - cell["scale_nll"])
        for cell in arrays
    ]

    config_hashes = {row["config_sha256"] for row in results}
    task_hashes = {row["task_sha256"] for row in results}
    commits = {row["runtime"]["git_commit"] for row in results}
    rankings = {row["rankings_sha256"] for row in results}
    state_ratios = [
        float(row["persistent_state"]["mean_state_bytes_per_model_parameter"])
        for row in results
    ]
    all_prompt_lengths_native = all(
        maximum <= int(config["query_context_tokens"])
        for row in results
        for maximum in [
            max(value["maximum"] for value in row["prompt_lengths"].values())
        ]
    )
    scale_gate_open = all(row["arms"]["scale"]["gate"]["enabled"] for row in results)
    system_gates = {
        "single_config_hash": len(config_hashes) == 1,
        "registered_task_hash": task_hashes == {config["task_sha256"]},
        "single_frozen_git_commit": len(commits) == 1,
        "all_worktrees_clean": all(not row["runtime"]["git_dirty"] for row in results),
        "identical_query_conditioned_rankings": len(rankings) == 1,
        "reader_never_receives_answer_or_support_labels": all(
            row["answer_and_support_labels_absent_from_reader"] for row in results
        ),
        "native_untruncated_prompts": all_prompt_lengths_native
        and all(row["native_context"] for row in results),
        "nested_scale_document_prefix": all(
            row["nested_document_prefix"] for row in results
        )
        and all(
            config["endpoints"][index + 1]["scale_documents"]
            > config["endpoints"][index]["scale_documents"]
            for index in range(2)
        ),
        "relative_persistent_state_strictly_decreases": all(
            state_ratios[index + 1] < state_ratios[index] for index in range(2)
        ),
    }
    primary_gates = {
        "scale_safe_gate_open_at_every_endpoint": scale_gate_open,
        "scale_nll_gain_lcb_positive_at_every_endpoint": all(
            interval["ci95_low"] > 0.0
            for interval in endpoint_scale_intervals.values()
        ),
        "both_adjacent_scale_gain_lcbs_positive": all(
            interval["ci95_low"] > 0.0 for interval in adjacent_intervals
        ),
        "log_parameter_slope_lcb_positive": slope_interval["ci95_low"] > 0.0,
        "scale_beats_fixed_at_mid_and_large": all(
            fixed_advantage[index]["ci95_low"] > 0.0 for index in (1, 2)
        ),
        "graph_scale_beats_query_only_at_large": query_graph_advantage[2]["ci95_low"]
        > 0.0,
        "graph_scale_beats_irrelevant_at_every_endpoint": all(
            interval["ci95_low"] > 0.0 for interval in irrelevant_advantage
        ),
    }
    official_accuracy = {
        endpoint: {
            "foundation": result["foundation_score"],
            "scale_ascent": result["arms"]["scale"]["official_score"],
            "absolute_gain": result["arms"]["scale"]["paired_official_score_gain"],
            "remaining_error_elimination": result["arms"]["scale"][
                "remaining_error_elimination"
            ],
            "saturated_ascent": result["arms"]["scale"]["official_score"]["mean"]
            >= 1.0,
        }
        for endpoint, result in zip(endpoints, results, strict=True)
    }
    gates = {**system_gates, **primary_gates}
    gates["development_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": config["status"],
        "endpoints": endpoints,
        "endpoint_scale_nll_gain_intervals": endpoint_scale_intervals,
        "adjacent_scale_nll_gain_difference_intervals": adjacent_intervals,
        "scale_gain_log_parameter_slope_interval": slope_interval,
        "scale_minus_fixed_nll_intervals": dict(zip(endpoints, fixed_advantage, strict=True)),
        "graph_scale_minus_query_only_nll_intervals": dict(
            zip(endpoints, query_graph_advantage, strict=True)
        ),
        "graph_scale_minus_irrelevant_nll_intervals": dict(
            zip(endpoints, irrelevant_advantage, strict=True)
        ),
        "official_accuracy_ceiling_aware_diagnostic": official_accuracy,
        "state_bytes_per_parameter": dict(zip(endpoints, state_ratios, strict=True)),
        "config_hashes": sorted(config_hashes),
        "task_hashes": sorted(task_hashes),
        "git_commits": sorted(commits),
        "ranking_hashes": sorted(rankings),
        "gates": gates,
        "interpretation": (
            "Teacher-forced answer NLL is the continuous primary. Official accuracy is "
            "reported with absolute gain and the prospectively frozen remaining-error "
            "elimination; a 1.0 ASCENT score is ceiling success, never a demand for >1.0."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--arrays-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(args.config, args.results_root, args.arrays_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
