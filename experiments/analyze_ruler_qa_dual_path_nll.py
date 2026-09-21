#!/usr/bin/env python3
"""Analyze the frozen 3x3 ASCENT/Foundation HotpotQA NLL development panel."""

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
        raise ValueError("a paired interval requires a nontrivial vector")
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
    if gains.ndim != 2 or gains.shape[0] != parameters.size:
        raise ValueError("gains must be endpoint by shared example")
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
    states = [row["name"] for row in config["memory_states"]]
    if len(endpoints) != 3 or states != ["graph_small", "graph_mid", "graph_large"]:
        raise ValueError("the analyzer requires the frozen 3x3 endpoint/state panel")
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
    expected_samples = int(config["test_samples"])
    for cell in arrays:
        if any(values.shape != (expected_samples,) for values in cell.values()):
            raise ValueError("all result arrays must align on the frozen test panel")

    gain_matrix = np.stack(
        [
            np.stack([cell["base_nll"] - cell[f"{state}_nll"] for state in states])
            for cell in arrays
        ]
    )
    diagonal = np.stack([gain_matrix[index, index] for index in range(3)])
    diagonal_intervals = {
        endpoint: paired_interval(diagonal[index])
        for index, endpoint in enumerate(endpoints)
    }
    adjacent_diagonal = [
        paired_interval(diagonal[index + 1] - diagonal[index])
        for index in range(2)
    ]
    parameters = np.asarray(
        [row["model_parameters"] for row in config["endpoints"]], dtype=np.float64
    )
    slope_interval = paired_interval(per_example_slopes(diagonal, parameters))

    state_expansion_intervals: dict[str, dict[str, Any]] = {}
    for foundation_index, endpoint in enumerate(endpoints):
        state_expansion_intervals[endpoint] = {
            "small_to_mid": paired_interval(
                gain_matrix[foundation_index, 1] - gain_matrix[foundation_index, 0]
            ),
            "mid_to_large": paired_interval(
                gain_matrix[foundation_index, 2] - gain_matrix[foundation_index, 1]
            ),
            "small_to_large": paired_interval(
                gain_matrix[foundation_index, 2] - gain_matrix[foundation_index, 0]
            ),
        }
    adjacent_factorial_interactions = [
        paired_interval(
            (gain_matrix[index + 1, index + 1] - gain_matrix[index + 1, index])
            - (gain_matrix[index, index + 1] - gain_matrix[index, index])
        )
        for index in range(2)
    ]
    global_factorial_interaction = paired_interval(
        (gain_matrix[2, 2] - gain_matrix[2, 0])
        - (gain_matrix[0, 2] - gain_matrix[0, 0])
    )

    fixed_advantage = [
        paired_interval(cell["graph_small_nll"] - cell[f"{results[index]['scale_state']}_nll"])
        for index, cell in enumerate(arrays)
    ]
    query_advantage = [
        paired_interval(cell["query_only_nll"] - cell[f"{results[index]['scale_state']}_nll"])
        for index, cell in enumerate(arrays)
    ]
    irrelevant_advantage = [
        paired_interval(cell["irrelevant_nll"] - cell[f"{results[index]['scale_state']}_nll"])
        for index, cell in enumerate(arrays)
    ]

    scale_gates = [
        result["arms"][result["scale_state"]]["gate"] for result in results
    ]
    diagonal_state_ratios = [
        float(
            result["arms"][result["scale_state"]][
                "mean_state_bytes_per_model_parameter"
            ]
        )
        for result in results
    ]
    diagonal_state_bytes = [
        float(result["arms"][result["scale_state"]]["mean_charged_state_bytes"])
        for result in results
    ]
    fixed_state_ratios = [
        float(result["arms"]["graph_small"]["mean_state_bytes_per_model_parameter"])
        for result in results
    ]
    state_hashes = {
        state: [result["arms"][state]["state_token_ids_sha256"] for result in results]
        for state in states
    }
    prompt_native = all(
        max(value["maximum"] for value in result["prompt_lengths"].values())
        <= int(config["query_context_tokens"])
        for result in results
    )
    system_gates = {
        "single_config_hash": len({row["config_sha256"] for row in results}) == 1,
        "registered_task_hash": {row["task_sha256"] for row in results}
        == {config["task_sha256"]},
        "single_frozen_git_commit": len(
            {row["runtime"]["git_commit"] for row in results}
        )
        == 1,
        "all_worktrees_clean": all(not row["runtime"]["git_dirty"] for row in results),
        "reader_never_receives_answer_or_support_labels": all(
            row["answer_and_support_labels_absent_from_reader"] for row in results
        ),
        "certified_and_latent_paths_share_identical_state": all(
            row["state_is_shared_by_certified_and_latent_paths"] for row in results
        ),
        "exact_nested_state_prefixes": all(
            row["exact_nested_graph_state_token_prefixes"] for row in results
        ),
        "identical_state_token_ids_across_foundations": all(
            len(set(hashes)) == 1 for hashes in state_hashes.values()
        ),
        "native_untruncated_prompts": prompt_native
        and all(row["native_context"] for row in results),
        "co_scaled_state_ratio_strictly_decreases": all(
            diagonal_state_ratios[index + 1] < diagonal_state_ratios[index]
            for index in range(2)
        ),
        "co_scaled_absolute_state_bytes_strictly_increases": all(
            diagonal_state_bytes[index + 1] > diagonal_state_bytes[index]
            for index in range(2)
        ),
        "fixed_state_ratio_strictly_decreases": all(
            fixed_state_ratios[index + 1] < fixed_state_ratios[index]
            for index in range(2)
        ),
    }
    primary_gates = {
        "co_scaled_safe_gate_open_at_every_endpoint": all(
            gate["enabled"] for gate in scale_gates
        ),
        "co_scaled_gain_lcb_positive_at_every_endpoint": all(
            interval["ci95_low"] > 0.0 for interval in diagonal_intervals.values()
        ),
        "both_adjacent_co_scaled_gain_lcbs_positive": all(
            interval["ci95_low"] > 0.0 for interval in adjacent_diagonal
        ),
        "co_scaled_log_parameter_slope_lcb_positive": slope_interval["ci95_low"]
        > 0.0,
        "both_adjacent_foundation_state_interaction_lcbs_positive": all(
            interval["ci95_low"] > 0.0 for interval in adjacent_factorial_interactions
        ),
        "global_foundation_state_interaction_lcb_positive": global_factorial_interaction[
            "ci95_low"
        ]
        > 0.0,
        "co_scaled_beats_fixed_small_state_at_mid_and_large": all(
            fixed_advantage[index]["ci95_low"] > 0.0 for index in (1, 2)
        ),
        "graph_beats_query_only_at_large": query_advantage[2]["ci95_low"] > 0.0,
        "graph_beats_irrelevant_at_every_endpoint": all(
            interval["ci95_low"] > 0.0 for interval in irrelevant_advantage
        ),
        "certified_path_selected_somewhere": any(
            gate["certified_weight"] > 0.0 for gate in scale_gates
        ),
        "latent_path_selected_somewhere": any(
            gate["latent_weight"] > 0.0 for gate in scale_gates
        ),
    }
    gates = {**system_gates, **primary_gates}
    gates["development_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": config["status"],
        "endpoints": endpoints,
        "memory_states": states,
        "co_scaled_gain_intervals": diagonal_intervals,
        "adjacent_co_scaled_gain_difference_intervals": adjacent_diagonal,
        "co_scaled_log_parameter_slope_interval": slope_interval,
        "state_expansion_intervals_by_foundation": state_expansion_intervals,
        "adjacent_foundation_state_factorial_interactions": adjacent_factorial_interactions,
        "global_small_large_foundation_state_interaction": global_factorial_interaction,
        "co_scaled_minus_fixed_small_state_intervals": dict(
            zip(endpoints, fixed_advantage, strict=True)
        ),
        "graph_minus_query_only_intervals": dict(
            zip(endpoints, query_advantage, strict=True)
        ),
        "graph_minus_irrelevant_intervals": dict(
            zip(endpoints, irrelevant_advantage, strict=True)
        ),
        "co_scaled_gates": dict(zip(endpoints, scale_gates, strict=True)),
        "co_scaled_state_bytes_per_parameter": dict(
            zip(endpoints, diagonal_state_ratios, strict=True)
        ),
        "co_scaled_mean_charged_state_bytes": dict(
            zip(endpoints, diagonal_state_bytes, strict=True)
        ),
        "fixed_state_bytes_per_parameter": dict(
            zip(endpoints, fixed_state_ratios, strict=True)
        ),
        "state_token_hashes": state_hashes,
        "gates": gates,
        "promotion": (
            "Official autoregressive decoding and untouched confirmation seeds are "
            "authorized only when development_pass is true without changing this panel."
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
