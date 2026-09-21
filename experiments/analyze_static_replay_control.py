#!/usr/bin/env python3
"""Analyze ASCENT dynamic replay against a same-suffix-FLOP static adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / np.sqrt(values.size))
    return {
        "episodes": int(values.size),
        "mean": mean,
        "standard_error": standard_error,
        "ci95_low": mean - 1.96 * standard_error,
        "ci95_high": mean + 1.96 * standard_error,
    }


def analyze_endpoint(result: dict[str, Any], arrays: Any) -> dict[str, Any]:
    dynamic_gain = arrays["base_nll"] - arrays["rich_fused_nll"]
    static_gain = arrays["base_nll"] - arrays["static_adapter_fused_nll"]
    advantage = arrays["static_adapter_fused_nll"] - arrays["rich_fused_nll"]
    adapter = result["same_flop_static_adapter"]
    return {
        "dynamic_rich_gain": summary(dynamic_gain),
        "static_adapter_gain": summary(static_gain),
        "dynamic_advantage_over_static": summary(advantage),
        "same_replay_token_count": adapter["same_replay_token_count_as_dynamic"],
        "same_suffix_layer": adapter["same_suffix_layer_index_as_dynamic"],
        "same_frozen_suffix_call": adapter["same_frozen_suffix_call_as_dynamic"],
        "static_adapter_training": {
            "steps": adapter["training_steps"],
            "initial_loss": adapter["initial_training_loss"],
            "final_loss": adapter["final_training_loss"],
            "inference_parameter_bytes_bf16": adapter[
                "inference_parameter_bytes_bf16"
            ],
        },
    }


def analyze(
    small_result_path: Path,
    small_arrays_path: Path,
    large_result_path: Path,
    large_arrays_path: Path,
) -> dict[str, Any]:
    small = json.loads(small_result_path.read_text())
    large = json.loads(large_result_path.read_text())
    small_arrays = np.load(small_arrays_path)
    large_arrays = np.load(large_arrays_path)
    invariant_fields = (
        "config_sha256",
        "registered_labels_sha256",
        "registered_full_signal_sha256",
        "identical_query_inputs_sha256",
    )
    invariants = {
        field: small[field] == large[field] for field in invariant_fields
    }
    invariants.update(
        {
            "test_labels_exact": bool(
                np.array_equal(small_arrays["labels"], large_arrays["labels"])
            ),
            "clean_source_trees": not (
                small["runtime"]["git_dirty"] or large["runtime"]["git_dirty"]
            ),
            "identical_source_commit": small["runtime"]["git_commit"]
            == large["runtime"]["git_commit"],
            "small_model_hash_verified": bool(small["model_files"]),
            "large_model_hash_verified": bool(large["model_files"]),
        }
    )
    if not all(invariants.values()):
        raise RuntimeError(f"static-control invariants failed: {invariants}")

    endpoints = {
        "pythia-410m": analyze_endpoint(small, small_arrays),
        "pythia-2.8b": analyze_endpoint(large, large_arrays),
    }
    small_dynamic_gain = small_arrays["base_nll"] - small_arrays["rich_fused_nll"]
    large_dynamic_gain = large_arrays["base_nll"] - large_arrays["rich_fused_nll"]
    interaction = large_dynamic_gain - small_dynamic_gain
    gates = {
        "invariants": all(invariants.values()),
        "exact_same_suffix_flop_path": all(
            endpoint["same_replay_token_count"]
            and endpoint["same_suffix_layer"]
            and endpoint["same_frozen_suffix_call"]
            for endpoint in endpoints.values()
        ),
        "dynamic_beats_static_small_lcb_positive": endpoints["pythia-410m"][
            "dynamic_advantage_over_static"
        ]["ci95_low"]
        > 0.0,
        "dynamic_beats_static_large_lcb_positive": endpoints["pythia-2.8b"][
            "dynamic_advantage_over_static"
        ]["ci95_low"]
        > 0.0,
        "dynamic_rich_gain_scale_interaction_positive": summary(interaction)[
            "ci95_low"
        ]
        > 0.0,
    }
    return {
        "schema_version": 1,
        "invariants": invariants,
        "endpoints": endpoints,
        "dynamic_rich_gain_large_minus_small": summary(interaction),
        "gates": gates,
        "development_pass": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small-result", type=Path, required=True)
    parser.add_argument("--small-arrays", type=Path, required=True)
    parser.add_argument("--large-result", type=Path, required=True)
    parser.add_argument("--large-arrays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.small_result,
        args.small_arrays,
        args.large_result,
        args.large_arrays,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
