#!/usr/bin/env python3
"""Verify two endpoint artifacts and compute paired factorial contrasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(values.size))
    return {
        "episodes": int(values.size),
        "mean": mean,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
    }


def run(
    small_json: Path,
    small_arrays: Path,
    large_json: Path,
    large_arrays: Path,
    output: Path,
) -> None:
    small = json.loads(small_json.read_text())
    large = json.loads(large_json.read_text())
    sa = np.load(small_arrays)
    la = np.load(large_arrays)
    invariant_fields = [
        "config_sha256",
        "registered_labels_sha256",
        "registered_full_signal_sha256",
        "identical_query_inputs_sha256",
    ]
    invariant_checks = {field: small[field] == large[field] for field in invariant_fields}
    invariant_checks["test_labels_exact"] = bool(np.array_equal(sa["labels"], la["labels"]))
    invariant_checks["clean_source_trees"] = not (
        small["runtime"]["git_dirty"] or large["runtime"]["git_dirty"]
    )
    invariant_checks["identical_source_commit"] = (
        small["runtime"]["git_commit"] == large["runtime"]["git_commit"]
    )
    if not all(invariant_checks.values()):
        raise RuntimeError(f"factorial invariants failed: {invariant_checks}")

    gains = {
        "small_fixed": sa["base_nll"] - sa["fixed_fused_nll"],
        "small_rich": sa["base_nll"] - sa["rich_fused_nll"],
        "large_fixed": la["base_nll"] - la["fixed_fused_nll"],
        "large_rich": la["base_nll"] - la["rich_fused_nll"],
    }
    interaction = (
        gains["large_rich"] - gains["large_fixed"]
        - gains["small_rich"] + gains["small_fixed"]
    )
    contrasts = {
        "primary_foundation_by_signal_interaction": interaction,
        "foundation_effect_at_fixed_signal": gains["large_fixed"] - gains["small_fixed"],
        "foundation_effect_at_rich_signal": gains["large_rich"] - gains["small_rich"],
        "small_memory_enrichment": gains["small_rich"] - gains["small_fixed"],
        "large_memory_enrichment": gains["large_rich"] - gains["large_fixed"],
        "co_scaled_large_rich_minus_small_fixed": gains["large_rich"] - gains["small_fixed"],
    }
    gain_summaries = {name: summary(values) for name, values in gains.items()}
    contrast_summaries = {name: summary(values) for name, values in contrasts.items()}
    primary_pass = contrast_summaries["primary_foundation_by_signal_interaction"]["ci95_low"] > 0
    secondary_gain_pass = all(row["ci95_low"] > 0 for row in gain_summaries.values())
    payload = {
        "schema_version": 1,
        "invariant_checks": invariant_checks,
        "gains": gain_summaries,
        "contrasts": contrast_summaries,
        "primary_pass": primary_pass,
        "secondary_all_gain_lcbs_positive": secondary_gain_pass,
        "factorial_confirmation_pass": primary_pass and secondary_gain_pass,
        "stage_b_promotion_eligible": False,
        "remaining_blockers": [
            "clean_exact_kv_capacity_pressure_tasks",
            "fpvm_v1_fixed_state",
            "overwrite_and_composition",
            "three_seed_scale_curve",
        ],
        "small_result": str(small_json),
        "large_result": str(large_json),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small-json", type=Path, required=True)
    parser.add_argument("--small-arrays", type=Path, required=True)
    parser.add_argument("--large-json", type=Path, required=True)
    parser.add_argument("--large-arrays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.small_json, args.small_arrays, args.large_json, args.large_arrays, args.output)


if __name__ == "__main__":
    main()
