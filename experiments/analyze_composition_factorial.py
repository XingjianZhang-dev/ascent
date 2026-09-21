#!/usr/bin/env python3
"""Audit and contrast the two-endpoint composition confirmation."""

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
        "episodes": int(values.size), "mean": mean,
        "ci95_low": mean - 1.96 * se, "ci95_high": mean + 1.96 * se,
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
    checks = {
        field: small[field] == large[field]
        for field in [
            "config_sha256", "registered_answers_sha256",
            "registered_full_signal_sha256", "identical_query_inputs_sha256",
        ]
    }
    checks.update({
        "answers_exact": bool(np.array_equal(sa["answers"], la["answers"])),
        "clean_source_trees": not (small["runtime"]["git_dirty"] or large["runtime"]["git_dirty"]),
        "identical_source_commit": small["runtime"]["git_commit"] == large["runtime"]["git_commit"],
        "oracle_nll_zero": small["clean_kv_oracle_nll"] == large["clean_kv_oracle_nll"] == 0.0,
    })
    if not all(checks.values()):
        raise RuntimeError(f"composition invariants failed: {checks}")

    gains = {
        "small_fixed": sa["base_nll"] - sa["fixed_latent_fused_nll"],
        "small_rich": sa["base_nll"] - sa["rich_latent_fused_nll"],
        "large_fixed": la["base_nll"] - la["fixed_latent_fused_nll"],
        "large_rich": la["base_nll"] - la["rich_latent_fused_nll"],
    }
    interaction = (
        gains["large_rich"] - gains["large_fixed"]
        - gains["small_rich"] + gains["small_fixed"]
    )
    raw_advantages = {
        "small_ascent_rich_minus_clean_kv_text":
            sa["clean_text_fused_nll"] - sa["rich_latent_fused_nll"],
        "large_ascent_rich_minus_clean_kv_text":
            la["clean_text_fused_nll"] - la["rich_latent_fused_nll"],
    }
    gain_summaries = {name: summary(values) for name, values in gains.items()}
    interaction_summary = summary(interaction)
    raw_summaries = {name: summary(values) for name, values in raw_advantages.items()}
    primary_pass = interaction_summary["ci95_low"] > 0
    all_gain_pass = all(row["ci95_low"] > 0 for row in gain_summaries.values())
    strong_raw_superiority = all(row["ci95_low"] > 0 for row in raw_summaries.values())
    payload = {
        "schema_version": 1, "invariant_checks": checks,
        "gains": gain_summaries,
        "primary_foundation_by_signal_interaction": interaction_summary,
        "primary_pass": primary_pass, "all_gain_lcbs_positive": all_gain_pass,
        "composition_confirmation_pass": primary_pass and all_gain_pass,
        "ascent_rich_advantage_over_clean_kv_text": raw_summaries,
        "strong_raw_text_superiority_pass": strong_raw_superiority,
        "clean_kv_oracle_nll": 0.0,
        "ascent_can_beat_clean_kv_oracle": False,
        "stage_b_promotion_eligible": False,
        "remaining_blockers": [
            "clean_kv_boundary_not_superiority", "overwrite",
            "fpvm_v1_same_task_or_external_validity", "natural_text_long_context_cross_family",
        ],
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
