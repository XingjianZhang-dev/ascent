#!/usr/bin/env python3
"""Audit a two-endpoint separated address/value overwrite interaction."""

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

from experiments.analyze_composition_factorial import summary


def analyze_cell(json_path: Path, arrays_path: Path) -> tuple[dict[str, Any], Any]:
    return json.loads(json_path.read_text()), np.load(arrays_path)


def run(
    small_json: Path,
    small_arrays: Path,
    large_json: Path,
    large_arrays: Path,
    output: Path,
) -> None:
    small, sa = analyze_cell(small_json, small_arrays)
    large, la = analyze_cell(large_json, large_arrays)
    checks = {
        field: small[field] == large[field]
        for field in [
            "config_sha256",
            "registered_current_sha256",
            "registered_full_signal_sha256",
            "identical_query_inputs_sha256",
            "registered_address_order_sha256",
            "target_key_sha256",
            "distractor_key_sha256",
        ]
    }
    checks.update(
        {
            "current_exact": bool(np.array_equal(sa["current"], la["current"])),
            "address_order_exact": bool(
                np.array_equal(sa["target_first"], la["target_first"])
            ),
            "clean_source_trees": not (
                small["runtime"]["git_dirty"] or large["runtime"]["git_dirty"]
            ),
            "identical_source_commit": small["runtime"]["git_commit"]
            == large["runtime"]["git_commit"],
            "latest_version_contract": bool(
                small["read_before_write_and_later_version_target"]
                and large["read_before_write_and_later_version_target"]
                and small["two_current_addresses_in_primary_replay"]
                and large["two_current_addresses_in_primary_replay"]
            ),
            "target_position_randomized": bool(
                small["target_position_randomized"]
                and large["target_position_randomized"]
            ),
            "registered_architecture": small["architecture"]
            == large["architecture"]
            == "separated_version_address_and_value_posterior",
            "registered_address_pooling": small["address_pooling"]
            == large["address_pooling"]
            == "mean_valid_tokens",
        }
    )
    if not all(checks.values()):
        raise RuntimeError(f"address/value overwrite invariants failed: {checks}")

    gains = {
        "small_fixed": sa["base_nll"] - sa["fixed_latent_fused_nll"],
        "small_rich": sa["base_nll"] - sa["rich_latent_fused_nll"],
        "large_fixed": la["base_nll"] - la["fixed_latent_fused_nll"],
        "large_rich": la["base_nll"] - la["rich_latent_fused_nll"],
    }
    interaction = (
        gains["large_rich"]
        - gains["large_fixed"]
        - gains["small_rich"]
        + gains["small_fixed"]
    )
    history_penalties = {
        "small_fixed": sa["fixed_history_fused_nll"]
        - sa["fixed_latent_fused_nll"],
        "small_rich": sa["rich_history_fused_nll"]
        - sa["rich_latent_fused_nll"],
        "large_fixed": la["fixed_history_fused_nll"]
        - la["fixed_latent_fused_nll"],
        "large_rich": la["rich_history_fused_nll"]
        - la["rich_latent_fused_nll"],
    }
    binding_penalties = {
        "small_fixed": sa["fixed_shuffled_binding_fused_nll"]
        - sa["fixed_latent_fused_nll"],
        "small_rich": sa["rich_shuffled_binding_fused_nll"]
        - sa["rich_latent_fused_nll"],
        "large_fixed": la["fixed_shuffled_binding_fused_nll"]
        - la["fixed_latent_fused_nll"],
        "large_rich": la["rich_shuffled_binding_fused_nll"]
        - la["rich_latent_fused_nll"],
    }
    gain_summaries = {name: summary(value) for name, value in gains.items()}
    interaction_summary = summary(interaction)
    history_summaries = {
        name: summary(value) for name, value in history_penalties.items()
    }
    binding_summaries = {
        name: summary(value) for name, value in binding_penalties.items()
    }
    gates = {
        "all_invariants": all(checks.values()),
        "all_gain_lcbs_positive": all(
            row["ci95_low"] > 0.0 for row in gain_summaries.values()
        ),
        "interaction_lcb_positive": interaction_summary["ci95_low"] > 0.0,
        "all_latest_over_history_lcbs_positive": all(
            row["ci95_low"] > 0.0 for row in history_summaries.values()
        ),
        "all_binding_penalty_lcbs_positive": all(
            row["ci95_low"] > 0.0 for row in binding_summaries.values()
        ),
    }
    gates["development_pass"] = all(gates.values())
    payload = {
        "schema_version": 1,
        "invariant_checks": checks,
        "gains": gain_summaries,
        "primary_foundation_by_signal_interaction": interaction_summary,
        "latest_only_advantage_over_unmasked_history": history_summaries,
        "correct_binding_advantage_over_shuffled_binding": binding_summaries,
        "gates": gates,
        "interpretation": (
            "development pass authorizes only an untouched confirmation seed"
            if gates["development_pass"]
            else "development failed; do not tune this frozen interface"
        ),
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
    run(
        args.small_json,
        args.small_arrays,
        args.large_json,
        args.large_arrays,
        args.output,
    )


if __name__ == "__main__":
    main()
