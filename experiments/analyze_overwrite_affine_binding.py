#!/usr/bin/env python3
"""Audit a two-endpoint binding-dependent overwrite composition screen."""

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


def load(json_path: Path, arrays_path: Path) -> tuple[dict[str, Any], Any]:
    return json.loads(json_path.read_text()), np.load(arrays_path)


def run(
    small_json: Path,
    small_arrays: Path,
    large_json: Path,
    large_arrays: Path,
    output: Path,
) -> None:
    small, sa = load(small_json, small_arrays)
    large, la = load(large_json, large_arrays)
    checks = {
        field: small[field] == large[field]
        for field in [
            "config_sha256",
            "registered_answer_sha256",
            "registered_current_sha256",
            "registered_full_signal_sha256",
            "identical_query_inputs_sha256",
            "first_key_sha256",
            "second_key_sha256",
            "registered_storage_order_sha256",
            "address_decoder",
            "target_function",
        ]
    }
    checks.update(
        {
            "answers_exact": bool(np.array_equal(sa["answers"], la["answers"])),
            "currents_exact": bool(
                np.array_equal(sa["first_current"], la["first_current"])
                and np.array_equal(sa["second_current"], la["second_current"])
            ),
            "storage_order_exact": bool(
                np.array_equal(sa["first_stored_first"], la["first_stored_first"])
            ),
            "clean_source_trees": not (
                small["runtime"]["git_dirty"] or large["runtime"]["git_dirty"]
            ),
            "identical_source_commit": small["runtime"]["git_commit"]
            == large["runtime"]["git_commit"],
            "registered_architecture": small["architecture"]
            == large["architecture"]
            == "exact_address_latest_version_ordered_affine_latent_replay",
            "continuous_nll_metric": bool(
                small["accuracy_ceiling_not_used"]
                and large["accuracy_ceiling_not_used"]
                and small["continuous_primary_metric"]
                == large["continuous_primary_metric"]
                == "paired_nll_gain_nats"
            ),
            "latest_version_contract": bool(
                small["read_before_write_and_later_version_target"]
                and large["read_before_write_and_later_version_target"]
            ),
            "noncommutative_binding_contract": bool(
                small["noncommutative_binding_target"]
                and large["noncommutative_binding_target"]
                and small["two_distinct_query_keys"]
                and large["two_distinct_query_keys"]
                and small["event_storage_order_randomized"]
                and large["event_storage_order_randomized"]
            ),
        }
    )
    if not all(checks.values()):
        raise RuntimeError(f"overwrite affine invariants failed: {checks}")

    gains = {
        "small_fixed": sa["base_nll"] - sa["fixed_latent_fused_nll"],
        "small_rich": sa["base_nll"] - sa["rich_latent_fused_nll"],
        "large_fixed": la["base_nll"] - la["fixed_latent_fused_nll"],
        "large_rich": la["base_nll"] - la["rich_latent_fused_nll"],
    }
    within_endpoint_expansion = {
        "small": gains["small_rich"] - gains["small_fixed"],
        "large": gains["large_rich"] - gains["large_fixed"],
    }
    interaction = (
        within_endpoint_expansion["large"]
        - within_endpoint_expansion["small"]
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
    expansion_summaries = {
        name: summary(value) for name, value in within_endpoint_expansion.items()
    }
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
        "both_expansion_lcbs_positive": all(
            row["ci95_low"] > 0.0 for row in expansion_summaries.values()
        ),
        "interaction_lcb_positive": interaction_summary["ci95_low"] > 0.0,
        "rich_latest_over_history_lcbs_positive": all(
            history_summaries[name]["ci95_low"] > 0.0
            for name in ("small_rich", "large_rich")
        ),
        "rich_correct_binding_lcbs_positive": all(
            binding_summaries[name]["ci95_low"] > 0.0
            for name in ("small_rich", "large_rich")
        ),
    }
    gates["development_pass"] = all(gates.values())
    payload = {
        "schema_version": 1,
        "invariant_checks": checks,
        "gains": gain_summaries,
        "within_endpoint_signal_expansion": expansion_summaries,
        "primary_foundation_by_signal_interaction": interaction_summary,
        "latest_only_advantage_over_unmasked_history": history_summaries,
        "correct_binding_advantage_over_shuffled_binding": binding_summaries,
        "gates": gates,
        "interpretation": (
            "development pass authorizes only untouched confirmation seeds"
            if gates["development_pass"]
            else "development failed; retain unchanged and do not tune this interface"
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
