#!/usr/bin/env python3
"""Audit and classify a two-endpoint overwrite interaction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_composition_factorial import summary


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
            "config_sha256", "registered_current_sha256",
            "registered_full_signal_sha256", "identical_query_inputs_sha256",
        ]
    }
    checks.update({
        "current_exact": bool(np.array_equal(sa["current"], la["current"])),
        "clean_source_trees": not (small["runtime"]["git_dirty"] or large["runtime"]["git_dirty"]),
        "identical_source_commit": small["runtime"]["git_commit"] == large["runtime"]["git_commit"],
        "latest_version_contract": bool(
            small["read_before_write_and_later_version_target"]
            and large["read_before_write_and_later_version_target"]
        ),
    })
    if not all(checks.values()):
        raise RuntimeError(f"overwrite invariants failed: {checks}")

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
    history_penalties = {
        "small_fixed_latest_minus_unmasked":
            sa["fixed_history_fused_nll"] - sa["fixed_latent_fused_nll"],
        "small_rich_latest_minus_unmasked":
            sa["rich_history_fused_nll"] - sa["rich_latent_fused_nll"],
        "large_fixed_latest_minus_unmasked":
            la["fixed_history_fused_nll"] - la["fixed_latent_fused_nll"],
        "large_rich_latest_minus_unmasked":
            la["rich_history_fused_nll"] - la["rich_latent_fused_nll"],
    }
    gain_summaries = {name: summary(values) for name, values in gains.items()}
    interaction_summary = summary(interaction)
    history_summaries = {name: summary(values) for name, values in history_penalties.items()}
    positive = interaction_summary["ci95_low"] > 0
    negative = interaction_summary["ci95_high"] < 0
    payload = {
        "schema_version": 1, "invariant_checks": checks,
        "gains": gain_summaries,
        "primary_foundation_by_signal_interaction": interaction_summary,
        "scale_complementarity_pass": positive,
        "scale_complementarity_falsified": negative,
        "interaction_inconclusive": not positive and not negative,
        "all_gain_lcbs_positive": all(row["ci95_low"] > 0 for row in gain_summaries.values()),
        "latest_mask_advantage_over_unmasked_history": history_summaries,
        "all_latest_mask_lcbs_positive": all(row["ci95_low"] > 0 for row in history_summaries.values()),
        "stage_b_promotion_eligible": False,
        "interpretation": (
            "overwrite scale complementarity is falsified for the locked interface"
            if negative else "overwrite interaction did not yield a decisive negative interval"
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
    run(args.small_json, args.small_arrays, args.large_json, args.large_arrays, args.output)


if __name__ == "__main__":
    main()
