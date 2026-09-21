#!/usr/bin/env python3
"""Analyze the frozen repeat of the isolated BABILong retrieval tail."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


ORDERS = ("foundation_first", "ascent_first")


def _prediction_signature(result: dict[str, Any]) -> list[tuple[Any, ...]]:
    keys = (
        "row_id",
        "foundation_output",
        "foundation_score",
        "ascent_output",
        "ascent_score",
        "retained_facts",
        "foundation_prompt_tokens",
        "ascent_prompt_tokens",
    )
    return [tuple(row[key] for key in keys) for row in result["predictions"]]


def _load_new(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads((root / f"{order}_{rep}.json").read_text())
        for order in ORDERS
        for rep in (1, 2, 3)
    ]


def analyze(config_path: Path, new_root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    results = _load_new(new_root)
    signatures = [_prediction_signature(result) for result in results]
    new_retrieval = [
        result["systems"]["retrieval_microseconds_per_sample"]
        for result in results
    ]
    prior_retrieval = config["prior_result_retained"][
        "reference_retrieval_microseconds_per_sample"
    ]
    combined = [*prior_retrieval, *new_retrieval]
    decode_ratios = [
        result["systems"]["ascent_to_foundation_decode_time_ratio"]
        for result in results
    ]
    result = {
        "schema_version": 1,
        "prior_observation_retained": config["prior_result_retained"],
        "new_runs": {
            "retrieval_microseconds_per_sample": new_retrieval,
            "median": statistics.median(new_retrieval),
            "maximum": max(new_retrieval),
            "count_above_1ms": sum(value >= 1000.0 for value in new_retrieval),
            "decode_ratios": decode_ratios,
            "decode_ratio_median": statistics.median(decode_ratios),
        },
        "combined_old_plus_new": {
            "retrieval_microseconds_per_sample": combined,
            "median": statistics.median(combined),
            "maximum": max(combined),
            "count_above_1ms": sum(value >= 1000.0 for value in combined),
        },
    }
    gates = {
        "six_new_runs": len(results) == 6,
        "all_new_predictions_identical": all(
            signature == signatures[0] for signature in signatures[1:]
        ),
        "all_new_accuracies_identical": len(
            {
                (
                    row["foundation"]["mean"],
                    row["ascent"]["mean"],
                    row["gain"]["mean"],
                )
                for row in results
            }
        )
        == 1,
        "all_new_worktrees_clean": all(
            not row["environment"]["git_dirty"] for row in results
        ),
        "combined_median_below_1ms": statistics.median(combined) < 1000.0,
        "at_most_one_combined_measurement_above_1ms": sum(
            value >= 1000.0 for value in combined
        )
        <= 1,
    }
    gates["isolated_tail_audit_pass"] = all(gates.values())
    gates["original_strict_every_run_gate_remains_failed"] = True
    result["gates"] = gates
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.config, args.new_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
