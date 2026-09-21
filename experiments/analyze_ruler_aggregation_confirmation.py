#!/usr/bin/env python3
"""Audit the frozen three-seed official RULER aggregation confirmation panel.

The absolute-gain gates remain the originally frozen confirmatory test.  A
separate, explicitly post-hoc ceiling-aware view reports the fraction of the
Foundation model's remaining error removed by ASCENT.  This prevents a perfect
ASCENT score from being misread as a scaling failure merely because accuracy
cannot exceed one.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


ENDPOINTS = (
    "smollm2-135m-instruct",
    "smollm2-360m-instruct",
    "smollm2-1p7b-instruct",
)
T_CRITICAL_DF2_95_TWO_SIDED = 4.302652729911275


def t_interval(values: list[float]) -> dict[str, float | int]:
    mean = statistics.mean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    margin = T_CRITICAL_DF2_95_TWO_SIDED * standard_error
    return {
        "clusters": len(values),
        "mean": mean,
        "standard_error": standard_error,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def slope_on_log_parameters(gains: list[float], parameters: list[int]) -> float:
    x = [math.log(value) for value in parameters]
    x_mean = statistics.mean(x)
    y_mean = statistics.mean(gains)
    return sum((a - x_mean) * (b - y_mean) for a, b in zip(x, gains, strict=True)) / sum(
        (value - x_mean) ** 2 for value in x
    )


def remaining_error_elimination(foundation: float, ascent: float) -> float | None:
    """Return (ASCENT - Foundation) / (1 - Foundation), if headroom exists."""

    if foundation >= 1.0:
        return None
    return (ascent - foundation) / (1.0 - foundation)


def analyze(config_path: Path, results_root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    endpoints = tuple(row["name"] for row in config["endpoints"])
    if len(endpoints) != 3:
        raise ValueError("the confirmation analyzer requires exactly three endpoints")
    seeds = [int(value) for value in config["data_seeds"]]
    seed_rows: list[dict[str, Any]] = []
    config_hashes: set[str] = set()
    commits: set[str] = set()
    observed_data_hashes: dict[str, str] = {}
    all_worktrees_clean = True
    all_parsers_exact = True
    all_no_regressions = True
    all_native_length_safe = True
    all_relative_state_decreases = True
    all_scale_beats_fixed = True
    for seed in seeds:
        cells: list[dict[str, Any]] = []
        for endpoint in endpoints:
            path = results_root / f"seed{seed}_{endpoint}.json"
            result = json.loads(path.read_text())
            config_hashes.add(result["config"]["sha256"])
            commits.add(result["environment"]["git_commit"])
            observed_data_hashes[str(seed)] = result["data"]["sha256"]
            all_worktrees_clean &= not result["environment"]["git_dirty"]
            all_parsers_exact &= result["parser_accuracy"] == 1.0
            all_no_regressions &= result["scale_regressions"] == 0
            all_native_length_safe &= (
                result["systems"]["maximum_foundation_prompt_tokens"]
                + int(config["max_new_tokens"])
                <= int(config["native_context_tokens"])
            )
            cells.append(
                {
                    "endpoint": endpoint,
                    "model_parameters": result["endpoint"]["model_parameters"],
                    "foundation": result["foundation"]["mean"],
                    "scale_ascent": result["scale_ascent"]["mean"],
                    "fixed_gain": result["fixed_gain"]["mean"],
                    "scale_gain": result["scale_gain"]["mean"],
                    "remaining_error_elimination": remaining_error_elimination(
                        result["foundation"]["mean"],
                        result["scale_ascent"]["mean"],
                    ),
                    "scale_minus_fixed": result["scale_gain"]["mean"] - result["fixed_gain"]["mean"],
                    "wins": result["scale_wins"],
                    "regressions": result["scale_regressions"],
                    "maximum_prompt_tokens": result["systems"]["maximum_foundation_prompt_tokens"],
                    "state_bytes_per_parameter": result["state"]["scale_total_state_bytes_per_model_parameter"],
                }
            )
        gains = [cell["scale_gain"] for cell in cells]
        adjacent = [gains[index + 1] - gains[index] for index in range(2)]
        error_elimination = [
            cell["remaining_error_elimination"] for cell in cells
        ]
        if any(value is None for value in error_elimination):
            raise ValueError(
                "Ceiling-aware analysis is undefined when Foundation is already perfect"
            )
        error_elimination_values = [float(value) for value in error_elimination]
        adjacent_error_elimination = [
            error_elimination_values[index + 1] - error_elimination_values[index]
            for index in range(2)
        ]
        relative_state = [cell["state_bytes_per_parameter"] for cell in cells]
        all_relative_state_decreases &= all(
            relative_state[index + 1] < relative_state[index]
            for index in range(2)
        )
        all_scale_beats_fixed &= all(
            cells[index]["scale_minus_fixed"] > 0.0 for index in (1, 2)
        )
        seed_rows.append(
            {
                "seed": seed,
                "cells": cells,
                "adjacent_gain_differences": adjacent,
                "adjacent_remaining_error_elimination_differences": adjacent_error_elimination,
                "slope_on_log_parameters": slope_on_log_parameters(
                    gains, [cell["model_parameters"] for cell in cells]
                ),
                "all_endpoint_gains_positive": all(value > 0.0 for value in gains),
                "both_adjacent_differences_positive": all(value > 0.0 for value in adjacent),
            }
        )

    adjacent_intervals = [
        t_interval(
            [row["adjacent_gain_differences"][index] for row in seed_rows]
        )
        for index in range(2)
    ]
    slope_interval = t_interval(
        [row["slope_on_log_parameters"] for row in seed_rows]
    )
    endpoint_gain_intervals = {
        endpoint: t_interval(
            [row["cells"][index]["scale_gain"] for row in seed_rows]
        )
        for index, endpoint in enumerate(endpoints)
    }
    endpoint_error_elimination_intervals = {
        endpoint: t_interval(
            [
                float(row["cells"][index]["remaining_error_elimination"])
                for row in seed_rows
            ]
        )
        for index, endpoint in enumerate(endpoints)
    }
    adjacent_error_elimination_intervals = [
        t_interval(
            [
                row["adjacent_remaining_error_elimination_differences"][index]
                for row in seed_rows
            ]
        )
        for index in range(2)
    ]
    gates = {
        "single_config_hash": len(config_hashes) == 1,
        "single_git_commit": len(commits) == 1,
        "exact_registered_data_hashes": observed_data_hashes
        == config["task_sha256_by_seed"],
        "all_worktrees_clean": all_worktrees_clean,
        "all_parsers_exact": all_parsers_exact,
        "all_native_lengths_safe": all_native_length_safe,
        "every_seed_has_positive_endpoint_gains": all(
            row["all_endpoint_gains_positive"] for row in seed_rows
        ),
        "every_seed_has_positive_adjacent_differences": all(
            row["both_adjacent_differences_positive"] for row in seed_rows
        ),
        "all_endpoint_gain_t_lcbs_positive": all(
            interval["ci95_low"] > 0.0
            for interval in endpoint_gain_intervals.values()
        ),
        "both_adjacent_t_lcbs_positive": all(
            interval["ci95_low"] > 0.0 for interval in adjacent_intervals
        ),
        "slope_t_lcb_positive": slope_interval["ci95_low"] > 0.0,
        "no_scale_regressions": all_no_regressions,
        "relative_state_cost_strictly_decreases": all_relative_state_decreases,
        "scale_expansion_beats_fixed_after_small_endpoint": all_scale_beats_fixed,
    }
    gates["confirmation_pass"] = all(gates.values())
    frozen_ceiling_config = config.get("ceiling_aware_secondary_estimand")
    ceiling_aware_status = (
        frozen_ceiling_config.get("status")
        if frozen_ceiling_config
        else "posthoc_metric_added_after_user_identified_accuracy_ceiling"
    )
    ceiling_aware_posthoc = {
        "status": ceiling_aware_status,
        "estimand": "(scale_ascent_accuracy - foundation_accuracy) / (1 - foundation_accuracy)",
        "interpretation": "fraction of Foundation's remaining error eliminated by scale ASCENT",
        "endpoint_t_intervals": endpoint_error_elimination_intervals,
        "adjacent_difference_t_intervals": adjacent_error_elimination_intervals,
        "both_adjacent_means_positive": all(
            interval["mean"] > 0.0
            for interval in adjacent_error_elimination_intervals
        ),
        "both_adjacent_t_lcbs_positive": all(
            interval["ci95_low"] > 0.0
            for interval in adjacent_error_elimination_intervals
        ),
        "secondary_gate_pass": all(
            interval["ci95_low"] > 0.0
            for interval in adjacent_error_elimination_intervals
        ),
        "frozen_before_scores": frozen_ceiling_config is not None,
        "not_part_of_original_confirmation_pass": True,
    }
    return {
        "schema_version": 1,
        "seeds": seed_rows,
        "endpoint_gain_t_intervals": endpoint_gain_intervals,
        "adjacent_gain_difference_t_intervals": adjacent_intervals,
        "slope_t_interval": slope_interval,
        "ceiling_aware_posthoc": ceiling_aware_posthoc,
        "config_hashes": sorted(config_hashes),
        "git_commits": sorted(commits),
        "observed_data_hashes": observed_data_hashes,
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.config, args.results_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
