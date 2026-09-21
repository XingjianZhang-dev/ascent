#!/usr/bin/env python3
"""Audit a frozen three-scale RULER aggregation development screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ENDPOINTS = (
    "smollm2-135m-instruct",
    "smollm2-360m-instruct",
    "smollm2-1p7b-instruct",
)


def analyze(results_root: Path, config_path: Path | None = None) -> dict[str, Any]:
    endpoints = ENDPOINTS
    if config_path is not None:
        config = json.loads(config_path.read_text())
        endpoints = tuple(row["name"] for row in config["endpoints"])
        if len(endpoints) != 3:
            raise ValueError("the development analyzer requires exactly three endpoints")
    rows: list[dict[str, Any]] = []
    config_hashes: set[str] = set()
    data_hashes: set[str] = set()
    commits: set[str] = set()
    for endpoint in endpoints:
        result = json.loads((results_root / f"{endpoint}.json").read_text())
        config_hashes.add(result["config"]["sha256"])
        data_hashes.add(result["data"]["sha256"])
        commits.add(result["environment"]["git_commit"])
        rows.append(
            {
                "endpoint": endpoint,
                "model_parameters": result["endpoint"]["model_parameters"],
                "certified_word_slots": result["state"]["scale_certified_word_slots"],
                "foundation": result["foundation"]["mean"],
                "fixed_gain": result["fixed_gain"]["mean"],
                "scale_gain": result["scale_gain"]["mean"],
                "scale_minus_fixed": result["scale_gain"]["mean"] - result["fixed_gain"]["mean"],
                "scale_wins": result["scale_wins"],
                "scale_regressions": result["scale_regressions"],
                "parser_accuracy": result["parser_accuracy"],
                "mean_total_state_bytes": result["state"]["mean_scale_total_state_bytes"],
                "state_bytes_per_parameter": result["state"]["scale_total_state_bytes_per_model_parameter"],
                "worktree_clean": not result["environment"]["git_dirty"],
            }
        )
    gains = [row["scale_gain"] for row in rows]
    adjacent = [gains[index + 1] - gains[index] for index in range(2)]
    relative_state = [row["state_bytes_per_parameter"] for row in rows]
    gates = {
        "single_config_hash": len(config_hashes) == 1,
        "single_data_hash": len(data_hashes) == 1,
        "single_git_commit": len(commits) == 1,
        "all_worktrees_clean": all(row["worktree_clean"] for row in rows),
        "all_parsers_exact": all(row["parser_accuracy"] == 1.0 for row in rows),
        "all_scale_gains_positive": all(value > 0.0 for value in gains),
        "both_adjacent_gain_differences_positive": all(value > 0.0 for value in adjacent),
        "no_scale_regressions": all(row["scale_regressions"] == 0 for row in rows),
        "relative_state_cost_strictly_decreases": all(
            relative_state[index + 1] < relative_state[index]
            for index in range(2)
        ),
        "scale_expansion_beats_fixed_after_small_endpoint": all(
            rows[index]["scale_minus_fixed"] > 0.0 for index in (1, 2)
        ),
    }
    gates["development_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "endpoints": rows,
        "adjacent_scale_gain_differences": adjacent,
        "config_hashes": sorted(config_hashes),
        "data_hashes": sorted(data_hashes),
        "git_commits": sorted(commits),
        "gates": gates,
        "interpretation": (
            "Development only. A pass authorizes three entirely new official-data "
            "seeds; it is not confirmatory evidence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.results_root, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
