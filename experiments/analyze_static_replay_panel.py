#!/usr/bin/env python3
"""Cluster three confirmation seeds for the same-FLOP static replay control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_babilong_scale_panel import PANEL_T_95, mean_ci
from experiments.analyze_static_replay_control import analyze


SEEDS = (20260941, 20260951, 20260961)


def paths(root: Path, seed: int, endpoint: str) -> tuple[Path, Path]:
    node = "node2" if endpoint == "pythia-410m" else "node1"
    stem = root / node / f"seed{seed}_{endpoint}"
    return Path(str(stem) + ".json"), Path(str(stem) + ".npz")


def run(root: Path) -> dict[str, Any]:
    seeds: list[dict[str, Any]] = []
    raw_results: list[dict[str, Any]] = []
    for seed in SEEDS:
        small_json, small_npz = paths(root, seed, "pythia-410m")
        large_json, large_npz = paths(root, seed, "pythia-2.8b")
        small = json.loads(small_json.read_text())
        large = json.loads(large_json.read_text())
        raw_results.extend((small, large))
        analysis = analyze(small_json, small_npz, large_json, large_npz)
        if small["effective_seed"] != seed or large["effective_seed"] != seed:
            raise RuntimeError(f"effective seed mismatch for {seed}")
        seeds.append({"seed": seed, "analysis": analysis})

    config_hashes = {result["config_sha256"] for result in raw_results}
    commits = {result["runtime"]["git_commit"] for result in raw_results}
    provenance = {
        "single_config_hash": len(config_hashes) == 1,
        "config_hashes": sorted(config_hashes),
        "single_git_commit": len(commits) == 1,
        "git_commits": sorted(commits),
        "all_worktrees_clean": all(
            not result["runtime"]["git_dirty"] for result in raw_results
        ),
        "all_model_hashes_verified": all(result["model_files"] for result in raw_results),
        "all_seeds_registered": all(
            result["seed_is_registered_override"] for result in raw_results
        ),
    }
    endpoint_intervals: dict[str, Any] = {}
    for endpoint in ("pythia-410m", "pythia-2.8b"):
        values = [
            row["analysis"]["endpoints"][endpoint][
                "dynamic_advantage_over_static"
            ]["mean"]
            for row in seeds
        ]
        endpoint_intervals[endpoint] = mean_ci(values, critical=PANEL_T_95)
    scale_values = [
        row["analysis"]["dynamic_rich_gain_large_minus_small"]["mean"]
        for row in seeds
    ]
    scale_interval = mean_ci(scale_values, critical=PANEL_T_95)
    gates = {
        "provenance": all(provenance.values()),
        "every_seed_passes": all(
            row["analysis"]["development_pass"] for row in seeds
        ),
        "every_seed_endpoint_paired_lcb_positive": all(
            row["analysis"]["endpoints"][endpoint][
                "dynamic_advantage_over_static"
            ]["ci95_low"]
            > 0.0
            for row in seeds
            for endpoint in ("pythia-410m", "pythia-2.8b")
        ),
        "both_endpoint_seed_t_lcbs_positive": all(
            interval["ci95_low"] > 0.0 for interval in endpoint_intervals.values()
        ),
        "scale_interaction_seed_t_lcb_positive": scale_interval["ci95_low"]
        > 0.0,
    }
    return {
        "schema_version": 1,
        "seeds": seeds,
        "dynamic_advantage_over_static_seed_t_intervals": endpoint_intervals,
        "dynamic_rich_gain_large_minus_small_seed_t_interval": scale_interval,
        "provenance": provenance,
        "gates": gates,
        "confirmation_pass": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
