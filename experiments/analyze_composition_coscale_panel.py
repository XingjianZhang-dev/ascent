#!/usr/bin/env python3
"""Seed-clustered analysis of the two-fact ASCENT composition scale curve."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_ruler_vt_coscale_panel import cluster_summary


def analyze(root: Path, seeds: list[int]) -> dict[str, Any]:
    analyses = {
        str(seed): json.loads((root / f"seed{seed}_analysis.json").read_text())
        for seed in seeds
    }
    first = next(iter(analyses.values()))
    endpoint_names = list(first["endpoint_results"])
    adjacent_names = list(first["adjacent_differences"])
    endpoints = {
        endpoint: cluster_summary([
            analyses[str(seed)]["endpoint_results"][endpoint]["gain"]["mean"] for seed in seeds
        ]) for endpoint in endpoint_names
    }
    adjacent = {
        name: cluster_summary([
            analyses[str(seed)]["adjacent_differences"][name]["difference"]["mean"] for seed in seeds
        ]) for name in adjacent_names
    }
    slope = cluster_summary([
        analyses[str(seed)]["slope_nats_per_log_parameter"]["mean"] for seed in seeds
    ])
    gates = {
        "all_endpoint_cluster_lcbs_positive": all(row["ci95_low"] > 0 for row in endpoints.values()),
        "all_adjacent_cluster_lcbs_positive": all(row["ci95_low"] > 0 for row in adjacent.values()),
        "slope_cluster_lcb_positive": slope["ci95_low"] > 0,
        "every_seed_slope_positive": all(value > 0 for value in slope["seed_values"]),
        "every_seed_adjacent_positive": all(
            value > 0 for row in adjacent.values() for value in row["seed_values"]
        ),
    }
    return {
        "schema_version": 1,
        "status": "passed" if all(gates.values()) else "failed",
        "seeds": seeds,
        "endpoint_gain_clusters": endpoints,
        "adjacent_gain_difference_clusters": adjacent,
        "slope_cluster": slope,
        "individual_seed_status": {seed: value["status"] for seed, value in analyses.items()},
        "gates": gates,
        "strong_raw_boundary": first["strong_raw_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(args.analysis_root, args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
