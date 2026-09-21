#!/usr/bin/env python3
"""Seed-clustered analysis for the three-scale ASCENT RULER-VT curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


T_CRITICAL_975 = {3: 4.302652729696142, 4: 3.182446305284263}


def cluster_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size not in T_CRITICAL_975:
        raise ValueError("registered panel supports exactly three or four data seeds")
    mean = float(array.mean())
    standard_error = float(array.std(ddof=1) / np.sqrt(array.size))
    half_width = T_CRITICAL_975[int(array.size)] * standard_error
    return {
        "seed_values": [float(value) for value in array],
        "mean": mean,
        "standard_error": standard_error,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "degrees_of_freedom": int(array.size - 1),
    }


def analyze(analysis_root: Path, seeds: list[int]) -> dict[str, Any]:
    analyses = {
        str(seed): json.loads((analysis_root / str(seed) / "analysis.json").read_text())
        for seed in seeds
    }
    endpoint_names = list(next(iter(analyses.values()))["endpoint_results"])
    adjacent_names = list(next(iter(analyses.values()))["adjacent_differences"])

    endpoint_results = {
        endpoint: cluster_summary([
            analyses[str(seed)]["endpoint_results"][endpoint]["gain"][
                "mean_nats_per_sample_token_average"
            ]
            for seed in seeds
        ])
        for endpoint in endpoint_names
    }
    adjacent_results = {
        contrast: cluster_summary([
            analyses[str(seed)]["adjacent_differences"][contrast]["gain_difference"][
                "mean_nats_per_sample_token_average"
            ]
            for seed in seeds
        ])
        for contrast in adjacent_names
    }
    slope = cluster_summary([
        analyses[str(seed)]["slope_nats_per_log_parameter"][
            "mean_nats_per_sample_token_average"
        ]
        for seed in seeds
    ])
    gates = {
        "all_endpoint_cluster_lcbs_positive": all(
            result["ci95_low"] > 0 for result in endpoint_results.values()
        ),
        "all_adjacent_cluster_lcbs_positive": all(
            result["ci95_low"] > 0 for result in adjacent_results.values()
        ),
        "slope_cluster_lcb_positive": slope["ci95_low"] > 0,
        "every_seed_slope_positive": all(value > 0 for value in slope["seed_values"]),
        "every_seed_adjacent_difference_positive": all(
            value > 0
            for result in adjacent_results.values()
            for value in result["seed_values"]
        ),
    }
    return {
        "schema_version": 1,
        "status": "passed" if all(gates.values()) else "failed",
        "seeds": seeds,
        "seed_weighting": "Each independently generated data seed receives equal weight.",
        "endpoint_gain_clusters": endpoint_results,
        "adjacent_gain_difference_clusters": adjacent_results,
        "slope_cluster": slope,
        "gates": gates,
        "individual_seed_status": {
            seed: analysis["status"] for seed, analysis in analyses.items()
        },
        "metric_boundary": "Teacher-forced answer-token NLL, not official autoregressive exact match.",
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
