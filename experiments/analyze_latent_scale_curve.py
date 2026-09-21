#!/usr/bin/env python3
"""Audit and aggregate the preregistered three-scale latent-replay curve."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

T975_DF2 = 4.302652729749


def normal_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(values.size))
    return {
        "episodes": int(values.size), "mean": mean,
        "ci95_low": mean - 1.96 * se, "ci95_high": mean + 1.96 * se,
    }


def three_seed_cluster_summary(seed_estimates: list[float]) -> dict[str, object]:
    values = np.asarray(seed_estimates, dtype=np.float64)
    if values.size != 3:
        raise ValueError("the preregistered cluster interval requires exactly three seeds")
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(3))
    return {
        "seed_estimates": values.tolist(), "clusters": 3, "mean": mean,
        "t_critical_df2": T975_DF2,
        "ci95_low": mean - T975_DF2 * se,
        "ci95_high": mean + T975_DF2 * se,
    }


def slug(name: str) -> str:
    return name.replace(".", "p").replace("-", "_")


def run(config_path: Path, result_root: Path, output: Path) -> None:
    config = json.loads(config_path.read_text())
    seeds = [int(value) for value in config["seeds"]]
    endpoints = list(config["endpoints"])
    log_parameters = np.log([float(row["model_parameters"]) for row in endpoints])
    centered = log_parameters - log_parameters.mean()
    slope_denominator = float(np.square(centered).sum())
    seed_slopes: list[float] = []
    adjacent_seed_interactions: list[list[float]] = [[], []]
    cell_summaries: dict[str, object] = {}
    audits: dict[str, object] = {}

    for seed in seeds:
        payloads = []
        arrays = []
        for endpoint in endpoints:
            prefix = f"{slug(endpoint['name'])}_seed{seed}"
            payloads.append(json.loads((result_root / f"{prefix}.json").read_text()))
            arrays.append(np.load(result_root / f"{prefix}.npz"))
        reference = payloads[0]
        equality_fields = [
            "config_sha256", "registered_labels_sha256",
            "registered_full_signal_sha256", "identical_query_inputs_sha256",
        ]
        checks = {
            field: all(row[field] == reference[field] for row in payloads)
            for field in equality_fields
        }
        checks.update({
            "labels_exact": all(np.array_equal(row["labels"], arrays[0]["labels"]) for row in arrays),
            "effective_seed": all(row["effective_seed"] == seed for row in payloads),
            "clean": all(not row["runtime"]["git_dirty"] for row in payloads),
            "same_commit": len({row["runtime"]["git_commit"] for row in payloads}) == 1,
            "no_depth_override": all(not row["relative_depth_is_posthoc_override"] for row in payloads),
            "no_replay_override": all(not row["replay_mode_is_posthoc_override"] for row in payloads),
        })
        if not all(checks.values()):
            raise RuntimeError(f"seed {seed} audit failed: {checks}")
        audits[str(seed)] = checks

        fixed_gains = [row["base_nll"] - row["fixed_fused_nll"] for row in arrays]
        rich_gains = [row["base_nll"] - row["rich_fused_nll"] for row in arrays]
        rich_matrix = np.stack(rich_gains)
        per_episode_slope = centered @ rich_matrix / slope_denominator
        seed_slopes.append(float(per_episode_slope.mean()))
        cell_summaries[str(seed)] = {
            endpoint["name"]: {
                "fixed_gain": normal_summary(fixed), "rich_gain": normal_summary(rich)
            }
            for endpoint, fixed, rich in zip(endpoints, fixed_gains, rich_gains, strict=True)
        }
        for index in range(2):
            interaction = (
                rich_gains[index + 1] - fixed_gains[index + 1]
                - rich_gains[index] + fixed_gains[index]
            )
            adjacent_seed_interactions[index].append(float(interaction.mean()))

    slope = three_seed_cluster_summary(seed_slopes)
    adjacent = {
        f"{endpoints[i]['name']}_to_{endpoints[i + 1]['name']}":
            three_seed_cluster_summary(adjacent_seed_interactions[i])
        for i in range(2)
    }
    primary_pass = float(slope["ci95_low"]) > 0
    all_seed_slopes_positive = all(value > 0 for value in seed_slopes)
    adjacent_pass = all(float(row["ci95_low"]) > 0 for row in adjacent.values())
    payload = {
        "schema_version": 1, "protocol_status": config["status"],
        "audits": audits, "cells": cell_summaries,
        "primary_rich_gain_log_parameter_slope": slope,
        "adjacent_foundation_by_signal_interactions": adjacent,
        "primary_pass": primary_pass,
        "all_seed_slopes_positive": all_seed_slopes_positive,
        "all_adjacent_interaction_lcbs_positive": adjacent_pass,
        "scale_curve_confirmation_pass": primary_pass and all_seed_slopes_positive and adjacent_pass,
        "promotion_eligible": False,
        "remaining_blockers": config["promotion_blockers_after_scale_curve"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.result_root, args.output)


if __name__ == "__main__":
    main()
