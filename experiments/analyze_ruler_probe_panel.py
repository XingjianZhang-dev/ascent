#!/usr/bin/env python3
"""Seed-clustered official-score analysis for ASCENT's learned evidence path."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


T_CRITICAL_975_DF2 = 4.302652729696142
ENDPOINTS = (
    ("qwen2p5-0p5b", 494032768),
    ("qwen2p5-1p5b", 1543714304),
    ("qwen2p5-3b", 3085938688),
)


def endpoints_from_config(config: dict[str, Any]) -> tuple[tuple[str, int], ...]:
    """Return result-file stems and parameter counts in registered scale order."""
    endpoints = tuple(
        (
            str(endpoint["name"]).removesuffix("-instruct"),
            int(endpoint["model_parameters"]),
        )
        for endpoint in config["endpoints"]
    )
    if len(endpoints) != 3:
        raise ValueError("scale confirmation requires exactly three endpoints")
    return endpoints


def state_budget_audit(config: dict[str, Any]) -> dict[str, Any]:
    """Audit the persistent latent payload, including model hidden width."""
    rows = []
    for endpoint in config["endpoints"]:
        tokens = int(endpoint["scale_replay_tokens"])
        width = int(endpoint["hidden_size"])
        parameters = int(endpoint["model_parameters"])
        elements = tokens * width
        rows.append(
            {
                "endpoint": endpoint["name"],
                "state_tokens": tokens,
                "hidden_size": width,
                "persistent_state_elements": elements,
                "model_parameters": parameters,
                "state_elements_per_model_parameter": elements / parameters,
            }
        )
    ratios = [row["state_elements_per_model_parameter"] for row in rows]
    decreasing = all(left > right for left, right in zip(ratios, ratios[1:]))
    rule = config.get("state_budget_rule", {})
    alpha = float(rule.get("alpha", float("nan")))
    base_tokens = int(rule.get("base_state_tokens", -1))
    base_width = int(rule.get("base_hidden_size", -1))
    expected_tokens = []
    if 0.0 < alpha <= 1.0 and base_tokens > 0 and base_width > 0:
        expected_tokens = [
            int(np.floor(base_tokens * (row["hidden_size"] / base_width) ** alpha + 0.5))
            for row in rows
        ]
    observed_tokens = [row["state_tokens"] for row in rows]
    width_law_matches = bool(expected_tokens == observed_tokens)
    passed = decreasing and width_law_matches
    return {
        "rows": rows,
        "gate": "passed" if passed else "failed",
        "relative_state_ratio_strictly_decreases": decreasing,
        "proposal_width_law_matches": width_law_matches,
        "proposal_width_law": {
            "base_state_tokens": base_tokens,
            "base_hidden_size": base_width,
            "alpha": alpha,
            "expected_state_tokens": expected_tokens,
            "observed_state_tokens": observed_tokens,
        },
        "accounting_identity": "persistent_state_elements = state_tokens * hidden_size",
    }


def cluster_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size != 3:
        raise ValueError("confirmation requires exactly three untouched seeds")
    mean = float(array.mean())
    standard_error = float(array.std(ddof=1) / np.sqrt(array.size))
    half_width = T_CRITICAL_975_DF2 * standard_error
    return {
        "seed_values": array.tolist(),
        "mean": mean,
        "standard_error": standard_error,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "degrees_of_freedom": 2,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def confirmation_audit(
    payloads: dict[str, dict[str, dict[str, Any]]],
    seeds: list[int],
    config_path: Path,
    manifest_path: Path,
    endpoints: tuple[tuple[str, int], ...] = ENDPOINTS,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    config_sha256 = sha256_file(config_path)
    expected_tasks = config["task_sha256_by_seed"]
    expected_endpoints = {
        endpoint["name"].removesuffix("-instruct"): endpoint
        for endpoint in config["endpoints"]
    }
    cells = []
    probe_signatures: dict[str, set[str]] = {endpoint: set() for endpoint, _ in endpoints}
    commits = set()
    for seed in seeds:
        for endpoint, _ in endpoints:
            payload = payloads[str(seed)][endpoint]
            expected = expected_endpoints[endpoint]
            predictions = payload["predictions"]
            wins = sum(
                row["ascent_score"] > row["foundation_score"]
                for row in predictions
            )
            regressions = sum(
                row["ascent_score"] < row["foundation_score"]
                for row in predictions
            )
            runtime = payload["runtime"]
            commits.add(runtime["git_commit"])
            probe_signatures[endpoint].add(
                json.dumps(payload["probe"], sort_keys=True, separators=(",", ":"))
            )
            expected_artifacts = expected.get(
                "model_artifacts", [expected.get("model_artifact")]
            )
            cell_gates = {
                "registered_seed": payload["data_seed"] == seed,
                "registered_task_sha256": payload["task_sha256"]
                == expected_tasks[str(seed)]
                == manifest["seeds"][str(seed)]["validation_sha256"],
                "registered_config_sha256": payload["config_sha256"]
                == config_sha256,
                "clean_runtime_commit": runtime["git_dirty"] is False,
                "registered_foundation_decode_cache": runtime.get(
                    "foundation_decode_cache", False
                )
                is bool(config.get("foundation_decode_cache", False)),
                "registered_endpoint_identity": all(
                    payload["endpoint"][key] == expected[key]
                    for key in ("repo_id", "revision", "model_parameters", "scale_replay_tokens")
                ),
                "registered_model_artifacts": payload["model_files"]
                == expected_artifacts,
                "registered_test_count": len(predictions)
                == config["test_samples_per_task"],
                "no_foundation_only_regression": regressions == 0,
            }
            cells.append(
                {
                    "seed": seed,
                    "endpoint": endpoint,
                    "runtime_commit": runtime["git_commit"],
                    "wins": wins,
                    "regressions": regressions,
                    "prediction_count": len(predictions),
                    "gates": cell_gates,
                }
            )
    gates = {
        "every_cell_matches_registration": all(
            all(cell["gates"].values()) for cell in cells
        ),
        "single_clean_runtime_commit": len(commits) == 1,
        "probe_fit_identical_across_evaluation_seeds": all(
            len(signatures) == 1 for signatures in probe_signatures.values()
        ),
    }
    state_audit = None
    if int(config.get("query_context_tokens", 0)) >= 4096:
        try:
            state_audit = state_budget_audit(config)
        except (KeyError, TypeError, ValueError):
            state_audit = {
                "gate": "failed",
                "relative_state_ratio_strictly_decreases": False,
                "error": "full-context confirmation requires hidden_size, scale_replay_tokens, and model_parameters for every endpoint",
            }
        gates["relative_state_ratio_strictly_decreases"] = state_audit[
            "relative_state_ratio_strictly_decreases"
        ]
        gates["proposal_width_law_matches"] = state_audit.get(
            "proposal_width_law_matches", False
        )
    return {
        "config_sha256": config_sha256,
        "manifest_sha256": sha256_file(manifest_path),
        "runtime_commits": sorted(commits),
        "cells": cells,
        "state_budget_audit": state_audit,
        "gates": gates,
        "status": "passed" if all(gates.values()) else "failed",
    }


def analyze(
    results_root: Path,
    seeds: list[int],
    config_path: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    if (config_path is None) != (manifest_path is None):
        raise ValueError("config_path and manifest_path must be provided together")
    config = json.loads(config_path.read_text()) if config_path is not None else None
    endpoints = endpoints_from_config(config) if config is not None else ENDPOINTS
    seed_gains: dict[str, list[float]] = {}
    endpoint_payloads: dict[str, dict[str, Any]] = {}
    raw_payloads: dict[str, dict[str, dict[str, Any]]] = {}
    for seed in seeds:
        gains = []
        endpoint_payloads[str(seed)] = {}
        raw_payloads[str(seed)] = {}
        for endpoint, _ in endpoints:
            payload = json.loads(
                (results_root / str(seed) / f"{endpoint}.json").read_text()
            )
            raw_payloads[str(seed)][endpoint] = payload
            gain = float(payload["paired_score_gain"]["mean_nats"])
            gains.append(gain)
            endpoint_payloads[str(seed)][endpoint] = {
                "foundation_score": payload["foundation_score"]["mean"],
                "ascent_score": payload["ascent_scale_score"]["mean"],
                "gain": gain,
                "probe_digit_accuracy": payload["probe"][
                    "validation_digit_token_accuracy"
                ],
                "task_sha256": payload["task_sha256"],
            }
        seed_gains[str(seed)] = gains
    endpoint_clusters = {
        endpoint: cluster_summary([
            seed_gains[str(seed)][index] for seed in seeds
        ])
        for index, (endpoint, _) in enumerate(endpoints)
    }
    low_to_mid = [seed_gains[str(seed)][1] - seed_gains[str(seed)][0] for seed in seeds]
    mid_to_high = [seed_gains[str(seed)][2] - seed_gains[str(seed)][1] for seed in seeds]
    log_parameters = np.log([parameters for _, parameters in endpoints])
    slopes = [
        float(np.polyfit(log_parameters, seed_gains[str(seed)], 1)[0])
        for seed in seeds
    ]
    adjacent = {
        f"{endpoints[0][0]}_to_{endpoints[1][0]}": cluster_summary(low_to_mid),
        f"{endpoints[1][0]}_to_{endpoints[2][0]}": cluster_summary(mid_to_high),
    }
    slope = cluster_summary(slopes)
    gates = {
        "every_seed_endpoint_gain_positive": all(
            gain > 0 for gains in seed_gains.values() for gain in gains
        ),
        "every_seed_adjacent_difference_positive": all(
            value > 0 for value in low_to_mid + mid_to_high
        ),
        "all_adjacent_cluster_lcbs_positive": all(
            summary["ci95_low"] > 0 for summary in adjacent.values()
        ),
        "slope_cluster_lcb_positive": slope["ci95_low"] > 0,
    }
    audit = None
    metric_boundary = (
        "Official NVIDIA RULER string_match_all under a short-query plus "
        "external-state protocol; not a full-context leaderboard score."
    )
    if config_path is not None and manifest_path is not None:
        audit = confirmation_audit(
            raw_payloads, seeds, config_path, manifest_path, endpoints
        )
        gates["confirmation_audit_passed"] = audit["status"] == "passed"
        context_tokens = int(config.get("query_context_tokens", 0))
        if context_tokens >= 4096:
            metric_boundary = (
                "Official NVIDIA RULER string_match_all with the complete "
                f"registered {context_tokens}-token current context visible to "
                "both arms. Any task grammar or evidence decoder remains part "
                "of the reported method and is not a stock leaderboard decoder."
            )
    return {
        "schema_version": 1,
        "status": "passed" if all(gates.values()) else "failed",
        "seeds": seeds,
        "seed_weighting": "Each independently generated data seed receives equal weight.",
        "endpoint_results": endpoint_payloads,
        "endpoint_gain_clusters": endpoint_clusters,
        "adjacent_gain_difference_clusters": adjacent,
        "slope_cluster": slope,
        "gates": gates,
        "confirmation_audit": audit,
        "metric_boundary": metric_boundary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    payload = analyze(
        args.results_root, args.seeds, args.config, args.manifest
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
