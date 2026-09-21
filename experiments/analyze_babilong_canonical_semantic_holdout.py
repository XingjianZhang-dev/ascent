#!/usr/bin/env python3
"""Analyze the frozen BABILong semantic-holdout canonical confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.analyze_babilong_canonical_coscale_16k_confirmation import (
    _load,
    _one_sided_positive_p,
    holm_adjust,
)
from experiments.analyze_babilong_canonical_coscale_development import _scope
from experiments.analyze_babilong_qrag_direct_peer import cluster_summary
from experiments.run_natural_repeat import sha256_file


def analyze(config_path: Path, roots: dict[str, Path]) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    if config["status"] != (
        "prospective_semantic_holdout_frozen_before_any_split_decoder_score"
    ):
        raise RuntimeError("unexpected semantic-holdout status")
    config_hash = sha256_file(config_path)
    manifest = Path(config["panel_manifest"]["path"])
    if sha256_file(manifest) != config["panel_manifest"]["sha256"]:
        raise RuntimeError("panel manifest hash mismatch")
    endpoints = [row["name"] for row in config["endpoints"]]
    if set(roots) != set(endpoints):
        raise RuntimeError("roots do not match endpoints")
    loaded: dict[str, list[dict[str, Any]]] = {}
    provenance: dict[str, Any] = {}
    for endpoint in endpoints:
        loaded[endpoint], provenance[endpoint] = _load(
            config,
            config_hash,
            endpoint,
            "canonical_coscale",
            roots[endpoint],
        )
    identities = [{row["row_id"] for row in loaded[e]} for e in endpoints]
    if any(ids != identities[0] for ids in identities[1:]):
        raise RuntimeError("endpoints do not cover identical rows")

    panels = config["panels"]
    tasks = set(config["tasks"])
    endpoint_results = {
        endpoint: {
            "primary": _scope(loaded[endpoint], panels, tasks),
            "by_task": {
                task: _scope(loaded[endpoint], panels, {task})
                for task in config["tasks"]
            },
        }
        for endpoint in endpoints
    }
    adjacent: list[dict[str, Any]] = []
    p_values: list[float] = []
    for lower, upper in zip(endpoints, endpoints[1:]):
        lo = endpoint_results[lower]["primary"]["gain"]["panel_values"]
        hi = endpoint_results[upper]["primary"]["gain"]["panel_values"]
        values = [h - l for l, h in zip(lo, hi, strict=True)]
        p_value = _one_sided_positive_p(values)
        p_values.append(p_value)
        adjacent.append(
            {
                "lower_endpoint": lower,
                "upper_endpoint": upper,
                "gain_increment": cluster_summary(values),
                "one_sided_p_unadjusted": p_value,
            }
        )
    for row, adjusted in zip(adjacent, holm_adjust(p_values), strict=True):
        row["one_sided_p_holm"] = adjusted
        row["passes_registered_contrast"] = (
            row["gain_increment"]["mean"] > 0.0 and adjusted < 0.05
        )
    return {
        "schema_version": 1,
        "experiment": config["experiment"],
        "status": config["status"],
        "config": {"path": str(config_path), "sha256": config_hash},
        "cluster_unit": "panel",
        "endpoint_results": endpoint_results,
        "adjacent_scale_contrasts": adjacent,
        "primary_gate_pass": all(
            row["passes_registered_contrast"] for row in adjacent
        ),
        "provenance": provenance,
        "interpretation_guardrails": {
            "semantic_overlap_with_official_8k_test": 0,
            "public_training_generation_split": True,
            "official_test_score": False,
            "project_trained_or_tuned_on_split": False,
            "foundation_pretraining_contamination_unknown": True,
            "eligible_for_qrag_fair_test_comparison": False,
            "all_panels_and_tasks_retained": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--small-root", type=Path, required=True)
    parser.add_argument("--mid-root", type=Path, required=True)
    parser.add_argument("--large-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.config,
        {
            "qwen2p5-0p5b-instruct": args.small_root,
            "qwen2p5-1p5b-instruct": args.mid_root,
            "qwen2p5-3b-instruct": args.large_root,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "endpoint_results": result["endpoint_results"],
        "adjacent_scale_contrasts": result["adjacent_scale_contrasts"],
        "primary_gate_pass": result["primary_gate_pass"],
    }, indent=2))


if __name__ == "__main__":
    main()
