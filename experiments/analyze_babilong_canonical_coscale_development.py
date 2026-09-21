#!/usr/bin/env python3
"""Analyze the frozen, non-promotional canonical co-scale development run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.analyze_babilong_qrag_direct_peer import cluster_summary
from experiments.run_natural_repeat import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def _load_endpoint(
    config: dict[str, Any], endpoint: str, root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_panels = config["panels"]
    expected_hashes = config["panel_sha256_by_name"]
    rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    for panel in expected_panels:
        path = root / f"{panel}.json"
        document = json.loads(path.read_text())
        if document["endpoint"]["name"] != endpoint:
            raise RuntimeError(f"endpoint mismatch in {path}")
        if document["data"]["sha256"] != expected_hashes[panel]:
            raise RuntimeError(f"panel hash mismatch in {path}")
        if document["data"]["rows"] != config["evaluation_samples"]:
            raise RuntimeError(f"row count mismatch in {path}")
        if document["condition"]["memory_representation"] != "canonical_event_facts":
            raise RuntimeError(f"non-canonical result in {path}")
        if document["environment"]["git_dirty"]:
            raise RuntimeError(f"dirty worktree result in {path}")
        panel_rows = document["predictions"]
        if len(panel_rows) != config["evaluation_samples"]:
            raise RuntimeError(f"prediction count mismatch in {path}")
        if {row["task"] for row in panel_rows} != set(config["tasks"]):
            raise RuntimeError(f"task coverage mismatch in {path}")
        for row in panel_rows:
            rows.append({"panel": panel, **row})
        provenance[panel] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "git_commit": document["environment"]["git_commit"],
            "config": document["config"],
            "model_files": document["model_files"],
        }
    expected = len(expected_panels) * config["evaluation_samples"]
    if len(rows) != expected or len({row["row_id"] for row in rows}) != expected:
        raise RuntimeError(f"row coverage mismatch for {endpoint}")
    return rows, provenance


def _panel_values(
    rows: list[dict[str, Any]], panels: list[str], tasks: set[str], field: str
) -> list[float]:
    values: list[float] = []
    for panel in panels:
        selected = [
            row for row in rows if row["panel"] == panel and row["task"] in tasks
        ]
        if not selected:
            raise RuntimeError(f"empty panel cell: {panel}")
        if field == "gain":
            value = np.mean(
                [row["ascent_score"] - row["foundation_score"] for row in selected]
            )
        elif field == "foundation":
            value = np.mean([row["foundation_score"] for row in selected])
        elif field == "ascent":
            value = np.mean([row["ascent_score"] for row in selected])
        else:
            raise ValueError(field)
        values.append(float(value))
    return values


def _scope(
    rows: list[dict[str, Any]], panels: list[str], tasks: set[str]
) -> dict[str, Any]:
    foundation = _panel_values(rows, panels, tasks, "foundation")
    ascent = _panel_values(rows, panels, tasks, "ascent")
    gain = _panel_values(rows, panels, tasks, "gain")
    rwe = [g / (1.0 - f) for g, f in zip(gain, foundation, strict=True)]
    return {
        "tasks": sorted(tasks),
        "foundation": cluster_summary(foundation),
        "ascent": cluster_summary(ascent),
        "gain": cluster_summary(gain),
        "remaining_error_elimination": cluster_summary(rwe),
    }


def analyze(
    config_path: Path, endpoint_roots: dict[str, Path]
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    if config["promotion"] is not False or not config["status"].endswith("no_promotion"):
        raise RuntimeError("development config must remain non-promotional")
    endpoints = [row["name"] for row in config["endpoints"]]
    if set(endpoint_roots) != set(endpoints):
        raise RuntimeError("endpoint roots do not match frozen endpoints")
    loaded: dict[str, list[dict[str, Any]]] = {}
    provenance: dict[str, Any] = {}
    for endpoint in endpoints:
        loaded[endpoint], provenance[endpoint] = _load_endpoint(
            config, endpoint, endpoint_roots[endpoint]
        )
    row_ids = [{row["row_id"] for row in loaded[endpoint]} for endpoint in endpoints]
    if any(ids != row_ids[0] for ids in row_ids[1:]):
        raise RuntimeError("endpoints do not cover identical examples")

    panels = config["panels"]
    primary_tasks = set(config["analysis_scope"]["primary_tasks"])
    endpoint_results = {
        endpoint: {
            "primary": _scope(loaded[endpoint], panels, primary_tasks),
            "by_task": {
                task: _scope(loaded[endpoint], panels, {task})
                for task in config["tasks"]
            },
        }
        for endpoint in endpoints
    }

    adjacent: list[dict[str, Any]] = []
    for lower, upper in zip(endpoints, endpoints[1:]):
        lo = endpoint_results[lower]["primary"]
        hi = endpoint_results[upper]["primary"]
        gain_increment = cluster_summary(
            [
                h - l
                for l, h in zip(
                    lo["gain"]["panel_values"],
                    hi["gain"]["panel_values"],
                    strict=True,
                )
            ]
        )
        rwe_increment = cluster_summary(
            [
                h - l
                for l, h in zip(
                    lo["remaining_error_elimination"]["panel_values"],
                    hi["remaining_error_elimination"]["panel_values"],
                    strict=True,
                )
            ]
        )
        adjacent.append(
            {
                "lower_endpoint": lower,
                "upper_endpoint": upper,
                "gain_increment": gain_increment,
                "gain_mean_positive": gain_increment["mean"] > 0.0,
                "gain_ci95_positive": gain_increment["ci95_low"] > 0.0,
                "remaining_error_elimination_increment": rwe_increment,
                "rwe_mean_positive": rwe_increment["mean"] > 0.0,
                "rwe_ci95_positive": rwe_increment["ci95_low"] > 0.0,
            }
        )

    return {
        "schema_version": 1,
        "experiment": config["experiment"],
        "status": config["status"],
        "promotion": False,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "cluster_unit": "panel",
        "confidence": 0.95,
        "endpoint_results": endpoint_results,
        "adjacent_scale_contrasts": adjacent,
        "required_development_pattern_pass": all(
            row["gain_mean_positive"] for row in adjacent
        ),
        "strong_ci_pattern_pass": all(row["gain_ci95_positive"] for row in adjacent),
        "provenance": provenance,
        "interpretation_guardrails": {
            "posthoc_development_only": True,
            "large_endpoint_was_observed_before_freeze": True,
            "eligible_as_independent_confirmation": False,
            "new_hashed_panels_required_for_promotion": True,
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
        "required_development_pattern_pass": result["required_development_pattern_pass"],
        "strong_ci_pattern_pass": result["strong_ci_pattern_pass"],
    }, indent=2))


if __name__ == "__main__":
    main()
