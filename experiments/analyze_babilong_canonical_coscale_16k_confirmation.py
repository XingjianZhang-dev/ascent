#!/usr/bin/env python3
"""Analyze the prospectively frozen 16K canonical co-scale confirmation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from experiments.analyze_babilong_canonical_coscale_development import (
    _panel_values,
    _scope,
)
from experiments.analyze_babilong_qrag_direct_peer import cluster_summary
from experiments.run_natural_repeat import sha256_file


def _load(
    config: dict[str, Any],
    config_hash: str,
    endpoint: str,
    condition: str,
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_representation = config["conditions"][condition]["memory_representation"]
    rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    for panel in config["panels"]:
        path = root / f"{panel}.json"
        document = json.loads(path.read_text())
        if document["status"] != config["status"]:
            raise RuntimeError(f"status mismatch: {path}")
        if document["endpoint"]["name"] != endpoint:
            raise RuntimeError(f"endpoint mismatch: {path}")
        if document["condition"]["memory_representation"] != expected_representation:
            raise RuntimeError(f"condition mismatch: {path}")
        if document["config"]["sha256"] != config_hash:
            raise RuntimeError(f"config hash mismatch: {path}")
        if document["data"]["sha256"] != config["panel_sha256_by_name"][panel]:
            raise RuntimeError(f"panel hash mismatch: {path}")
        if document["parser_accuracy"] != 1.0:
            raise RuntimeError(f"parser mismatch: {path}")
        if document["environment"]["git_dirty"]:
            raise RuntimeError(f"dirty worktree: {path}")
        if len(document["predictions"]) != config["evaluation_samples"]:
            raise RuntimeError(f"prediction count mismatch: {path}")
        rows.extend({"panel": panel, **row} for row in document["predictions"])
        provenance[panel] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "git_commit": document["environment"]["git_commit"],
            "model_files": document["model_files"],
            "config": document["config"],
        }
    expected = len(config["panels"]) * config["evaluation_samples"]
    if len(rows) != expected or len({row["row_id"] for row in rows}) != expected:
        raise RuntimeError(f"row coverage mismatch: {endpoint}/{condition}")
    return rows, provenance


def _one_sided_positive_p(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    standard_error = float(array.std(ddof=1) / math.sqrt(array.size))
    mean = float(array.mean())
    if standard_error == 0.0:
        return 0.0 if mean > 0.0 else 1.0
    return float(student_t.sf(mean / standard_error, array.size - 1))


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _paired_summary(
    left: list[float], right: list[float]
) -> dict[str, Any]:
    if len(left) != len(right):
        raise RuntimeError("paired vectors differ in length")
    values = [a - b for a, b in zip(left, right, strict=True)]
    return cluster_summary(values)


def analyze(
    config_path: Path,
    canonical_roots: dict[str, Path],
    raw_3b_root: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    config_hash = sha256_file(config_path)
    if config["status"] != "prospective_16k_confirmation_frozen_before_any_16k_decoder_score":
        raise RuntimeError("unexpected confirmatory status")
    manifest = Path(config["panel_manifest"]["path"])
    if sha256_file(manifest) != config["panel_manifest"]["sha256"]:
        raise RuntimeError("panel manifest hash mismatch")
    endpoints = [row["name"] for row in config["endpoints"]]
    if set(canonical_roots) != set(endpoints):
        raise RuntimeError("canonical roots do not match endpoints")

    canonical: dict[str, list[dict[str, Any]]] = {}
    provenance: dict[str, Any] = {"canonical_coscale": {}}
    for endpoint in endpoints:
        canonical[endpoint], endpoint_provenance = _load(
            config,
            config_hash,
            endpoint,
            "canonical_coscale",
            canonical_roots[endpoint],
        )
        provenance["canonical_coscale"][endpoint] = endpoint_provenance
    raw, provenance["raw_slots_3_control"] = _load(
        config, config_hash, endpoints[-1], "raw_slots_3_control", raw_3b_root
    )
    identity_sets = [
        {row["row_id"] for row in canonical[endpoint]} for endpoint in endpoints
    ] + [{row["row_id"] for row in raw}]
    if any(ids != identity_sets[0] for ids in identity_sets[1:]):
        raise RuntimeError("result cells do not cover identical examples")

    panels = config["panels"]
    tasks = set(config["tasks"])
    endpoint_results = {
        endpoint: {
            "primary": _scope(canonical[endpoint], panels, tasks),
            "by_task": {
                task: _scope(canonical[endpoint], panels, {task})
                for task in config["tasks"]
            },
        }
        for endpoint in endpoints
    }

    adjacent: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    for lower, upper in zip(endpoints, endpoints[1:]):
        lo = endpoint_results[lower]["primary"]["gain"]["panel_values"]
        hi = endpoint_results[upper]["primary"]["gain"]["panel_values"]
        values = [h - l for l, h in zip(lo, hi, strict=True)]
        summary = cluster_summary(values)
        p_value = _one_sided_positive_p(values)
        raw_p_values.append(p_value)
        adjacent.append(
            {
                "lower_endpoint": lower,
                "upper_endpoint": upper,
                "gain_increment": summary,
                "one_sided_p_unadjusted": p_value,
            }
        )
    adjusted = holm_adjust(raw_p_values)
    for row, adjusted_p in zip(adjacent, adjusted, strict=True):
        row["one_sided_p_holm"] = adjusted_p
        row["passes_registered_contrast"] = (
            row["gain_increment"]["mean"] > 0.0 and adjusted_p < 0.05
        )

    large = endpoints[-1]
    canonical_rows = {row["row_id"]: row for row in canonical[large]}
    raw_rows = {row["row_id"]: row for row in raw}
    foundation_exact = all(
        canonical_rows[row_id]["foundation_output"]
        == raw_rows[row_id]["foundation_output"]
        and canonical_rows[row_id]["foundation_score"]
        == raw_rows[row_id]["foundation_score"]
        for row_id in canonical_rows
    )

    def representation_contrast(task: str | None) -> dict[str, Any]:
        selected_tasks = tasks if task is None else {task}
        canonical_accuracy = _panel_values(
            canonical[large], panels, selected_tasks, "ascent"
        )
        raw_accuracy = _panel_values(raw, panels, selected_tasks, "ascent")
        return {
            "canonical_accuracy": cluster_summary(canonical_accuracy),
            "raw_accuracy": cluster_summary(raw_accuracy),
            "canonical_minus_raw": _paired_summary(canonical_accuracy, raw_accuracy),
        }

    raw_control = {
        "overall": representation_contrast(None),
        "by_task": {
            task: representation_contrast(task) for task in config["tasks"]
        },
        "foundation_predictions_exact_match": foundation_exact,
    }

    return {
        "schema_version": 1,
        "experiment": config["experiment"],
        "status": config["status"],
        "config": {"path": str(config_path), "sha256": config_hash},
        "cluster_unit": "panel",
        "confidence": 0.95,
        "endpoint_results": endpoint_results,
        "adjacent_scale_contrasts": adjacent,
        "primary_gate_pass": all(
            row["passes_registered_contrast"] for row in adjacent
        ),
        "raw_3b_control": raw_control,
        "provenance": provenance,
        "interpretation_guardrails": {
            "untouched_16k_decoder_scores": True,
            "semantic_episode_independence_from_8k": False,
            "independent_context_length_and_distractor_instantiation": True,
            "all_panels_and_tasks_retained": True,
            "raw_control_is_secondary": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--small-root", type=Path, required=True)
    parser.add_argument("--mid-root", type=Path, required=True)
    parser.add_argument("--large-root", type=Path, required=True)
    parser.add_argument("--raw-large-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.config,
        {
            "qwen2p5-0p5b-instruct": args.small_root,
            "qwen2p5-1p5b-instruct": args.mid_root,
            "qwen2p5-3b-instruct": args.large_root,
        },
        args.raw_large_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "endpoint_results": result["endpoint_results"],
        "adjacent_scale_contrasts": result["adjacent_scale_contrasts"],
        "primary_gate_pass": result["primary_gate_pass"],
        "raw_3b_control": result["raw_3b_control"],
    }, indent=2))


if __name__ == "__main__":
    main()
