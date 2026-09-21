#!/usr/bin/env python3
"""Analyze the prospectively frozen matched Q-RAG direct-peer comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import t as student_t

from experiments.run_natural_repeat import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def cluster_summary(values: list[float], confidence: float = 0.95) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        raise ValueError("cluster inference requires at least two panels")
    mean = float(array.mean())
    standard_error = float(array.std(ddof=1) / math.sqrt(array.size))
    critical = float(student_t.ppf((1.0 + confidence) / 2.0, array.size - 1))
    return {
        "clusters": int(array.size),
        "degrees_of_freedom": int(array.size - 1),
        "mean": mean,
        "standard_error": standard_error,
        "ci95_low": mean - critical * standard_error,
        "ci95_high": mean + critical * standard_error,
        "panel_values": [float(value) for value in array],
    }


def panel_means(
    rows: list[dict[str, Any]],
    panels: list[str],
    value: Callable[[dict[str, Any]], float],
    task: str | None = None,
) -> list[float]:
    means: list[float] = []
    for panel in panels:
        selected = [
            row
            for row in rows
            if row["panel"] == panel and (task is None or row["task"] == task)
        ]
        if not selected:
            raise RuntimeError(f"empty panel/task cell: {panel}/{task}")
        means.append(float(np.mean([value(row) for row in selected])))
    return means


def paired_effect(
    rows: list[dict[str, Any]],
    panels: list[str],
    left: str,
    right: str,
    task: str | None = None,
) -> dict[str, Any]:
    return cluster_summary(
        panel_means(rows, panels, lambda row: row[left] - row[right], task)
    )


def load_reader_results(
    manifest_path: Path, reader_root: Path
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = sha256_file(manifest_path)
    endpoints = manifest["evaluation"]["reader_endpoints"]
    panels = manifest["evaluation"]["panels"]
    tasks = manifest["evaluation"]["tasks"]
    expected_rows = int(manifest["evaluation"]["retrieval_rows"])
    loaded: dict[str, list[dict[str, Any]]] = {}
    provenance: dict[str, Any] = {}
    for endpoint in endpoints:
        path = reader_root / f"{endpoint}.json"
        document = json.loads(path.read_text())
        if document["status"] != manifest["status"]:
            raise RuntimeError(f"reader status mismatch: {endpoint}")
        if document["manifest"]["sha256"] != manifest_hash:
            raise RuntimeError(f"reader manifest mismatch: {endpoint}")
        if document["endpoint"]["name"] != endpoint:
            raise RuntimeError(f"reader endpoint mismatch: {endpoint}")
        if document["environment"]["git_dirty"]:
            raise RuntimeError(f"dirty reader worktree: {endpoint}")
        rows = document["predictions"]
        if len(rows) != expected_rows or len({row["row_id"] for row in rows}) != expected_rows:
            raise RuntimeError(f"reader row coverage mismatch: {endpoint}")
        if {row["panel"] for row in rows} != set(panels):
            raise RuntimeError(f"reader panel coverage mismatch: {endpoint}")
        if {row["task"] for row in rows} != set(tasks):
            raise RuntimeError(f"reader task coverage mismatch: {endpoint}")
        loaded[endpoint] = rows
        provenance[endpoint] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "git_commit": document["environment"]["git_commit"],
            "model_files": document["model_files"],
            "systems": document["systems"],
        }
    return manifest, loaded, provenance


def analyze(manifest_path: Path, reader_root: Path) -> dict[str, Any]:
    manifest, loaded, provenance = load_reader_results(manifest_path, reader_root)
    panels = manifest["evaluation"]["panels"]
    tasks = manifest["evaluation"]["tasks"]
    margin = float(manifest["analysis"]["peer_noninferiority_margin"])
    endpoints = manifest["evaluation"]["reader_endpoints"]

    endpoint_results: dict[str, Any] = {}
    for endpoint in endpoints:
        rows = loaded[endpoint]

        def scope(task: str | None) -> dict[str, Any]:
            selected = rows if task is None else [row for row in rows if row["task"] == task]
            means = {
                field: float(np.mean([row[field] for row in selected]))
                for field in ("foundation_score", "ascent_score", "qrag_score")
            }
            qrag_foundation = paired_effect(
                rows, panels, "qrag_score", "foundation_score", task
            )
            ascent_qrag = paired_effect(rows, panels, "ascent_score", "qrag_score", task)
            ascent_foundation = paired_effect(
                rows, panels, "ascent_score", "foundation_score", task
            )
            return {
                "rows": len(selected),
                "accuracy": means,
                "qrag_minus_foundation": qrag_foundation,
                "ascent_minus_foundation": ascent_foundation,
                "ascent_minus_qrag": ascent_qrag,
                "ascent_superior_to_qrag": ascent_qrag["ci95_low"] > 0.0,
                "ascent_noninferior_to_qrag_at_5_points": (
                    ascent_qrag["ci95_low"] > -margin
                ),
            }

        endpoint_results[endpoint] = {
            "overall": scope(None),
            "by_task": {task: scope(task) for task in tasks},
        }

    adjacent: list[dict[str, Any]] = []
    for lower, upper in zip(endpoints, endpoints[1:]):
        lower_rows = {row["row_id"]: row for row in loaded[lower]}
        upper_rows = {row["row_id"]: row for row in loaded[upper]}
        if lower_rows.keys() != upper_rows.keys():
            raise RuntimeError("reader endpoints do not have identical row IDs")

        def method_increment(method: str, task: str | None) -> dict[str, Any]:
            values: list[float] = []
            for panel in panels:
                row_ids = [
                    row_id
                    for row_id, row in lower_rows.items()
                    if row["panel"] == panel and (task is None or row["task"] == task)
                ]
                per_row = []
                for row_id in row_ids:
                    lo = lower_rows[row_id]
                    hi = upper_rows[row_id]
                    lo_gain = lo[f"{method}_score"] - lo["foundation_score"]
                    hi_gain = hi[f"{method}_score"] - hi["foundation_score"]
                    per_row.append(hi_gain - lo_gain)
                values.append(float(np.mean(per_row)))
            return cluster_summary(values)

        def contrast(task: str | None) -> dict[str, Any]:
            ascent = method_increment("ascent", task)
            qrag = method_increment("qrag", task)
            difference = cluster_summary(
                [
                    a - q
                    for a, q in zip(
                        ascent["panel_values"], qrag["panel_values"], strict=True
                    )
                ]
            )
            return {
                "ascent_gain_increment": ascent,
                "qrag_gain_increment": qrag,
                "ascent_minus_qrag_increment": difference,
            }

        adjacent.append(
            {
                "lower_endpoint": lower,
                "upper_endpoint": upper,
                "overall": contrast(None),
                "by_task": {task: contrast(task) for task in tasks},
            }
        )

    return {
        "schema_version": 1,
        "experiment": manifest["experiment"],
        "status": manifest["status"],
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "cluster_unit": "panel",
        "confidence": 0.95,
        "noninferiority_margin": margin,
        "endpoint_results": endpoint_results,
        "adjacent_reader_scale_contrasts": adjacent,
        "provenance": provenance,
        "interpretation_guardrails": {
            "qrag_is_task_trained": True,
            "ascent_is_not_trained_on_qa2_or_qa3": True,
            "matched_reader_and_rows": True,
            "failure_retained": True,
            "does_not_redefine_ascent_primary_claim": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reader-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.manifest, args.reader_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["endpoint_results"], indent=2))


if __name__ == "__main__":
    main()
