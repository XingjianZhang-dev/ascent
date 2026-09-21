#!/usr/bin/env python3
"""Analyze the frozen ten-panel Qwen2.5 BABILong 3x3 replication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ENDPOINTS = (
    ("qwen2p5-0p5b-instruct", "node2", 494_032_768),
    ("qwen2p5-1p5b-instruct", "node2", 1_543_714_304),
    ("qwen2p5-3b-instruct", "node1", 3_085_938_688),
)
STATE_SLOTS = (1, 2, 3)
T95_DF9 = 2.2621571627409915
T_ONE_SIDED_BONFERRONI_DF9_M6 = 2.9333240883739897


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_ci(values: list[float], critical: float = T95_DF9) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size != 10:
        raise ValueError("Qwen replication requires exactly ten panel values")
    standard_error = float(array.std(ddof=1) / np.sqrt(array.size))
    mean = float(array.mean())
    return {
        "values": array.tolist(),
        "mean": mean,
        "standard_error": standard_error,
        "ci95_low": mean - critical * standard_error,
        "ci95_high": mean + critical * standard_error,
    }


def paired(left: list[float], right: list[float]) -> list[float]:
    return (
        np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    ).tolist()


def values(
    results: dict[tuple[str, int], list[dict[str, Any]]],
    endpoint: str,
    slots: int,
    field: str,
) -> list[float]:
    return [float(row[field]["mean"]) for row in results[(endpoint, slots)]]


def expected_model_files(endpoint: dict[str, Any]) -> list[dict[str, Any]]:
    singular = endpoint.get("model_artifact")
    plural = endpoint.get("model_artifacts")
    if (singular is None) == (plural is None):
        raise RuntimeError("endpoint must freeze exactly one weight-manifest field")
    return [singular] if singular is not None else list(plural)


def referenced_document_checks(config: dict[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for key in ("factorial_design", "source_panel_config"):
        spec = config[key]
        path = ROOT / spec["path"]
        checks[f"{key}_exists"] = path.is_file()
        checks[f"{key}_sha256"] = path.is_file() and sha256_file(path) == spec["sha256"]
    return checks


def collect(
    root: Path, config_path: Path, run_commit: str
) -> tuple[
    dict[tuple[str, int], list[dict[str, Any]]],
    list[str],
    dict[str, Any],
]:
    config = load(config_path)
    config_sha256 = sha256_file(config_path)
    panels = list(config["panels"])
    if len(panels) != 10:
        raise RuntimeError("expected exactly ten frozen panels")
    endpoints = {endpoint["name"]: endpoint for endpoint in config["endpoints"]}
    results: dict[tuple[str, int], list[dict[str, Any]]] = {}
    checks: list[bool] = []
    for endpoint, node, _ in ENDPOINTS:
        for slots in STATE_SLOTS:
            rows = [
                load(
                    root
                    / node
                    / f"generative_slots_{slots}_{panel}_{endpoint}.json"
                )
                for panel in panels
            ]
            results[(endpoint, slots)] = rows
            for panel, row in zip(panels, rows, strict=True):
                checks.extend(
                    (
                        row["endpoint"] == endpoints[endpoint],
                        row["condition"]["name"] == f"generative_slots_{slots}",
                        row["condition"]["fact_slots"] == slots,
                        row["condition"]["readout_path"] == "foundation_generation",
                        row["condition"]["memory_source"] == "relevant",
                        row["environment"]["git_commit"] == run_commit,
                        not row["environment"]["git_dirty"],
                        row["config"]["sha256"] == config_sha256,
                        row["data"]["sha256"]
                        == config["panel_sha256_by_name"][panel],
                        row["model_files"] == expected_model_files(endpoints[endpoint]),
                        row["status"] == config["status"],
                        row["parser_accuracy"] == 1.0,
                        len(row["predictions"]) == 120,
                        row["data"]["path"].endswith(f"{panel}.jsonl"),
                    )
                )

    def foundation_projection(row: dict[str, Any]) -> list[tuple[Any, ...]]:
        return [
            (
                prediction["row_id"],
                prediction["task"],
                prediction["target"],
                prediction["foundation_output"],
                prediction["foundation_answer"],
                prediction["foundation_score"],
                prediction["foundation_prompt_tokens"],
            )
            for prediction in row["predictions"]
        ]

    foundation_equal = True
    for endpoint, _, _ in ENDPOINTS:
        for panel_index in range(len(panels)):
            rows = [results[(endpoint, slots)][panel_index] for slots in STATE_SLOTS]
            foundation_equal &= all(
                foundation_projection(row) == foundation_projection(rows[0])
                for row in rows[1:]
            )

    panel_alignment = True
    for panel_index in range(len(panels)):
        identities = [
            [
                (prediction["row_id"], prediction["task"], prediction["target"])
                for prediction in rows[panel_index]["predictions"]
            ]
            for rows in results.values()
        ]
        panel_alignment &= all(
            identity == identities[0] for identity in identities[1:]
        )

    document_checks = referenced_document_checks(config)
    provenance = {
        "config_sha256": config_sha256,
        "run_commit": run_commit,
        "all_cell_checks_pass": all(checks),
        "foundation_equal_across_state_sizes": foundation_equal,
        "all_cells_row_task_target_aligned": panel_alignment,
        "referenced_document_checks": document_checks,
    }
    provenance["provenance_and_alignment_pass"] = (
        all(checks)
        and foundation_equal
        and panel_alignment
        and all(document_checks.values())
    )
    return results, panels, provenance


def registered_estimand_values(
    results: dict[tuple[str, int], list[dict[str, Any]]]
) -> dict[str, list[float]]:
    small, middle, large = (endpoint for endpoint, _, _ in ENDPOINTS)
    first_interaction = paired(
        paired(
            values(results, middle, 2, "ascent"),
            values(results, middle, 1, "ascent"),
        ),
        paired(
            values(results, small, 2, "ascent"),
            values(results, small, 1, "ascent"),
        ),
    )
    second_interaction = paired(
        paired(
            values(results, large, 3, "ascent"),
            values(results, large, 2, "ascent"),
        ),
        paired(
            values(results, middle, 3, "ascent"),
            values(results, middle, 2, "ascent"),
        ),
    )
    first_diagonal = paired(
        values(results, middle, 2, "gain"),
        values(results, small, 1, "gain"),
    )
    second_diagonal = paired(
        values(results, large, 3, "gain"),
        values(results, middle, 2, "gain"),
    )
    first_memory_only = paired(
        values(results, small, 2, "gain"),
        values(results, small, 1, "gain"),
    )
    second_memory_only = paired(
        values(results, middle, 3, "gain"),
        values(results, middle, 2, "gain"),
    )
    return {
        "first_registered_interaction": first_interaction,
        "second_registered_interaction": second_interaction,
        "first_diagonal_gain_increment": first_diagonal,
        "second_diagonal_gain_increment": second_diagonal,
        "first_coscale_excess": paired(first_diagonal, first_memory_only),
        "second_coscale_excess": paired(second_diagonal, second_memory_only),
    }


def analyze(root: Path, config_path: Path, run_commit: str) -> dict[str, Any]:
    results, panels, provenance = collect(root, config_path, run_commit)
    cells = {
        f"{endpoint}__slots_{slots}": {
            "model_parameters": parameters,
            "state_slots": slots,
            "foundation_accuracy": mean_ci(
                values(results, endpoint, slots, "foundation")
            ),
            "ascent_accuracy": mean_ci(values(results, endpoint, slots, "ascent")),
            "absolute_gain": mean_ci(values(results, endpoint, slots, "gain")),
        }
        for endpoint, _, parameters in ENDPOINTS
        for slots in STATE_SLOTS
    }
    raw_estimands = registered_estimand_values(results)
    estimands = {key: mean_ci(value) for key, value in raw_estimands.items()}
    individual_positive = {
        key: value["ci95_low"] > 0.0 for key, value in estimands.items()
    }
    familywise = {
        key: mean_ci(value, T_ONE_SIDED_BONFERRONI_DF9_M6)
        for key, value in raw_estimands.items()
    }
    familywise_positive = {
        key: value["ci95_low"] > 0.0 for key, value in familywise.items()
    }
    primary_pass = provenance["provenance_and_alignment_pass"] and all(
        individual_positive.values()
    )
    familywise_pass = all(familywise_positive.values())
    return {
        "schema_version": 1,
        "root": str(root),
        "panels": panels,
        "panel_count": len(panels),
        "interval": "two-sided Student-t 95% interval over panels (df=9)",
        "familywise_interval": "Bonferroni simultaneous one-sided 95% familywise lower bound over six directional estimands (df=9)",
        "provenance": provenance,
        "cells": cells,
        "registered_estimands": estimands,
        "individual_lcbs_positive": individual_positive,
        "primary_gate_pass": primary_pass,
        "familywise_directional_estimands": familywise,
        "familywise_lcbs_positive": familywise_positive,
        "familywise_gate_pass": familywise_pass,
        "same_task_cross_architecture_replication_pass": primary_pass
        and familywise_pass,
        "interpretation": "Prospectively frozen Qwen2.5 same-task replication; all prior SmolLM2 outcomes remain unchanged.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.root, args.config, args.run_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
