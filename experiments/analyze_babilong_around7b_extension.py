#!/usr/bin/env python3
"""Analyze the prospectively frozen around-7B BABILong extension."""

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
from experiments.analyze_babilong_canonical_coscale_16k_confirmation import (
    _load as _load_qwen_source,
    holm_adjust,
)
from experiments.analyze_babilong_qrag_direct_peer import cluster_summary
from experiments.freeze_babilong_around7b_extension import EXECUTABLE_STATUS
from experiments.run_natural_repeat import sha256_file


def one_sided_positive_p(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        raise RuntimeError("a panel-cluster test requires at least two panels")
    standard_error = float(array.std(ddof=1) / math.sqrt(array.size))
    mean = float(array.mean())
    if standard_error == 0.0:
        return 0.0 if mean > 0.0 else 1.0
    return float(student_t.sf(mean / standard_error, array.size - 1))


def paired_summary(left: list[float], right: list[float]) -> dict[str, Any]:
    if len(left) != len(right):
        raise RuntimeError("paired vectors differ in length")
    return cluster_summary([a - b for a, b in zip(left, right, strict=True)])


def _load(
    config: dict[str, Any],
    config_hash: str,
    endpoint: dict[str, Any],
    condition_name: str,
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    condition = config["conditions"][condition_name]
    expected_slots = condition["state_fact_slots_by_endpoint"].get(endpoint["name"])
    if expected_slots is None:
        raise RuntimeError(f"condition {condition_name} is not registered for {endpoint['name']}")
    rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    for panel in config["panels"]:
        path = root / f"{panel}.json"
        document = json.loads(path.read_text())
        if document["status"] != EXECUTABLE_STATUS:
            raise RuntimeError(f"status mismatch: {path}")
        if document["endpoint"] != endpoint:
            raise RuntimeError(f"exact endpoint specification mismatch: {path}")
        if document["condition"]["name"] != condition_name:
            raise RuntimeError(f"condition-name mismatch: {path}")
        if document["condition"]["memory_representation"] != condition["memory_representation"]:
            raise RuntimeError(f"condition representation mismatch: {path}")
        if int(document["condition"]["fact_slots"]) != int(expected_slots):
            raise RuntimeError(f"state-slot mismatch: {path}")
        if document["config"]["sha256"] != config_hash:
            raise RuntimeError(f"config hash mismatch: {path}")
        if document["data"]["sha256"] != config["panel_sha256_by_name"][panel]:
            raise RuntimeError(f"panel hash mismatch: {path}")
        if document["parser_accuracy"] != 1.0:
            raise RuntimeError(f"parser mismatch: {path}")
        if document["environment"]["git_dirty"]:
            raise RuntimeError(f"dirty worktree: {path}")
        for package, version in config["runtime_dependencies"].items():
            if document["environment"].get(package) != version:
                raise RuntimeError(f"runtime dependency mismatch in result: {path}")
        identity = document.get("model_identity", {})
        if identity.get("loaded_parameters") != endpoint["model_parameters"]:
            raise RuntimeError(f"loaded parameter identity mismatch: {path}")
        if identity.get("model_type") != endpoint["model_type"]:
            raise RuntimeError(f"loaded model type mismatch: {path}")
        if document["model_files"] != endpoint["model_artifacts"]:
            raise RuntimeError(f"weight manifest mismatch: {path}")
        panel_rows = document["predictions"]
        if len(panel_rows) != config["evaluation_samples"]:
            raise RuntimeError(f"prediction count mismatch: {path}")
        if {row["task"] for row in panel_rows} != set(config["tasks"]):
            raise RuntimeError(f"task coverage mismatch: {path}")
        rows.extend({"panel": panel, **row} for row in panel_rows)
        provenance[panel] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "git_commit": document["environment"]["git_commit"],
            "model_files": document["model_files"],
            "model_identity": identity,
        }
    expected = len(config["panels"]) * config["evaluation_samples"]
    if len(rows) != expected or len({row["row_id"] for row in rows}) != expected:
        raise RuntimeError(f"row coverage mismatch: {endpoint['name']}/{condition_name}")
    return rows, provenance


def failure_accounting(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(rows),
        "foundation_invalid_outputs": sum(row["foundation_answer"] is None for row in rows),
        "ascent_invalid_outputs": sum(row["ascent_answer"] is None for row in rows),
        "wins": sum(row["ascent_score"] > row["foundation_score"] for row in rows),
        "regressions": sum(row["ascent_score"] < row["foundation_score"] for row in rows),
        "ties": sum(row["ascent_score"] == row["foundation_score"] for row in rows),
    }


def analyze(
    config_path: Path,
    canonical_roots: dict[str, Path],
    qwen_fixed3_root: Path,
    qwen_raw4_root: Path,
    source_config_path: Path,
    frozen_qwen3b_root: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    config_hash = sha256_file(config_path)
    if config["status"] != EXECUTABLE_STATUS:
        raise RuntimeError("unexpected extension status")
    if config["execution_authorization"]["authorized"] is not True:
        raise RuntimeError("extension config is not executable")
    if sha256_file(Path(config["panel_manifest"]["path"])) != config["panel_manifest"]["sha256"]:
        raise RuntimeError("panel manifest hash mismatch")
    endpoints = {row["name"]: row for row in config["endpoints"]}
    if set(canonical_roots) != set(endpoints):
        raise RuntimeError("canonical roots do not match all frozen endpoints")

    canonical: dict[str, list[dict[str, Any]]] = {}
    provenance: dict[str, Any] = {"canonical_slots_4": {}}
    for name, endpoint in endpoints.items():
        canonical[name], provenance["canonical_slots_4"][name] = _load(
            config, config_hash, endpoint, "canonical_slots_4", canonical_roots[name]
        )
    identity_sets = [{row["row_id"] for row in rows} for rows in canonical.values()]
    if any(ids != identity_sets[0] for ids in identity_sets[1:]):
        raise RuntimeError("canonical endpoints do not cover identical rows")

    panels = config["panels"]
    tasks = set(config["tasks"])
    endpoint_results: dict[str, Any] = {}
    model_p: list[float] = []
    for name, rows in canonical.items():
        primary = _scope(rows, panels, tasks)
        p_value = one_sided_positive_p(primary["gain"]["panel_values"])
        model_p.append(p_value)
        endpoint_results[name] = {
            "primary": primary,
            "by_task": {task: _scope(rows, panels, {task}) for task in config["tasks"]},
            "one_sided_gain_p_unadjusted": p_value,
            "failure_accounting": failure_accounting(rows),
        }
    adjusted = holm_adjust(model_p)
    for (name, result), adjusted_p in zip(endpoint_results.items(), adjusted, strict=True):
        result["one_sided_gain_p_holm"] = adjusted_p
        result["passes_registered_gain"] = (
            result["primary"]["gain"]["mean"] > 0 and adjusted_p < 0.05
        )

    qwen_name = "qwen2p5-7b-instruct"
    qwen_endpoint = endpoints[qwen_name]
    fixed3, provenance["canonical_fixed_slots_3"] = _load(
        config, config_hash, qwen_endpoint, "canonical_fixed_slots_3", qwen_fixed3_root
    )
    raw4, provenance["raw_slots_4"] = _load(
        config, config_hash, qwen_endpoint, "raw_slots_4", qwen_raw4_root
    )
    qwen4 = canonical[qwen_name]
    for other in (fixed3, raw4):
        if {row["row_id"] for row in other} != {row["row_id"] for row in qwen4}:
            raise RuntimeError("Qwen control rows do not match canonical rows")
    maps = [{row["row_id"]: row for row in rows} for rows in (qwen4, fixed3, raw4)]
    foundation_exact = all(
        left[row_id]["foundation_output"] == right[row_id]["foundation_output"]
        and left[row_id]["foundation_score"] == right[row_id]["foundation_score"]
        for left, right in ((maps[0], maps[1]), (maps[0], maps[2]))
        for row_id in maps[0]
    )
    if not foundation_exact:
        raise RuntimeError("Qwen Foundation outputs differ across memory-only controls")

    source_config = json.loads(source_config_path.read_text())
    source_hash = sha256_file(source_config_path)
    if source_hash != config["source_protocol"]["sha256"]:
        raise RuntimeError("frozen Qwen source config hash mismatch")
    qwen3b, provenance["frozen_qwen2p5_3b"] = _load_qwen_source(
        source_config,
        source_hash,
        "qwen2p5-3b-instruct",
        "canonical_coscale",
        frozen_qwen3b_root,
    )
    if {row["row_id"] for row in qwen3b} != {row["row_id"] for row in qwen4}:
        raise RuntimeError("Qwen 3B and 7B rows do not match")
    gain3 = _panel_values(qwen3b, panels, tasks, "gain")
    gain7 = _panel_values(qwen4, panels, tasks, "gain")
    increment_values = [high - low for low, high in zip(gain3, gain7, strict=True)]
    increment_p = one_sided_positive_p(increment_values)
    qwen_fourth_scale = {
        "lower_endpoint": "qwen2p5-3b-instruct",
        "upper_endpoint": qwen_name,
        "lower_gain": cluster_summary(gain3),
        "upper_gain": cluster_summary(gain7),
        "gain_increment": cluster_summary(increment_values),
        "one_sided_p": increment_p,
        "passes_registered_contrast": float(np.mean(increment_values)) > 0 and increment_p < 0.05,
    }

    def control(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "accuracy_difference": paired_summary(
                _panel_values(left, panels, tasks, "ascent"),
                _panel_values(right, panels, tasks, "ascent"),
            ),
            "gain_difference": paired_summary(
                _panel_values(left, panels, tasks, "gain"),
                _panel_values(right, panels, tasks, "gain"),
            ),
        }

    return {
        "schema_version": 1,
        "experiment": config["experiment"],
        "status": config["status"],
        "config": {"path": str(config_path), "sha256": config_hash},
        "cluster_unit": "panel",
        "endpoint_results": endpoint_results,
        "qwen_fourth_scale": qwen_fourth_scale,
        "qwen_controls": {
            "slots4_minus_slots3": control(qwen4, fixed3),
            "canonical_minus_raw_at_slots4": control(qwen4, raw4),
            "foundation_outputs_exact_across_conditions": foundation_exact,
            "fixed3_failure_accounting": failure_accounting(fixed3),
            "raw4_failure_accounting": failure_accounting(raw4),
        },
        "registered_gate_pass": qwen_fourth_scale["passes_registered_contrast"]
        and all(row["passes_registered_gain"] for row in endpoint_results.values()),
        "provenance": provenance,
        "retention_audit": {
            "all_models_retained": len(endpoint_results) == len(endpoints),
            "all_panels_and_tasks_retained": True,
            "invalid_outputs_and_regressions_retained": True,
        },
    }


def _mapping(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in result:
            raise ValueError(f"invalid or duplicate NAME=PATH mapping: {value}")
        result[name] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--canonical-root", action="append", required=True)
    parser.add_argument("--qwen-fixed3-root", type=Path, required=True)
    parser.add_argument("--qwen-raw4-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--frozen-qwen3b-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.config,
        _mapping(args.canonical_root),
        args.qwen_fixed3_root,
        args.qwen_raw4_root,
        args.source_config,
        args.frozen_qwen3b_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "endpoint_results": result["endpoint_results"],
        "qwen_fourth_scale": result["qwen_fourth_scale"],
        "qwen_controls": result["qwen_controls"],
        "registered_gate_pass": result["registered_gate_pass"],
    }, indent=2))


if __name__ == "__main__":
    main()
