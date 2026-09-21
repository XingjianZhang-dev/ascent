#!/usr/bin/env python3
"""Verify exact prediction replication across two physical GPU nodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.freeze_babilong_around7b_extension import EXECUTABLE_STATUS
from experiments.run_natural_repeat import sha256_file


def _validate(
    document: dict[str, Any],
    config: dict[str, Any],
    config_hash: str,
    *,
    endpoint: str,
    condition: str,
    panel: str,
) -> None:
    if document["status"] != EXECUTABLE_STATUS:
        raise RuntimeError("replication input has unexpected status")
    expected_endpoint = next(
        (row for row in config["endpoints"] if row["name"] == endpoint), None
    )
    if expected_endpoint is None or document["endpoint"] != expected_endpoint:
        raise RuntimeError("replication exact endpoint specification mismatch")
    if document["condition"]["name"] != condition:
        raise RuntimeError("replication condition mismatch")
    if document["config"]["sha256"] != config_hash:
        raise RuntimeError("replication config hash mismatch")
    if document["data"]["sha256"] != config["panel_sha256_by_name"][panel]:
        raise RuntimeError("replication panel hash mismatch")
    if document["parser_accuracy"] != 1.0:
        raise RuntimeError("replication parser mismatch")
    if document["environment"]["git_dirty"]:
        raise RuntimeError("replication result came from a dirty worktree")
    for package, version in config["runtime_dependencies"].items():
        if document["environment"].get(package) != version:
            raise RuntimeError("replication runtime dependency mismatch")
    if len(document["predictions"]) != config["evaluation_samples"]:
        raise RuntimeError("replication prediction count mismatch")


def verify(
    config_path: Path,
    left_path: Path,
    right_path: Path,
    *,
    endpoint: str,
    condition: str,
    panel: str,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    config_hash = sha256_file(config_path)
    if config.get("status") != EXECUTABLE_STATUS:
        raise RuntimeError("cross-node config has unexpected status")
    if config.get("execution_authorization", {}).get("authorized") is not True:
        raise RuntimeError("cross-node config is not authorized")
    left = json.loads(left_path.read_text())
    right = json.loads(right_path.read_text())
    for document in (left, right):
        _validate(
            document,
            config,
            config_hash,
            endpoint=endpoint,
            condition=condition,
            panel=panel,
        )
    exact_fields = (
        "endpoint",
        "condition",
        "config",
        "data",
        "model_files",
        "model_identity",
        "parser_accuracy",
        "foundation",
        "ascent",
        "gain",
        "remaining_error_elimination",
        "wins",
        "regressions",
        "by_task",
        "state",
        "predictions",
    )
    mismatches = [field for field in exact_fields if left[field] != right[field]]
    if mismatches:
        raise RuntimeError(f"cross-node exact-field mismatch: {mismatches}")
    left_commit = left["environment"]["git_commit"]
    right_commit = right["environment"]["git_commit"]
    if left_commit != right_commit:
        raise RuntimeError("cross-node git commits differ")
    return {
        "schema_version": 1,
        "status": "cross_node_exact_replication_passed",
        "endpoint": endpoint,
        "condition": condition,
        "panel": panel,
        "config": {"path": str(config_path), "sha256": config_hash},
        "left": {"path": str(left_path), "sha256": sha256_file(left_path)},
        "right": {"path": str(right_path), "sha256": sha256_file(right_path)},
        "git_commit": left_commit,
        "exact_fields": list(exact_fields),
        "prediction_rows": len(left["predictions"]),
        "exact_prediction_match": True,
        "audit_pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        args.config,
        args.left,
        args.right,
        endpoint=args.endpoint,
        condition=args.condition,
        panel=args.panel,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
