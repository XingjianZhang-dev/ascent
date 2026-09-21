#!/usr/bin/env python3
"""Score-free preflight for the frozen Qwen2.5 BABILong replication."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ascent.babilong_memory import read_babilong
from experiments.run_babilong_prompt import (
    configure_decoder_tokenizer,
    frozen_model_artifact_spec,
)
from experiments.run_natural_repeat import git_value, sha256_file
from experiments.run_ruler_niah import verify_model_artifact


ROOT = Path(__file__).resolve().parents[1]


def verify_referenced_document(spec: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / spec["path"]
    observed = sha256_file(path)
    if observed != spec["sha256"]:
        raise RuntimeError(f"referenced document hash mismatch: {path}")
    return {"path": str(path), "sha256": observed}


def audit(
    config_path: Path,
    data_root: Path,
    model_root: Path,
    endpoint_names: list[str],
    results_root: Path | None,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    config = json.loads(config_path.read_text())
    if config["status"] != (
        "prospective_qwen_same_task_factorial_before_any_babilong_decoder_score"
    ):
        raise RuntimeError("unexpected Qwen replication status")
    referenced = {
        key: verify_referenced_document(config[key])
        for key in ("factorial_design", "source_panel_config")
    }

    tasks = tuple(config["tasks"])
    row_identities: set[tuple[str, str]] = set()
    panels: list[dict[str, Any]] = []
    parser_correct = 0
    total_rows = 0
    for panel in config["panels"]:
        path = data_root / f"{panel}.jsonl"
        observed_hash = sha256_file(path)
        if observed_hash != config["panel_sha256_by_name"][panel]:
            raise RuntimeError(f"panel hash mismatch: {panel}")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != int(config["evaluation_samples"]):
            raise RuntimeError(f"row count mismatch: {panel}")
        counts = {task: sum(row["task"] == task for row in rows) for task in tasks}
        if len(set(counts.values())) != 1:
            raise RuntimeError(f"panel is not task balanced: {panel}")
        for row in rows:
            identity = (row["task"], row["row_id"])
            if identity in row_identities:
                raise RuntimeError(f"duplicate task/row identity: {identity}")
            row_identities.add(identity)
            read = read_babilong(
                row["input"],
                row["question"],
                history_slots=int(config["history_slots"]),
                enabled_tasks=tasks,
            )
            parser_correct += int(read.answer == row["target"])
            total_rows += 1
        panels.append(
            {
                "name": panel,
                "path": str(path),
                "sha256": observed_hash,
                "rows": len(rows),
                "task_counts": counts,
            }
        )
    if parser_correct != total_rows:
        raise RuntimeError(
            f"target-blind parser mismatch: {parser_correct}/{total_rows}"
        )

    endpoints = {endpoint["name"]: endpoint for endpoint in config["endpoints"]}
    if not endpoint_names or len(set(endpoint_names)) != len(endpoint_names):
        raise RuntimeError("endpoint arguments must be nonempty and unique")
    model_checks: list[dict[str, Any]] = []
    for name in endpoint_names:
        endpoint = endpoints[name]
        model_dir = model_root / name
        manifest = verify_model_artifact(
            model_dir, frozen_model_artifact_spec(endpoint)
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        configure_decoder_tokenizer(tokenizer)
        rendered = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": "Reply with exactly one lowercase location word.",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        if not rendered:
            raise RuntimeError(f"empty chat template rendering: {name}")
        model_checks.append(
            {
                "endpoint": name,
                "model_dir": str(model_dir),
                "model_files": manifest,
                "chat_template_rendered": True,
                "model_max_length": int(tokenizer.model_max_length),
                "padding_side": tokenizer.padding_side,
                "truncation_side": tokenizer.truncation_side,
            }
        )

    prior_result_files: list[str] = []
    if results_root is not None and results_root.exists():
        prior_result_files = sorted(str(path) for path in results_root.rglob("*.json"))
    if prior_result_files:
        raise RuntimeError(
            f"Qwen replication result files already exist: {len(prior_result_files)}"
        )

    dirty = bool(git_value(["status", "--porcelain"]))
    if dirty:
        raise RuntimeError("preflight requires a clean worktree")
    return {
        "schema_version": 1,
        "audit": "score_free_qwen_babilong_preflight",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "referenced_documents": referenced,
        "panels": panels,
        "panel_count": len(panels),
        "rows": total_rows,
        "unique_task_row_ids": len(row_identities),
        "target_blind_parser_correct": parser_correct,
        "model_checks": model_checks,
        "prior_result_files": prior_result_files,
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_dirty": dirty,
        "model_loaded": False,
        "generation_called": False,
        "preflight_pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.config,
        args.data_root,
        args.model_root,
        args.endpoint,
        args.results_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
