#!/usr/bin/env python3
"""Freeze the executable around-7B extension from score-free audits only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.run_natural_repeat import sha256_file


EXECUTABLE_STATUS = (
    "prospective_around7b_extension_frozen_before_any_new_endpoint_decoder_score"
)


def _audit_by_model(
    design: dict[str, Any], design_hash: str, audit_paths: list[Path]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    audits: dict[str, dict[str, Any]] = {}
    artifacts: list[dict[str, Any]] = []
    roster = {row["name"]: row for row in design["frozen_model_roster"]}
    for path in audit_paths:
        document = json.loads(path.read_text())
        name = document["model"]["name"]
        if name not in roster or name in audits:
            raise RuntimeError(f"unexpected or duplicate audit model: {name}")
        if document["status"] != "score_free_checkpoint_and_template_audit":
            raise RuntimeError(f"unexpected audit status: {path}")
        if document.get("decoder_scores_observed") is not False:
            raise RuntimeError(f"audit is not score-free: {path}")
        if document.get("audit_pass") is not True:
            raise RuntimeError(f"checkpoint audit did not pass: {path}")
        if document.get("official_lfs_manifest_exact_match") is not True:
            raise RuntimeError(f"official LFS manifest did not match: {path}")
        if document.get("official_critical_manifest_exact_match") is not True:
            raise RuntimeError(f"official config/tokenizer manifest did not match: {path}")
        index_validation = document.get("safetensors_index_validation", {})
        if index_validation.get("tensor_count_match") is not True:
            raise RuntimeError(f"safetensors tensor count did not match: {path}")
        if index_validation.get("resolved_by_official_index_and_exact_model_load") is not True:
            raise RuntimeError(f"safetensors index metadata was not resolved: {path}")
        load_validation = document.get("model_load_validation", {})
        if load_validation.get("load_pass") is not True:
            raise RuntimeError(f"BF16/CUDA model-load audit did not pass: {path}")
        if load_validation.get("generation_invoked") is not False:
            raise RuntimeError(f"score-free load audit unexpectedly generated: {path}")
        if document["design"]["sha256"] != design_hash:
            raise RuntimeError(f"design hash mismatch: {path}")
        if document["model"] != roster[name]:
            raise RuntimeError(f"roster metadata mismatch: {path}")
        if document["environment"].get("git_dirty"):
            raise RuntimeError(f"audit was created from a dirty worktree: {path}")
        audits[name] = document
        artifacts.append(
            {"model": name, "path": str(path), "sha256": sha256_file(path)}
        )
    if set(audits) != set(roster):
        missing = sorted(set(roster) - set(audits))
        raise RuntimeError(f"missing model audits: {missing}")
    return audits, sorted(artifacts, key=lambda row: row["model"])


def build_config(
    design_path: Path,
    source_path: Path,
    audit_paths: list[Path],
    *,
    frozen_utc: str,
) -> dict[str, Any]:
    design = json.loads(design_path.read_text())
    source = json.loads(source_path.read_text())
    design_hash = sha256_file(design_path)
    source_hash = sha256_file(source_path)
    protocol = design["source_panel_protocol"]
    if source_hash != protocol["config_sha256"]:
        raise RuntimeError("source protocol config hash mismatch")
    if source["panel_manifest"]["sha256"] != protocol["manifest_sha256"]:
        raise RuntimeError("source panel manifest mismatch")
    if design["execution_authorization"]["authorized"] is not False:
        raise RuntimeError("design must remain non-executable")
    audits, audit_artifacts = _audit_by_model(design, design_hash, audit_paths)
    dependency_sets = {
        json.dumps(audit["environment"]["libraries"], sort_keys=True)
        for audit in audits.values()
    }
    if len(dependency_sets) != 1:
        raise RuntimeError("score-free audits used different runtime dependency versions")
    runtime_dependencies = next(iter(audits.values()))["environment"]["libraries"]

    endpoints: list[dict[str, Any]] = []
    for roster_row in design["frozen_model_roster"]:
        name = roster_row["name"]
        audit = audits[name]
        architectures = audit["model_config"]["architectures"]
        if len(architectures) != 1:
            raise RuntimeError(f"expected exactly one architecture for {name}")
        endpoint = {
            "name": name,
            "repo_id": roster_row["repo_id"],
            "revision": roster_row["revision"],
            "model_parameters": audit["weights"]["parameters"],
            "model_type": audit["model_config"]["model_type"],
            "architecture": architectures[0],
            "native_context_tokens": audit["model_config"]["native_context_tokens"],
            "state_fact_slots": 4,
            "batch_size": 1,
            "model_artifacts": [
                {"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
                for row in audit["weights"]["files"]
            ],
            "checkpoint_audit_sha256": next(
                row["sha256"] for row in audit_artifacts if row["model"] == name
            ),
            "chat_template_probe_sha256": audit["chat_template"][
                "rendered_probe_sha256"
            ],
        }
        if roster_row.get("chat_template_kwargs"):
            endpoint["chat_template_kwargs"] = roster_row["chat_template_kwargs"]
        if roster_row.get("system_prompt"):
            endpoint["system_prompt"] = roster_row["system_prompt"]
        endpoints.append(endpoint)

    qwen_name = "qwen2p5-7b-instruct"
    all_slots = {row["name"]: 4 for row in endpoints}
    return {
        "schema_version": 1,
        "experiment": "ascent_babilong_around7b_16k_extension",
        "status": EXECUTABLE_STATUS,
        "frozen_utc": frozen_utc,
        "seed": design["seed"],
        "tasks": source["tasks"],
        "panels": source["panels"],
        "panel_sha256_by_name": source["panel_sha256_by_name"],
        "panel_manifest": source["panel_manifest"],
        "official_source": source["official_source"],
        "evaluation_samples": source["evaluation_samples"],
        "history_slots": source["history_slots"],
        "context_tokens": source["context_tokens"],
        "max_new_tokens": source["max_new_tokens"],
        "allowed_decode_orders": ["foundation_first"],
        "conditions": {
            "canonical_slots_4": {
                "memory_source": "relevant",
                "memory_representation": "canonical_event_facts",
                "state_fact_slots_by_endpoint": all_slots,
            },
            "canonical_fixed_slots_3": {
                "memory_source": "relevant",
                "memory_representation": "canonical_event_facts",
                "state_fact_slots_by_endpoint": {qwen_name: 3},
            },
            "raw_slots_4": {
                "memory_source": "relevant",
                "memory_representation": "event_facts",
                "state_fact_slots_by_endpoint": {qwen_name: 4},
            },
        },
        "endpoints": endpoints,
        "prospective_design": {
            "path": str(design_path),
            "sha256": design_hash,
            "score_visibility": design["score_visibility_at_freeze"],
        },
        "source_protocol": {
            "path": str(source_path),
            "sha256": source_hash,
            "frozen_qwen_3b_analysis_sha256": source["development_evidence"][
                "analysis_artifact_sha256"
            ],
        },
        "checkpoint_audits": audit_artifacts,
        "runtime_dependencies": runtime_dependencies,
        "primary_analysis": design["primary_analysis"],
        "secondary_analysis": design["secondary_analysis"],
        "execution_authorization": {
            "authorized": True,
            "basis": "All five score-free checkpoint/template audits passed before any new endpoint score.",
            "cross_node_qwen_requirement": "The first Qwen2.5-7B formal panel must reproduce exactly on both physical nodes before the remaining panels execute.",
        },
        "retention_rule": "Report every frozen model, panel, task, invalid output, and regression regardless of sign.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--frozen-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_config(
        args.design, args.source, args.audit, frozen_utc=args.frozen_utc
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "endpoints": [row["name"] for row in result["endpoints"]],
        "authorized": result["execution_authorization"]["authorized"],
    }, indent=2))


if __name__ == "__main__":
    main()
