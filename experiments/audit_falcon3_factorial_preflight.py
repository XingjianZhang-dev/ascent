#!/usr/bin/env python3
"""Run score-free model, data, parser, nesting, and prompt audits for Falcon3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from ascent.babilong_memory import read_babilong
from experiments.audit_around7b_checkpoint import (
    frozen_weight_paths,
    git_blob_sha1,
    tensor_manifest,
    validate_cuda_model_load,
    validate_safetensors_index_metadata,
)
from experiments.run_babilong_prompt import (
    answer_instruction,
    prompt_input_token_budget,
    render_chat_prompt,
)
from experiments.run_natural_repeat import sha256_file


def endpoint_by_name(config: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [row for row in config["endpoints"] if row["name"] == name]
    if len(matches) != 1:
        raise RuntimeError(f"endpoint count is {len(matches)} for {name}")
    return matches[0]


def expected_model_files(endpoint: dict[str, Any]) -> list[dict[str, Any]]:
    singular = endpoint.get("model_artifact")
    plural = endpoint.get("model_artifacts")
    if (singular is None) == (plural is None):
        raise RuntimeError("endpoint must freeze exactly one weight-manifest field")
    return [singular] if singular is not None else list(plural)


def verify_critical_files(model_dir: Path, endpoint: dict[str, Any]) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for expected in endpoint["official_critical_artifacts"]:
        path = model_dir / expected["path"]
        actual = {
            "path": expected["path"],
            "bytes": path.stat().st_size,
            "git_blob_sha1": git_blob_sha1(path),
            "sha256": sha256_file(path),
        }
        if actual["bytes"] != expected["bytes"]:
            raise RuntimeError(f"critical-file byte mismatch: {path.name}")
        if actual["git_blob_sha1"] != expected["git_blob_sha1"]:
            raise RuntimeError(f"critical-file Git blob mismatch: {path.name}")
        verified.append(actual)
    return verified


def prompt_audit(
    config: dict[str, Any],
    endpoint: dict[str, Any],
    tokenizer: Any,
    data_root: Path,
) -> dict[str, Any]:
    budget = prompt_input_token_budget(
        int(config["context_tokens"]), int(config["max_new_tokens"])
    )
    tokenizer.truncation_side = "left"
    maximum_foundation = 0
    maximum_ascent = 0
    foundation_over_cap = 0
    over_cap_rows: list[dict[str, Any]] = []
    ascent_over_cap = 0
    parser_rows = 0
    nested_rows = 0
    state_slots = [int(value) for value in config["factorial"]["state_fact_slots"]]
    refinement_rows = {
        f"slots_{low}_to_{high}": 0
        for low, high in zip(state_slots[:-1], state_slots[1:], strict=True)
    }
    for panel in config["panels"]:
        path = data_root / config.get("panel_path_by_name", {}).get(
            panel, f"{panel}.jsonl"
        )
        if sha256_file(path) != config["panel_sha256_by_name"][panel]:
            raise RuntimeError(f"panel hash mismatch: {panel}")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != config["evaluation_samples"]:
            raise RuntimeError(f"panel row count mismatch: {panel}")
        for row in rows:
            read = read_babilong(
                row["input"],
                row["question"],
                history_slots=int(config["history_slots"]),
                enabled_tasks=tuple(config["tasks"]),
            )
            if read.answer != row["target"]:
                raise RuntimeError(f"parser mismatch: {row['row_id']}")
            parser_rows += 1
            retained_by_slots = [list(read.facts[-slots:]) for slots in state_slots]
            if not all(
                high[-len(low) :] == low
                for low, high in zip(
                    retained_by_slots[:-1], retained_by_slots[1:], strict=True
                )
            ):
                raise RuntimeError(f"nonnested state: {row['row_id']}")
            nested_rows += 1
            for index, (low, high) in enumerate(
                zip(state_slots[:-1], state_slots[1:], strict=True)
            ):
                refinement_rows[f"slots_{low}_to_{high}"] += (
                    retained_by_slots[index] != retained_by_slots[index + 1]
                )
            instruction = answer_instruction(row["task"])
            foundation_prompt = (
                "Find the short facts hidden in the distractor text and answer the question. "
                f"{instruction}\n\nText:\n{row['input']}\n\nQuestion: "
                f"{row['question']}\nAnswer:"
            )
            foundation_chat = render_chat_prompt(tokenizer, foundation_prompt, endpoint)
            foundation_length = len(
                tokenizer(foundation_chat, add_special_tokens=False)["input_ids"]
            )
            maximum_foundation = max(maximum_foundation, foundation_length)
            foundation_over_cap += foundation_length > budget
            if foundation_length > budget:
                full_ids = tokenizer(
                    foundation_chat, add_special_tokens=False
                )["input_ids"]
                retained_ids = full_ids[-budget:]

                def contains_subsequence(haystack: list[int], needle: list[int]) -> bool:
                    return any(
                        haystack[index : index + len(needle)] == needle
                        for index in range(len(haystack) - len(needle) + 1)
                    )

                support_preserved = []
                for fact in read.facts:
                    fact_ids = tokenizer(
                        fact.source, add_special_tokens=False
                    )["input_ids"]
                    support_preserved.append(
                        contains_subsequence(retained_ids, fact_ids)
                    )
                over_cap_rows.append(
                    {
                        "panel": panel,
                        "row_id": row["row_id"],
                        "prompt_tokens": foundation_length,
                        "left_truncated_tokens": foundation_length - budget,
                        "supporting_fact_count": len(read.facts),
                        "all_supporting_facts_preserved": all(support_preserved),
                        "supporting_facts_preserved": support_preserved,
                    }
                )
            for retained in retained_by_slots:
                memory_text = "\n".join(f"- {fact.source}" for fact in retained)
                ascent_prompt = (
                    "A bounded causal memory retained the following relevant facts in "
                    "chronological order. Answer using only these facts. "
                    f"{instruction}\n\nMemory:\n{memory_text}\n\nQuestion: "
                    f"{row['question']}\nAnswer:"
                )
                ascent_chat = render_chat_prompt(tokenizer, ascent_prompt, endpoint)
                ascent_length = len(
                    tokenizer(ascent_chat, add_special_tokens=False)["input_ids"]
                )
                maximum_ascent = max(maximum_ascent, ascent_length)
                ascent_over_cap += ascent_length > budget
    return {
        "parser_rows": parser_rows,
        "nested_rows": nested_rows,
        "nonredundant_rows_by_transition": refinement_rows,
        "every_transition_changes_at_least_one_row": all(
            count > 0 for count in refinement_rows.values()
        ),
        "prompt_input_cap": budget,
        "maximum_foundation_prompt_tokens_before_truncation": maximum_foundation,
        "maximum_ascent_prompt_tokens_before_truncation": maximum_ascent,
        "foundation_prompts_over_cap": foundation_over_cap,
        "foundation_over_cap_rows": over_cap_rows,
        "all_over_cap_rows_preserve_supporting_facts": all(
            row["all_supporting_facts_preserved"] for row in over_cap_rows
        ),
        "ascent_prompts_over_cap": ascent_over_cap,
        "left_truncation_frozen": tokenizer.truncation_side == "left",
    }


def audit(
    config_path: Path,
    endpoint_name: str,
    model_dir: Path,
    data_root: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    endpoint = endpoint_by_name(config, endpoint_name)
    for directory, expected_hash in config[
        "data_manifest_sha256_by_root"
    ].items():
        if sha256_file(data_root / directory / "MANIFEST.json") != expected_hash:
            raise RuntimeError(f"data manifest hash mismatch: {directory}")
    weight_paths, indexed_tensors, indexed_payload_bytes = frozen_weight_paths(model_dir)
    weights = tensor_manifest(weight_paths)
    actual_files = [
        {"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
        for row in weights["files"]
    ]
    if actual_files != expected_model_files(endpoint):
        raise RuntimeError("local weights differ from the frozen official manifest")
    if weights["parameters"] != endpoint["model_parameters"]:
        raise RuntimeError("safetensors parameter count differs from the endpoint")
    index_validation = validate_safetensors_index_metadata(
        weights, indexed_tensors, indexed_payload_bytes
    )
    critical_files = verify_critical_files(model_dir, endpoint)
    model_config = json.loads((model_dir / "config.json").read_text())
    if model_config.get("model_type") != endpoint["model_type"]:
        raise RuntimeError("model type mismatch")
    if model_config.get("architectures") != [endpoint["architecture"]]:
        raise RuntimeError("architecture mismatch")
    if int(model_config.get("max_position_embeddings", 0)) != endpoint[
        "native_context_tokens"
    ]:
        raise RuntimeError("native context mismatch")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    probe = render_chat_prompt(tokenizer, "Return exactly: kitchen", endpoint)
    if probe.count("Return exactly: kitchen") != 1:
        raise RuntimeError("chat template did not preserve the probe")
    prompts = prompt_audit(config, endpoint, tokenizer, data_root)
    model_load = validate_cuda_model_load(
        model_dir,
        model_config,
        weights,
        registered_context=int(config["context_tokens"]),
    )
    return {
        "schema_version": 1,
        "status": "score_free_falcon3_factorial_preflight",
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "endpoint": endpoint,
        "model_dir": str(model_dir),
        "weights": weights,
        "index_validation": index_validation,
        "critical_files": critical_files,
        "chat_template_probe_tokens": len(
            tokenizer(probe, add_special_tokens=False)["input_ids"]
        ),
        "prompt_and_data_audit": prompts,
        "model_load": model_load,
        "generation_invoked": False,
        "preflight_pass": all(
            (
                prompts["parser_rows"] == 1200,
                prompts["nested_rows"] == 1200,
                prompts["every_transition_changes_at_least_one_row"],
                prompts["all_over_cap_rows_preserve_supporting_facts"],
                prompts["ascent_prompts_over_cap"] == 0,
                model_load["load_pass"],
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.config, args.endpoint, args.model_dir, args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "endpoint": args.endpoint,
        "config_sha256": result["config"]["sha256"],
        "parameters": result["weights"]["parameters"],
        "prompt_and_data_audit": result["prompt_and_data_audit"],
        "preflight_pass": result["preflight_pass"],
    }, indent=2))


if __name__ == "__main__":
    main()
