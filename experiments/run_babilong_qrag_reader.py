#!/usr/bin/env python3
"""Read frozen Q-RAG selections with the exact Qwen2.5 ASCENT endpoints."""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_babilong_prompt import (
    answer_instruction,
    configure_decoder_tokenizer,
    extract_task_answer,
    frozen_model_artifact_spec,
    prompt_input_token_budget,
    summary,
)
from experiments.run_natural_repeat import git_value, sha256_file
from experiments.run_ruler_niah import verify_model_artifact


ROOT = Path(__file__).resolve().parents[1]


def qrag_prompt(task: str, question: str, retained_chunks: list[str]) -> str:
    if retained_chunks:
        memory = "\n".join(f"- {chunk}" for chunk in retained_chunks)
    else:
        memory = "- No passage survived the frozen Q-value stopping rule."
    return (
        "A task-trained Q-RAG retriever selected the following passages in "
        "chronological document order. Answer using only these passages. "
        f"{answer_instruction(task)}\n\n"
        f"Retrieved passages:\n{memory}\n\nQuestion: {question}\nAnswer:"
    )


def load_retrieval_document(
    path: Path,
    manifest: dict[str, Any],
    manifest_sha256: str,
    panel: str,
    task: str,
) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if document["status"] != manifest["status"]:
        raise RuntimeError("Q-RAG retrieval status mismatch")
    if document["manifest"]["sha256"] != manifest_sha256:
        raise RuntimeError("Q-RAG retrieval manifest hash mismatch")
    if document["panel"] != panel or document["task"] != task:
        raise RuntimeError("Q-RAG retrieval panel/task mismatch")
    if document["qrag"]["code_commit"] != manifest["qrag_code"]["commit"]:
        raise RuntimeError("Q-RAG retrieval code commit mismatch")
    if document["qrag"]["checkpoint_model_sha256"] != (
        manifest["qrag_checkpoints"][task]["model_sha256"]
    ):
        raise RuntimeError("Q-RAG retrieval checkpoint mismatch")
    expected_rows = int(manifest["evaluation"]["rows_per_task_per_panel"])
    if document["data"]["rows"] != expected_rows:
        raise RuntimeError("Q-RAG retrieval row count mismatch")
    if len(document["rows"]) != expected_rows:
        raise RuntimeError("Q-RAG retrieval output count mismatch")
    if len({row["row_id"] for row in document["rows"]}) != expected_rows:
        raise RuntimeError("Q-RAG retrieval row IDs are not unique")
    return document


def find_ascent_result(
    ascent_root: Path,
    endpoint_name: str,
    slots: int,
    panel: str,
    source_config: dict[str, Any],
    executed_commit: str,
) -> tuple[Path, dict[str, Any]]:
    filename = f"generative_slots_{slots}_{panel}_{endpoint_name}.json"
    matches = list(ascent_root.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f"expected one ASCENT reference result for {filename}")
    path = matches[0]
    result = json.loads(path.read_text())
    checks = (
        result["status"] == source_config["status"],
        result["endpoint"]["name"] == endpoint_name,
        result["condition"]["fact_slots"] == slots,
        result["config"]["sha256"]
        == sha256_file(
            ROOT
            / "configs"
            / "babilong_qwen2p5_8k_generative_factorial_replication_frozen.json"
        ),
        result["parser_accuracy"] == 1.0,
        result["environment"]["git_commit"] == executed_commit,
        not result["environment"]["git_dirty"],
        len(result["predictions"]) == int(source_config["evaluation_samples"]),
    )
    if not all(checks):
        raise RuntimeError(f"ASCENT reference result failed validation: {path}")
    return path, result


def prepare_rows(
    manifest_path: Path,
    retrieval_root: Path,
    ascent_root: Path,
    endpoint_name: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text())
    manifest_sha256 = sha256_file(manifest_path)
    source_spec = manifest["source_panel_config"]
    source_path = ROOT / source_spec["path"]
    if sha256_file(source_path) != source_spec["sha256"]:
        raise RuntimeError("source Qwen config hash mismatch")
    source_config = json.loads(source_path.read_text())
    endpoint_by_name = {row["name"]: row for row in source_config["endpoints"]}
    if endpoint_name not in manifest["evaluation"]["reader_endpoints"]:
        raise RuntimeError("reader endpoint is absent from the peer manifest")
    endpoint = endpoint_by_name[endpoint_name]
    slots = int(endpoint["state_fact_slots"])
    prepared: list[dict[str, Any]] = []
    for panel in manifest["evaluation"]["panels"]:
        _, ascent = find_ascent_result(
            ascent_root,
            endpoint_name,
            slots,
            panel,
            source_config,
            source_spec["executed_commit"],
        )
        ascent_predictions = {
            row["row_id"]: row for row in ascent["predictions"]
        }
        for task in manifest["evaluation"]["tasks"]:
            retrieval_path = retrieval_root / f"{panel}_{task}.json"
            retrieval = load_retrieval_document(
                retrieval_path,
                manifest,
                manifest_sha256,
                panel,
                task,
            )
            if retrieval["data"]["sha256"] != ascent["data"]["sha256"]:
                raise RuntimeError("Q-RAG/ASCENT panel data mismatch")
            for row in retrieval["rows"]:
                reference = ascent_predictions.get(row["row_id"])
                if reference is None or reference["task"] != task:
                    raise RuntimeError("Q-RAG row is absent from ASCENT reference")
                if reference["target"] != row["target"]:
                    raise RuntimeError("Q-RAG/ASCENT target mismatch")
                prepared.append(
                    {
                        "panel": panel,
                        **row,
                        "prompt": qrag_prompt(
                            task, row["question"], row["retained_chunks"]
                        ),
                        "foundation_score": reference["foundation_score"],
                        "foundation_output": reference["foundation_output"],
                        "ascent_score": reference["ascent_score"],
                        "ascent_output": reference["ascent_output"],
                    }
                )
    expected = int(manifest["evaluation"]["retrieval_rows"])
    if len(prepared) != expected or len({row["row_id"] for row in prepared}) != expected:
        raise RuntimeError("prepared Q-RAG reader rows are incomplete or duplicated")
    return manifest, source_config, endpoint, prepared


def run(
    manifest_path: Path,
    retrieval_root: Path,
    ascent_root: Path,
    endpoint_name: str,
    model_dir: Path,
    output: Path,
) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = datetime.now(timezone.utc).isoformat()
    run_start = time.perf_counter()
    manifest, source_config, endpoint, prepared = prepare_rows(
        manifest_path, retrieval_root, ascent_root, endpoint_name
    )
    model_files = verify_model_artifact(
        model_dir, frozen_model_artifact_spec(endpoint)
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    configure_decoder_tokenizer(tokenizer)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, dtype=torch.bfloat16
        )
        .to("cuda")
        .eval()
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.manual_seed(int(source_config["seed"]))
    batch_size = int(endpoint["batch_size"])
    max_new_tokens = int(source_config["max_new_tokens"])
    input_budget = prompt_input_token_budget(
        int(source_config["context_tokens"]), max_new_tokens
    )

    decoded: list[str] = []
    token_lengths: list[int] = []
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    decode_start = time.perf_counter()
    for start in range(0, len(prepared), batch_size):
        prompts = [row["prompt"] for row in prepared[start : start + batch_size]]
        chat_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        encoded = tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=input_budget,
            add_special_tokens=False,
        ).to("cuda")
        token_lengths.extend(int(value) for value in encoded.attention_mask.sum(dim=1))
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        suffix = generated[:, encoded.input_ids.shape[1] :]
        decoded.extend(tokenizer.batch_decode(suffix, skip_special_tokens=True))
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - decode_start

    predictions: list[dict[str, Any]] = []
    qrag_scores: list[float] = []
    for row, raw_output, prompt_tokens in zip(
        prepared, decoded, token_lengths, strict=True
    ):
        answer = extract_task_answer(raw_output, row["task"])
        score = float(answer == row["target"])
        qrag_scores.append(score)
        predictions.append(
            {
                "panel": row["panel"],
                "row_id": row["row_id"],
                "task": row["task"],
                "target": row["target"],
                "foundation_score": row["foundation_score"],
                "foundation_output": row["foundation_output"],
                "ascent_score": row["ascent_score"],
                "ascent_output": row["ascent_output"],
                "qrag_score": score,
                "qrag_output": raw_output,
                "qrag_answer": answer,
                "qrag_prompt_tokens": prompt_tokens,
                "selected_indices_in_retrieval_order": row[
                    "selected_indices_in_retrieval_order"
                ],
                "q_values": row["q_values"],
                "retained_indices_in_document_order": row[
                    "retained_indices_in_document_order"
                ],
                "retained_chunks": row["retained_chunks"],
            }
        )

    def grouped(field: str, value: str) -> dict[str, Any]:
        rows = [row for row in predictions if row[field] == value]
        foundation = [row["foundation_score"] for row in rows]
        ascent = [row["ascent_score"] for row in rows]
        qrag = [row["qrag_score"] for row in rows]
        return {
            "rows": len(rows),
            "foundation": summary(foundation),
            "ascent": summary(ascent),
            "qrag": summary(qrag),
            "qrag_minus_foundation": summary(
                [q - f for q, f in zip(qrag, foundation, strict=True)]
            ),
            "ascent_minus_qrag": summary(
                [a - q for a, q in zip(ascent, qrag, strict=True)]
            ),
        }

    document = {
        "schema_version": 1,
        "experiment": manifest["experiment"],
        "status": manifest["status"],
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "end_to_end_seconds": time.perf_counter() - run_start,
        "endpoint": endpoint,
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "ascent_reference_commit": manifest["source_panel_config"][
            "executed_commit"
        ],
        "model_files": model_files,
        "overall": {
            "rows": len(predictions),
            "foundation": summary([row["foundation_score"] for row in predictions]),
            "ascent": summary([row["ascent_score"] for row in predictions]),
            "qrag": summary(qrag_scores),
            "qrag_minus_foundation": summary(
                [row["qrag_score"] - row["foundation_score"] for row in predictions]
            ),
            "ascent_minus_qrag": summary(
                [row["ascent_score"] - row["qrag_score"] for row in predictions]
            ),
        },
        "by_task": {
            task: grouped("task", task) for task in manifest["evaluation"]["tasks"]
        },
        "by_panel": {
            panel: grouped("panel", panel)
            for panel in manifest["evaluation"]["panels"]
        },
        "systems": {
            "decode_seconds": decode_seconds,
            "samples_per_second": len(prepared) / decode_seconds,
            "maximum_prompt_tokens": max(token_lengths),
            "samples_at_prompt_token_cap": sum(
                length == input_budget for length in token_lengths
            ),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
            "git_commit": git_value(["rev-parse", "HEAD"]),
            "git_dirty": bool(git_value(["status", "--porcelain"])),
        },
        "predictions": predictions,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n")
    print(
        json.dumps(
            {
                "endpoint": endpoint_name,
                "foundation": document["overall"]["foundation"]["mean"],
                "ascent": document["overall"]["ascent"]["mean"],
                "qrag": document["overall"]["qrag"]["mean"],
                "output": str(output),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--retrieval-root", type=Path, required=True)
    parser.add_argument("--ascent-root", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.manifest,
        args.retrieval_root,
        args.ascent_root,
        args.endpoint,
        args.model_dir,
        args.output,
    )


if __name__ == "__main__":
    main()
