#!/usr/bin/env python3
"""Run official RULER FWE with a target-blind certified ASCENT counter."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ascent.ruler_aggregation_memory import (
    read_ruler_common_words,
    read_ruler_frequency,
)
from experiments.run_natural_repeat import git_value, sha256_file
from experiments.run_ruler_niah import verify_model_artifact


def summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    standard_error = float(array.std(ddof=1) / np.sqrt(array.size))
    return {
        "samples": int(array.size),
        "mean": float(array.mean()),
        "standard_error": standard_error,
        "ci95_low": float(array.mean() - 1.96 * standard_error),
        "ci95_high": float(array.mean() + 1.96 * standard_error),
    }


def string_match_all(prediction: str, references: list[str]) -> float:
    """NVIDIA RULER's reference-substring fraction on one sample."""
    lowered = prediction.lower()
    return sum(reference.lower() in lowered for reference in references) / len(
        references
    )


def run(
    config_path: Path,
    data_path: Path,
    endpoint_name: str,
    model_dir: Path,
    output: Path,
    data_seed: int | None = None,
) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config: dict[str, Any] = json.loads(config_path.read_text())
    endpoint = {row["name"]: row for row in config["endpoints"]}[
        endpoint_name
    ]
    expected_task_hash = config.get("task_sha256")
    if "task_sha256_by_seed" in config:
        if data_seed is None:
            raise RuntimeError("seed-indexed task hashes require --data-seed")
        expected_task_hash = config["task_sha256_by_seed"].get(str(data_seed))
        if expected_task_hash is None:
            raise RuntimeError("data seed is absent from the frozen task-hash map")
    if sha256_file(data_path) != expected_task_hash:
        raise RuntimeError("RULER FWE task hash mismatch")
    rows = [json.loads(line) for line in data_path.read_text().splitlines()]
    if len(rows) != int(config["evaluation_samples"]):
        raise RuntimeError("unexpected RULER FWE row count")

    model_files = verify_model_artifact(
        model_dir,
        endpoint.get("model_artifacts", endpoint.get("model_artifact")),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.bfloat16
    ).to("cuda").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.manual_seed(int(config["seed"]))

    fixed_slots = int(config["fixed_state_word_slots"])
    scale_slots = int(config["state_word_slots_by_endpoint"][endpoint_name])
    prepared: list[dict[str, Any]] = []
    parser_matches = 0
    reader_name = str(config.get("reader", "frequency_top3"))
    readers = {
        "frequency_top3": read_ruler_frequency,
        "common_top10": read_ruler_common_words,
    }
    if reader_name not in readers:
        raise RuntimeError(f"unsupported aggregation reader: {reader_name}")
    memory_reader = readers[reader_name]
    retrieval_start = time.perf_counter()
    for row in rows:
        read = memory_reader(
            row["input"], memory_slots=int(config["memory_slots"])
        )
        parser_words = list(read.top_words[: len(row["outputs"])])
        parser_matches += int(set(parser_words) == set(row["outputs"]))
        prepared.append(
            {
                "index": row["index"],
                "references": row["outputs"],
                "prompt": row["input"] + row.get("answer_prefix", ""),
                "parser_words": parser_words,
                "counter_entries": len(read.entries),
                "write_payload_bytes": read.persistent_payload_bytes,
            }
        )
    retrieval_seconds = time.perf_counter() - retrieval_start
    if parser_matches != len(rows):
        raise RuntimeError(
            f"target-blind frequency parser mismatch on {len(rows)-parser_matches} rows"
        )

    batch_size = int(endpoint["batch_size"])
    decoded: list[str] = []
    token_lengths: list[int] = []
    generated_tokens = 0
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    allocated_before = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    decode_start = time.perf_counter()
    for start in range(0, len(prepared), batch_size):
        prompts = [
            row["prompt"] for row in prepared[start : start + batch_size]
        ]
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
            max_length=int(config["query_context_tokens"]),
            add_special_tokens=False,
        ).to("cuda")
        token_lengths.extend(
            int(value) for value in encoded.attention_mask.sum(dim=1)
        )
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=int(config["max_new_tokens"]),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        suffix = generated[:, encoded.input_ids.shape[1] :]
        generated_tokens += int(
            (suffix != tokenizer.pad_token_id).sum().item()
        )
        decoded.extend(tokenizer.batch_decode(suffix, skip_special_tokens=True))
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - decode_start
    peak_bytes = int(torch.cuda.max_memory_allocated())

    foundation_scores: list[float] = []
    fixed_scores: list[float] = []
    scale_scores: list[float] = []
    predictions: list[dict[str, Any]] = []
    for row, foundation_output, prompt_tokens in zip(
        prepared, decoded, token_lengths, strict=True
    ):
        fixed_words = row["parser_words"][:fixed_slots]
        scale_words = row["parser_words"][:scale_slots]
        fixed_output = foundation_output + " " + " ".join(fixed_words)
        scale_output = foundation_output + " " + " ".join(scale_words)
        foundation_score = string_match_all(
            foundation_output, row["references"]
        )
        fixed_score = string_match_all(fixed_output, row["references"])
        scale_score = string_match_all(scale_output, row["references"])
        foundation_scores.append(foundation_score)
        fixed_scores.append(fixed_score)
        scale_scores.append(scale_score)
        predictions.append(
            {
                "index": row["index"],
                "references": row["references"],
                "foundation_output": foundation_output,
                "foundation_score": foundation_score,
                "fixed_certified_words": fixed_words,
                "fixed_score": fixed_score,
                "scale_certified_words": scale_words,
                "scale_score": scale_score,
                "counter_entries": row["counter_entries"],
                "write_payload_bytes": row["write_payload_bytes"],
                "prompt_tokens": prompt_tokens,
            }
        )

    fixed_gains = [
        value - base
        for value, base in zip(fixed_scores, foundation_scores, strict=True)
    ]
    scale_gains = [
        value - base
        for value, base in zip(scale_scores, foundation_scores, strict=True)
    ]
    mean_write_bytes = float(
        np.mean([row["write_payload_bytes"] for row in prepared])
    )
    fixed_read_bytes = float(
        np.mean(
            [
                sum(len(word.encode("utf-8")) + 8 for word in row["parser_words"][:fixed_slots])
                for row in prepared
            ]
        )
    )
    scale_read_bytes = float(
        np.mean(
            [
                sum(len(word.encode("utf-8")) + 8 for word in row["parser_words"][:scale_slots])
                for row in prepared
            ]
        )
    )
    result = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "status": config["status"],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "data": {"path": str(data_path), "sha256": sha256_file(data_path), "rows": len(rows)},
        "data_seed": data_seed,
        "model_files": model_files,
        "parser_accuracy": parser_matches / len(rows),
        "foundation": summary(foundation_scores),
        "fixed_ascent": summary(fixed_scores),
        "fixed_gain": summary(fixed_gains),
        "scale_ascent": summary(scale_scores),
        "scale_gain": summary(scale_gains),
        "fixed_wins": sum(value > base for value, base in zip(fixed_scores, foundation_scores, strict=True)),
        "fixed_regressions": sum(value < base for value, base in zip(fixed_scores, foundation_scores, strict=True)),
        "scale_wins": sum(value > base for value, base in zip(scale_scores, foundation_scores, strict=True)),
        "scale_regressions": sum(value < base for value, base in zip(scale_scores, foundation_scores, strict=True)),
        "state": {
            "memory_slots": int(config["memory_slots"]),
            "fixed_certified_word_slots": fixed_slots,
            "scale_certified_word_slots": scale_slots,
            "mean_write_payload_bytes": mean_write_bytes,
            "maximum_write_payload_bytes": max(row["write_payload_bytes"] for row in prepared),
            "mean_fixed_read_payload_bytes": fixed_read_bytes,
            "mean_scale_read_payload_bytes": scale_read_bytes,
            "mean_scale_total_state_bytes": mean_write_bytes + scale_read_bytes,
            "scale_total_state_bytes_per_model_parameter": (mean_write_bytes + scale_read_bytes) / int(endpoint["model_parameters"]),
        },
        "systems": {
            "target_blind_counter_seconds": retrieval_seconds,
            "counter_microseconds_per_sample": retrieval_seconds * 1e6 / len(rows),
            "foundation_decode_seconds": decode_seconds,
            "foundation_prompt_tokens": sum(token_lengths),
            "maximum_foundation_prompt_tokens": max(token_lengths),
            "foundation_generated_tokens": generated_tokens,
            "allocated_before_bytes": allocated_before,
            "peak_allocated_bytes": peak_bytes,
            "incremental_peak_bytes": peak_bytes - allocated_before,
        },
        "measurement_boundary": config["measurement_boundary"],
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
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "endpoint": endpoint_name,
                "foundation": result["foundation"]["mean"],
                "fixed_gain": result["fixed_gain"]["mean"],
                "scale_gain": result["scale_gain"]["mean"],
                "scale_wins": result["scale_wins"],
                "scale_regressions": result["scale_regressions"],
                "output": str(output),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-seed", type=int)
    args = parser.parse_args()
    run(
        args.config,
        args.data,
        args.endpoint,
        args.model_dir,
        args.output,
        args.data_seed,
    )


if __name__ == "__main__":
    main()
