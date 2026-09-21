#!/usr/bin/env python3
"""Profile supported-operator FLOPs for exact BABILong generate calls."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ascent.babilong_memory import read_babilong
from experiments.run_babilong_prompt import extract_location
from experiments.run_natural_repeat import git_value, sha256_file
from experiments.run_ruler_niah import verify_model_artifact


def prepare_prompts(
    rows: list[dict[str, Any]], *, history_slots: int, fact_slots: int
) -> tuple[list[str], list[str], list[dict[str, Any]], float]:
    foundation: list[str] = []
    ascent: list[str] = []
    metadata: list[dict[str, Any]] = []
    retrieval_start = time.perf_counter()
    for row in rows:
        read = read_babilong(
            row["input"], row["question"], history_slots=history_slots
        )
        if read.answer != row["target"]:
            raise RuntimeError(f"parser mismatch: {row['row_id']}")
        facts = [fact.source for fact in read.facts[-fact_slots:]]
        foundation.append(
            "Find the short facts hidden in the distractor text and answer the "
            "question. Reply with exactly one lowercase location word.\n\n"
            f"Text:\n{row['input']}\n\nQuestion: {row['question']}\nAnswer:"
        )
        memory_text = "\n".join(f"- {source}" for source in facts)
        ascent.append(
            "A bounded causal memory retained the following relevant facts in "
            "chronological order. Answer using only these facts. Reply with "
            "exactly one lowercase location word.\n\n"
            f"Memory:\n{memory_text}\n\nQuestion: {row['question']}\nAnswer:"
        )
        metadata.append(
            {
                "row_id": row["row_id"],
                "target": row["target"],
                "retained_facts": facts,
            }
        )
    return foundation, ascent, metadata, time.perf_counter() - retrieval_start


def run(
    config_path: Path,
    data_path: Path,
    endpoint_name: str,
    model_dir: Path,
    output: Path,
) -> None:
    import torch
    import transformers
    from torch.profiler import ProfilerActivity, profile
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = json.loads(config_path.read_text())
    endpoint = {row["name"]: row for row in config["endpoints"]}[endpoint_name]
    if sha256_file(data_path) != config["panel_sha256"]:
        raise RuntimeError("panel hash mismatch")
    rows = [json.loads(line) for line in data_path.read_text().splitlines()]
    if len(rows) != int(config["evaluation_samples"]):
        raise RuntimeError("row-count mismatch")
    model_files = verify_model_artifact(model_dir, endpoint["model_artifact"])
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
    fact_slots = int(
        config["conditions"]["scale_relevant"][
            "state_fact_slots_by_endpoint"
        ][endpoint_name]
    )
    foundation_prompts, ascent_prompts, metadata, retrieval_seconds = (
        prepare_prompts(
            rows,
            history_slots=int(config["history_slots"]),
            fact_slots=fact_slots,
        )
    )
    batch_size = int(endpoint["batch_size"])

    def encode(batch: list[str]) -> Any:
        chat = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt_text in batch
        ]
        return tokenizer(
            chat,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(config["context_tokens"]),
            add_special_tokens=False,
        ).to("cuda")

    def generate(encoded: Any) -> Any:
        with torch.inference_mode():
            return model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=int(config["max_new_tokens"]),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

    for prompts in (foundation_prompts, ascent_prompts):
        warmup = encode(prompts[:batch_size])
        generate(warmup)
        torch.cuda.synchronize()

    def profile_arm(prompts: list[str]) -> dict[str, Any]:
        operator_flops: defaultdict[str, int] = defaultdict(int)
        outputs: list[str] = []
        unpadded_prompt_tokens = 0
        padded_prompt_tokens = 0
        generated_tokens = 0
        wall_start = time.perf_counter()
        for start in range(0, len(prompts), batch_size):
            encoded = encode(prompts[start : start + batch_size])
            current_batch = int(encoded.input_ids.shape[0])
            unpadded_prompt_tokens += int(encoded.attention_mask.sum().item())
            padded_prompt_tokens += current_batch * int(encoded.input_ids.shape[1])
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                with_flops=True,
                record_shapes=True,
            ) as trace:
                generated = generate(encoded)
                torch.cuda.synchronize()
            for event in trace.key_averages():
                if event.flops:
                    operator_flops[event.key] += int(event.flops)
            suffix = generated[:, encoded.input_ids.shape[1] :]
            generated_tokens += int(
                (suffix != tokenizer.pad_token_id).sum().item()
            )
            outputs.extend(tokenizer.batch_decode(suffix, skip_special_tokens=True))
        wall_seconds = time.perf_counter() - wall_start
        locations = [extract_location(text) for text in outputs]
        scores = [
            float(location == row["target"])
            for location, row in zip(locations, metadata, strict=True)
        ]
        return {
            "supported_operator_flops": sum(operator_flops.values()),
            "supported_operator_flops_by_name": dict(
                sorted(operator_flops.items(), key=lambda item: -item[1])
            ),
            "wall_seconds": wall_seconds,
            "unpadded_prompt_tokens": unpadded_prompt_tokens,
            "padded_prompt_tokens": padded_prompt_tokens,
            "generated_tokens": generated_tokens,
            "accuracy": sum(scores) / len(scores),
            "normalized_predictions": locations,
        }

    foundation_result = profile_arm(foundation_prompts)
    ascent_result = profile_arm(ascent_prompts)
    result = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "endpoint": endpoint,
        "fact_slots": fact_slots,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "data": {"path": str(data_path), "sha256": sha256_file(data_path)},
        "model_files": model_files,
        "foundation": foundation_result,
        "ascent": ascent_result,
        "ratios": {
            "supported_operator_flops": ascent_result[
                "supported_operator_flops"
            ]
            / foundation_result["supported_operator_flops"],
            "wall_seconds": ascent_result["wall_seconds"]
            / foundation_result["wall_seconds"],
            "padded_prompt_tokens": ascent_result["padded_prompt_tokens"]
            / foundation_result["padded_prompt_tokens"],
        },
        "prediction_equality_reference_metrics": {
            "foundation_accuracy": foundation_result["accuracy"],
            "ascent_accuracy": ascent_result["accuracy"],
        },
        "retrieval_seconds": retrieval_seconds,
        "measurement_boundary": config["measurement_boundary"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
            "git_commit": git_value(["rev-parse", "HEAD"]),
            "git_dirty": bool(git_value(["status", "--porcelain"])),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "endpoint": endpoint_name,
                "foundation_accuracy": foundation_result["accuracy"],
                "ascent_accuracy": ascent_result["accuracy"],
                "flop_ratio": result["ratios"]["supported_operator_flops"],
                "wall_ratio": result["ratios"]["wall_seconds"],
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
    args = parser.parse_args()
    run(args.config, args.data, args.endpoint, args.model_dir, args.output)


if __name__ == "__main__":
    main()
