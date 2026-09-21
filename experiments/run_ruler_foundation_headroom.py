#!/usr/bin/env python3
"""Foundation-only headroom screen for an official NVIDIA RULER task."""

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

from ascent.ruler_metrics import string_match_all, string_match_part
from experiments.run_natural_repeat import git_value, sha256_file
from experiments.run_ruler_autoregressive import score_summary
from experiments.run_ruler_niah import verify_model_artifact


def headroom_decision(
    scores: list[float], minimum: float, maximum: float
) -> dict[str, Any]:
    """Apply a prospectively frozen two-sided non-saturation gate."""

    if not scores:
        raise ValueError("headroom scores cannot be empty")
    if not 0.0 <= minimum < maximum <= 1.0:
        raise ValueError("headroom bounds must satisfy 0 <= min < max <= 1")
    mean = float(np.mean(scores))
    return {
        "minimum_score": minimum,
        "maximum_score": maximum,
        "observed_score": mean,
        "observed_errors": int(sum(score < 1.0 for score in scores)),
        "passes": bool(minimum <= mean <= maximum),
        "failure_reason": (
            None
            if minimum <= mean <= maximum
            else ("too_hard_or_decode_invalid" if mean < minimum else "saturated")
        ),
    }


def run(
    config_path: Path,
    data_root: Path,
    endpoint_name: str,
    model_dir: Path,
    output: Path,
) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    raw: dict[str, Any] = json.loads(config_path.read_text())
    endpoint = {row["name"]: row for row in raw["endpoints"]}[endpoint_name]
    task_path = data_root / str(raw["task_name"]) / "validation.jsonl"
    observed_task_sha = sha256_file(task_path)
    if observed_task_sha != raw["task_sha256"]:
        raise RuntimeError("RULER task hash mismatch")
    rows = [json.loads(line) for line in task_path.read_text().splitlines()]
    if len(rows) != int(raw["samples_per_task"]):
        raise RuntimeError("unexpected RULER row count")
    rows = rows[: int(raw.get("evaluation_samples", len(rows)))]
    model_files = verify_model_artifact(
        model_dir, endpoint.get("model_artifacts", endpoint.get("model_artifact"))
    )

    started = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.bfloat16
    ).to("cuda").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    context_tokens = int(raw["query_context_tokens"])
    max_new_tokens = int(raw["max_new_tokens"])
    metric_name = str(raw["official_metric"])
    metrics = {
        "string_match_all": string_match_all,
        "string_match_part": string_match_part,
    }
    if metric_name not in metrics:
        raise ValueError(f"unsupported official RULER metric: {metric_name}")
    metric = metrics[metric_name]

    def prompt_ids(row: dict[str, Any]) -> tuple[Any, int, bool]:
        if raw.get("prompt_mode", "raw") == "chat":
            user_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["input"]}],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )[0]
            prefix_ids = tokenizer(
                row["answer_prefix"],
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0]
            full = torch.cat([user_ids, prefix_ids])
        else:
            full = tokenizer(
                row["input"] + row["answer_prefix"],
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0]
        length = int(full.numel())
        return full[-context_tokens:].to("cuda"), length, length > context_tokens

    def greedy_decode(ids: Any) -> str:
        generated: list[int] = []
        eos_config = model.generation_config.eos_token_id
        if eos_config is None:
            eos_ids: set[int] = set()
        elif isinstance(eos_config, int):
            eos_ids = {int(eos_config)}
        else:
            eos_ids = {int(value) for value in eos_config}
        query = ids.unsqueeze(0)
        attention_mask = torch.ones_like(query)
        with torch.inference_mode():
            result = model(
                input_ids=query,
                attention_mask=attention_mask,
                use_cache=True,
            )
        past_key_values = result.past_key_values
        logits = result.logits[0, -1].float()
        for step in range(max_new_tokens):
            next_id = int(torch.argmax(logits).item())
            generated.append(next_id)
            if next_id in eos_ids or step + 1 == max_new_tokens:
                break
            next_token = torch.tensor(
                [[next_id]], dtype=ids.dtype, device=ids.device
            )
            attention_mask = torch.ones(
                (1, int(ids.numel()) + len(generated)),
                dtype=ids.dtype,
                device=ids.device,
            )
            with torch.inference_mode():
                result = model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = result.past_key_values
            logits = result.logits[0, -1].float()
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    scores: list[float] = []
    predictions: list[dict[str, Any]] = []
    full_lengths: list[int] = []
    truncated_rows = 0
    for row in rows:
        ids, full_length, truncated = prompt_ids(row)
        prediction = greedy_decode(ids)
        score = metric(prediction, row["outputs"])
        scores.append(score)
        full_lengths.append(full_length)
        truncated_rows += int(truncated)
        predictions.append(
            {
                "index": row["index"],
                "outputs": row["outputs"],
                "prediction": prediction,
                "score": score,
                "full_prompt_tokens": full_length,
                "evaluated_prompt_tokens": int(ids.numel()),
                "truncated": truncated,
            }
        )

    gate = headroom_decision(
        scores,
        float(raw["headroom_minimum_score"]),
        float(raw["headroom_maximum_score"]),
    )
    payload = {
        "schema_version": 1,
        "status": raw["status"],
        "started_at_utc": started,
        "task_name": raw["task_name"],
        "task_sha256": observed_task_sha,
        "config_sha256": sha256_file(config_path),
        "data_generator_commit": raw["data_generator_commit"],
        "data_seed": raw["data_seed"],
        "endpoint": endpoint,
        "metric": f"NVIDIA RULER {metric_name} fraction",
        "foundation_score": score_summary(scores),
        "headroom_gate": gate,
        "prompt_audit": {
            "minimum_full_prompt_tokens": min(full_lengths),
            "maximum_full_prompt_tokens": max(full_lengths),
            "truncated_rows": truncated_rows,
            "native_full_context": truncated_rows == 0,
        },
        "predictions": predictions,
        "model_files": model_files,
        "runtime": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "git_commit": git_value(["rev-parse", "HEAD"]),
            "git_dirty": bool(git_value(["status", "--porcelain"])),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "wall_seconds": time.perf_counter() - wall_start,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.data_root, args.endpoint, args.model_dir, args.output)


if __name__ == "__main__":
    main()
