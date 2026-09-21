#!/usr/bin/env python3
"""Teacher-forced ASCENT latent-replay diagnostic on official RULER tasks."""

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

from ascent.gpt_neox_replay import relative_layer_index, replay_from_hidden
from ascent.ruler_memory import read_ruler_niah
from ascent.ruler_vt_memory import read_ruler_vt
from experiments.run_natural_repeat import (
    array_sha256,
    evaluate_arm,
    git_value,
    paired_summary,
    probability_summary,
    sha256_file,
    true_token_probabilities,
)


def verify_model_artifact(
    model_dir: Path, expected: dict[str, Any] | list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Hash model weights before loading and enforce an optional frozen artifact."""
    weights = sorted(model_dir.glob("*.safetensors")) + sorted(
        model_dir.glob("pytorch_model*.bin")
    )
    if not weights:
        raise RuntimeError(f"no model weights found in {model_dir}")
    manifest = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in weights
    ]
    if expected is not None:
        expected_rows = expected if isinstance(expected, list) else [expected]
        for row in expected_rows:
            required = {
                "path": str(row["path"]),
                "bytes": int(row["bytes"]),
                "sha256": str(row["sha256"]),
            }
            if required not in manifest:
                raise RuntimeError(
                    f"model artifact mismatch: required {required}, observed {manifest}"
                )
    return manifest


def run(
    config_path: Path,
    data_root: Path,
    endpoint_name: str,
    model_dir: Path,
    output: Path,
    arrays: Path,
) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    raw: dict[str, Any] = json.loads(config_path.read_text())
    reader_name = raw.get("reader", "niah")
    readers = {"niah": read_ruler_niah, "vt": read_ruler_vt}
    if reader_name not in readers:
        raise ValueError(f"unsupported RULER reader: {reader_name}")
    reader = readers[reader_name]
    output_separator = str(raw.get("output_separator", ", "))
    replay_token_selection = str(raw.get("replay_token_selection", "suffix"))
    if replay_token_selection not in {"prefix", "suffix"}:
        raise ValueError("replay_token_selection must be 'prefix' or 'suffix'")
    endpoint = {row["name"]: row for row in raw["endpoints"]}[endpoint_name]
    manifest_path = data_root / "DATA_MANIFEST.json"
    expected_manifest_hash = raw.get("dataset_manifest_sha256")
    if expected_manifest_hash is not None:
        if sha256_file(manifest_path) != expected_manifest_hash:
            raise RuntimeError("RULER manifest hash mismatch")
    task_rows: dict[str, list[dict[str, Any]]] = {}
    for task in raw["tasks"]:
        path = data_root / task / "validation.jsonl"
        if sha256_file(path) != raw["task_sha256"][task]:
            raise RuntimeError(f"RULER task hash mismatch: {task}")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != int(raw["samples_per_task"]):
            raise RuntimeError(f"unexpected row count for {task}")
        task_rows[task] = rows

    model_files = verify_model_artifact(model_dir, endpoint.get("model_artifact"))

    started = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.bfloat16
    ).to("cuda").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layer = relative_layer_index(
        len(model.gpt_neox.layers), float(raw["relative_injection_depth"])
    )
    fixed_span = int(raw["fixed_replay_tokens"])
    scale_span = int(endpoint["scale_replay_tokens"])
    spans = sorted({fixed_span, scale_span})
    context_tokens = int(raw["query_context_tokens"])
    epsilon = float(raw["epsilon"])
    grid_size = int(raw["fusion_grid_size"])
    split_samples = int(raw["calibration_samples_per_task"])
    results: dict[str, Any] = {}
    saved: dict[str, np.ndarray] = {}
    all_scale_gains: list[np.ndarray] = []
    all_latent_scale_gains: list[np.ndarray] = []

    for task, rows in task_rows.items():
        base_by_sample: list[np.ndarray] = []
        cert_by_sample: list[np.ndarray] = []
        latent_by_span: dict[int, list[np.ndarray]] = {span: [] for span in spans}
        base_top1: list[np.ndarray] = []
        latent_top1: dict[int, list[np.ndarray]] = {span: [] for span in spans}
        payload_bytes: list[int] = []
        parser_correct: list[bool] = []
        source_token_lengths: list[int] = []
        for row in rows:
            read = reader(row["input"], memory_slots=int(raw["memory_slots"]))
            correct = list(read.values) == row["outputs"]
            parser_correct.append(correct)
            payload_bytes.append(read.persistent_payload_bytes)
            source_text = "\n".join(fact.source for fact in read.facts)
            source_ids_all = tokenizer(
                source_text, add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0]
            source_token_lengths.append(int(source_ids_all.numel()))
            target_text = " " + output_separator.join(row["outputs"])
            target_ids = tokenizer(
                target_text, add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0]
            prompt_ids = tokenizer(
                row["input"] + row["answer_prefix"],
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0, -context_tokens:]
            input_ids = torch.cat([prompt_ids, target_ids[:-1]]).unsqueeze(0).to("cuda")
            targets = target_ids.to("cuda")
            attention_mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                query_output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                start = prompt_ids.numel() - 1
                base_logits = query_output.logits[0, start : start + targets.numel()]
                base_by_sample.append(true_token_probabilities(base_logits, targets))
                base_top1.append(
                    (base_logits.argmax(dim=-1) == targets).cpu().numpy().astype(np.float64)
                )
                query_hidden = query_output.hidden_states[layer]
                for registered_span in spans:
                    actual_span = min(registered_span, source_ids_all.numel())
                    selected_source_ids = (
                        source_ids_all[:actual_span]
                        if replay_token_selection == "prefix"
                        else source_ids_all[-actual_span:]
                    )
                    source_ids = selected_source_ids.unsqueeze(0).to("cuda")
                    source_output = model(
                        input_ids=source_ids,
                        attention_mask=torch.ones_like(source_ids),
                        output_hidden_states=True,
                        use_cache=False,
                    )
                    hidden = replay_from_hidden(
                        model, query_hidden, attention_mask,
                        source_output.hidden_states[layer], layer,
                    )
                    replay_logits = model.embed_out(hidden)[
                        0,
                        actual_span + start : actual_span + start + targets.numel(),
                    ]
                    latent_by_span[registered_span].append(
                        true_token_probabilities(replay_logits, targets)
                    )
                    latent_top1[registered_span].append(
                        (replay_logits.argmax(dim=-1) == targets).cpu().numpy().astype(np.float64)
                    )
                    del source_output, hidden, replay_logits
            cert_by_sample.append(np.ones(target_ids.numel()) if correct else np.zeros(target_ids.numel()))
            del query_output

        cal_tokens = sum(len(values) for values in base_by_sample[:split_samples])
        base = np.concatenate(base_by_sample[:split_samples] + base_by_sample[split_samples:])
        certified = np.concatenate(cert_by_sample[:split_samples] + cert_by_sample[split_samples:])
        task_result: dict[str, Any] = {
            "samples": len(rows),
            "calibration_samples": split_samples,
            "test_samples": len(rows) - split_samples,
            "matched_raw_exact_parser_accuracy": float(np.mean(parser_correct[split_samples:])),
            "persistent_payload_bytes_max": max(payload_bytes),
            "persistent_payload_bytes_mean": float(np.mean(payload_bytes)),
            "source_token_length_min": min(source_token_lengths),
            "source_token_length_mean": float(np.mean(source_token_lengths)),
            "source_token_length_max": max(source_token_lengths),
            "foundation_test": probability_summary(base[cal_tokens:], epsilon=epsilon),
            "foundation_teacher_forced_top1_accuracy": float(
                np.concatenate(base_top1[split_samples:]).mean()
            ),
            "arms": {},
        }
        for name, span in (("ascent_fixed", fixed_span), ("ascent_scale", scale_span)):
            latent_samples = latent_by_span[span]
            latent = np.concatenate(latent_samples[:split_samples] + latent_samples[split_samples:])
            arm, fused = evaluate_arm(
                base, certified, latent, split=cal_tokens,
                grid_size=grid_size, epsilon=epsilon,
            )
            arm["registered_replay_tokens"] = span
            actual_replay_tokens = [min(span, length) for length in source_token_lengths]
            arm["actual_replay_tokens_min"] = min(actual_replay_tokens)
            arm["actual_replay_tokens_mean"] = float(np.mean(actual_replay_tokens))
            arm["actual_replay_tokens_max"] = max(actual_replay_tokens)
            arm["latent_teacher_forced_top1_accuracy"] = float(
                np.concatenate(latent_top1[span][split_samples:]).mean()
            )
            latent_only, latent_fused = evaluate_arm(
                base, base, latent, split=cal_tokens,
                grid_size=grid_size, epsilon=epsilon,
            )
            certified_only, certified_fused = evaluate_arm(
                base, certified, base, split=cal_tokens,
                grid_size=grid_size, epsilon=epsilon,
            )
            arm["latent_only"] = latent_only
            arm["certified_only"] = certified_only
            task_result["arms"][name] = arm
            saved[f"{task}_{name}_gain"] = (
                np.log(np.clip(fused, epsilon, 1.0))
                - np.log(np.clip(base[cal_tokens:], epsilon, 1.0))
            )
            saved[f"{task}_{name}_latent_only_gain"] = (
                np.log(np.clip(latent_fused, epsilon, 1.0))
                - np.log(np.clip(base[cal_tokens:], epsilon, 1.0))
            )
            saved[f"{task}_{name}_certified_only_gain"] = (
                np.log(np.clip(certified_fused, epsilon, 1.0))
                - np.log(np.clip(base[cal_tokens:], epsilon, 1.0))
            )
        task_result["scale_minus_fixed"] = paired_summary(
            saved[f"{task}_ascent_scale_gain"] - saved[f"{task}_ascent_fixed_gain"]
        )
        task_result["latent_only_scale_minus_fixed"] = paired_summary(
            saved[f"{task}_ascent_scale_latent_only_gain"]
            - saved[f"{task}_ascent_fixed_latent_only_gain"]
        )
        all_scale_gains.append(saved[f"{task}_ascent_scale_gain"])
        all_latent_scale_gains.append(saved[f"{task}_ascent_scale_latent_only_gain"])
        results[task] = task_result

    arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays, **saved)
    payload = {
        "schema_version": 1,
        "status": raw["status"],
        "started_at_utc": started,
        "endpoint": endpoint,
        "config_sha256": sha256_file(config_path),
        "dataset_manifest_sha256": (
            sha256_file(manifest_path) if expected_manifest_hash is not None else None
        ),
        "tasks": results,
        "micro_token_pooled_ascent_scale_gain": paired_summary(np.concatenate(all_scale_gains)),
        "micro_token_pooled_latent_only_scale_gain": paired_summary(
            np.concatenate(all_latent_scale_gains)
        ),
        "matched_raw_boundary": "The causal exact parser is expected to be perfect; ASCENT cannot claim superiority to it on direct retrieval.",
        "metric_boundary": "Neural paths use teacher-forced answer-token NLL/top-1, not official autoregressive exact match.",
        "model_files": model_files,
        "runtime": {
            "hostname": platform.node(), "python": platform.python_version(),
            "numpy": np.__version__, "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "relative_layer_index": layer,
            "git_commit": git_value(["rev-parse", "HEAD"]),
            "git_dirty": bool(git_value(["status", "--porcelain"])),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "wall_seconds": time.perf_counter() - wall_start,
        },
        "arrays_path": str(arrays),
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
    parser.add_argument("--arrays", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.data_root, args.endpoint, args.model_dir, args.output, args.arrays)


if __name__ == "__main__":
    main()
