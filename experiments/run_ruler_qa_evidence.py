#!/usr/bin/env python3
"""Official RULER QA with query-conditioned ASCENT evidence replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from ascent.ruler_metrics import string_match_part
from ascent.ruler_qa_memory import read_ruler_qa_graph, serialize_documents
from experiments.run_natural_repeat import git_value, paired_summary, sha256_file
from experiments.run_ruler_autoregressive import score_summary
from experiments.run_ruler_niah import verify_model_artifact
from experiments.run_stage_b_endpoint import array_sha256


def safe_nll_gate(base_nll: np.ndarray, path_nll: np.ndarray) -> dict[str, Any]:
    """Enable a frozen path only when its paired calibration NLL LCB is positive."""

    if base_nll.shape != path_nll.shape or base_nll.size < 2:
        raise ValueError("calibration NLL arrays must align and be nontrivial")
    gain = base_nll - path_nll
    mean = float(gain.mean())
    standard_error = float(gain.std(ddof=1) / math.sqrt(gain.size))
    low = mean - 1.96 * standard_error
    high = mean + 1.96 * standard_error
    return {
        "samples": int(gain.size),
        "mean_gain_nats": mean,
        "standard_error": standard_error,
        "ci95_low": low,
        "ci95_high": high,
        "enabled": bool(low > 0.0),
    }


def remaining_error_elimination(foundation: float, ascent: float) -> float | None:
    """Measure removed Foundation error without demanding accuracy above one."""

    if foundation >= 1.0:
        return None
    return (ascent - foundation) / (1.0 - foundation)


def evidence_content(source: str) -> str:
    return (
        "\n\nASCENT query-conditioned memory read. These documents were written "
        "before the question and retrieved without using the answer:\n\n"
        + source
        + "\n\nReturn only the answer to the original question."
    )


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
    endpoint = {row["name"]: row for row in raw["endpoints"]}[endpoint_name]
    task_path = data_root / str(raw["task_name"]) / "validation.jsonl"
    if sha256_file(task_path) != raw["task_sha256"]:
        raise RuntimeError("RULER QA task hash mismatch")
    rows = [json.loads(line) for line in task_path.read_text().splitlines()]
    if len(rows) != int(raw["samples_per_task"]):
        raise RuntimeError("unexpected RULER QA row count")
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
    context_limit = int(raw["query_context_tokens"])
    fixed_documents = int(raw["fixed_documents"])
    scale_documents = int(endpoint["scale_documents"])
    if not 0 < fixed_documents < scale_documents:
        raise ValueError("scale document count must strictly exceed fixed")
    split = int(raw["calibration_samples"])
    if not 1 < split < len(rows):
        raise ValueError("calibration split must leave test rows")
    max_new_tokens = int(raw["max_new_tokens"])

    prepared: list[dict[str, Any]] = []
    for row in rows:
        read = read_ruler_qa_graph(
            row["input"], seed_documents=int(raw["bm25_seed_documents"])
        )
        sources = {
            "fixed": serialize_documents(read.graph_ranked[:fixed_documents]),
            "scale": serialize_documents(read.graph_ranked[:scale_documents]),
            "query_only": serialize_documents(
                read.query_only_ranked[:scale_documents]
            ),
            "irrelevant": serialize_documents(
                read.irrelevant_ranked[:scale_documents]
            ),
        }
        prepared.append({"row": row, "read": read, "sources": sources})

    def prompt_ids(row: dict[str, Any], source: str | None) -> tuple[Any, int]:
        content = row["input"] if source is None else row["input"] + evidence_content(source)
        user_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )[0]
        prefix_ids = tokenizer(
            row["answer_prefix"], add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]
        full = torch.cat([user_ids, prefix_ids])
        if int(full.numel()) > context_limit:
            raise RuntimeError("native QA prompt exceeded the frozen context limit")
        return full.to("cuda"), int(full.numel())

    def target_ids(row: dict[str, Any]) -> Any:
        if len(row["outputs"]) != 1:
            raise RuntimeError("the frozen HotpotQA panel requires one answer per row")
        return tokenizer(
            " " + row["outputs"][0],
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0].to("cuda")

    def teacher_nll(ids: Any, target: Any) -> float:
        query = torch.cat([ids, target[:-1]]).unsqueeze(0)
        with torch.inference_mode():
            result = model(
                input_ids=query,
                attention_mask=torch.ones_like(query),
                use_cache=False,
            )
        start = int(ids.numel()) - 1
        logits = result.logits[0, start : start + int(target.numel())].float()
        losses = torch.nn.functional.cross_entropy(
            logits, target, reduction="none"
        )
        return float(losses.mean().item())

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
        with torch.inference_mode():
            result = model(
                input_ids=query,
                attention_mask=torch.ones_like(query),
                use_cache=True,
            )
        cache = result.past_key_values
        logits = result.logits[0, -1].float()
        for step in range(max_new_tokens):
            next_id = int(torch.argmax(logits).item())
            generated.append(next_id)
            if next_id in eos_ids or step + 1 == max_new_tokens:
                break
            next_token = torch.tensor([[next_id]], dtype=ids.dtype, device="cuda")
            attention_mask = torch.ones(
                (1, int(ids.numel()) + len(generated)),
                dtype=ids.dtype,
                device="cuda",
            )
            with torch.inference_mode():
                result = model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=cache,
                    use_cache=True,
                )
            cache = result.past_key_values
            logits = result.logits[0, -1].float()
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    arm_names = ("fixed", "scale", "query_only", "irrelevant")
    calibration_base: list[float] = []
    calibration_paths: dict[str, list[float]] = {name: [] for name in arm_names}
    prompt_lengths: dict[str, list[int]] = {
        "foundation": [],
        **{name: [] for name in arm_names},
    }
    source_token_counts: dict[str, list[int]] = {name: [] for name in arm_names}
    source_byte_counts: dict[str, list[int]] = {name: [] for name in arm_names}

    for item in prepared[:split]:
        row = item["row"]
        target = target_ids(row)
        foundation_ids, length = prompt_ids(row, None)
        prompt_lengths["foundation"].append(length)
        calibration_base.append(teacher_nll(foundation_ids, target))
        for name in arm_names:
            source = item["sources"][name]
            ids, length = prompt_ids(row, source)
            prompt_lengths[name].append(length)
            source_token_counts[name].append(
                len(tokenizer(source, add_special_tokens=False)["input_ids"])
            )
            source_byte_counts[name].append(len(source.encode()))
            calibration_paths[name].append(teacher_nll(ids, target))
    base_cal = np.asarray(calibration_base, dtype=np.float64)
    gates = {
        name: safe_nll_gate(base_cal, np.asarray(values, dtype=np.float64))
        for name, values in calibration_paths.items()
    }

    base_nll: list[float] = []
    path_nll: dict[str, list[float]] = {name: [] for name in arm_names}
    foundation_scores: list[float] = []
    path_scores: dict[str, list[float]] = {name: [] for name in arm_names}
    predictions: list[dict[str, Any]] = []
    persistent_state_bytes: list[int] = []
    ranking_hashes: list[str] = []
    for item in prepared[split:]:
        row = item["row"]
        read = item["read"]
        target = target_ids(row)
        foundation_ids, length = prompt_ids(row, None)
        prompt_lengths["foundation"].append(length)
        foundation_loss = teacher_nll(foundation_ids, target)
        base_nll.append(foundation_loss)
        foundation_prediction = greedy_decode(foundation_ids)
        foundation_score = string_match_part(
            foundation_prediction, row["outputs"]
        )
        foundation_scores.append(foundation_score)
        row_predictions: dict[str, Any] = {
            "index": row["index"],
            "outputs": row["outputs"],
            "foundation_prediction": foundation_prediction,
            "foundation_score": foundation_score,
        }
        ranking_hashes.append(
            hashlib.sha256(
                "\n".join(str(document.number) for document in read.graph_ranked).encode()
            ).hexdigest()
        )
        persistent_state_bytes.append(
            len(serialize_documents(read.documents).encode())
        )
        for name in arm_names:
            source = item["sources"][name]
            ids, length = prompt_ids(row, source)
            prompt_lengths[name].append(length)
            source_token_counts[name].append(
                len(tokenizer(source, add_special_tokens=False)["input_ids"])
            )
            source_byte_counts[name].append(len(source.encode()))
            loss = teacher_nll(ids, target)
            path_nll[name].append(loss)
            if gates[name]["enabled"]:
                prediction = greedy_decode(ids)
                score = string_match_part(prediction, row["outputs"])
            else:
                prediction = foundation_prediction
                score = foundation_score
            path_scores[name].append(score)
            row_predictions[f"{name}_prediction"] = prediction
            row_predictions[f"{name}_score"] = score
        predictions.append(row_predictions)

    base_array = np.asarray(base_nll, dtype=np.float64)
    path_arrays = {
        name: np.asarray(values, dtype=np.float64) for name, values in path_nll.items()
    }
    foundation_score_array = np.asarray(foundation_scores, dtype=np.float64)
    path_score_arrays = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in path_scores.items()
    }
    saved: dict[str, np.ndarray] = {
        "base_nll": base_array,
        "foundation_score": foundation_score_array,
    }
    for name in arm_names:
        saved[f"{name}_nll"] = path_arrays[name]
        saved[f"{name}_score"] = path_score_arrays[name]
    arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays, **saved)

    arms: dict[str, Any] = {}
    for name in arm_names:
        nll_gain = base_array - path_arrays[name]
        score_gain = path_score_arrays[name] - foundation_score_array
        foundation_mean_score = float(foundation_score_array.mean())
        path_mean_score = float(path_score_arrays[name].mean())
        arms[name] = {
            "gate": gates[name],
            "teacher_forced_path_nll": paired_summary(path_arrays[name]),
            "paired_nll_gain_over_foundation": paired_summary(nll_gain),
            "official_score": score_summary(path_score_arrays[name].tolist()),
            "paired_official_score_gain": paired_summary(score_gain),
            "wins": int(np.sum(score_gain > 0.0)),
            "regressions": int(np.sum(score_gain < 0.0)),
            "remaining_error_elimination": remaining_error_elimination(
                foundation_mean_score, path_mean_score
            ),
            "minimum_source_tokens": min(source_token_counts[name]),
            "maximum_source_tokens": max(source_token_counts[name]),
            "mean_source_bytes": float(np.mean(source_byte_counts[name])),
        }
    model_parameters = int(endpoint["model_parameters"])
    payload = {
        "schema_version": 1,
        "status": raw["status"],
        "started_at_utc": started,
        "task_name": raw["task_name"],
        "task_sha256": sha256_file(task_path),
        "config_sha256": sha256_file(config_path),
        "endpoint": endpoint,
        "metric": "NVIDIA RULER string_match_part fraction",
        "ceiling_policy": {
            "absolute_accuracy_bounded_at_one": True,
            "secondary_estimand": "(ASCENT - Foundation) / (1 - Foundation)",
            "perfect_ascent_interpretation": "100% of remaining Foundation error eliminated",
            "continuous_primary": "paired teacher-forced answer NLL gain",
        },
        "architecture": "query_bm25_title_graph_nested_evidence_replay",
        "calibration_samples": split,
        "test_samples": len(rows) - split,
        "foundation_teacher_forced_nll": paired_summary(base_array),
        "foundation_score": score_summary(foundation_scores),
        "arms": arms,
        "fixed_documents": fixed_documents,
        "scale_documents": scale_documents,
        "rankings_sha256": hashlib.sha256(
            "\n".join(ranking_hashes).encode()
        ).hexdigest(),
        "nested_document_prefix": True,
        "answer_and_support_labels_absent_from_reader": True,
        "native_context": True,
        "prompt_lengths": {
            name: {"minimum": min(values), "maximum": max(values)}
            for name, values in prompt_lengths.items()
        },
        "persistent_state": {
            "mean_raw_document_bytes": float(np.mean(persistent_state_bytes)),
            "maximum_raw_document_bytes": max(persistent_state_bytes),
            "mean_state_bytes_per_model_parameter": float(
                np.mean(persistent_state_bytes) / model_parameters
            ),
        },
        "arrays_sha256": {
            name: array_sha256(value) for name, value in saved.items()
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
    run(
        args.config,
        args.data_root,
        args.endpoint,
        args.model_dir,
        args.output,
        args.arrays,
    )


if __name__ == "__main__":
    main()
