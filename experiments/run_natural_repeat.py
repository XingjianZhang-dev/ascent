#!/usr/bin/env python3
"""Direct ASCENT evaluation on causal repeated contexts in natural text."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ascent.fusion import (
    SimplexGateSelection,
    apply_true_probability_simplex,
    select_safe_true_probability_simplex,
)
from ascent.gpt_neox_replay import relative_layer_index, replay_from_hidden
from ascent.natural_repeat import collect_natural_repeat_events, persistent_state_bytes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def git_value(arguments: list[str]) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=False, capture_output=True, text=True
    ).stdout.strip()


def paired_summary(gains: np.ndarray) -> dict[str, float]:
    values = np.asarray(gains, dtype=np.float64)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else float("nan")
    return {
        "mean_nats": mean,
        "standard_error": se,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
    }


def probability_summary(probabilities: np.ndarray, *, epsilon: float) -> dict[str, float]:
    losses = -np.log(np.clip(probabilities, epsilon, 1.0))
    return {
        "mean_nll": float(losses.mean()),
        "perplexity": float(np.exp(min(50.0, losses.mean()))),
        "mean_true_probability": float(probabilities.mean()),
    }


def gate_payload(gate: SimplexGateSelection) -> dict[str, Any]:
    return {
        "enabled": gate.enabled,
        "foundation_weight": 1.0 - gate.certified_weight - gate.latent_weight,
        "certified_weight": gate.certified_weight,
        "latent_weight": gate.latent_weight,
        "calibration_gain_nats": gate.calibration_gain_nats,
        "calibration_gain_lcb95": gate.calibration_gain_lcb95,
    }


def evaluate_arm(
    foundation: np.ndarray,
    certified: np.ndarray,
    latent: np.ndarray,
    *,
    split: int,
    grid_size: int,
    epsilon: float,
) -> tuple[dict[str, Any], np.ndarray]:
    gate = select_safe_true_probability_simplex(
        foundation[:split], certified[:split], latent[:split],
        grid_size=grid_size, epsilon=epsilon,
    )
    fused = apply_true_probability_simplex(
        foundation[split:], certified[split:], latent[split:],
        gate.certified_weight, gate.latent_weight,
    )
    base = foundation[split:]
    gains = np.log(np.clip(fused, epsilon, 1.0)) - np.log(np.clip(base, epsilon, 1.0))
    return {
        "gate": gate_payload(gate),
        "test": probability_summary(fused, epsilon=epsilon),
        "paired_nll_gain": paired_summary(gains),
    }, fused


def true_token_probabilities(logits: Any, targets: Any) -> np.ndarray:
    import torch

    probabilities = logits.float().softmax(dim=-1)
    selected = probabilities.gather(1, targets[:, None]).squeeze(1)
    return selected.detach().cpu().numpy().astype(np.float64)


def run(
    config_path: Path,
    dataset_name: str,
    dataset_path: Path,
    endpoint_name: str,
    model_dir: Path,
    output: Path,
    arrays: Path,
) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    raw: dict[str, Any] = json.loads(config_path.read_text())
    dataset = raw["datasets"][dataset_name]
    endpoint = {row["name"]: row for row in raw["endpoints"]}[endpoint_name]
    observed_dataset_hash = sha256_file(dataset_path)
    if observed_dataset_hash != dataset["sha256"]:
        raise RuntimeError(
            f"dataset hash mismatch: expected {dataset['sha256']}, got {observed_dataset_hash}"
        )
    started = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    total = int(raw["calibration_events"]) + int(raw["test_events"])
    split = int(raw["calibration_events"])
    tokens = np.memmap(dataset_path, mode="r", dtype=np.dtype(dataset["dtype"]))
    events = collect_natural_repeat_events(
        tokens,
        event_offset=int(raw["event_offset"]),
        num_events=total,
        key_length=int(raw["key_length"]),
        context_length=int(raw["context_length"]),
        source_tokens=int(raw["maximum_source_tokens"]),
        min_distance=int(raw["min_distance"]),
        memory_slots=int(raw["memory_slots"]),
        eos_token_id=0,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.eos_token_id != 0:
        raise RuntimeError(f"expected GPT-NeoX EOS token 0, got {tokenizer.eos_token_id}")
    if int(max(events.targets.max(), events.query_contexts.max())) >= len(tokenizer):
        raise RuntimeError("dataset contains a token outside the registered tokenizer vocabulary")
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.bfloat16
    ).to("cuda").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layer = relative_layer_index(
        len(model.gpt_neox.layers), float(raw["relative_injection_depth"])
    )
    fixed_span = int(raw["fixed_source_tokens"])
    scale_span = int(endpoint["scale_source_tokens"])
    spans = sorted({fixed_span, scale_span})
    batch_size = int(endpoint["batch_size"])
    foundation_rows: list[np.ndarray] = []
    latent_rows: dict[int, list[np.ndarray]] = {span: [] for span in spans}

    for start in range(0, total, batch_size):
        stop = min(total, start + batch_size)
        query_ids = torch.as_tensor(
            events.query_contexts[start:stop].astype(np.int64), device="cuda"
        )
        targets = torch.as_tensor(
            events.targets[start:stop].astype(np.int64), device="cuda"
        )
        query_mask = torch.ones_like(query_ids)
        with torch.inference_mode():
            query_output = model(
                input_ids=query_ids,
                attention_mask=query_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            foundation_rows.append(true_token_probabilities(query_output.logits[:, -1], targets))
            query_hidden = query_output.hidden_states[layer]
            for span in spans:
                source_ids = torch.as_tensor(
                    events.source_fragments[start:stop, -span:].astype(np.int64),
                    device="cuda",
                )
                source_output = model(
                    input_ids=source_ids,
                    attention_mask=torch.ones_like(source_ids),
                    output_hidden_states=True,
                    use_cache=False,
                )
                replay = source_output.hidden_states[layer]
                hidden = replay_from_hidden(model, query_hidden, query_mask, replay, layer)
                replay_logits = model.embed_out(hidden)[:, span + query_ids.shape[1] - 1]
                latent_rows[span].append(true_token_probabilities(replay_logits, targets))
                del source_output, hidden, replay_logits
        del query_output

    foundation = np.concatenate(foundation_rows)
    latent = {span: np.concatenate(rows) for span, rows in latent_rows.items()}
    certified = (events.retrieved_targets == events.targets).astype(np.float64)
    epsilon = float(raw["epsilon"])
    grid_size = int(raw["fusion_grid_size"])
    foundation_test = foundation[split:]
    base_nll = -np.log(np.clip(foundation_test, epsilon, 1.0))
    arms: dict[str, Any] = {}
    saved: dict[str, np.ndarray] = {
        "foundation_true_probability": foundation_test,
        "foundation_nll": base_nll,
        "targets": events.targets[split:],
        "retrieved_targets": events.retrieved_targets[split:],
        "query_positions": events.query_positions[split:],
        "source_positions": events.source_positions[split:],
    }
    for name, span in (("ascent_fixed", fixed_span), ("ascent_scale", scale_span)):
        result, fused = evaluate_arm(
            foundation, certified, latent[span], split=split,
            grid_size=grid_size, epsilon=epsilon,
        )
        result["source_tokens"] = span
        result["packed_persistent_state_bytes"] = persistent_state_bytes(
            memory_slots=int(raw["memory_slots"]),
            key_length=int(raw["key_length"]),
            source_tokens=span,
        )
        arms[name] = result
        saved[f"{name}_fused_true_probability"] = fused
        saved[f"{name}_nll_gain"] = (
            np.log(np.clip(fused, epsilon, 1.0))
            - np.log(np.clip(foundation_test, epsilon, 1.0))
        )

    cert_result, cert_fused = evaluate_arm(
        foundation, certified, foundation, split=split,
        grid_size=grid_size, epsilon=epsilon,
    )
    latent_result, latent_fused = evaluate_arm(
        foundation, foundation, latent[scale_span], split=split,
        grid_size=grid_size, epsilon=epsilon,
    )
    saved["certified_only_fused_true_probability"] = cert_fused
    saved["latent_only_fused_true_probability"] = latent_fused
    arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays, **saved)

    fixed_gain = saved["ascent_fixed_nll_gain"]
    scale_gain = saved["ascent_scale_nll_gain"]
    weights = sorted(model_dir.glob("*.safetensors")) + sorted(model_dir.glob("pytorch_model*.bin"))
    payload = {
        "schema_version": 1,
        "status": raw["status"],
        "started_at_utc": started,
        "dataset": {
            "name": dataset_name,
            "path": str(dataset_path),
            "bytes": dataset_path.stat().st_size,
            "sha256": observed_dataset_hash,
            "tokens": int(tokens.size),
        },
        "endpoint": endpoint,
        "config_sha256": sha256_file(config_path),
        "event_protocol": {
            "event_offset": int(raw["event_offset"]),
            "calibration_events": split,
            "test_events": int(raw["test_events"]),
            "key_length": int(raw["key_length"]),
            "context_length": int(raw["context_length"]),
            "min_distance": int(raw["min_distance"]),
            "memory_slots": int(raw["memory_slots"]),
            "read_before_write": True,
            "document_boundary_reset": True,
            "target_absent_from_query": True,
            "hits_seen": events.hits_seen,
            "tokens_scanned": events.tokens_scanned,
            "event_identity_sha256": array_sha256(np.stack([
                events.source_positions, events.query_positions,
                events.targets.astype(np.int64), events.retrieved_targets.astype(np.int64),
            ], axis=1)),
        },
        "foundation": probability_summary(foundation_test, epsilon=epsilon),
        "matched_raw_exact_retrieval": {
            "top1_accuracy": float(certified[split:].mean()),
            "mean_true_probability": float(certified[split:].mean()),
            "note": "Exact retrieved next token; zeros are clipped only when computing log loss.",
        },
        "arms": arms,
        "ablations": {
            "certified_only": cert_result,
            "latent_only_scale_span": latent_result,
        },
        "scale_minus_fixed_paired_gain": paired_summary(scale_gain - fixed_gain),
        "model_files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in weights
        ],
        "runtime": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "relative_layer_index": layer,
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
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.config, args.dataset, args.dataset_path, args.endpoint,
        args.model_dir, args.output, args.arrays,
    )


if __name__ == "__main__":
    main()
