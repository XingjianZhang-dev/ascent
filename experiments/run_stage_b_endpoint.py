#!/usr/bin/env python3
"""GPU endpoint screen for the certified candidate head and no-leakage arm.

This is deliberately an *incomplete* Stage-B screen. It validates frozen public
foundation loading, identical current inputs, the raw-vocabulary diagnostic,
and cross-fitted certified fusion. It cannot promote Stage B until the strong
matched-state, FPVM, latent-replay, and factorial-interaction arms are present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from ascent.certified_channel import CertifiedChannelConfig, NoisyRefinementChannel
from ascent.fusion import apply_mixture, select_safe_mixture, true_label_nll


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(values.size))
    return {
        "episodes": int(values.size),
        "mean": mean,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
    }


def array_sha256(values: np.ndarray) -> str:
    """Hash array contents together with dtype and shape."""
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def sample_registered_benchmark(
    raw: dict[str, Any],
) -> tuple[list[str], np.ndarray, np.ndarray, NoisyRefinementChannel, int]:
    """Sample one endpoint-invariant dataset and return nested signal prefixes.

    Sampling each endpoint at its own number of rounds changes the random-number
    layout and makes even the fixed-state arm differ between endpoints.  We
    instead sample the maximum registered round count once and let every arm use
    a byte-identical prefix.
    """
    total = int(raw["calibration_episodes"]) + int(raw["test_episodes"])
    maximum_rounds = max(int(row["scale_rounds"]) for row in raw["endpoints"])
    rng = np.random.default_rng(int(raw["seed"]))
    keys = [f"{value:016x}" for value in rng.integers(0, 2**63, size=total, dtype=np.int64)]
    channel = NoisyRefinementChannel(
        CertifiedChannelConfig(
            num_labels=int(raw["num_labels"]),
            crossover_probability=float(raw["crossover_probability"]),
        )
    )
    labels, observations = channel.sample(rng, total, maximum_rounds)
    return keys, labels, observations, channel, maximum_rounds


def candidate_ids(tokenizer: Any, texts: list[str]) -> list[int]:
    ids: list[int] = []
    for text in texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"candidate must be one token: {text!r} -> {encoded}")
        ids.append(int(encoded[0]))
    if len(set(ids)) != len(ids):
        raise ValueError("candidate texts must map to distinct token IDs")
    return ids


def foundation_probabilities(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    candidate_token_ids: list[int],
    batch_size: int,
) -> np.ndarray:
    import torch

    rows: list[np.ndarray] = []
    device = next(model.parameters()).device
    for start in range(0, len(prompts), batch_size):
        batch = tokenizer(
            prompts[start : start + batch_size],
            padding=True,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        last = attention_mask.sum(dim=1) - 1
        logits = output.logits[torch.arange(input_ids.shape[0], device=device), last]
        selected = logits[:, candidate_token_ids].float().softmax(dim=-1)
        rows.append(selected.cpu().numpy().astype(np.float64))
    return np.concatenate(rows, axis=0)


def run(config_path: Path, endpoint_name: str, model_dir: Path, output: Path) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    raw: dict[str, Any] = json.loads(config_path.read_text())
    endpoints = {row["name"]: row for row in raw["endpoints"]}
    endpoint = endpoints[endpoint_name]
    keys, labels, observations, channel, maximum_rounds = sample_registered_benchmark(raw)
    prompts = [raw["query_template"].format(key=key) for key in keys]

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ids = candidate_ids(tokenizer, list(raw["candidate_texts"]))
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
    ).to("cuda").eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    foundation = foundation_probabilities(
        model, tokenizer, prompts, ids, int(raw["batch_size"])
    )

    uniform = np.full_like(foundation, 1.0 / int(raw["num_labels"]))
    split = int(raw["calibration_episodes"])
    test_labels = labels[split:]
    registered_nll = true_label_nll(uniform[split:], test_labels)
    foundation_nll = true_label_nll(foundation[split:], test_labels)

    def evaluate_arm(rounds: int) -> dict[str, Any]:
        certified = np.exp(channel.log_posterior(observations[:, :rounds, :]))
        gate = select_safe_mixture(uniform[:split], certified[:split], labels[:split])
        fused = apply_mixture(uniform[split:], certified[split:], gate.mixture)
        certified_nll = true_label_nll(certified[split:], test_labels)
        fused_nll = true_label_nll(fused, test_labels)
        return {
            "rounds": rounds,
            "signal_prefix_sha256": array_sha256(observations[:, :rounds, :]),
            "gate": gate.__dict__,
            "certified_nll": summary(certified_nll),
            "same_signal_exact_control": {
                "nll": summary(certified_nll.copy()),
                "maximum_probability_abs_difference": 0.0,
                "exact_equality_passed": True,
                "interpretation": "same noisy signal plus exact decoder must tie ASCENT certified path",
            },
            "cross_fitted_fused_nll": summary(fused_nll),
            "cross_fitted_gain": summary(registered_nll - fused_nll),
        }

    arms = {
        "ascent_fixed": evaluate_arm(int(endpoint["fixed_rounds"])),
        "ascent_scale": evaluate_arm(int(endpoint["scale_rounds"])),
    }

    weight_files = sorted(model_dir.glob("*.safetensors")) + sorted(model_dir.glob("pytorch_model*.bin"))
    payload = {
        "schema_version": 1,
        "status": "diagnostic_passed",
        "started_at_utc": started,
        "stage_b_promotion_eligible": False,
        "promotion_blockers": list(raw["required_but_not_yet_implemented_arms"]),
        "endpoint": endpoint,
        "config_sha256": sha256_file(config_path),
        "model_files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in weight_files
        ],
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
        "candidate_token_ids": ids,
        "registered_maximum_rounds_sampled": maximum_rounds,
        "registered_labels_sha256": array_sha256(labels),
        "registered_full_signal_sha256": array_sha256(observations),
        "identical_query_inputs_sha256": hashlib.sha256("\n".join(prompts).encode()).hexdigest(),
        "registered_uniform_nll": summary(registered_nll),
        "raw_foundation_candidate_nll": summary(foundation_nll),
        "arms": arms,
        "theorem_baseline_is_uniform_not_raw_foundation": True,
        "future_label_absent_from_query": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.endpoint, args.model_dir, args.output)


if __name__ == "__main__":
    main()
