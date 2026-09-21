#!/usr/bin/env python3
"""Two-fact addition diagnostic with latent, raw-text, and oracle controls."""

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

from ascent.certified_channel import CertifiedChannelConfig, NoisyRefinementChannel
from ascent.fusion import apply_mixture, select_safe_mixture, true_label_nll
from ascent.gpt_neox_replay import relative_layer_index, replay_from_hidden
from experiments.run_stage_b_endpoint import (
    array_sha256,
    candidate_ids,
    foundation_probabilities,
    sha256_file,
    summary,
)
from experiments.run_ruler_niah import verify_model_artifact


def git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sum_posterior(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Convolve independent categorical posteriors along their label axis."""
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("value posteriors must have equal [episodes, values] shapes")
    result = np.zeros((first.shape[0], first.shape[1] * 2 - 1), dtype=np.float64)
    for left in range(first.shape[1]):
        for right in range(second.shape[1]):
            result[:, left + right] += first[:, left] * second[:, right]
    return result


def candidate_probabilities(logits: Any, ids: list[int]) -> np.ndarray:
    return logits[:, ids].float().softmax(dim=-1).cpu().numpy().astype(np.float64)


def extract_codebook(model: Any, tokenizer: Any, prompts: list[str], layer: int) -> Any:
    import torch

    batch = tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")
    lengths = batch["attention_mask"].sum(dim=1)
    if not torch.all(lengths == lengths[0]):
        raise ValueError("role codebook prompts must tokenize to equal lengths")
    with torch.inference_mode():
        output = model(**batch, output_hidden_states=True, use_cache=False)
    return output.hidden_states[layer][:, : int(lengths[0].item())].detach()


def evaluate_path(
    base: np.ndarray,
    path: np.ndarray,
    labels: np.ndarray,
    split: int,
) -> tuple[dict[str, Any], np.ndarray]:
    gate = select_safe_mixture(base[:split], path[:split], labels[:split])
    fused = apply_mixture(base[split:], path[split:], gate.mixture)
    base_nll = true_label_nll(base[split:], labels[split:])
    path_nll = true_label_nll(path[split:], labels[split:])
    fused_nll = true_label_nll(fused, labels[split:])
    return {
        "gate": gate.__dict__,
        "path_only_nll": summary(path_nll),
        "cross_fitted_fused_nll": summary(fused_nll),
        "cross_fitted_gain_over_foundation": summary(base_nll - fused_nll),
    }, fused_nll


def run(
    config_path: Path,
    endpoint_name: str,
    model_dir: Path,
    output: Path,
    arrays: Path,
    seed_override: int | None = None,
) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    raw: dict[str, Any] = json.loads(config_path.read_text())
    endpoint = {row["name"]: row for row in raw["endpoints"]}[endpoint_name]
    model_files = verify_model_artifact(model_dir, endpoint.get("model_artifact"))
    started = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    if seed_override is None:
        effective_seed = int(raw["seed"])
    else:
        registered_seeds = [int(value) for value in raw.get("seeds", [])]
        if seed_override not in registered_seeds:
            raise ValueError("seed override is not present in the frozen config")
        effective_seed = seed_override
    total = int(raw["calibration_episodes"]) + int(raw["test_episodes"])
    split = int(raw["calibration_episodes"])
    rng = np.random.default_rng(effective_seed)
    keys = [f"{value:016x}" for value in rng.integers(0, 2**63, size=total, dtype=np.int64)]
    queries = [raw["query_template"].format(key=key) for key in keys]
    channel = NoisyRefinementChannel(
        CertifiedChannelConfig(int(raw["num_values"]), float(raw["crossover_probability"]))
    )
    maximum_rounds = int(raw.get("maximum_rounds", raw["rich_rounds"]))
    first, first_signal = channel.sample(rng, total, maximum_rounds)
    second, second_signal = channel.sample(rng, total, maximum_rounds)
    answers = first + second

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    value_ids = candidate_ids(tokenizer, list(raw["value_texts"]))
    answer_ids = candidate_ids(tokenizer, list(raw["answer_texts"]))
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.bfloat16
    ).to("cuda").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layer = relative_layer_index(
        len(model.gpt_neox.layers), float(raw["relative_injection_depth"])
    )
    first_codebook = extract_codebook(
        model, tokenizer,
        [raw["first_codebook_template"].format(value=value) for value in raw["value_texts"]],
        layer,
    )
    second_codebook = extract_codebook(
        model, tokenizer,
        [raw["second_codebook_template"].format(value=value) for value in raw["value_texts"]],
        layer,
    )

    arm_rounds = {
        "fixed": int(raw["fixed_rounds"]),
        "rich": int(endpoint.get("scale_rounds", raw["rich_rounds"])),
    }
    if not all(0 < rounds <= maximum_rounds for rounds in arm_rounds.values()):
        raise ValueError("every arm round count must be in [1, maximum_rounds]")
    posterior_by_arm: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, rounds in arm_rounds.items():
        posterior_by_arm[name] = (
            np.exp(channel.log_posterior(first_signal[:, :rounds, :])),
            np.exp(channel.log_posterior(second_signal[:, :rounds, :])),
        )

    base_rows: list[np.ndarray] = []
    latent_rows: dict[str, list[np.ndarray]] = {"fixed": [], "rich": []}
    for start in range(0, total, int(raw["batch_size"])):
        stop = min(total, start + int(raw["batch_size"]))
        batch = tokenizer(queries[start:stop], padding=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            query_output = model(**batch, output_hidden_states=True, use_cache=False)
            last = batch["attention_mask"].sum(dim=1) - 1
            base_logits = query_output.logits[torch.arange(stop - start, device="cuda"), last]
            base_rows.append(candidate_probabilities(base_logits, answer_ids))
            query_hidden = query_output.hidden_states[layer]
            for name, (first_p, second_p) in posterior_by_arm.items():
                first_map = torch.as_tensor(first_p[start:stop].argmax(axis=1), device="cuda")
                second_map = torch.as_tensor(second_p[start:stop].argmax(axis=1), device="cuda")
                replay = torch.cat(
                    [first_codebook[first_map], second_codebook[second_map]], dim=1
                )
                hidden = replay_from_hidden(
                    model, query_hidden, batch["attention_mask"], replay, layer
                )
                logits = model.embed_out(hidden)[
                    torch.arange(stop - start, device="cuda"), replay.shape[1] + last
                ]
                latent_rows[name].append(candidate_probabilities(logits, answer_ids))
        del query_output

    base = np.concatenate(base_rows)
    latent = {name: np.concatenate(rows) for name, rows in latent_rows.items()}
    text: dict[str, np.ndarray] = {}
    exact: dict[str, np.ndarray] = {}
    for name, (first_p, second_p) in posterior_by_arm.items():
        first_map = first_p.argmax(axis=1)
        second_map = second_p.argmax(axis=1)
        prompts = [
            raw["text_replay_template"].format(
                first=raw["value_texts"][int(a)],
                second=raw["value_texts"][int(b)],
                query=query,
            )
            for a, b, query in zip(first_map, second_map, queries, strict=True)
        ]
        text[name] = foundation_probabilities(
            model, tokenizer, prompts, answer_ids, int(raw["batch_size"])
        )
        exact[name] = sum_posterior(first_p, second_p)
    clean_prompts = [
        raw["text_replay_template"].format(
            first=raw["value_texts"][int(a)], second=raw["value_texts"][int(b)], query=query
        )
        for a, b, query in zip(first, second, queries, strict=True)
    ]
    clean_text = foundation_probabilities(
        model, tokenizer, clean_prompts, answer_ids, int(raw["batch_size"])
    )

    base_nll = true_label_nll(base[split:], answers[split:])
    result_arms: dict[str, Any] = {}
    saved: dict[str, np.ndarray] = {"answers": answers[split:], "base_nll": base_nll}
    for name in ("fixed", "rich"):
        latent_result, latent_fused_nll = evaluate_path(base, latent[name], answers, split)
        text_result, text_fused_nll = evaluate_path(base, text[name], answers, split)
        exact_nll = true_label_nll(exact[name][split:], answers[split:])
        saved[f"{name}_latent_fused_nll"] = latent_fused_nll
        saved[f"{name}_text_fused_nll"] = text_fused_nll
        result_arms[name] = {
            "rounds_per_fact": arm_rounds[name],
            "persistent_signal_bits_per_episode": 2 * arm_rounds[name] * int(raw["num_values"]),
            "latent_replay": latent_result,
            "same_signal_map_text_replay": text_result,
            "same_signal_exact_sum_posterior_nll": summary(exact_nll),
            "signal_sha256": array_sha256(
                np.stack([
                    first_signal[:, : arm_rounds[name], :],
                    second_signal[:, : arm_rounds[name], :],
                ], axis=1)
            ),
        }
    clean_text_result, clean_text_fused_nll = evaluate_path(
        base, clean_text, answers, split
    )
    saved["clean_text_fused_nll"] = clean_text_fused_nll
    saved["fixed_to_rich_latent_improvement"] = (
        saved["fixed_latent_fused_nll"] - saved["rich_latent_fused_nll"]
    )
    arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays, **saved)

    prior_counts = np.concatenate([
        np.arange(1, int(raw["num_values"]) + 1),
        np.arange(int(raw["num_values"]) - 1, 0, -1),
    ]).astype(np.float64)
    prior = np.broadcast_to(prior_counts / prior_counts.sum(), (total, prior_counts.size))
    prior_nll = true_label_nll(prior[split:], answers[split:])
    payload = {
        "schema_version": 1, "status": raw["status"], "started_at_utc": started,
        "endpoint": endpoint, "config_sha256": sha256_file(config_path),
        "effective_seed": effective_seed, "relative_layer_index": layer,
        "num_model_layers": len(model.gpt_neox.layers),
        "registered_answers_sha256": array_sha256(answers),
        "registered_full_signal_sha256": array_sha256(np.stack([first_signal, second_signal], axis=1)),
        "identical_query_inputs_sha256": hashlib.sha256("\n".join(queries).encode()).hexdigest(),
        "foundation_nll": summary(base_nll), "bayes_no_memory_prior_nll": summary(prior_nll),
        "arms": result_arms,
        "clean_kv_text_replay": clean_text_result,
        "clean_kv_oracle_nll": 0.0,
        "clean_kv_oracle_dominates_every_nonperfect_method": True,
        "future_values_absent_from_query": True,
        "model_files": model_files,
        "runtime": {
            "hostname": platform.node(), "python": platform.python_version(),
            "numpy": np.__version__, "torch": torch.__version__, "transformers": transformers.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
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
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    run(args.config, args.endpoint, args.model_dir, args.output, args.arrays, args.seed)


if __name__ == "__main__":
    main()
