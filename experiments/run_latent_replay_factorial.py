#!/usr/bin/env python3
"""Development screen for foundation-by-memory latent-replay interaction."""

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


def normalized_candidates(logits: Any, ids: list[int]) -> np.ndarray:
    return logits[:, ids].float().softmax(dim=-1).cpu().numpy().astype(np.float64)


def run(
    config_path: Path,
    endpoint_name: str,
    model_dir: Path,
    output: Path,
    arrays: Path,
    relative_depth_override: float | None = None,
    replay_mode_override: str | None = None,
    seed_override: int | None = None,
) -> None:
    import torch
    import torch.nn.functional as functional
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    raw: dict[str, Any] = json.loads(config_path.read_text())
    endpoints = {row["name"]: row for row in raw["endpoints"]}
    endpoint = endpoints[endpoint_name]
    started = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    total = int(raw["calibration_episodes"]) + int(raw["test_episodes"])
    split = int(raw["calibration_episodes"])
    if seed_override is None:
        seed = int(raw["seed"])
    else:
        registered_seeds = [int(value) for value in raw.get("seeds", [])]
        if seed_override not in registered_seeds:
            raise ValueError(f"seed {seed_override} is not registered in the config")
        seed = int(seed_override)
    rng = np.random.default_rng(seed)
    keys = [f"{value:016x}" for value in rng.integers(0, 2**63, size=total, dtype=np.int64)]
    prompts = [raw["query_template"].format(key=key) for key in keys]
    channel = NoisyRefinementChannel(
        CertifiedChannelConfig(int(raw["num_labels"]), float(raw["crossover_probability"]))
    )
    labels, observations = channel.sample(rng, total, int(raw["rich_rounds"]))

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ids = candidate_ids(tokenizer, list(raw["candidate_texts"]))
    if "model_artifact" in endpoint:
        model_files = verify_model_artifact(model_dir, endpoint["model_artifact"])
    else:
        weights = sorted(model_dir.glob("*.safetensors")) + sorted(
            model_dir.glob("pytorch_model*.bin")
        )
        model_files = [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in weights
        ]
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.bfloat16
    ).to("cuda").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    relative_depth = (
        float(raw["relative_injection_depth"])
        if relative_depth_override is None
        else float(relative_depth_override)
    )
    replay_mode = (
        str(raw["replay_mode"])
        if replay_mode_override is None
        else replay_mode_override
    )
    layer_index = relative_layer_index(len(model.gpt_neox.layers), relative_depth)

    codebook_prompts = [
        raw["codebook_template"].format(candidate=text)
        for text in raw["candidate_texts"]
    ]
    codebook_batch = tokenizer(codebook_prompts, padding=True, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        codebook_output = model(**codebook_batch, output_hidden_states=True, use_cache=False)
    codebook_last = codebook_batch["attention_mask"].sum(dim=1) - 1
    codebook_last_token = codebook_output.hidden_states[layer_index][
        torch.arange(len(ids), device="cuda"), codebook_last
    ].detach()
    codebook_lengths = codebook_batch["attention_mask"].sum(dim=1)
    if not torch.all(codebook_lengths == codebook_lengths[0]):
        raise ValueError("full-sequence replay requires equal-length codebook prompts")
    codebook_sequence = codebook_output.hidden_states[layer_index][
        :, : int(codebook_lengths[0].item())
    ].detach()
    del codebook_output
    if replay_mode not in {"posterior_mean_last", "map_full_sequence"}:
        raise ValueError(f"unknown replay mode: {replay_mode}")

    static_steps = int(raw.get("static_adapter_steps", 0))
    if static_steps < 0:
        raise ValueError("static_adapter_steps must be nonnegative")
    static_learning_rate = float(raw.get("static_adapter_learning_rate", 0.0))
    static_training_batch_size = int(
        raw.get("static_adapter_batch_size", raw["batch_size"])
    )
    if static_training_batch_size != int(raw["batch_size"]):
        raise ValueError(
            "static adapter batch size must equal evaluation batch size for the frozen protocol"
        )
    static_initial = (
        codebook_last_token.mean(dim=0, keepdim=True)
        if replay_mode == "posterior_mean_last"
        else codebook_sequence.mean(dim=0)
    )
    static_parameter = torch.nn.Parameter(static_initial.float().clone())
    calibration_batches: list[tuple[Any, Any, Any, Any]] = []
    static_optimizer = None
    if static_steps:
        static_optimizer = torch.optim.AdamW(
            [static_parameter], lr=static_learning_rate, weight_decay=0.0
        )
        for start in range(0, split, static_training_batch_size):
            stop = min(split, start + static_training_batch_size)
            batch = tokenizer(
                prompts[start:stop], padding=True, return_tensors="pt"
            ).to("cuda")
            with torch.inference_mode():
                query_output = model(
                    **batch, output_hidden_states=True, use_cache=False
                )
            calibration_batches.append(
                (
                    query_output.hidden_states[layer_index].detach().cpu(),
                    batch["attention_mask"].detach().cpu(),
                    (batch["attention_mask"].sum(dim=1) - 1).detach().cpu(),
                    torch.as_tensor(labels[start:stop], dtype=torch.long),
                )
            )
            del query_output
    static_training_losses: list[float] = []
    static_checkpoint_losses: list[float] = []
    best_static_parameter = static_parameter.detach().clone()
    best_static_loss = float("inf")

    def evaluate_static_calibration() -> float:
        losses: list[float] = []
        with torch.inference_mode():
            for hidden_cpu, mask_cpu, last_cpu, label_cpu in calibration_batches:
                query_hidden = hidden_cpu.to("cuda")
                query_mask = mask_cpu.to("cuda")
                query_last = last_cpu.to("cuda")
                target = label_cpu.to("cuda")
                replay = static_parameter.to(dtype=query_hidden.dtype).unsqueeze(0)
                replay = replay.expand(query_hidden.shape[0], -1, -1)
                hidden = replay_from_hidden(
                    model, query_hidden, query_mask, replay, layer_index
                )
                logits = model.embed_out(hidden)[
                    torch.arange(query_hidden.shape[0], device="cuda"),
                    replay.shape[1] + query_last,
                ][:, ids]
                losses.append(
                    float(functional.cross_entropy(logits.float(), target).cpu())
                )
        return float(np.mean(losses))

    if static_steps:
        best_static_loss = evaluate_static_calibration()
        static_checkpoint_losses.append(best_static_loss)
    for step in range(static_steps):
        hidden_cpu, mask_cpu, last_cpu, label_cpu = calibration_batches[
            step % len(calibration_batches)
        ]
        query_hidden = hidden_cpu.to("cuda")
        query_mask = mask_cpu.to("cuda")
        query_last = last_cpu.to("cuda")
        target = label_cpu.to("cuda")
        static_replay = static_parameter.to(dtype=query_hidden.dtype).unsqueeze(0)
        static_replay = static_replay.expand(query_hidden.shape[0], -1, -1)
        static_hidden = replay_from_hidden(
            model, query_hidden, query_mask, static_replay, layer_index
        )
        static_logits = model.embed_out(static_hidden)[
            torch.arange(query_hidden.shape[0], device="cuda"),
            static_replay.shape[1] + query_last,
        ][:, ids]
        loss = functional.cross_entropy(static_logits.float(), target)
        assert static_optimizer is not None
        static_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        static_optimizer.step()
        static_training_losses.append(float(loss.detach().cpu()))
        del static_hidden, static_logits, loss
        if (step + 1) % len(calibration_batches) == 0 or step + 1 == static_steps:
            checkpoint_loss = evaluate_static_calibration()
            static_checkpoint_losses.append(checkpoint_loss)
            if checkpoint_loss < best_static_loss:
                best_static_loss = checkpoint_loss
                best_static_parameter = static_parameter.detach().clone()
    if static_steps:
        with torch.no_grad():
            static_parameter.copy_(best_static_parameter)
    static_replay_value = static_parameter.detach().to(
        device="cuda", dtype=codebook_last_token.dtype
    )
    del calibration_batches, static_optimizer

    base_rows: list[np.ndarray] = []
    arm_rows: dict[str, list[np.ndarray]] = {"fixed": [], "rich": []}
    static_rows: list[np.ndarray] = []
    posterior_by_arm = {
        "fixed": np.exp(channel.log_posterior(observations[:, : int(raw["fixed_rounds"]), :])),
        "rich": np.exp(channel.log_posterior(observations)),
    }
    for start in range(0, total, int(raw["batch_size"])):
        stop = min(total, start + int(raw["batch_size"]))
        batch = tokenizer(prompts[start:stop], padding=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            query_output = model(**batch, output_hidden_states=True, use_cache=False)
            query_last = batch["attention_mask"].sum(dim=1) - 1
            base_logits = query_output.logits[
                torch.arange(stop - start, device="cuda"), query_last
            ]
            base_rows.append(normalized_candidates(base_logits, ids))
            query_hidden = query_output.hidden_states[layer_index]
            for arm_name, posterior in posterior_by_arm.items():
                weights = torch.as_tensor(
                    posterior[start:stop], dtype=codebook_last_token.dtype, device="cuda"
                )
                if replay_mode == "posterior_mean_last":
                    replay = (weights @ codebook_last_token).unsqueeze(1)
                else:
                    map_labels = weights.argmax(dim=1)
                    replay = codebook_sequence[map_labels]
                hidden = replay_from_hidden(
                    model, query_hidden, batch["attention_mask"], replay, layer_index
                )
                replay_logits = model.embed_out(hidden)[
                    torch.arange(stop - start, device="cuda"), replay.shape[1] + query_last
                ]
                arm_rows[arm_name].append(normalized_candidates(replay_logits, ids))
            static_replay = static_replay_value.unsqueeze(0).expand(
                stop - start, -1, -1
            )
            static_hidden = replay_from_hidden(
                model,
                query_hidden,
                batch["attention_mask"],
                static_replay,
                layer_index,
            )
            static_logits = model.embed_out(static_hidden)[
                torch.arange(stop - start, device="cuda"),
                static_replay.shape[1] + query_last,
            ]
            static_rows.append(normalized_candidates(static_logits, ids))
        del query_output

    base = np.concatenate(base_rows)
    latent = {name: np.concatenate(rows) for name, rows in arm_rows.items()}
    static_adapter = np.concatenate(static_rows)
    text_replay: dict[str, np.ndarray] = {}
    for name, posterior in posterior_by_arm.items():
        map_labels = posterior.argmax(axis=1)
        text_prompts = [
            raw["text_replay_template"].format(
                candidate=raw["candidate_texts"][int(label)], query=query
            )
            for label, query in zip(map_labels, prompts, strict=True)
        ]
        text_replay[name] = foundation_probabilities(
            model, tokenizer, text_prompts, ids, int(raw["batch_size"])
        )
    label_test = labels[split:]
    base_nll = true_label_nll(base[split:], label_test)
    result_arms: dict[str, Any] = {}
    test_arrays: dict[str, np.ndarray] = {"labels": label_test, "base_nll": base_nll}
    for name in ("fixed", "rich"):
        gate = select_safe_mixture(base[:split], latent[name][:split], labels[:split])
        fused = apply_mixture(base[split:], latent[name][split:], gate.mixture)
        latent_nll = true_label_nll(latent[name][split:], label_test)
        fused_nll = true_label_nll(fused, label_test)
        certified_nll = true_label_nll(posterior_by_arm[name][split:], label_test)
        text_gate = select_safe_mixture(
            base[:split], text_replay[name][:split], labels[:split]
        )
        text_fused = apply_mixture(
            base[split:], text_replay[name][split:], text_gate.mixture
        )
        text_nll = true_label_nll(text_replay[name][split:], label_test)
        text_fused_nll = true_label_nll(text_fused, label_test)
        test_arrays[f"{name}_text_fused_nll"] = text_fused_nll
        test_arrays[f"{name}_fused_nll"] = fused_nll
        result_arms[name] = {
            "rounds": int(raw[f"{name}_rounds"]),
            "persistent_signal_bits_per_episode": int(raw[f"{name}_rounds"]) * int(raw["num_labels"]),
            "transient_replay_bytes_per_episode": int(
                (1 if replay_mode == "posterior_mean_last" else codebook_sequence.shape[1])
                * codebook_last_token.shape[1]
                * codebook_last_token.element_size()
            ),
            "signal_prefix_sha256": array_sha256(
                observations[:, : int(raw[f"{name}_rounds"]), :]
            ),
            "gate": gate.__dict__,
            "latent_only_nll": summary(latent_nll),
            "cross_fitted_fused_nll": summary(fused_nll),
            "cross_fitted_gain_over_foundation": summary(base_nll - fused_nll),
            "same_signal_exact_posterior_nll": summary(certified_nll),
            "exact_posterior_advantage_over_latent": summary(latent_nll - certified_nll),
            "same_signal_map_text_replay": {
                "gate": text_gate.__dict__,
                "text_only_nll": summary(text_nll),
                "cross_fitted_fused_nll": summary(text_fused_nll),
                "cross_fitted_gain_over_foundation": summary(base_nll - text_fused_nll),
            },
        }
    static_gate = select_safe_mixture(
        base[:split], static_adapter[:split], labels[:split]
    )
    static_fused = apply_mixture(
        base[split:], static_adapter[split:], static_gate.mixture
    )
    static_only_nll = true_label_nll(static_adapter[split:], label_test)
    static_fused_nll = true_label_nll(static_fused, label_test)
    test_arrays["static_adapter_fused_nll"] = static_fused_nll
    test_arrays["static_adapter_only_nll"] = static_only_nll
    test_arrays["rich_dynamic_advantage_over_static_adapter"] = (
        static_fused_nll - test_arrays["rich_fused_nll"]
    )
    test_arrays["fixed_to_rich_nll_improvement"] = (
        test_arrays["fixed_fused_nll"] - test_arrays["rich_fused_nll"]
    )
    test_arrays["fixed_to_rich_text_nll_improvement"] = (
        test_arrays["fixed_text_fused_nll"] - test_arrays["rich_text_fused_nll"]
    )
    arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays, **test_arrays)

    payload = {
        "schema_version": 1,
        "status": "development_diagnostic_complete",
        "confirmatory_eligible": False,
        "endpoint": endpoint,
        "config_sha256": sha256_file(config_path),
        "started_at_utc": started,
        "effective_seed": seed,
        "seed_is_registered_override": seed_override is not None,
        "candidate_token_ids": ids,
        "relative_layer_index": layer_index,
        "relative_injection_depth": relative_depth,
        "relative_depth_is_posthoc_override": relative_depth_override is not None,
        "replay_mode": replay_mode,
        "replay_mode_is_posthoc_override": replay_mode_override is not None,
        "protocol_status": raw.get("status", "unspecified"),
        "num_model_layers": len(model.gpt_neox.layers),
        "registered_labels_sha256": array_sha256(labels),
        "registered_full_signal_sha256": array_sha256(observations),
        "identical_query_inputs_sha256": hashlib.sha256("\n".join(prompts).encode()).hexdigest(),
        "foundation_nll": summary(base_nll),
        "arms": result_arms,
        "same_flop_static_adapter": {
            "definition": "One calibration-trained replay block shared by every episode; it receives no episode observation, key, target, or query-dependent parameters.",
            "training_episodes": split,
            "training_steps": static_steps,
            "learning_rate": static_learning_rate,
            "initial_training_loss": (
                static_training_losses[0] if static_training_losses else None
            ),
            "final_training_loss": (
                static_training_losses[-1] if static_training_losses else None
            ),
            "full_calibration_checkpoint_losses": static_checkpoint_losses,
            "selected_best_full_calibration_loss": (
                best_static_loss if static_checkpoint_losses else None
            ),
            "replay_tokens": int(static_replay_value.shape[0]),
            "hidden_width": int(static_replay_value.shape[1]),
            "persistent_parameter_bytes_fp32": int(static_parameter.numel() * 4),
            "inference_parameter_bytes_bf16": int(static_parameter.numel() * 2),
            "same_suffix_layer_index_as_dynamic": True,
            "same_replay_token_count_as_dynamic": int(static_replay_value.shape[0])
            == int(
                1
                if replay_mode == "posterior_mean_last"
                else codebook_sequence.shape[1]
            ),
            "same_frozen_suffix_call_as_dynamic": True,
            "gate": static_gate.__dict__,
            "adapter_only_nll": summary(static_only_nll),
            "cross_fitted_fused_nll": summary(static_fused_nll),
            "cross_fitted_gain_over_foundation": summary(
                base_nll - static_fused_nll
            ),
            "rich_dynamic_advantage_over_static_adapter": summary(
                test_arrays["rich_dynamic_advantage_over_static_adapter"]
            ),
        },
        "paired_fixed_to_rich_nll_improvement": summary(
            test_arrays["fixed_to_rich_nll_improvement"]
        ),
        "model_files": model_files,
        "runtime": {
            "hostname": platform.node(), "python": platform.python_version(),
            "numpy": np.__version__, "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "git_commit": git_value(["rev-parse", "HEAD"]),
            "git_dirty": bool(git_value(["status", "--porcelain"])),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "wall_seconds": time.perf_counter() - wall_start,
        },
        "future_label_absent_from_query": True,
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
    parser.add_argument("--relative-depth", type=float)
    parser.add_argument(
        "--replay-mode",
        choices=["posterior_mean_last", "map_full_sequence"],
    )
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    run(
        args.config,
        args.endpoint,
        args.model_dir,
        args.output,
        args.arrays,
        args.relative_depth,
        args.replay_mode,
        args.seed,
    )


if __name__ == "__main__":
    main()
