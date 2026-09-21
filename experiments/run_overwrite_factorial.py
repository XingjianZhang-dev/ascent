#!/usr/bin/env python3
"""Versioned-overwrite diagnostic with obsolete-state distractors."""

from __future__ import annotations

import argparse
import hashlib
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

from ascent.certified_channel import CertifiedChannelConfig, NoisyRefinementChannel
from ascent.fusion import true_label_nll
from ascent.gpt_neox_replay import relative_layer_index, replay_from_hidden
from experiments.run_composition_factorial import (
    candidate_probabilities,
    evaluate_path,
    extract_codebook,
    git_value,
)
from experiments.run_stage_b_endpoint import (
    array_sha256,
    candidate_ids,
    foundation_probabilities,
    sha256_file,
    summary,
)


def run(config_path: Path, endpoint_name: str, model_dir: Path, output: Path, arrays: Path) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    raw: dict[str, Any] = json.loads(config_path.read_text())
    endpoint = {row["name"]: row for row in raw["endpoints"]}[endpoint_name]
    replay_mode = str(raw.get("latent_replay_mode", "argmax"))
    if replay_mode not in {"argmax", "posterior_mean"}:
        raise ValueError(f"unsupported latent replay mode: {replay_mode}")
    started = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    total = int(raw["calibration_episodes"]) + int(raw["test_episodes"])
    split = int(raw["calibration_episodes"])
    rng = np.random.default_rng(int(raw["seed"]))
    keys = [f"{value:016x}" for value in rng.integers(0, 2**63, size=total, dtype=np.int64)]
    queries = [raw["query_template"].format(key=key) for key in keys]
    channel = NoisyRefinementChannel(
        CertifiedChannelConfig(int(raw["num_labels"]), float(raw["crossover_probability"]))
    )
    obsolete, obsolete_signal = channel.sample(rng, total, int(raw["rich_rounds"]))
    current, current_signal = channel.sample(rng, total, int(raw["rich_rounds"]))

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ids = candidate_ids(tokenizer, list(raw["candidate_texts"]))
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.bfloat16
    ).to("cuda").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layer = relative_layer_index(
        len(model.gpt_neox.layers), float(raw["relative_injection_depth"])
    )
    old_codebook = extract_codebook(
        model, tokenizer,
        [raw["old_codebook_template"].format(value=value) for value in raw["candidate_texts"]],
        layer,
    )
    new_codebook = extract_codebook(
        model, tokenizer,
        [raw["new_codebook_template"].format(value=value) for value in raw["candidate_texts"]],
        layer,
    )
    posteriors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in ("fixed", "rich"):
        rounds = int(raw[f"{name}_rounds"])
        posteriors[name] = (
            np.exp(channel.log_posterior(obsolete_signal[:, :rounds, :])),
            np.exp(channel.log_posterior(current_signal[:, :rounds, :])),
        )

    base_rows: list[np.ndarray] = []
    latent_rows: dict[str, list[np.ndarray]] = {"fixed": [], "rich": []}
    history_rows: dict[str, list[np.ndarray]] = {"fixed": [], "rich": []}
    reversed_rows: dict[str, list[np.ndarray]] = {"fixed": [], "rich": []}
    argmax_rows: dict[str, list[np.ndarray]] = {"fixed": [], "rich": []}
    for start in range(0, total, int(raw["batch_size"])):
        stop = min(total, start + int(raw["batch_size"]))
        batch = tokenizer(queries[start:stop], padding=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            query_output = model(**batch, output_hidden_states=True, use_cache=False)
            last = batch["attention_mask"].sum(dim=1) - 1
            logits = query_output.logits[torch.arange(stop - start, device="cuda"), last]
            base_rows.append(candidate_probabilities(logits, ids))
            query_hidden = query_output.hidden_states[layer]
            for name, (old_p, new_p) in posteriors.items():
                old_map = torch.as_tensor(old_p[start:stop].argmax(axis=1), device="cuda")
                new_map = torch.as_tensor(new_p[start:stop].argmax(axis=1), device="cuda")
                old_argmax_replay = old_codebook[old_map]
                new_argmax_replay = new_codebook[new_map]
                if replay_mode == "posterior_mean":
                    old_weights = torch.as_tensor(
                        old_p[start:stop], device="cuda", dtype=old_codebook.dtype
                    )
                    new_weights = torch.as_tensor(
                        new_p[start:stop], device="cuda", dtype=new_codebook.dtype
                    )
                    old_replay = torch.einsum("bv,vth->bth", old_weights, old_codebook)
                    latest_replay = torch.einsum("bv,vth->bth", new_weights, new_codebook)
                else:
                    old_replay = old_argmax_replay
                    latest_replay = new_argmax_replay
                history_replay = torch.cat([old_replay, latest_replay], dim=1)
                reversed_replay = torch.cat([latest_replay, old_replay], dim=1)
                for destination, replay in (
                    (latent_rows[name], latest_replay),
                    (history_rows[name], history_replay),
                    (reversed_rows[name], reversed_replay),
                ):
                    hidden = replay_from_hidden(
                        model, query_hidden, batch["attention_mask"], replay, layer
                    )
                    replay_logits = model.embed_out(hidden)[
                        torch.arange(stop - start, device="cuda"), replay.shape[1] + last
                    ]
                    destination.append(candidate_probabilities(replay_logits, ids))
                if replay_mode == "posterior_mean":
                    hidden = replay_from_hidden(
                        model,
                        query_hidden,
                        batch["attention_mask"],
                        new_argmax_replay,
                        layer,
                    )
                    replay_logits = model.embed_out(hidden)[
                        torch.arange(stop - start, device="cuda"),
                        new_argmax_replay.shape[1] + last,
                    ]
                    argmax_rows[name].append(candidate_probabilities(replay_logits, ids))
        del query_output

    base = np.concatenate(base_rows)
    latent = {name: np.concatenate(rows) for name, rows in latent_rows.items()}
    history_latent = {name: np.concatenate(rows) for name, rows in history_rows.items()}
    reversed_latent = {name: np.concatenate(rows) for name, rows in reversed_rows.items()}
    argmax_latent = (
        {name: np.concatenate(rows) for name, rows in argmax_rows.items()}
        if replay_mode == "posterior_mean"
        else {}
    )
    text: dict[str, np.ndarray] = {}
    for name, (old_p, new_p) in posteriors.items():
        prompts = [
            raw["text_replay_template"].format(
                old=raw["candidate_texts"][int(a)],
                new=raw["candidate_texts"][int(b)], query=query,
            )
            for a, b, query in zip(old_p.argmax(axis=1), new_p.argmax(axis=1), queries, strict=True)
        ]
        text[name] = foundation_probabilities(
            model, tokenizer, prompts, ids, int(raw["batch_size"])
        )
    clean_prompts = [
        raw["text_replay_template"].format(
            old=raw["candidate_texts"][int(a)],
            new=raw["candidate_texts"][int(b)], query=query,
        )
        for a, b, query in zip(obsolete, current, queries, strict=True)
    ]
    clean_text = foundation_probabilities(
        model, tokenizer, clean_prompts, ids, int(raw["batch_size"])
    )

    base_nll = true_label_nll(base[split:], current[split:])
    saved: dict[str, np.ndarray] = {"current": current[split:], "base_nll": base_nll}
    arms: dict[str, Any] = {}
    for name in ("fixed", "rich"):
        latent_result, latent_nll = evaluate_path(base, latent[name], current, split)
        history_result, history_nll = evaluate_path(base, history_latent[name], current, split)
        reversed_result, reversed_nll = evaluate_path(base, reversed_latent[name], current, split)
        text_result, text_nll = evaluate_path(base, text[name], current, split)
        exact_nll = true_label_nll(posteriors[name][1][split:], current[split:])
        saved[f"{name}_latent_fused_nll"] = latent_nll
        saved[f"{name}_history_fused_nll"] = history_nll
        saved[f"{name}_reversed_fused_nll"] = reversed_nll
        if replay_mode == "posterior_mean":
            argmax_result, argmax_nll = evaluate_path(
                base, argmax_latent[name], current, split
            )
            saved[f"{name}_argmax_fused_nll"] = argmax_nll
        else:
            argmax_result = latent_result
        saved[f"{name}_text_fused_nll"] = text_nll
        arms[name] = {
            "rounds_per_version": int(raw[f"{name}_rounds"]),
            "latest_version_only_latent_replay": latent_result,
            "unmasked_history_ablation": history_result,
            "reversed_replay_order_ablation": reversed_result,
            "same_signal_map_text_replay": text_result,
            "same_signal_argmax_latest_ablation": argmax_result,
            "same_signal_exact_current_posterior_nll": summary(exact_nll),
            "signal_sha256": array_sha256(np.stack([
                obsolete_signal[:, : int(raw[f"{name}_rounds"]), :],
                current_signal[:, : int(raw[f"{name}_rounds"]), :],
            ], axis=1)),
        }
    clean_result, clean_nll = evaluate_path(base, clean_text, current, split)
    saved["clean_text_fused_nll"] = clean_nll
    saved["fixed_to_rich_latent_improvement"] = (
        saved["fixed_latent_fused_nll"] - saved["rich_latent_fused_nll"]
    )
    arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays, **saved)

    weights = sorted(model_dir.glob("*.safetensors")) + sorted(model_dir.glob("pytorch_model*.bin"))
    payload = {
        "schema_version": 1, "status": raw["status"], "started_at_utc": started,
        "latent_replay_mode": replay_mode,
        "endpoint": endpoint, "config_sha256": sha256_file(config_path),
        "effective_seed": int(raw["seed"]), "relative_layer_index": layer,
        "registered_current_sha256": array_sha256(current),
        "registered_full_signal_sha256": array_sha256(np.stack([obsolete_signal, current_signal], axis=1)),
        "identical_query_inputs_sha256": hashlib.sha256("\n".join(queries).encode()).hexdigest(),
        "foundation_nll": summary(base_nll), "arms": arms,
        "clean_kv_text_replay": clean_result, "clean_kv_oracle_nll": 0.0,
        "read_before_write_and_later_version_target": True,
        "obsolete_and_current_values_absent_from_query": True,
        "model_files": [
            {"path": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in weights
        ],
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
    args = parser.parse_args()
    run(args.config, args.endpoint, args.model_dir, args.output, args.arrays)


if __name__ == "__main__":
    main()
