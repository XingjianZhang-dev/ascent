#!/usr/bin/env python3
"""Version-address/value separated ASCENT overwrite development runner."""

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
from experiments.run_ruler_niah import verify_model_artifact
from experiments.run_stage_b_endpoint import (
    array_sha256,
    candidate_ids,
    sha256_file,
    summary,
)


def ordered_pair(
    first: Any,
    second: Any,
    target_first: np.ndarray,
) -> Any:
    """Return two equal-length replay events in registered per-row order."""

    import torch

    if first.shape != second.shape:
        raise ValueError("paired replay events must have equal shapes")
    batch, tokens, width = first.shape
    pair = torch.stack([first, second], dim=1)
    order = torch.as_tensor(
        np.stack(
            [
                np.where(target_first, 0, 1),
                np.where(target_first, 1, 0),
            ],
            axis=1,
        ),
        device=first.device,
        dtype=torch.long,
    )
    ordered = pair.gather(
        1, order[:, :, None, None].expand(batch, 2, tokens, width)
    )
    return ordered.reshape(batch, 2 * tokens, width)


def posterior_mean(codebook: Any, probabilities: np.ndarray) -> Any:
    """Preserve a categorical posterior as a codebook-weighted latent value."""

    import torch

    weights = torch.as_tensor(
        probabilities, device=codebook.device, dtype=codebook.dtype
    )
    return torch.einsum("bv,vth->bth", weights, codebook)


def masked_mean_hidden(hidden: Any, attention_mask: Any) -> Any:
    """Pool variable-length address tokens without using padding or length as state."""

    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def run(
    config_path: Path,
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
    model_files = verify_model_artifact(model_dir, endpoint.get("model_artifact"))
    started = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    total = int(raw["calibration_episodes"]) + int(raw["test_episodes"])
    split = int(raw["calibration_episodes"])
    rich_rounds = int(raw["rich_rounds"])
    rng = np.random.default_rng(int(raw["seed"]))

    target_key_int = rng.integers(0, 2**63, size=total, dtype=np.int64)
    distractor_key_int = rng.integers(0, 2**63, size=total, dtype=np.int64)
    equal = distractor_key_int == target_key_int
    distractor_key_int[equal] ^= np.int64(1)
    target_keys = [f"{value:016x}" for value in target_key_int]
    distractor_keys = [f"{value:016x}" for value in distractor_key_int]
    target_first = rng.integers(0, 2, size=total, dtype=np.int8).astype(bool)
    queries = [raw["query_template"].format(key=key) for key in target_keys]

    channel = NoisyRefinementChannel(
        CertifiedChannelConfig(
            int(raw["num_labels"]), float(raw["crossover_probability"])
        )
    )
    target_old, target_old_signal = channel.sample(rng, total, rich_rounds)
    target_current, target_current_signal = channel.sample(rng, total, rich_rounds)
    distractor_old, distractor_old_signal = channel.sample(rng, total, rich_rounds)
    distractor_current, distractor_current_signal = channel.sample(
        rng, total, rich_rounds
    )

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
        model,
        tokenizer,
        [
            raw["old_value_template"].format(value=value)
            for value in raw["candidate_texts"]
        ],
        layer,
    )
    current_codebook = extract_codebook(
        model,
        tokenizer,
        [
            raw["current_value_template"].format(value=value)
            for value in raw["candidate_texts"]
        ],
        layer,
    )

    posteriors: dict[str, dict[str, np.ndarray]] = {}
    for name in ("fixed", "rich"):
        rounds = int(raw[f"{name}_rounds"])
        posteriors[name] = {
            "target_old": np.exp(
                channel.log_posterior(target_old_signal[:, :rounds, :])
            ),
            "target_current": np.exp(
                channel.log_posterior(target_current_signal[:, :rounds, :])
            ),
            "distractor_old": np.exp(
                channel.log_posterior(distractor_old_signal[:, :rounds, :])
            ),
            "distractor_current": np.exp(
                channel.log_posterior(distractor_current_signal[:, :rounds, :])
            ),
        }

    base_rows: list[np.ndarray] = []
    primary_rows = {"fixed": [], "rich": []}
    history_rows = {"fixed": [], "rich": []}
    shuffled_rows = {"fixed": [], "rich": []}
    batch_size = int(raw["batch_size"])
    for start in range(0, total, batch_size):
        stop = min(total, start + batch_size)
        size = stop - start
        query_batch = tokenizer(
            queries[start:stop], padding=True, return_tensors="pt"
        ).to("cuda")
        target_key_batch = target_keys[start:stop]
        distractor_key_batch = distractor_keys[start:stop]
        address_prompts: list[str] = []
        for target_key, distractor_key in zip(
            target_key_batch, distractor_key_batch, strict=True
        ):
            address_prompts.extend(
                [
                    raw["address_template"].format(key=target_key, version=1),
                    raw["address_template"].format(key=target_key, version=2),
                    raw["address_template"].format(key=distractor_key, version=1),
                    raw["address_template"].format(key=distractor_key, version=2),
                ]
            )
        address_batch = tokenizer(
            address_prompts, padding=True, return_tensors="pt"
        ).to("cuda")
        with torch.inference_mode():
            query_output = model(
                **query_batch, output_hidden_states=True, use_cache=False
            )
            address_output = model(
                **address_batch, output_hidden_states=True, use_cache=False
            )
            last = query_batch["attention_mask"].sum(dim=1) - 1
            logits = query_output.logits[torch.arange(size, device="cuda"), last]
            base_rows.append(candidate_probabilities(logits, ids))
            query_hidden = query_output.hidden_states[layer]
            address_hidden = masked_mean_hidden(
                address_output.hidden_states[layer], address_batch["attention_mask"]
            )
            address_hidden = address_hidden[:, None, :].reshape(size, 4, 1, -1)
            target_old_address = address_hidden[:, 0]
            target_current_address = address_hidden[:, 1]
            distractor_old_address = address_hidden[:, 2]
            distractor_current_address = address_hidden[:, 3]

            for name, posterior in posteriors.items():
                target_old_value = posterior_mean(
                    old_codebook, posterior["target_old"][start:stop]
                )
                target_current_value = posterior_mean(
                    current_codebook, posterior["target_current"][start:stop]
                )
                distractor_old_value = posterior_mean(
                    old_codebook, posterior["distractor_old"][start:stop]
                )
                distractor_current_value = posterior_mean(
                    current_codebook,
                    posterior["distractor_current"][start:stop],
                )

                target_current_event = torch.cat(
                    [target_current_address, target_current_value], dim=1
                )
                distractor_current_event = torch.cat(
                    [distractor_current_address, distractor_current_value], dim=1
                )
                latest_replay = ordered_pair(
                    target_current_event,
                    distractor_current_event,
                    target_first[start:stop],
                )

                target_history = torch.cat(
                    [
                        target_old_address,
                        target_old_value,
                        target_current_address,
                        target_current_value,
                    ],
                    dim=1,
                )
                distractor_history = torch.cat(
                    [
                        distractor_old_address,
                        distractor_old_value,
                        distractor_current_address,
                        distractor_current_value,
                    ],
                    dim=1,
                )
                history_replay = ordered_pair(
                    target_history,
                    distractor_history,
                    target_first[start:stop],
                )

                target_shuffled = torch.cat(
                    [distractor_current_address, target_current_value], dim=1
                )
                distractor_shuffled = torch.cat(
                    [target_current_address, distractor_current_value], dim=1
                )
                shuffled_replay = ordered_pair(
                    target_shuffled,
                    distractor_shuffled,
                    target_first[start:stop],
                )

                for destination, replay in (
                    (primary_rows[name], latest_replay),
                    (history_rows[name], history_replay),
                    (shuffled_rows[name], shuffled_replay),
                ):
                    hidden = replay_from_hidden(
                        model,
                        query_hidden,
                        query_batch["attention_mask"],
                        replay,
                        layer,
                    )
                    replay_logits = model.embed_out(hidden)[
                        torch.arange(size, device="cuda"),
                        replay.shape[1] + last,
                    ]
                    destination.append(candidate_probabilities(replay_logits, ids))
        del query_output, address_output

    base = np.concatenate(base_rows)
    primary = {name: np.concatenate(rows) for name, rows in primary_rows.items()}
    history = {name: np.concatenate(rows) for name, rows in history_rows.items()}
    shuffled = {name: np.concatenate(rows) for name, rows in shuffled_rows.items()}
    base_nll = true_label_nll(base[split:], target_current[split:])
    saved: dict[str, np.ndarray] = {
        "current": target_current[split:],
        "base_nll": base_nll,
        "target_first": target_first[split:],
    }
    arms: dict[str, Any] = {}
    for name in ("fixed", "rich"):
        primary_result, primary_nll = evaluate_path(
            base, primary[name], target_current, split
        )
        history_result, history_nll = evaluate_path(
            base, history[name], target_current, split
        )
        shuffled_result, shuffled_nll = evaluate_path(
            base, shuffled[name], target_current, split
        )
        exact_nll = true_label_nll(
            posteriors[name]["target_current"][split:], target_current[split:]
        )
        saved[f"{name}_latent_fused_nll"] = primary_nll
        saved[f"{name}_history_fused_nll"] = history_nll
        saved[f"{name}_reversed_fused_nll"] = shuffled_nll
        saved[f"{name}_shuffled_binding_fused_nll"] = shuffled_nll
        arms[name] = {
            "rounds_per_version": int(raw[f"{name}_rounds"]),
            "latest_version_only_latent_replay": primary_result,
            "unmasked_history_ablation": history_result,
            "reversed_replay_order_ablation": shuffled_result,
            "shuffled_address_value_binding_ablation": shuffled_result,
            "same_signal_exact_current_posterior_nll": summary(exact_nll),
            "signal_sha256": array_sha256(
                np.stack(
                    [
                        target_old_signal[:, : int(raw[f"{name}_rounds"]), :],
                        target_current_signal[:, : int(raw[f"{name}_rounds"]), :],
                        distractor_old_signal[:, : int(raw[f"{name}_rounds"]), :],
                        distractor_current_signal[:, : int(raw[f"{name}_rounds"]), :],
                    ],
                    axis=1,
                )
            ),
        }
    saved["fixed_to_rich_latent_improvement"] = (
        saved["fixed_latent_fused_nll"] - saved["rich_latent_fused_nll"]
    )
    arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays, **saved)

    payload = {
        "schema_version": 1,
        "status": raw["status"],
        "started_at_utc": started,
        "architecture": "separated_version_address_and_value_posterior",
        "address_pooling": "mean_valid_tokens",
        "endpoint": endpoint,
        "config_sha256": sha256_file(config_path),
        "effective_seed": int(raw["seed"]),
        "relative_layer_index": layer,
        "registered_current_sha256": array_sha256(target_current),
        "registered_full_signal_sha256": array_sha256(
            np.stack(
                [
                    target_old_signal,
                    target_current_signal,
                    distractor_old_signal,
                    distractor_current_signal,
                ],
                axis=1,
            )
        ),
        "identical_query_inputs_sha256": hashlib.sha256(
            "\n".join(queries).encode()
        ).hexdigest(),
        "registered_address_order_sha256": array_sha256(target_first),
        "target_key_sha256": hashlib.sha256(
            "\n".join(target_keys).encode()
        ).hexdigest(),
        "distractor_key_sha256": hashlib.sha256(
            "\n".join(distractor_keys).encode()
        ).hexdigest(),
        "foundation_nll": summary(base_nll),
        "arms": arms,
        "clean_kv_oracle_nll": 0.0,
        "read_before_write_and_later_version_target": True,
        "obsolete_and_current_values_absent_from_query": True,
        "two_current_addresses_in_primary_replay": True,
        "target_position_randomized": True,
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
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.endpoint, args.model_dir, args.output, args.arrays)


if __name__ == "__main__":
    main()
