#!/usr/bin/env python3
"""Binding-dependent ordered overwrite composition development runner."""

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
from experiments.run_overwrite_address_value import posterior_mean
from experiments.run_ruler_niah import verify_model_artifact
from experiments.run_stage_b_endpoint import (
    array_sha256,
    candidate_ids,
    sha256_file,
    summary,
)


def affine_posterior(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Distribution of ``2 * first + second`` for independent posteriors."""

    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("value posteriors must have equal [episodes, values] shapes")
    values = first.shape[1]
    result = np.zeros((first.shape[0], 3 * values - 2), dtype=np.float64)
    for left in range(values):
        for right in range(values):
            result[:, 2 * left + right] += first[:, left] * second[:, right]
    return result


def ordered_history(
    first_old: Any,
    first_current: Any,
    second_old: Any,
    second_current: Any,
    first_stored_first: np.ndarray,
) -> Any:
    """Materialize randomized storage order while retaining within-key chronology."""

    import torch

    first = torch.cat([first_old, first_current], dim=1)
    second = torch.cat([second_old, second_current], dim=1)
    pair = torch.stack([first, second], dim=1)
    order = torch.as_tensor(
        np.stack(
            [
                np.where(first_stored_first, 0, 1),
                np.where(first_stored_first, 1, 0),
            ],
            axis=1,
        ),
        device=first.device,
        dtype=torch.long,
    )
    batch, _, tokens, width = pair.shape
    return pair.gather(
        1, order[:, :, None, None].expand(batch, 2, tokens, width)
    ).reshape(batch, 2 * tokens, width)


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

    first_key_int = rng.integers(0, 2**63, size=total, dtype=np.int64)
    second_key_int = rng.integers(0, 2**63, size=total, dtype=np.int64)
    equal = second_key_int == first_key_int
    second_key_int[equal] ^= np.int64(1)
    first_keys = [f"{value:016x}" for value in first_key_int]
    second_keys = [f"{value:016x}" for value in second_key_int]
    first_stored_first = rng.integers(0, 2, size=total, dtype=np.int8).astype(bool)
    queries = [
        raw["query_template"].format(first_key=first, second_key=second)
        for first, second in zip(first_keys, second_keys, strict=True)
    ]

    channel = NoisyRefinementChannel(
        CertifiedChannelConfig(
            int(raw["num_values"]), float(raw["crossover_probability"])
        )
    )
    first_old, first_old_signal = channel.sample(rng, total, rich_rounds)
    first_current, first_current_signal = channel.sample(rng, total, rich_rounds)
    second_old, second_old_signal = channel.sample(rng, total, rich_rounds)
    second_current, second_current_signal = channel.sample(rng, total, rich_rounds)
    answers = 2 * first_current + second_current

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
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

    def codebook(template_name: str) -> Any:
        return extract_codebook(
            model,
            tokenizer,
            [
                raw[template_name].format(value=value)
                for value in raw["value_texts"]
            ],
            layer,
        )

    first_old_codebook = codebook("first_old_codebook_template")
    first_current_codebook = codebook("first_current_codebook_template")
    second_old_codebook = codebook("second_old_codebook_template")
    second_current_codebook = codebook("second_current_codebook_template")

    posteriors: dict[str, dict[str, np.ndarray]] = {}
    for name in ("fixed", "rich"):
        rounds = int(raw[f"{name}_rounds"])
        posteriors[name] = {
            "first_old": np.exp(
                channel.log_posterior(first_old_signal[:, :rounds, :])
            ),
            "first_current": np.exp(
                channel.log_posterior(first_current_signal[:, :rounds, :])
            ),
            "second_old": np.exp(
                channel.log_posterior(second_old_signal[:, :rounds, :])
            ),
            "second_current": np.exp(
                channel.log_posterior(second_current_signal[:, :rounds, :])
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
        with torch.inference_mode():
            query_output = model(
                **query_batch, output_hidden_states=True, use_cache=False
            )
            last = query_batch["attention_mask"].sum(dim=1) - 1
            base_logits = query_output.logits[
                torch.arange(size, device="cuda"), last
            ]
            base_rows.append(candidate_probabilities(base_logits, answer_ids))
            query_hidden = query_output.hidden_states[layer]

            for name, posterior in posteriors.items():
                first_old_value = posterior_mean(
                    first_old_codebook, posterior["first_old"][start:stop]
                )
                first_current_value = posterior_mean(
                    first_current_codebook,
                    posterior["first_current"][start:stop],
                )
                second_old_value = posterior_mean(
                    second_old_codebook, posterior["second_old"][start:stop]
                )
                second_current_value = posterior_mean(
                    second_current_codebook,
                    posterior["second_current"][start:stop],
                )

                latest_replay = torch.cat(
                    [first_current_value, second_current_value], dim=1
                )
                history_replay = ordered_history(
                    first_old_value,
                    first_current_value,
                    second_old_value,
                    second_current_value,
                    first_stored_first[start:stop],
                )
                shuffled_replay = torch.cat(
                    [
                        posterior_mean(
                            first_current_codebook,
                            posterior["second_current"][start:stop],
                        ),
                        posterior_mean(
                            second_current_codebook,
                            posterior["first_current"][start:stop],
                        ),
                    ],
                    dim=1,
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
                    destination.append(
                        candidate_probabilities(replay_logits, answer_ids)
                    )
        del query_output

    base = np.concatenate(base_rows)
    primary = {name: np.concatenate(rows) for name, rows in primary_rows.items()}
    history = {name: np.concatenate(rows) for name, rows in history_rows.items()}
    shuffled = {name: np.concatenate(rows) for name, rows in shuffled_rows.items()}
    base_nll = true_label_nll(base[split:], answers[split:])
    saved: dict[str, np.ndarray] = {
        "answers": answers[split:],
        "first_current": first_current[split:],
        "second_current": second_current[split:],
        "base_nll": base_nll,
        "first_stored_first": first_stored_first[split:],
    }
    arms: dict[str, Any] = {}
    for name in ("fixed", "rich"):
        primary_result, primary_nll = evaluate_path(
            base, primary[name], answers, split
        )
        history_result, history_nll = evaluate_path(
            base, history[name], answers, split
        )
        shuffled_result, shuffled_nll = evaluate_path(
            base, shuffled[name], answers, split
        )
        exact = affine_posterior(
            posteriors[name]["first_current"],
            posteriors[name]["second_current"],
        )
        exact_nll = true_label_nll(exact[split:], answers[split:])
        saved[f"{name}_latent_fused_nll"] = primary_nll
        saved[f"{name}_history_fused_nll"] = history_nll
        saved[f"{name}_shuffled_binding_fused_nll"] = shuffled_nll
        arms[name] = {
            "rounds_per_version": int(raw[f"{name}_rounds"]),
            "latest_only_ordered_affine_replay": primary_result,
            "unmasked_random_order_history_ablation": history_result,
            "shuffled_role_binding_ablation": shuffled_result,
            "same_signal_exact_affine_posterior_nll": summary(exact_nll),
            "persistent_signal_bits_per_episode": (
                4 * int(raw[f"{name}_rounds"]) * int(raw["num_values"])
            ),
            "signal_sha256": array_sha256(
                np.stack(
                    [
                        first_old_signal[:, : int(raw[f"{name}_rounds"]), :],
                        first_current_signal[:, : int(raw[f"{name}_rounds"]), :],
                        second_old_signal[:, : int(raw[f"{name}_rounds"]), :],
                        second_current_signal[:, : int(raw[f"{name}_rounds"]), :],
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
        "architecture": "exact_address_latest_version_ordered_affine_latent_replay",
        "endpoint": endpoint,
        "config_sha256": sha256_file(config_path),
        "effective_seed": int(raw["seed"]),
        "relative_layer_index": layer,
        "registered_answer_sha256": array_sha256(answers),
        "registered_current_sha256": array_sha256(
            np.stack([first_current, second_current], axis=1)
        ),
        "registered_full_signal_sha256": array_sha256(
            np.stack(
                [
                    first_old_signal,
                    first_current_signal,
                    second_old_signal,
                    second_current_signal,
                ],
                axis=1,
            )
        ),
        "identical_query_inputs_sha256": hashlib.sha256(
            "\n".join(queries).encode()
        ).hexdigest(),
        "first_key_sha256": hashlib.sha256("\n".join(first_keys).encode()).hexdigest(),
        "second_key_sha256": hashlib.sha256(
            "\n".join(second_keys).encode()
        ).hexdigest(),
        "registered_storage_order_sha256": array_sha256(first_stored_first),
        "foundation_nll": summary(base_nll),
        "arms": arms,
        "address_decoder": "exact_key_match_then_latest_version",
        "target_function": "2 * first_current + second_current",
        "continuous_primary_metric": "paired_nll_gain_nats",
        "accuracy_ceiling_not_used": True,
        "read_before_write_and_later_version_target": True,
        "obsolete_and_current_values_absent_from_query": True,
        "two_distinct_query_keys": True,
        "event_storage_order_randomized": True,
        "noncommutative_binding_target": True,
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
