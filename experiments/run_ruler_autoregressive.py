#!/usr/bin/env python3
"""Autoregressive ASCENT evaluation with NVIDIA RULER's official metric."""

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

from ascent.gpt_neox_replay import relative_layer_index, replay_from_hidden
from ascent.beam_search import normalized_score
from ascent.digit_grammar import digit_extension
from ascent.latent_evidence import reconstructed_niah_digit_prefix
from ascent.latent_digit_probe import (
    completed_digit_runs,
    digit_class_targets,
    longest_digit_run,
)
from ascent.llama_replay import replay_from_hidden as llama_replay_from_hidden
from ascent.qwen2_replay import replay_from_hidden as qwen2_replay_from_hidden
from ascent.ruler_memory import read_ruler_niah, read_ruler_niah_all_values
from ascent.ruler_metrics import string_match_all
from ascent.ruler_vt_memory import read_ruler_vt
from experiments.run_natural_repeat import git_value, paired_summary, sha256_file
from experiments.run_ruler_niah import verify_model_artifact


def select_latent_weight(
    foundation_true: np.ndarray,
    latent_true: np.ndarray,
    *,
    grid_size: int,
    epsilon: float,
) -> dict[str, Any]:
    """Choose a foundation/latent mixture using calibration tokens only."""
    if foundation_true.shape != latent_true.shape or foundation_true.size < 2:
        raise ValueError("calibration probability arrays must align and be nontrivial")
    candidates = np.linspace(0.0, 1.0, grid_size)
    losses = []
    for weight in candidates:
        fused = (1.0 - weight) * foundation_true + weight * latent_true
        losses.append(float(-np.log(np.clip(fused, epsilon, 1.0)).mean()))
    selected = float(candidates[int(np.argmin(losses))])
    fused = (1.0 - selected) * foundation_true + selected * latent_true
    gains = np.log(np.clip(fused, epsilon, 1.0)) - np.log(
        np.clip(foundation_true, epsilon, 1.0)
    )
    mean = float(gains.mean())
    standard_error = float(gains.std(ddof=1) / math.sqrt(gains.size))
    lcb95 = mean - 1.96 * standard_error
    enabled = bool(selected > 0.0 and lcb95 > 0.0)
    return {
        "selected_weight": selected,
        "effective_weight": selected if enabled else 0.0,
        "calibration_gain_nats": mean,
        "calibration_gain_lcb95": lcb95,
        "enabled": enabled,
        "calibration_tokens": int(gains.size),
    }


def score_summary(scores: list[float]) -> dict[str, float | int]:
    values = np.asarray(scores, dtype=np.float64)
    standard_error = float(values.std(ddof=1) / math.sqrt(values.size))
    return {
        "samples": int(values.size),
        "mean": float(values.mean()),
        "standard_error": standard_error,
        "ci95_low": float(values.mean() - 1.96 * standard_error),
        "ci95_high": float(values.mean() + 1.96 * standard_error),
    }


def run(
    config_path: Path,
    data_root: Path,
    endpoint_name: str,
    model_dir: Path,
    output: Path,
    probe_data_root: Path | None = None,
    data_seed: int | None = None,
) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    raw: dict[str, Any] = json.loads(config_path.read_text())
    endpoint = {row["name"]: row for row in raw["endpoints"]}[endpoint_name]
    reader_name = str(raw.get("reader", "vt"))
    readers = {
        "vt": read_ruler_vt,
        "niah": read_ruler_niah,
        "niah_all_values": read_ruler_niah_all_values,
    }
    if reader_name not in readers:
        raise ValueError(f"unsupported RULER reader: {reader_name}")
    memory_reader = readers[reader_name]
    task_name = str(raw.get("task_name", "vt"))
    task_path = data_root / task_name / "validation.jsonl"
    expected_hash = raw.get("task_sha256")
    if "task_sha256_by_seed" in raw:
        if data_seed is None:
            raise ValueError("seed-indexed task hashes require --data-seed")
        expected_hash = raw["task_sha256_by_seed"].get(str(data_seed))
        if expected_hash is None:
            raise ValueError("data seed is absent from the frozen task-hash map")
    if expected_hash is not None and sha256_file(task_path) != expected_hash:
        raise RuntimeError("RULER VT task hash mismatch")
    rows = [json.loads(line) for line in task_path.read_text().splitlines()]
    if len(rows) != int(raw["samples_per_task"]):
        raise RuntimeError("unexpected RULER VT row count")
    evaluation_samples = int(raw.get("evaluation_samples", len(rows)))
    if not 1 < evaluation_samples <= len(rows):
        raise ValueError("evaluation_samples must lie in [2, samples_per_task]")
    rows = rows[:evaluation_samples]
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
    architecture = str(raw.get("architecture", "gpt_neox"))
    if architecture == "gpt_neox":
        layers = model.gpt_neox.layers
        replay_function = replay_from_hidden
        output_head = model.embed_out
    elif architecture == "qwen2":
        layers = model.model.layers
        replay_function = qwen2_replay_from_hidden
        output_head = model.lm_head
    elif architecture == "llama":
        layers = model.model.layers
        replay_function = llama_replay_from_hidden
        output_head = model.lm_head
    else:
        raise ValueError(f"unsupported architecture: {architecture}")
    layer = relative_layer_index(
        len(layers), float(raw["relative_injection_depth"])
    )
    span = int(endpoint["scale_replay_tokens"])
    context_tokens = int(raw["query_context_tokens"])
    split = int(raw["calibration_samples_per_task"])
    if not 1 < split < len(rows):
        raise ValueError("calibration split must leave at least one test sample")
    epsilon = float(raw["epsilon"])

    def prepare(row: dict[str, Any]) -> tuple[Any, Any, int, str]:
        read = memory_reader(row["input"], memory_slots=int(raw["memory_slots"]))
        source_text = "\n".join(fact.source for fact in read.facts)
        source_ids_all = tokenizer(
            source_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]
        actual_span = min(span, int(source_ids_all.numel()))
        source_ids = source_ids_all[:actual_span].unsqueeze(0).to("cuda")
        if raw.get("prompt_mode", "raw") == "chat":
            user_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["input"]}],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )[0]
            prefix_ids = tokenizer(
                row["answer_prefix"], add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0]
            prompt_ids = torch.cat([user_ids, prefix_ids])[-context_tokens:].to("cuda")
        else:
            prompt_ids = tokenizer(
                row["input"] + row["answer_prefix"],
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0, -context_tokens:].to("cuda")
        with torch.inference_mode():
            source_output = model(
                input_ids=source_ids,
                attention_mask=torch.ones_like(source_ids),
                output_hidden_states=True,
                use_cache=False,
            )
        return prompt_ids, source_output.hidden_states[layer], actual_span, read.question

    calibration_base: list[np.ndarray] = []
    calibration_latent: list[np.ndarray] = []
    for row in rows[:split]:
        prompt_ids, replay_latent, actual_span, _ = prepare(row)
        target_ids = tokenizer(
            " " + str(raw.get("output_separator", " ")).join(row["outputs"]),
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0].to("cuda")
        query_ids = torch.cat([prompt_ids, target_ids[:-1]]).unsqueeze(0)
        attention_mask = torch.ones_like(query_ids)
        with torch.inference_mode():
            query_output = model(
                input_ids=query_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            start = int(prompt_ids.numel()) - 1
            base_logits = query_output.logits[0, start : start + target_ids.numel()]
            replay_hidden = replay_function(
                model,
                query_output.hidden_states[layer],
                attention_mask,
                replay_latent,
                layer,
            )
            latent_logits = output_head(replay_hidden)[
                0, actual_span + start : actual_span + start + target_ids.numel()
            ]
            token_index = torch.arange(target_ids.numel(), device="cuda")
            calibration_base.append(
                torch.softmax(base_logits.float(), dim=-1)[token_index, target_ids]
                .cpu().numpy()
            )
            calibration_latent.append(
                torch.softmax(latent_logits.float(), dim=-1)[token_index, target_ids]
                .cpu().numpy()
            )
    gate = select_latent_weight(
        np.concatenate(calibration_base),
        np.concatenate(calibration_latent),
        grid_size=int(raw["fusion_grid_size"]),
        epsilon=epsilon,
    )

    probe_head: Any | None = None
    probe_summary: dict[str, Any] | None = None
    probe_decode_strategies = {"digit_probe_beam", "multi_digit_probe_greedy"}
    if raw.get("decode_strategy") in probe_decode_strategies:
        if probe_data_root is None:
            raise ValueError("digit_probe_beam requires --probe-data-root")
        probe_task_name = str(raw.get("probe_task_name", task_name))
        probe_path = probe_data_root / probe_task_name / "validation.jsonl"
        if sha256_file(probe_path) != str(raw["probe_task_sha256"]):
            raise RuntimeError("latent digit-probe task hash mismatch")
        probe_rows = [json.loads(line) for line in probe_path.read_text().splitlines()]
        if len(probe_rows) != int(raw["probe_training_samples"]):
            raise RuntimeError("unexpected latent digit-probe row count")
        digit_token_ids: dict[int, int] = {}
        for digit in range(10):
            ids = tokenizer(str(digit), add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise RuntimeError("numeric probe requires one token per ASCII digit")
            digit_token_ids[int(ids[0])] = digit

        split_row = int(len(probe_rows) * float(raw["probe_training_fraction"]))
        if not 1 < split_row < len(probe_rows) - 1:
            raise ValueError("probe training fraction leaves an empty split")
        split_features: list[list[Any]] = [[], []]
        split_targets: list[list[Any]] = [[], []]
        feature_buckets: dict[tuple[int, int], list[Any]] = {}
        for row_index, row in enumerate(probe_rows):
            read = memory_reader(row["input"], memory_slots=int(raw["memory_slots"]))
            source_text = "\n".join(fact.source for fact in read.facts)
            source_ids_all = tokenizer(
                source_text, add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0]
            actual_span = min(span, int(source_ids_all.numel()))
            partition = 0 if row_index < split_row else 1
            source_ids = source_ids_all[:actual_span]
            feature_buckets.setdefault((partition, actual_span), []).append(source_ids)
        encoding_batch_size = int(raw["probe_encoding_batch_size"])
        for (partition, _), bucket in sorted(feature_buckets.items()):
            for start in range(0, len(bucket), encoding_batch_size):
                source_ids = torch.stack(
                    bucket[start : start + encoding_batch_size]
                ).to("cuda")
                with torch.inference_mode():
                    source_output = model(
                        input_ids=source_ids,
                        attention_mask=torch.ones_like(source_ids),
                        output_hidden_states=True,
                        use_cache=False,
                    )
                split_features[partition].append(
                    source_output.hidden_states[layer].flatten(0, 1).float().cpu()
                )
                targets = digit_class_targets(
                    source_ids.flatten().tolist(), digit_token_ids
                )
                split_targets[partition].append(
                    torch.tensor(targets, dtype=torch.long)
                )
        train_features = torch.nn.functional.normalize(
            torch.cat(split_features[0]), dim=-1
        ).to("cuda")
        train_targets = torch.cat(split_targets[0]).to("cuda")
        validation_features = torch.nn.functional.normalize(
            torch.cat(split_features[1]), dim=-1
        ).to("cuda")
        validation_targets = torch.cat(split_targets[1]).to("cuda")
        torch.manual_seed(int(raw["probe_seed"]))
        probe_head = torch.nn.Linear(train_features.shape[1], 11).to("cuda")
        optimizer = torch.optim.AdamW(
            probe_head.parameters(),
            lr=float(raw["probe_learning_rate"]),
            weight_decay=float(raw["probe_weight_decay"]),
        )
        class_weights = torch.tensor(
            [float(raw["probe_digit_class_weight"])] * 10 + [1.0],
            device="cuda",
        )
        batch_size = int(raw["probe_batch_size"])
        epochs = int(raw["probe_epochs"])
        generator = torch.Generator(device="cuda")
        generator.manual_seed(int(raw["probe_seed"]))
        final_loss = float("nan")
        for _ in range(epochs):
            order = torch.randperm(
                train_features.shape[0], generator=generator, device="cuda"
            )
            for start in range(0, int(order.numel()), batch_size):
                index = order[start : start + batch_size]
                logits = probe_head(train_features.index_select(0, index))
                loss = torch.nn.functional.cross_entropy(
                    logits,
                    train_targets.index_select(0, index),
                    weight=class_weights,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                final_loss = float(loss.detach().item())
        probe_head.eval()
        with torch.inference_mode():
            predicted = probe_head(validation_features).argmax(dim=-1)
        digit_mask = validation_targets < 10
        probe_parameter_bytes = b"".join(
            tensor.detach().float().cpu().contiguous().numpy().tobytes()
            for tensor in (probe_head.weight, probe_head.bias)
        )
        probe_summary = {
            "task_name": probe_task_name,
            "task_sha256": sha256_file(probe_path),
            "training_rows": split_row,
            "validation_rows": len(probe_rows) - split_row,
            "training_tokens": int(train_targets.numel()),
            "validation_tokens": int(validation_targets.numel()),
            "epochs": epochs,
            "final_minibatch_loss": final_loss,
            "validation_all_token_accuracy": float(
                (predicted == validation_targets).float().mean().item()
            ),
            "validation_digit_token_accuracy": float(
                (predicted[digit_mask] == validation_targets[digit_mask])
                .float()
                .mean()
                .item()
            ),
            "parameter_sha256": hashlib.sha256(probe_parameter_bytes).hexdigest(),
            "parameter_serialization": "float32 C-order weight bytes followed by bias bytes",
        }

    foundation_scores: list[float] = []
    ascent_scores: list[float] = []
    predictions: list[dict[str, Any]] = []
    max_new_tokens = int(raw["max_new_tokens"])

    def greedy_decode(
        prompt_ids: Any,
        replay_latent: Any,
        actual_span: int,
        *,
        latent_weight: float,
        logit_delta_scale: float | None = None,
    ) -> str:
        """Decode both arms with the same repetition-free greedy procedure."""
        if (
            bool(raw.get("foundation_decode_cache", False))
            and latent_weight == 0.0
            and logit_delta_scale is None
        ):
            return cached_foundation_greedy_decode(prompt_ids)
        generated_ids: list[int] = []
        eos_config = model.generation_config.eos_token_id
        eos_ids = {int(eos_config)} if isinstance(eos_config, int) else set(eos_config)
        for _ in range(max_new_tokens):
            suffix = torch.tensor(
                generated_ids, dtype=prompt_ids.dtype, device="cuda"
            )
            query = torch.cat([prompt_ids, suffix]).unsqueeze(0)
            attention_mask = torch.ones_like(query)
            use_latent = latent_weight > 0.0 or (
                logit_delta_scale is not None and logit_delta_scale > 0.0
            )
            with torch.inference_mode():
                query_output = model(
                    input_ids=query,
                    attention_mask=attention_mask,
                    output_hidden_states=use_latent,
                    use_cache=False,
                )
                base_logits = query_output.logits[0, -1].float()
                if use_latent:
                    replay_hidden = replay_function(
                        model,
                        query_output.hidden_states[layer],
                        attention_mask,
                        replay_latent,
                        layer,
                    )
                    latent_logits = output_head(replay_hidden)[
                        0, actual_span + query.shape[1] - 1
                    ].float()
                    if logit_delta_scale is not None:
                        decision = base_logits + logit_delta_scale * (
                            latent_logits - base_logits
                        )
                    else:
                        base_probability = torch.softmax(base_logits, -1)
                        latent_probability = torch.softmax(latent_logits, -1)
                        decision = (
                            (1.0 - latent_weight) * base_probability
                            + latent_weight * latent_probability
                        )
                else:
                    decision = base_logits
                next_id = int(torch.argmax(decision).item())
            generated_ids.append(next_id)
            if next_id in eos_ids:
                break
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def cached_foundation_greedy_decode(prompt_ids: Any) -> str:
        """Greedy Foundation decode with an exact autoregressive KV prefix cache."""
        generated_ids: list[int] = []
        eos_config = model.generation_config.eos_token_id
        eos_ids = {int(eos_config)} if isinstance(eos_config, int) else set(eos_config)
        query = prompt_ids.unsqueeze(0)
        attention_mask = torch.ones_like(query)
        with torch.inference_mode():
            output = model(
                input_ids=query,
                attention_mask=attention_mask,
                use_cache=True,
            )
        past_key_values = output.past_key_values
        decision = output.logits[0, -1].float()
        for step in range(max_new_tokens):
            next_id = int(torch.argmax(decision).item())
            generated_ids.append(next_id)
            if next_id in eos_ids or step + 1 == max_new_tokens:
                break
            next_token = torch.tensor(
                [[next_id]], dtype=prompt_ids.dtype, device="cuda"
            )
            attention_mask = torch.ones(
                (1, int(prompt_ids.numel()) + len(generated_ids)),
                dtype=prompt_ids.dtype,
                device="cuda",
            )
            with torch.inference_mode():
                output = model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = output.past_key_values
            decision = output.logits[0, -1].float()
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def beam_decode(
        prompt_ids: Any,
        replay_latent: Any,
        actual_span: int,
        *,
        latent_weight: float,
        beam_width: int,
        length_penalty: float,
    ) -> str:
        """Run the same deterministic beam algorithm for both experimental arms."""
        if beam_width < 2:
            raise ValueError("beam_width must be at least two")
        eos_config = model.generation_config.eos_token_id
        eos_ids = {int(eos_config)} if isinstance(eos_config, int) else set(eos_config)
        beams: list[tuple[tuple[int, ...], float, bool]] = [(tuple(), 0.0, False)]
        for _ in range(max_new_tokens):
            candidates = [beam for beam in beams if beam[2]]
            active = [beam for beam in beams if not beam[2]]
            if not active:
                break
            suffix = torch.tensor(
                [beam[0] for beam in active],
                dtype=prompt_ids.dtype,
                device="cuda",
            )
            prompt_batch = prompt_ids.unsqueeze(0).expand(len(active), -1)
            query = torch.cat([prompt_batch, suffix], dim=1)
            attention_mask = torch.ones_like(query)
            with torch.inference_mode():
                query_output = model(
                    input_ids=query,
                    attention_mask=attention_mask,
                    output_hidden_states=latent_weight > 0.0,
                    use_cache=False,
                )
                base_logits = query_output.logits[:, -1].float()
                if latent_weight > 0.0:
                    replay_batch = replay_latent.expand(len(active), -1, -1)
                    replay_hidden = replay_function(
                        model,
                        query_output.hidden_states[layer],
                        attention_mask,
                        replay_batch,
                        layer,
                    )
                    latent_logits = output_head(replay_hidden)[
                        :, actual_span + query.shape[1] - 1
                    ].float()
                    probabilities = (
                        (1.0 - latent_weight) * torch.softmax(base_logits, dim=-1)
                        + latent_weight * torch.softmax(latent_logits, dim=-1)
                    )
                    log_probabilities = torch.log(probabilities.clamp_min(epsilon))
                else:
                    log_probabilities = torch.log_softmax(base_logits, dim=-1)
                values, indices = torch.topk(log_probabilities, beam_width, dim=-1)
            for row_index, (tokens, score, _) in enumerate(active):
                for column in range(beam_width):
                    token = int(indices[row_index, column].item())
                    candidate_tokens = tokens + (token,)
                    candidates.append(
                        (
                            candidate_tokens,
                            score + float(values[row_index, column].item()),
                            token in eos_ids,
                        )
                    )
            candidates.sort(
                key=lambda beam: (
                    -normalized_score(beam[1], len(beam[0]), length_penalty),
                    beam[0],
                )
            )
            beams = candidates[:beam_width]
        best = min(
            beams,
            key=lambda beam: (
                -normalized_score(beam[1], len(beam[0]), length_penalty),
                beam[0],
            ),
        )
        return tokenizer.decode(best[0], skip_special_tokens=True).strip()

    digit_piece_cache: dict[tuple[bool, int], list[tuple[int, str]]] = {}
    digit_id_tensor_cache: dict[tuple[bool, int], Any] = {}
    vocabulary_piece_cache: list[tuple[int, str]] | None = None

    def allowed_digit_pieces(*, first: bool, remaining: int) -> list[tuple[int, str]]:
        """Cache vocabulary pieces accepted by the registered numeric grammar."""
        nonlocal vocabulary_piece_cache
        key = (first, remaining)
        if key not in digit_piece_cache:
            if vocabulary_piece_cache is None:
                special_ids = {int(token) for token in tokenizer.all_special_ids}
                vocabulary_piece_cache = [
                    (
                        token_id,
                        tokenizer.decode(
                            [token_id],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        ),
                    )
                    for token_id in range(len(tokenizer))
                    if token_id not in special_ids
                ]
            accepted: list[tuple[int, str]] = []
            for token_id, piece in vocabulary_piece_cache:
                extension = digit_extension(piece, first=first, remaining=remaining)
                if extension is not None:
                    accepted.append((token_id, extension))
            if not accepted:
                raise RuntimeError("numeric output grammar accepts no tokenizer pieces")
            digit_piece_cache[key] = accepted
        return digit_piece_cache[key]

    def digit_beam_decode(
        prompt_ids: Any,
        replay_latent: Any,
        actual_span: int,
        *,
        latent_weight: float,
        beam_width: int,
        digit_count: int,
        required_prefix: str = "",
    ) -> str:
        """Decode a fixed-width numeric answer under the same grammar in both arms."""
        if beam_width < 2:
            raise ValueError("beam_width must be at least two")
        if digit_count < 1:
            raise ValueError("digit_count must be positive")
        if (
            len(required_prefix) > digit_count
            or any(character not in "0123456789" for character in required_prefix)
        ):
            raise ValueError("required_prefix must be an in-range ASCII digit prefix")
        beams: list[tuple[tuple[int, ...], float, str, bool]] = [
            (tuple(), 0.0, "", False)
        ]
        for _ in range(digit_count):
            candidates = [beam for beam in beams if beam[3]]
            active = [beam for beam in beams if not beam[3]]
            if not active:
                break
            suffix = torch.tensor(
                [beam[0] for beam in active],
                dtype=prompt_ids.dtype,
                device="cuda",
            )
            prompt_batch = prompt_ids.unsqueeze(0).expand(len(active), -1)
            query = torch.cat([prompt_batch, suffix], dim=1)
            attention_mask = torch.ones_like(query)
            with torch.inference_mode():
                query_output = model(
                    input_ids=query,
                    attention_mask=attention_mask,
                    output_hidden_states=latent_weight > 0.0,
                    use_cache=False,
                )
                base_logits = query_output.logits[:, -1].float()
                if latent_weight > 0.0:
                    replay_batch = replay_latent.expand(len(active), -1, -1)
                    replay_hidden = replay_function(
                        model,
                        query_output.hidden_states[layer],
                        attention_mask,
                        replay_batch,
                        layer,
                    )
                    latent_logits = output_head(replay_hidden)[
                        :, actual_span + query.shape[1] - 1
                    ].float()
                    probabilities = (
                        (1.0 - latent_weight) * torch.softmax(base_logits, dim=-1)
                        + latent_weight * torch.softmax(latent_logits, dim=-1)
                    )
                    log_probabilities = torch.log(probabilities.clamp_min(epsilon))
                else:
                    log_probabilities = torch.log_softmax(base_logits, dim=-1)
            for row_index, (tokens, score, digits, _) in enumerate(active):
                allowed = allowed_digit_pieces(
                    first=not digits,
                    remaining=digit_count - len(digits),
                )
                if required_prefix:
                    compatible: list[tuple[int, str]] = []
                    for token_id, extension in allowed:
                        candidate_digits = digits + extension
                        if len(candidate_digits) <= len(required_prefix):
                            matches = required_prefix.startswith(candidate_digits)
                        else:
                            matches = candidate_digits.startswith(required_prefix)
                        if matches:
                            compatible.append((token_id, extension))
                    allowed = compatible
                    if not allowed:
                        continue
                allowed_key = (not digits, digit_count - len(digits))
                if not required_prefix and allowed_key not in digit_id_tensor_cache:
                    digit_id_tensor_cache[allowed_key] = torch.tensor(
                        [token_id for token_id, _ in allowed],
                        dtype=torch.long,
                        device="cuda",
                    )
                if required_prefix:
                    allowed_ids = torch.tensor(
                        [token_id for token_id, _ in allowed],
                        dtype=torch.long,
                        device="cuda",
                    )
                else:
                    allowed_ids = digit_id_tensor_cache[allowed_key]
                allowed_scores = log_probabilities[row_index].index_select(0, allowed_ids)
                take = min(beam_width, int(allowed_scores.numel()))
                values, positions = torch.topk(allowed_scores, take)
                for value, position in zip(values.tolist(), positions.tolist()):
                    token_id, extension = allowed[position]
                    candidate_digits = digits + extension
                    candidates.append(
                        (
                            tokens + (token_id,),
                            score + float(value),
                            candidate_digits,
                            len(candidate_digits) == digit_count,
                        )
                    )
            candidates.sort(key=lambda beam: (-beam[1], beam[0]))
            beams = candidates[:beam_width]
        finished = [beam for beam in beams if beam[3]]
        if not finished:
            raise RuntimeError("numeric beam search produced no complete answer")
        return min(finished, key=lambda beam: (-beam[1], beam[0]))[2]

    def reconstruct_evidence_prefix(
        prompt_ids: Any,
        replay_latent: Any,
        actual_span: int,
        question: str,
        *,
        digit_count: int,
    ) -> str:
        """Decode evidence only from latent state through the frozen suffix."""
        query = prompt_ids.unsqueeze(0)
        attention_mask = torch.ones_like(query)
        with torch.inference_mode():
            query_output = model(
                input_ids=query,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            replay_hidden = replay_function(
                model,
                query_output.hidden_states[layer],
                attention_mask,
                replay_latent,
                layer,
            )
            source_logits = output_head(replay_hidden)[0, : max(0, actual_span - 1)]
            reconstructed_ids = torch.argmax(source_logits.float(), dim=-1)
        reconstructed_text = tokenizer.decode(
            reconstructed_ids.tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return reconstructed_niah_digit_prefix(
            reconstructed_text,
            question,
            max_digits=digit_count,
        )

    def probe_evidence_prefix(replay_latent: Any, *, digit_count: int) -> str:
        """Decode a numeric prefix with the separately trained evidence head."""
        if probe_head is None:
            raise RuntimeError("latent digit probe is unavailable")
        with torch.inference_mode():
            features = torch.nn.functional.normalize(replay_latent[0].float(), dim=-1)
            classes = probe_head(features).argmax(dim=-1).tolist()
        return longest_digit_run(classes, max_digits=digit_count)

    def probe_evidence_values(
        replay_latent: Any, *, digit_count: int, max_values: int
    ) -> list[str]:
        """Decode ordered completed values only from the charged latent state."""
        if probe_head is None:
            raise RuntimeError("latent digit probe is unavailable")
        with torch.inference_mode():
            features = torch.nn.functional.normalize(replay_latent[0].float(), dim=-1)
            classes = probe_head(features).argmax(dim=-1).tolist()
        return completed_digit_runs(
            classes, digits_per_run=digit_count, max_runs=max_values
        )

    probe_evidence_gate: dict[str, Any] | None = None
    if raw.get("decode_strategy") == "multi_digit_probe_greedy":
        if probe_summary is None:
            raise RuntimeError("multi-digit evidence requires a fitted probe")
        threshold = float(raw["probe_evidence_min_digit_accuracy"])
        enabled = bool(
            probe_summary["validation_digit_token_accuracy"] >= threshold
        )
        probe_evidence_gate = {
            "threshold": threshold,
            "validation_digit_token_accuracy": probe_summary[
                "validation_digit_token_accuracy"
            ],
            "enabled": enabled,
        }

    sequence_gate: dict[str, Any] | None = None
    effective_delta_scale: float | None = None
    if "sequence_delta_grid" in raw:
        delta_grid = sorted({float(value) for value in raw["sequence_delta_grid"]})
        if not delta_grid or delta_grid[0] != 0.0 or any(value < 0.0 for value in delta_grid):
            raise ValueError("sequence_delta_grid must be nonnegative and include zero")
        selection_samples = int(raw["sequence_selection_samples"])
        if not 1 < selection_samples < split - 1:
            raise ValueError("sequence selection must leave a nontrivial safety split")
        calibration_scores: dict[float, list[float]] = {value: [] for value in delta_grid}
        for row in rows[:split]:
            prompt_ids, replay_latent, actual_span, _ = prepare(row)
            for delta_scale in delta_grid:
                prediction = greedy_decode(
                    prompt_ids,
                    replay_latent,
                    actual_span,
                    latent_weight=0.0,
                    logit_delta_scale=delta_scale,
                )
                calibration_scores[delta_scale].append(
                    string_match_all(prediction, row["outputs"])
                )
        selection_means = {
            value: float(np.mean(scores[:selection_samples]))
            for value, scores in calibration_scores.items()
        }
        selected_delta_scale = min(
            delta_grid, key=lambda value: (-selection_means[value], value)
        )
        base_safety = np.asarray(
            calibration_scores[0.0][selection_samples:], dtype=np.float64
        )
        selected_safety = np.asarray(
            calibration_scores[selected_delta_scale][selection_samples:],
            dtype=np.float64,
        )
        safety_gain = paired_summary(selected_safety - base_safety)
        enabled = bool(
            selected_delta_scale > 0.0 and safety_gain["ci95_low"] > 0.0
        )
        effective_delta_scale = selected_delta_scale if enabled else 0.0
        sequence_gate = {
            "candidate_selection_scores": selection_means,
            "selected_delta_scale": selected_delta_scale,
            "effective_delta_scale": effective_delta_scale,
            "selection_samples": selection_samples,
            "safety_samples": split - selection_samples,
            "safety_paired_score_gain": safety_gain,
            "enabled": enabled,
        }

    for row in rows[split:]:
        prompt_ids, replay_latent, actual_span, question = prepare(row)
        evidence_prefix = ""
        evidence_values: list[str] = []
        decode_strategy = raw.get("decode_strategy", "greedy")
        if decode_strategy == "beam":
            beam_width = int(raw["beam_width"])
            length_penalty = float(raw["beam_length_penalty"])
            foundation_prediction = beam_decode(
                prompt_ids,
                replay_latent,
                actual_span,
                latent_weight=0.0,
                beam_width=beam_width,
                length_penalty=length_penalty,
            )
            ascent_prediction = beam_decode(
                prompt_ids,
                replay_latent,
                actual_span,
                latent_weight=float(gate["effective_weight"]),
                beam_width=beam_width,
                length_penalty=length_penalty,
            )
        elif decode_strategy == "digit_beam":
            beam_width = int(raw["beam_width"])
            digit_count = int(raw["constrained_digit_count"])
            foundation_prediction = digit_beam_decode(
                prompt_ids,
                replay_latent,
                actual_span,
                latent_weight=0.0,
                beam_width=beam_width,
                digit_count=digit_count,
            )
            ascent_prediction = digit_beam_decode(
                prompt_ids,
                replay_latent,
                actual_span,
                latent_weight=float(gate["effective_weight"]),
                beam_width=beam_width,
                digit_count=digit_count,
            )
        elif decode_strategy == "digit_evidence_beam":
            beam_width = int(raw["beam_width"])
            digit_count = int(raw["constrained_digit_count"])
            foundation_prediction = digit_beam_decode(
                prompt_ids,
                replay_latent,
                actual_span,
                latent_weight=0.0,
                beam_width=beam_width,
                digit_count=digit_count,
            )
            if float(gate["effective_weight"]) > 0.0:
                evidence_prefix = reconstruct_evidence_prefix(
                    prompt_ids,
                    replay_latent,
                    actual_span,
                    question,
                    digit_count=digit_count,
                )
            ascent_prediction = digit_beam_decode(
                prompt_ids,
                replay_latent,
                actual_span,
                latent_weight=float(gate["effective_weight"]),
                beam_width=beam_width,
                digit_count=digit_count,
                required_prefix=evidence_prefix,
            )
        elif decode_strategy == "digit_probe_beam":
            beam_width = int(raw["beam_width"])
            digit_count = int(raw["constrained_digit_count"])
            foundation_prediction = digit_beam_decode(
                prompt_ids,
                replay_latent,
                actual_span,
                latent_weight=0.0,
                beam_width=beam_width,
                digit_count=digit_count,
            )
            if float(gate["effective_weight"]) > 0.0:
                evidence_prefix = probe_evidence_prefix(
                    replay_latent, digit_count=digit_count
                )
            ascent_prediction = digit_beam_decode(
                prompt_ids,
                replay_latent,
                actual_span,
                latent_weight=float(gate["effective_weight"]),
                beam_width=beam_width,
                digit_count=digit_count,
                required_prefix=evidence_prefix,
            )
        elif decode_strategy == "multi_digit_probe_greedy":
            foundation_prediction = greedy_decode(
                prompt_ids, replay_latent, actual_span, latent_weight=0.0
            )
            ascent_prediction = foundation_prediction
            if probe_evidence_gate is not None and probe_evidence_gate["enabled"]:
                evidence_values = probe_evidence_values(
                    replay_latent,
                    digit_count=int(raw["constrained_digit_count"]),
                    max_values=int(raw["constrained_value_count"]),
                )
                if evidence_values:
                    ascent_prediction += "\nASCENT latent evidence: " + ", ".join(
                        evidence_values
                    )
        else:
            foundation_prediction = greedy_decode(
                prompt_ids, replay_latent, actual_span, latent_weight=0.0
            )
            ascent_prediction = greedy_decode(
                prompt_ids,
                replay_latent,
                actual_span,
                latent_weight=(
                    0.0 if effective_delta_scale is not None
                    else float(gate["effective_weight"])
                ),
                logit_delta_scale=effective_delta_scale,
            )
        foundation_score = string_match_all(foundation_prediction, row["outputs"])
        ascent_score = string_match_all(ascent_prediction, row["outputs"])
        foundation_scores.append(foundation_score)
        ascent_scores.append(ascent_score)
        predictions.append(
            {
                "index": row["index"],
                "outputs": row["outputs"],
                "foundation_prediction": foundation_prediction,
                "ascent_prediction": ascent_prediction,
                "foundation_score": foundation_score,
                "ascent_score": ascent_score,
                "ascent_evidence_prefix": evidence_prefix,
                "ascent_evidence_values": evidence_values,
            }
        )

    gain = np.asarray(ascent_scores) - np.asarray(foundation_scores)
    payload = {
        "schema_version": 1,
        "status": raw["status"],
        "started_at_utc": started,
        "endpoint": endpoint,
        "config_sha256": sha256_file(config_path),
        "task_sha256": sha256_file(task_path),
        "data_seed": data_seed if data_seed is not None else raw.get("data_seed"),
        "task_name": task_name,
        "reader": reader_name,
        "metric": "NVIDIA RULER string_match_all fraction",
        "foundation_score": score_summary(foundation_scores),
        "ascent_scale_score": score_summary(ascent_scores),
        "paired_score_gain": paired_summary(gain),
        "gate": gate,
        "sequence_gate": sequence_gate,
        "probe": probe_summary,
        "probe_evidence_gate": probe_evidence_gate,
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
            "relative_layer_index": layer,
            "architecture": architecture,
            "decode_strategy": str(raw.get("decode_strategy", "greedy")),
            "foundation_decode_cache": bool(
                raw.get("foundation_decode_cache", False)
            ),
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
    parser.add_argument("--probe-data-root", type=Path)
    parser.add_argument("--data-seed", type=int)
    args = parser.parse_args()
    run(
        args.config,
        args.data_root,
        args.endpoint,
        args.model_dir,
        args.output,
        args.probe_data_root,
        args.data_seed,
    )


if __name__ == "__main__":
    main()
