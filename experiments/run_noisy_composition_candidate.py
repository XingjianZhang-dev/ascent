#!/usr/bin/env python3
"""Score noisy ASCENT evidence with a candidate-normalized neural head."""

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

from experiments.prepare_noisy_composition_factorial import (
    exact_nll,
    exact_sum_nll,
    posterior,
    sha256_file,
)
from experiments.run_babilong_prompt import (
    configure_decoder_tokenizer,
    frozen_model_artifact_spec,
    render_chat_prompt,
    validate_loaded_model,
    validate_runtime_dependencies,
)
from experiments.run_natural_repeat import git_value
from experiments.run_ruler_niah import verify_model_artifact
from experiments.run_stage_b_endpoint import candidate_ids


def candidate_mapping(candidate_texts: list[str]) -> str:
    return ", ".join(
        f"{candidate}={value}" for value, candidate in enumerate(candidate_texts)
    )


def foundation_prompt(
    row: dict[str, Any],
    style: str = "affine_readings",
    candidate_texts: list[str] | None = None,
) -> str:
    if style == "sum_posterior_candidate_labels":
        if candidate_texts is None:
            raise RuntimeError("sum candidate labels are required")
        return (
            "A fresh episode was created after model training. Hidden registers A and B "
            "are integers from 0 through 7. The requested answer is the ordinary integer "
            "sum A+B. No ASCENT state is available in this arm. Select exactly one label "
            "from the fixed answer code.\n"
            f"Answer code: {candidate_mapping(candidate_texts)}\n"
            f"Episode: {row['episode_id']}\nAnswer label:"
        )
    if style == "sum_posterior_candidate_numeric":
        return (
            "A fresh episode was created after model training. Hidden registers A and B "
            "are integers from 0 through 7. The requested answer is the ordinary integer "
            "sum A+B. No ASCENT state is available in this arm. Give the answer as exactly "
            "two digits from 00 through 14, using a leading zero when needed.\n"
            f"Episode: {row['episode_id']}"
        )
    if style != "affine_readings":
        raise RuntimeError(f"unknown candidate prompt style: {style}")
    return (
        "A and B are independent hidden integers from 0 through 7, sampled after model "
        "training. Compute R = (2*A + B) modulo 8. This arm has no observations. "
        "Return only your best result digit from 0 through 7.\n"
        f"Fresh episode key: {row['episode_id']}\nResult digit:"
    )


def ascent_prompt(
    row: dict[str, Any],
    rounds: int,
    eta: float,
    style: str = "affine_readings",
    candidate_texts: list[str] | None = None,
) -> str:
    first = ", ".join(str(value) for value in row["first_observations"][:rounds])
    second = ", ".join(str(value) for value in row["second_observations"][:rounds])
    if style == "sum_posterior_candidate_labels":
        if candidate_texts is None:
            raise RuntimeError("sum candidate labels are required")
        first_posterior = posterior(row["first_observations"][:rounds], 8, eta)
        second_posterior = posterior(row["second_observations"][:rounds], 8, eta)
        first_probabilities = ", ".join(
            f"{index}:{probability:.4f}"
            for index, probability in enumerate(first_posterior)
        )
        second_probabilities = ", ".join(
            f"{index}:{probability:.4f}"
            for index, probability in enumerate(second_posterior)
        )
        return (
            "A fresh episode was created after model training. ASCENT retained append-only "
            "noisy readings and its certified decoder computed the registered exact "
            "posteriors. Infer A and B and calculate the ordinary integer sum A+B. Select "
            "exactly one label from the fixed answer code.\n"
            f"Retained A readings: {first}\nRetained B readings: {second}\n"
            f"Certified A posterior: {first_probabilities}\n"
            f"Certified A MAP: {int(first_posterior.argmax())}\n"
            f"Certified B posterior: {second_probabilities}\n"
            f"Certified B MAP: {int(second_posterior.argmax())}\n"
            f"Answer code: {candidate_mapping(candidate_texts)}\n"
            f"Episode: {row['episode_id']}\nAnswer label:"
        )
    if style == "sum_posterior_candidate_numeric":
        first_posterior = posterior(row["first_observations"][:rounds], 8, eta)
        second_posterior = posterior(row["second_observations"][:rounds], 8, eta)
        first_probabilities = ", ".join(
            f"{index}:{probability:.4f}"
            for index, probability in enumerate(first_posterior)
        )
        second_probabilities = ", ".join(
            f"{index}:{probability:.4f}"
            for index, probability in enumerate(second_posterior)
        )
        return (
            "A fresh episode was created after model training. ASCENT retained append-only "
            "noisy readings and its certified decoder computed the registered exact "
            "posteriors. Infer A and B and calculate the ordinary integer sum A+B. Give the "
            "answer as exactly two digits from 00 through 14, using a leading zero when "
            "needed.\n"
            f"Retained A readings: {first}\nRetained B readings: {second}\n"
            f"Certified A posterior: {first_probabilities}\n"
            f"Certified A MAP: {int(first_posterior.argmax())}\n"
            f"Certified B posterior: {second_probabilities}\n"
            f"Certified B MAP: {int(second_posterior.argmax())}\n"
            f"Episode: {row['episode_id']}"
        )
    if style != "affine_readings":
        raise RuntimeError(f"unknown candidate prompt style: {style}")
    return (
        "A and B are independent hidden integers from 0 through 7, sampled after model "
        "training. ASCENT retained independent noisy readings. Each reading equals its "
        f"register with probability {1.0 - eta:.2f}; otherwise it is uniformly one of the "
        "other seven digits. Use every reading to infer A and B. Compute "
        "R = (2*A + B) modulo 8. Return only the result digit from 0 through 7.\n"
        f"ASCENT A readings: {first}\nASCENT B readings: {second}\n"
        f"Fresh episode key: {row['episode_id']}\nResult digit:"
    )


def summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    se = float(array.std(ddof=1) / np.sqrt(array.size))
    mean = float(array.mean())
    return {
        "samples": int(array.size),
        "mean": mean,
        "standard_error": se,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
    }


def candidate_probabilities(
    model: Any,
    tokenizer: Any,
    rendered_prompts: list[str],
    token_sequences: list[list[int]],
    batch_size: int,
) -> np.ndarray:
    import torch

    rows = []
    device = next(model.parameters()).device
    for start in range(0, len(rendered_prompts), batch_size):
        batch = tokenizer(
            rendered_prompts[start : start + batch_size],
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(device)
        with torch.inference_mode():
            logits = model(**batch, use_cache=False).logits
        positions = torch.arange(batch.attention_mask.shape[1], device=device)
        last = (batch.attention_mask * positions).argmax(dim=1)
        final_logits = logits[
            torch.arange(batch.input_ids.shape[0], device=device), last
        ].float()
        lengths = {len(sequence) for sequence in token_sequences}
        if lengths == {1}:
            token_ids = [sequence[0] for sequence in token_sequences]
            selected = final_logits[:, token_ids]
            probabilities = selected.softmax(dim=-1)
        elif lengths == {2}:
            first_ids = sorted({sequence[0] for sequence in token_sequences})
            first_log_probabilities = final_logits.log_softmax(dim=-1)
            conditional: dict[int, Any] = {}
            for first_id in first_ids:
                appended = torch.full(
                    (batch.input_ids.shape[0], 1),
                    first_id,
                    dtype=batch.input_ids.dtype,
                    device=device,
                )
                extended_ids = torch.cat([batch.input_ids, appended], dim=1)
                extended_mask = torch.cat(
                    [
                        batch.attention_mask,
                        torch.ones_like(appended, dtype=batch.attention_mask.dtype),
                    ],
                    dim=1,
                )
                with torch.inference_mode():
                    next_logits = model(
                        input_ids=extended_ids,
                        attention_mask=extended_mask,
                        use_cache=False,
                    ).logits[:, -1].float()
                conditional[first_id] = next_logits.log_softmax(dim=-1)
            scores = torch.stack(
                [
                    first_log_probabilities[:, first_id]
                    + conditional[first_id][:, second_id]
                    for first_id, second_id in token_sequences
                ],
                dim=1,
            )
            probabilities = scores.softmax(dim=-1)
        else:
            raise RuntimeError("candidate sequences must have equal length one or two")
        rows.append(probabilities.cpu().numpy().astype(np.float64))
    return np.concatenate(rows)


def run(
    config_path: Path,
    data_root: Path,
    panel: str,
    endpoint_name: str,
    model_dir: Path,
    rounds: int,
    output: Path,
    *,
    preflight_only: bool = False,
    audit_rerun: bool = False,
) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = json.loads(config_path.read_text())
    runtime_dependencies = validate_runtime_dependencies(config)
    if panel not in config["panels"] or rounds not in config["factorial"]["state_rounds"]:
        raise RuntimeError("unregistered panel or state size")
    data_path = data_root / config["panel_path_by_name"][panel]
    manifest_path = data_root / config.get(
        "data_manifest_path",
        "posttraining_noisy_composition_candidate/MANIFEST.json",
    )
    if sha256_file(data_path) != config["panel_sha256_by_name"][panel]:
        raise RuntimeError("panel hash mismatch")
    if sha256_file(manifest_path) != config["data_manifest_sha256"]:
        raise RuntimeError("manifest hash mismatch")
    rows = [json.loads(line) for line in data_path.read_text().splitlines()]
    if len(rows) != config["evaluation_samples"]:
        raise RuntimeError("row count mismatch")
    labels = int(config["construction"]["num_values"])
    target_classes = int(config.get("target_classes", labels))
    eta = float(config["construction"]["crossover_probability"])
    target_rule = config["construction"]["target_rule"]
    for row in rows:
        expected_target = (
            row["first_value"] + row["second_value"]
            if target_rule == "first_value + second_value"
            else (2 * row["first_value"] + row["second_value"]) % labels
        )
        if row["target"] != expected_target:
            raise RuntimeError("target construction mismatch")
    exact_value = (
        exact_sum_nll(rows, rounds, labels, eta)
        if target_rule == "first_value + second_value"
        else exact_nll(rows, rounds, labels, eta)
    )
    if exact_value != config["exact_posterior_mean_nll_by_panel_and_rounds"][panel][str(rounds)]:
        raise RuntimeError("exact posterior mismatch")

    endpoint = {row["name"]: row for row in config["endpoints"]}[endpoint_name]
    model_files = verify_model_artifact(model_dir, frozen_model_artifact_spec(endpoint))
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    configure_decoder_tokenizer(tokenizer)
    candidate_texts = config["candidate_texts"]
    if len(candidate_texts) != target_classes:
        raise RuntimeError("candidate count differs from target class count")
    token_sequences = [
        [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]
        for text in candidate_texts
    ]
    sequence_lengths = {len(sequence) for sequence in token_sequences}
    if sequence_lengths == {1}:
        # Retain the original strict audit for all existing one-token designs.
        candidate_ids(tokenizer, candidate_texts)
    if sequence_lengths not in ({1}, {2}):
        raise RuntimeError("candidate texts must all encode to one or two tokens")
    if len({tuple(sequence) for sequence in token_sequences}) != target_classes:
        raise RuntimeError("candidate token sequences must be distinct")
    prompt_style = config.get("prompt_style", "affine_readings")
    foundation_prompts = [
        foundation_prompt(row, prompt_style, candidate_texts) for row in rows
    ]
    ascent_prompts = [
        ascent_prompt(row, rounds, eta, prompt_style, candidate_texts) for row in rows
    ]
    scoring_prefix = config.get("scoring_prefix", "")
    rendered_foundation = [
        render_chat_prompt(tokenizer, value, endpoint) + scoring_prefix
        for value in foundation_prompts
    ]
    rendered_ascent = [
        render_chat_prompt(tokenizer, value, endpoint) + scoring_prefix
        for value in ascent_prompts
    ]
    boundary_stable = all(
        tokenizer.encode(rendered + candidate, add_special_tokens=False)
        == tokenizer.encode(rendered, add_special_tokens=False) + sequence
        for rendered in rendered_foundation + rendered_ascent
        for candidate, sequence in zip(candidate_texts, token_sequences, strict=True)
    )
    if not boundary_stable:
        raise RuntimeError("candidate tokenization changes at the scoring boundary")
    maximum_foundation_tokens = max(
        len(tokenizer(value, add_special_tokens=False)["input_ids"])
        for value in rendered_foundation
    )
    maximum_ascent_tokens = max(
        len(tokenizer(value, add_special_tokens=False)["input_ids"])
        for value in rendered_ascent
    )
    if max(maximum_foundation_tokens, maximum_ascent_tokens) > config["context_tokens"]:
        raise RuntimeError("prompt exceeds context")
    model = (
        AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=torch.bfloat16)
        .to("cuda")
        .eval()
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model_identity = validate_loaded_model(model, endpoint, context_tokens=config["context_tokens"])
    preflight = {
        "rows": len(rows),
        "candidate_texts": config["candidate_texts"],
        "candidate_token_sequences": token_sequences,
        "candidate_sequence_length": next(iter(sequence_lengths)),
        "candidate_support_valid": True,
        "candidate_boundary_stable": boundary_stable,
        "distinct_single_token_candidates": (
            sequence_lengths == {1}
            and len({sequence[0] for sequence in token_sequences}) == target_classes
        ),
        "target_values_not_rendered_by_prompt_builder": True,
        "exact_posterior_nll": exact_value,
        "maximum_foundation_tokens": maximum_foundation_tokens,
        "maximum_ascent_tokens": maximum_ascent_tokens,
        "generation_invoked": False,
        "pass": True,
    }
    environment = {
        "hostname": platform.node(),
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_dirty": bool(git_value(["status", "--porcelain"])),
        "audit_rerun": audit_rerun,
    }
    if preflight_only:
        payload = {
            "schema_version": 1,
            "status": "score_free_candidate_nll_preflight",
            "config_sha256": sha256_file(config_path),
            "panel": panel,
            "endpoint": endpoint,
            "rounds": rounds,
            "model_files": model_files,
            "model_identity": model_identity,
            "runtime_dependencies": runtime_dependencies,
            "preflight": preflight,
            "environment": environment,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return

    torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()
    foundation_probabilities = candidate_probabilities(
        model, tokenizer, rendered_foundation, token_sequences, endpoint["batch_size"]
    )
    ascent_probabilities = candidate_probabilities(
        model, tokenizer, rendered_ascent, token_sequences, endpoint["batch_size"]
    )
    targets = np.asarray([row["target"] for row in rows], dtype=np.int64)
    indices = np.arange(targets.size)
    foundation_nll = -np.log(np.clip(foundation_probabilities[indices, targets], 1e-300, 1.0))
    ascent_nll = -np.log(np.clip(ascent_probabilities[indices, targets], 1e-300, 1.0))
    gain_nll = foundation_nll - ascent_nll
    foundation_answers = foundation_probabilities.argmax(axis=1)
    ascent_answers = ascent_probabilities.argmax(axis=1)
    predictions = []
    for index, row in enumerate(rows):
        predictions.append(
            {
                "row_id": row["row_id"],
                "episode_id": row["episode_id"],
                "target": row["target"],
                "foundation_probabilities": foundation_probabilities[index].tolist(),
                "ascent_probabilities": ascent_probabilities[index].tolist(),
                "foundation_answer": int(foundation_answers[index]),
                "ascent_answer": int(ascent_answers[index]),
                "foundation_nll": float(foundation_nll[index]),
                "ascent_nll": float(ascent_nll[index]),
                "gain_nll": float(gain_nll[index]),
                "retained_first_observations": row["first_observations"][:rounds],
                "retained_second_observations": row["second_observations"][:rounds],
            }
        )
    environment.update(
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        }
    )
    payload = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "status": (
            "posthoc_cross_node_candidate_audit_no_independence_claim"
            if audit_rerun
            else config["status"]
        ),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(config_path),
        "data_sha256": sha256_file(data_path),
        "panel": panel,
        "endpoint": endpoint,
        "rounds": rounds,
        "model_files": model_files,
        "model_identity": model_identity,
        "preflight": preflight,
        "foundation_nll": summary(foundation_nll),
        "ascent_nll": summary(ascent_nll),
        "gain_nll": summary(gain_nll),
        "foundation_accuracy": float(np.mean(foundation_answers == targets)),
        "ascent_accuracy": float(np.mean(ascent_answers == targets)),
        "predictions": predictions,
        "systems": {
            "wall_seconds": time.perf_counter() - wall_start,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "environment": environment,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--audit-rerun", action="store_true")
    args = parser.parse_args()
    run(
        args.config,
        args.data_root,
        args.panel,
        args.endpoint,
        args.model_dir,
        args.rounds,
        args.output,
        preflight_only=args.preflight_only,
        audit_rerun=args.audit_rerun,
    )


if __name__ == "__main__":
    main()
