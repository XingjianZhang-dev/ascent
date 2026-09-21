#!/usr/bin/env python3
"""Evaluate frozen Falcon3 endpoints on append-only noisy ASCENT evidence."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
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
)
from experiments.run_natural_repeat import git_value
from experiments.run_ruler_niah import verify_model_artifact


ANSWER_PATTERN = re.compile(r"(?<!\d)([0-7])(?!\d)")
SUM_ANSWER_PATTERN = re.compile(r"(?<!\d)(1[0-4]|[0-9])(?!\d)")
FINAL_ANSWER_PATTERN = re.compile(r"FINAL\s*:\s*([0-7])", re.I)


def extract_answer(text: str, maximum: int = 7) -> int | None:
    if maximum not in {7, 14}:
        raise ValueError("answer maximum must be 7 or 14")
    final_matches = FINAL_ANSWER_PATTERN.findall(text)
    if final_matches:
        return int(final_matches[-1])
    matches = (ANSWER_PATTERN if maximum == 7 else SUM_ANSWER_PATTERN).findall(text)
    return int(matches[-1]) if matches else None


def foundation_prompt(row: dict[str, Any], style: str = "direct") -> str:
    if style == "qwen_cot":
        return (
            "A fresh episode was created after model training. Hidden registers A and B are "
            "independent integers from 0 through 7. Compute R = (2*A + B) modulo 8. "
            "No memory observations are available, so make your best prediction. Reason "
            "briefly, then end with a separate line exactly formatted FINAL: d, where d is "
            "one digit from 0 through 7.\n\n"
            f"Episode: {row['episode_id']}"
        )
    if style == "posterior_replay":
        return (
            "A fresh episode was created after model training. Hidden registers A and B are "
            "integers from 0 through 7 and R = (2*A + B) modulo 8. No ASCENT state is "
            "available in this arm. Reply with only one best-guess digit from 0 through 7; "
            "do not explain.\n"
            f"Episode: {row['episode_id']}\nR:"
        )
    if style == "sum_posterior_replay":
        return (
            "A fresh episode was created after model training. Hidden registers A and B are "
            "integers from 0 through 7. The requested answer is the ordinary integer sum A+B. "
            "No ASCENT state is available in this arm. Reply with only one best-guess integer "
            "from 0 through 14; do not explain.\n"
            f"Episode: {row['episode_id']}\nA+B:"
        )
    if style == "map_sum_replay":
        return (
            "A fresh episode was created after model training. Hidden registers A and B are "
            "integers from 0 through 7. The requested answer is the ordinary integer sum A+B. "
            "No ASCENT decoder output is available in this arm. Reply with only one best-guess "
            "integer from 0 through 14; do not explain.\n"
            f"Episode: {row['episode_id']}\nA+B:"
        )
    if style != "direct":
        raise RuntimeError(f"unknown prompt style: {style}")
    return (
        "A fresh episode was created after model training. Hidden registers A and B are "
        "independent integers from 0 through 7. The requested value is "
        "R = (2*A + B) modulo 8. No memory observations are available in this arm, so "
        "make your best prediction. Reply with exactly one digit from 0 through 7.\n\n"
        f"Episode: {row['episode_id']}\nR:"
    )


def ascent_prompt(
    row: dict[str, Any], rounds: int, eta: float, style: str = "direct"
) -> str:
    first = ", ".join(str(value) for value in row["first_observations"][:rounds])
    second = ", ".join(str(value) for value in row["second_observations"][:rounds])
    correct = 1.0 - eta
    if style == "qwen_cot":
        return (
            "A fresh episode was created after model training. Hidden registers A and B are "
            "independent integers from 0 through 7. ASCENT retained append-only independent "
            f"sensor readings. Each reading equals its register with probability {correct:.2f}; "
            "otherwise it is uniformly one of the other seven digits. Infer the most likely A "
            "and B from all readings, then compute R = (2*A + B) modulo 8. Reason briefly, "
            "then end with a separate line exactly formatted FINAL: d, where d is one digit "
            "from 0 through 7.\n\n"
            f"ASCENT A readings: {first}\nASCENT B readings: {second}\n"
            f"Episode: {row['episode_id']}"
        )
    if style == "posterior_replay":
        first_posterior = posterior(row["first_observations"][:rounds], 8, eta)
        second_posterior = posterior(row["second_observations"][:rounds], 8, eta)
        first_map = int(first_posterior.argmax())
        second_map = int(second_posterior.argmax())
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
            "noisy readings and its certified decoder computed the registered exact posteriors. "
            "Use the decoder MAP values A and B, then compute R = (2*A + B) modulo 8. "
            "Reply with only one digit from 0 through 7; do not explain.\n"
            f"Retained A readings: {first}\nRetained B readings: {second}\n"
            f"Certified A posterior: {first_probabilities}\nCertified A MAP: {first_map}\n"
            f"Certified B posterior: {second_probabilities}\nCertified B MAP: {second_map}\n"
            f"Episode: {row['episode_id']}\nR:"
        )
    if style == "sum_posterior_replay":
        first_posterior = posterior(row["first_observations"][:rounds], 8, eta)
        second_posterior = posterior(row["second_observations"][:rounds], 8, eta)
        first_map = int(first_posterior.argmax())
        second_map = int(second_posterior.argmax())
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
            "noisy readings and its certified decoder computed the registered exact posteriors. "
            "Use the decoder MAP values A and B, then calculate the ordinary integer sum A+B. "
            "Reply with only one integer from 0 through 14; do not explain.\n"
            f"Retained A readings: {first}\nRetained B readings: {second}\n"
            f"Certified A posterior: {first_probabilities}\nCertified A MAP: {first_map}\n"
            f"Certified B posterior: {second_probabilities}\nCertified B MAP: {second_map}\n"
            f"Episode: {row['episode_id']}\nA+B:"
        )
    if style == "map_sum_replay":
        first_posterior = posterior(row["first_observations"][:rounds], 8, eta)
        second_posterior = posterior(row["second_observations"][:rounds], 8, eta)
        first_map = int(first_posterior.argmax())
        second_map = int(second_posterior.argmax())
        return (
            "ASCENT's certified decoder has already aggregated its append-only noisy evidence. "
            "Add the two decoded MAP integers using ordinary addition. Reply with only one "
            "integer from 0 through 14; do not explain.\n"
            f"Decoded A: {first_map}\nDecoded B: {second_map}\nA+B:"
        )
    if style != "direct":
        raise RuntimeError(f"unknown prompt style: {style}")
    return (
        "A fresh episode was created after model training. Hidden registers A and B are "
        "independent integers from 0 through 7. ASCENT retained append-only independent "
        f"sensor readings. Each reading equals its register with probability {correct:.2f}; "
        "otherwise it is uniformly one of the other seven digits. Infer the most likely "
        "register values from all readings, then compute R = (2*A + B) modulo 8. "
        "Example only: if A=3 and B=6, R=4. Reply with exactly one digit from 0 through 7.\n\n"
        f"ASCENT A readings: {first}\nASCENT B readings: {second}\n"
        f"Episode: {row['episode_id']}\nR:"
    )


def summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    se = float(array.std(ddof=1) / np.sqrt(array.size))
    return {
        "samples": int(array.size),
        "mean": float(array.mean()),
        "standard_error": se,
        "ci95_low": float(array.mean() - 1.96 * se),
        "ci95_high": float(array.mean() + 1.96 * se),
    }


def validate_dependencies(config: dict[str, Any]) -> dict[str, str]:
    expected = config["runtime_dependencies"]
    actual = {name: importlib.metadata.version(name) for name in expected}
    if actual != expected:
        raise RuntimeError(f"runtime dependency mismatch: {actual!r} != {expected!r}")
    return actual


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
    runtime_dependencies = validate_dependencies(config)
    if panel not in config["panels"]:
        raise RuntimeError("panel is absent from frozen config")
    if rounds not in config["factorial"]["state_rounds"]:
        raise RuntimeError("round count is absent from frozen config")
    data_path = data_root / config["panel_path_by_name"][panel]
    if sha256_file(data_path) != config["panel_sha256_by_name"][panel]:
        raise RuntimeError("panel hash mismatch")
    manifest_path = data_root / config.get(
        "data_manifest_path", "posttraining_noisy_composition/MANIFEST.json"
    )
    if sha256_file(manifest_path) != config["data_manifest_sha256"]:
        raise RuntimeError("data manifest hash mismatch")
    rows = [json.loads(line) for line in data_path.read_text().splitlines()]
    if len(rows) != int(config["evaluation_samples"]):
        raise RuntimeError("unexpected row count")
    construction = config["construction"]
    labels = int(construction["num_values"])
    eta = float(construction["crossover_probability"])
    target_rule = construction["target_rule"]
    for row in rows:
        expected_target = (
            row["first_value"] + row["second_value"]
            if target_rule == "first_value + second_value"
            else (2 * row["first_value"] + row["second_value"]) % labels
        )
        if row["target"] != expected_target:
            raise RuntimeError(f"target mismatch: {row['row_id']}")
        if len(row["first_observations"]) < max(config["factorial"]["state_rounds"]):
            raise RuntimeError("incomplete first observation prefix")
        if len(row["second_observations"]) < max(config["factorial"]["state_rounds"]):
            raise RuntimeError("incomplete second observation prefix")
    expected_exact_nll = config["exact_posterior_mean_nll_by_panel_and_rounds"][panel][
        str(rounds)
    ]
    actual_exact_nll = (
        exact_sum_nll(rows, rounds, labels, eta)
        if target_rule == "first_value + second_value"
        else exact_nll(rows, rounds, labels, eta)
    )
    if actual_exact_nll != expected_exact_nll:
        raise RuntimeError("exact posterior audit differs from frozen value")

    endpoints = {endpoint["name"]: endpoint for endpoint in config["endpoints"]}
    endpoint = endpoints[endpoint_name]
    model_files = verify_model_artifact(model_dir, frozen_model_artifact_spec(endpoint))
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    configure_decoder_tokenizer(tokenizer)
    prompt_style = config.get("prompt_style", "direct")
    foundation_prompts = [foundation_prompt(row, prompt_style) for row in rows]
    ascent_prompts = [ascent_prompt(row, rounds, eta, prompt_style) for row in rows]
    rendered_foundation = [render_chat_prompt(tokenizer, value, endpoint) for value in foundation_prompts]
    rendered_ascent = [render_chat_prompt(tokenizer, value, endpoint) for value in ascent_prompts]
    lengths = {
        "maximum_foundation_tokens": max(
            len(tokenizer(value, add_special_tokens=False)["input_ids"])
            for value in rendered_foundation
        ),
        "maximum_ascent_tokens": max(
            len(tokenizer(value, add_special_tokens=False)["input_ids"])
            for value in rendered_ascent
        ),
    }
    cap = int(config["context_tokens"]) - int(config["max_new_tokens"])
    if max(lengths.values()) > cap:
        raise RuntimeError("registered prompt exceeds frozen input cap")

    model = (
        AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, dtype=torch.bfloat16
        )
        .to("cuda")
        .eval()
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model_identity = validate_loaded_model(
        model, endpoint, context_tokens=int(config["context_tokens"])
    )
    preflight = {
        "data_rows": len(rows),
        "target_rule_exact": True,
        "strict_prefix_transitions": all(
            high > low
            for low, high in zip(
                config["factorial"]["state_rounds"][:-1],
                config["factorial"]["state_rounds"][1:],
                strict=True,
            )
        ),
        "exact_posterior_nll": actual_exact_nll,
        "prompt_lengths": lengths,
        "prompt_cap": cap,
        "generation_invoked": not preflight_only,
        "pass": True,
    }
    if preflight_only:
        payload = {
            "schema_version": 1,
            "status": "score_free_noisy_composition_preflight",
            "config_sha256": sha256_file(config_path),
            "panel": panel,
            "endpoint": endpoint,
            "rounds": rounds,
            "model_files": model_files,
            "model_identity": model_identity,
            "runtime_dependencies": runtime_dependencies,
            "preflight": preflight,
            "environment": {
                "git_commit": git_value(["rev-parse", "HEAD"]),
                "git_dirty": bool(git_value(["status", "--porcelain"])),
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return

    batch_size = int(endpoint["batch_size"])
    max_new_tokens = int(config["max_new_tokens"])

    def decode(prompts: list[str]) -> tuple[list[str], list[int], float]:
        decoded: list[str] = []
        token_lengths: list[int] = []
        started = time.perf_counter()
        for start in range(0, len(prompts), batch_size):
            rendered = [
                render_chat_prompt(tokenizer, value, endpoint)
                for value in prompts[start : start + batch_size]
            ]
            encoded = tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
            ).to("cuda")
            token_lengths.extend(int(value) for value in encoded.attention_mask.sum(dim=1))
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            suffix = generated[:, encoded.input_ids.shape[1] :]
            decoded.extend(tokenizer.batch_decode(suffix, skip_special_tokens=True))
        torch.cuda.synchronize()
        return decoded, token_lengths, time.perf_counter() - started

    torch.manual_seed(20260815)
    torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()
    foundation_outputs, foundation_lengths, foundation_seconds = decode(foundation_prompts)
    ascent_outputs, ascent_lengths, ascent_seconds = decode(ascent_prompts)
    predictions = []
    foundation_scores: list[float] = []
    ascent_scores: list[float] = []
    for row, foundation_output, ascent_output, foundation_tokens, ascent_tokens in zip(
        rows,
        foundation_outputs,
        ascent_outputs,
        foundation_lengths,
        ascent_lengths,
        strict=True,
    ):
        answer_maximum = 14 if target_rule == "first_value + second_value" else 7
        foundation_answer = extract_answer(foundation_output, answer_maximum)
        ascent_answer = extract_answer(ascent_output, answer_maximum)
        foundation_score = float(foundation_answer == row["target"])
        ascent_score = float(ascent_answer == row["target"])
        foundation_scores.append(foundation_score)
        ascent_scores.append(ascent_score)
        predictions.append(
            {
                "row_id": row["row_id"],
                "episode_id": row["episode_id"],
                "target": row["target"],
                "foundation_output": foundation_output,
                "foundation_answer": foundation_answer,
                "foundation_final_marker_present": bool(
                    FINAL_ANSWER_PATTERN.search(foundation_output)
                ),
                "foundation_score": foundation_score,
                "ascent_output": ascent_output,
                "ascent_answer": ascent_answer,
                "ascent_final_marker_present": bool(
                    FINAL_ANSWER_PATTERN.search(ascent_output)
                ),
                "ascent_score": ascent_score,
                "retained_first_observations": row["first_observations"][:rounds],
                "retained_second_observations": row["second_observations"][:rounds],
                "decoded_first_map": int(
                    posterior(row["first_observations"][:rounds], labels, eta).argmax()
                ),
                "decoded_second_map": int(
                    posterior(row["second_observations"][:rounds], labels, eta).argmax()
                ),
                "foundation_prompt_tokens": foundation_tokens,
                "ascent_prompt_tokens": ascent_tokens,
            }
        )
    gain = [
        ascent - foundation
        for foundation, ascent in zip(foundation_scores, ascent_scores, strict=True)
    ]
    diagnostic_status = (
        "posthoc_cross_node_audit_rerun_no_independence_claim"
        if audit_rerun
        else config["status"]
    )
    payload = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "status": diagnostic_status,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "rounds": rounds,
        "panel": panel,
        "config_sha256": sha256_file(config_path),
        "data_sha256": sha256_file(data_path),
        "model_files": model_files,
        "model_identity": model_identity,
        "preflight": preflight,
        "foundation": summary(foundation_scores),
        "ascent": summary(ascent_scores),
        "gain": summary(gain),
        "final_marker_coverage": {
            "foundation": float(
                np.mean(
                    [row["foundation_final_marker_present"] for row in predictions]
                )
            ),
            "ascent": float(
                np.mean([row["ascent_final_marker_present"] for row in predictions])
            ),
        },
        "answer_coverage": {
            "foundation": float(
                np.mean([row["foundation_answer"] is not None for row in predictions])
            ),
            "ascent": float(
                np.mean([row["ascent_answer"] is not None for row in predictions])
            ),
        },
        "exact_posterior_nll": actual_exact_nll,
        "predictions": predictions,
        "systems": {
            "foundation_seconds": foundation_seconds,
            "ascent_seconds": ascent_seconds,
            "wall_seconds": time.perf_counter() - wall_start,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "git_commit": git_value(["rev-parse", "HEAD"]),
            "git_dirty": bool(git_value(["status", "--porcelain"])),
            "audit_rerun": audit_rerun,
        },
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
