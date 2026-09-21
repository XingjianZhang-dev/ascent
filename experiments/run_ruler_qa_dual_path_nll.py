#!/usr/bin/env python3
"""Official RULER QA NLL evaluation for ASCENT's certified/latent dual path."""

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

from ascent.fusion import (
    apply_true_probability_simplex,
    select_clustered_safe_true_probability_simplex,
)
from ascent.gpt_neox_replay import relative_layer_index
from ascent.qwen2_replay import replay_from_hidden
from ascent.ruler_qa_memory import (
    encode_ruler_qa_state,
    read_ruler_qa_graph,
)
from experiments.run_natural_repeat import git_value, paired_summary, sha256_file
from experiments.run_ruler_niah import verify_model_artifact
from experiments.run_stage_b_endpoint import array_sha256


GRAPH_STATE_NAMES = ("graph_small", "graph_mid", "graph_large")
ARM_NAMES = (*GRAPH_STATE_NAMES, "query_only", "irrelevant")


def gate_payload(gate: Any) -> dict[str, Any]:
    return {
        "enabled": bool(gate.enabled),
        "foundation_weight": float(
            1.0 - gate.certified_weight - gate.latent_weight
        ),
        "certified_weight": float(gate.certified_weight),
        "latent_weight": float(gate.latent_weight),
        "candidate_foundation_weight": float(
            1.0
            - gate.candidate_certified_weight
            - gate.candidate_latent_weight
        ),
        "candidate_certified_weight": float(gate.candidate_certified_weight),
        "candidate_latent_weight": float(gate.candidate_latent_weight),
        "calibration_gain_nats": float(gate.calibration_gain_nats),
        "calibration_gain_lcb95": float(gate.calibration_gain_lcb95),
    }


def compact_evidence_content(state_text: str, question: str) -> str:
    return (
        "ASCENT retrieved the following query-conditioned memory state without "
        "using the answer or support labels. Answer only from this evidence.\n\n"
        f"{state_text}\n\nQuestion: {question}\n"
        "Return only the answer and no explanation."
    )


def resolve_relative_injection_depth(
    registered_depth: float, diagnostic_override: float | None
) -> tuple[float, bool]:
    """Resolve an explicitly labeled post-hoc replay-depth diagnostic."""

    registered = float(registered_depth)
    if not 0.0 < registered < 1.0:
        raise ValueError("registered relative injection depth must lie in (0, 1)")
    if diagnostic_override is None:
        return registered, False
    diagnostic = float(diagnostic_override)
    if not 0.0 < diagnostic < 1.0:
        raise ValueError("diagnostic relative injection depth must lie in (0, 1)")
    return diagnostic, True


def run(
    config_path: Path,
    data_root: Path,
    endpoint_name: str,
    model_dir: Path,
    output: Path,
    arrays_path: Path,
    diagnostic_relative_injection_depth: float | None = None,
) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config: dict[str, Any] = json.loads(config_path.read_text())
    endpoint = {row["name"]: row for row in config["endpoints"]}[endpoint_name]
    task_path = data_root / config["task_name"] / "validation.jsonl"
    if sha256_file(task_path) != config["task_sha256"]:
        raise RuntimeError("RULER QA task hash mismatch")
    rows = [json.loads(line) for line in task_path.read_text().splitlines()]
    if len(rows) != int(config["samples_per_task"]):
        raise RuntimeError("unexpected RULER QA row count")
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

    registered_depth = float(config["relative_injection_depth"])
    executed_depth, diagnostic_depth_override = resolve_relative_injection_depth(
        registered_depth, diagnostic_relative_injection_depth
    )
    layer = relative_layer_index(len(model.model.layers), executed_depth)
    split = int(config["calibration_samples"])
    if not 1 < split < len(rows):
        raise ValueError("calibration split must leave test samples")
    context_limit = int(config["query_context_tokens"])
    epsilon = float(config["epsilon"])
    hidden_size = int(endpoint["hidden_size"])
    if hidden_size != int(model.config.hidden_size):
        raise RuntimeError("registered hidden size does not match checkpoint")

    memory_states = {
        row["name"]: {
            "documents": int(row["documents"]),
            "replay_tokens": int(row["replay_tokens"]),
        }
        for row in config["memory_states"]
    }
    if tuple(memory_states) != GRAPH_STATE_NAMES:
        raise ValueError("memory states must be the registered small/mid/large order")
    scale_state = str(endpoint["scale_state"])
    if scale_state not in memory_states:
        raise ValueError("endpoint scale_state is not registered")
    scale_spec = memory_states[scale_state]
    state_serialization = str(config.get("state_serialization", "document_sketches"))

    prepared: list[dict[str, Any]] = []
    ranking_rows: list[str] = []
    for row in rows:
        read = read_ruler_qa_graph(
            row["input"], seed_documents=int(config["bm25_seed_documents"])
        )
        ranked = {
            **{
                name: read.graph_ranked[: spec["documents"]]
                for name, spec in memory_states.items()
            },
            "query_only": read.query_only_ranked[: scale_spec["documents"]],
            "irrelevant": read.irrelevant_ranked[: scale_spec["documents"]],
        }
        ranking_rows.append(",".join(str(doc.number) for doc in read.graph_ranked))
        prepared.append({"row": row, "read": read, "documents": ranked})

    for item in prepared:
        graph_ids = {
            name: encode_ruler_qa_state(
                tokenizer,
                item["documents"][name],
                item["read"].question,
                mode=state_serialization,
            )[: memory_states[name]["replay_tokens"]]
            for name in GRAPH_STATE_NAMES
        }
        for smaller, larger in zip(GRAPH_STATE_NAMES, GRAPH_STATE_NAMES[1:]):
            if graph_ids[larger][: len(graph_ids[smaller])] != graph_ids[smaller]:
                raise RuntimeError("ASCENT graph state token prefixes are not nested")

    prompt_lengths: dict[str, list[int]] = {
        "foundation": [],
        **{f"{name}_certified": [] for name in ARM_NAMES},
    }
    state_tokens: dict[str, list[int]] = {name: [] for name in ARM_NAMES}
    state_bytes: dict[str, list[int]] = {name: [] for name in ARM_NAMES}
    state_hashes: dict[str, list[str]] = {name: [] for name in ARM_NAMES}

    def ids_with_prefix(content: str, answer_prefix: str) -> Any:
        user_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )[0]
        prefix_ids = tokenizer(
            answer_prefix, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]
        result = torch.cat([user_ids, prefix_ids])
        if int(result.numel()) > context_limit:
            raise RuntimeError("native QA prompt exceeded frozen context limit")
        return result.to("cuda")

    def target_ids(row: dict[str, Any]) -> Any:
        if len(row["outputs"]) != 1:
            raise RuntimeError("the frozen QA panel requires one target answer")
        return tokenizer(
            " " + row["outputs"][0],
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0].to("cuda")

    def true_probabilities(logits: Any, targets: Any) -> np.ndarray:
        index = torch.arange(int(targets.numel()), device="cuda")
        return (
            torch.softmax(logits.float(), dim=-1)[index, targets]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    def evaluate_row(item: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        row = item["row"]
        read = item["read"]
        targets = target_ids(row)
        full_prompt = ids_with_prefix(row["input"], row["answer_prefix"])
        prompt_lengths["foundation"].append(int(full_prompt.numel()))
        full_query = torch.cat([full_prompt, targets[:-1]]).unsqueeze(0)
        full_mask = torch.ones_like(full_query)
        with torch.inference_mode():
            full_output = model(
                input_ids=full_query,
                attention_mask=full_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        start = int(full_prompt.numel()) - 1
        base_logits = full_output.logits[0, start : start + targets.numel()]
        base_true = true_probabilities(base_logits, targets)
        base_replay_hidden = full_output.hidden_states[layer]
        del full_output, base_logits
        arms: dict[str, Any] = {}
        for name in ARM_NAMES:
            budget = (
                memory_states[name]["replay_tokens"]
                if name in memory_states
                else scale_spec["replay_tokens"]
            )
            source_all = torch.tensor(
                encode_ruler_qa_state(
                    tokenizer,
                    item["documents"][name],
                    read.question,
                    mode=state_serialization,
                ),
                dtype=torch.long,
            )
            actual_span = min(budget, int(source_all.numel()))
            if actual_span < 1:
                raise RuntimeError("empty ASCENT state")
            source_ids = source_all[:actual_span].unsqueeze(0).to("cuda")
            state_text = tokenizer.decode(
                source_ids[0], skip_special_tokens=True
            ).strip()
            compact_prompt = ids_with_prefix(
                compact_evidence_content(state_text, read.question),
                row["answer_prefix"],
            )
            prompt_lengths[f"{name}_certified"].append(
                int(compact_prompt.numel())
            )
            compact_query = torch.cat([compact_prompt, targets[:-1]]).unsqueeze(0)
            with torch.inference_mode():
                certified_output = model(
                    input_ids=compact_query,
                    attention_mask=torch.ones_like(compact_query),
                    use_cache=False,
                )
                source_output = model.model(
                    input_ids=source_ids,
                    attention_mask=torch.ones_like(source_ids),
                    output_hidden_states=True,
                    use_cache=False,
                )
                replay_hidden = replay_from_hidden(
                    model,
                    base_replay_hidden,
                    full_mask,
                    source_output.hidden_states[layer],
                    layer,
                )
            certified_start = int(compact_prompt.numel()) - 1
            certified_logits = certified_output.logits[
                0, certified_start : certified_start + targets.numel()
            ]
            latent_hidden = replay_hidden[
                0,
                actual_span + start : actual_span + start + targets.numel(),
            ]
            latent_logits = model.lm_head(latent_hidden)
            certified_true = true_probabilities(certified_logits, targets)
            latent_true = true_probabilities(latent_logits, targets)
            charged_bytes = actual_span * hidden_size * 2 + actual_span * 8
            state_tokens[name].append(actual_span)
            state_bytes[name].append(charged_bytes)
            state_hashes[name].append(
                hashlib.sha256(source_ids[0].cpu().numpy().tobytes()).hexdigest()
            )
            arms[name] = {
                "certified": certified_true,
                "latent": latent_true,
            }
            del certified_output, source_output, replay_hidden
        del base_replay_hidden
        return base_true, arms

    calibration_base: list[np.ndarray] = []
    calibration_paths: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"certified": [], "latent": []} for name in ARM_NAMES
    }
    calibration_sample_ids: list[np.ndarray] = []
    for sample_index, item in enumerate(prepared[:split]):
        base_true, arms = evaluate_row(item)
        calibration_base.append(base_true)
        calibration_sample_ids.append(
            np.full(base_true.size, sample_index, dtype=np.int64)
        )
        for name in ARM_NAMES:
            for path in ("certified", "latent"):
                calibration_paths[name][path].append(arms[name][path])
    base_calibration = np.concatenate(calibration_base)
    sample_ids = np.concatenate(calibration_sample_ids)
    gates = {
        name: select_clustered_safe_true_probability_simplex(
            base_calibration,
            np.concatenate(calibration_paths[name]["certified"]),
            np.concatenate(calibration_paths[name]["latent"]),
            sample_ids,
            grid_size=int(config["fusion_grid_size"]),
            epsilon=epsilon,
        )
        for name in ARM_NAMES
    }

    base_nll: list[float] = []
    fused_nll: dict[str, list[float]] = {name: [] for name in ARM_NAMES}
    candidate_nll: dict[str, list[float]] = {name: [] for name in ARM_NAMES}
    certified_nll: dict[str, list[float]] = {name: [] for name in ARM_NAMES}
    latent_nll: dict[str, list[float]] = {name: [] for name in ARM_NAMES}
    for item in prepared[split:]:
        base_true, arms = evaluate_row(item)
        base_nll.append(float(-np.log(np.clip(base_true, epsilon, 1.0)).mean()))
        for name in ARM_NAMES:
            certified = arms[name]["certified"]
            latent = arms[name]["latent"]
            gate = gates[name]
            fused = apply_true_probability_simplex(
                base_true,
                certified,
                latent,
                gate.certified_weight,
                gate.latent_weight,
            )
            candidate = apply_true_probability_simplex(
                base_true,
                certified,
                latent,
                gate.candidate_certified_weight,
                gate.candidate_latent_weight,
            )
            fused_nll[name].append(
                float(-np.log(np.clip(fused, epsilon, 1.0)).mean())
            )
            candidate_nll[name].append(
                float(-np.log(np.clip(candidate, epsilon, 1.0)).mean())
            )
            certified_nll[name].append(
                float(-np.log(np.clip(certified, epsilon, 1.0)).mean())
            )
            latent_nll[name].append(
                float(-np.log(np.clip(latent, epsilon, 1.0)).mean())
            )

    base_array = np.asarray(base_nll, dtype=np.float64)
    saved: dict[str, np.ndarray] = {"base_nll": base_array}
    arms_payload: dict[str, Any] = {}
    for name in ARM_NAMES:
        fused_array = np.asarray(fused_nll[name], dtype=np.float64)
        candidate_array = np.asarray(candidate_nll[name], dtype=np.float64)
        certified_array = np.asarray(certified_nll[name], dtype=np.float64)
        latent_array = np.asarray(latent_nll[name], dtype=np.float64)
        saved[f"{name}_nll"] = fused_array
        saved[f"{name}_candidate_nll"] = candidate_array
        saved[f"{name}_certified_nll"] = certified_array
        saved[f"{name}_latent_nll"] = latent_array
        arms_payload[name] = {
            "gate": gate_payload(gates[name]),
            "fused_nll": paired_summary(fused_array),
            "paired_fused_nll_gain_over_foundation": paired_summary(
                base_array - fused_array
            ),
            "candidate_fused_nll": paired_summary(candidate_array),
            "paired_candidate_nll_gain_over_foundation": paired_summary(
                base_array - candidate_array
            ),
            "certified_only_nll": paired_summary(certified_array),
            "paired_certified_nll_gain_over_foundation": paired_summary(
                base_array - certified_array
            ),
            "latent_only_nll": paired_summary(latent_array),
            "paired_latent_nll_gain_over_foundation": paired_summary(
                base_array - latent_array
            ),
            "minimum_state_tokens": min(state_tokens[name]),
            "maximum_state_tokens": max(state_tokens[name]),
            "mean_charged_state_bytes": float(np.mean(state_bytes[name])),
            "mean_state_bytes_per_model_parameter": float(
                np.mean(state_bytes[name]) / int(endpoint["model_parameters"])
            ),
            "state_token_ids_sha256": hashlib.sha256(
                "\n".join(state_hashes[name]).encode()
            ).hexdigest(),
        }
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **saved)
    payload = {
        "schema_version": 1,
        "status": (
            "posthoc_replay_depth_diagnostic_no_promotion"
            if diagnostic_depth_override
            else config["status"]
        ),
        "started_at_utc": started,
        "task_name": config["task_name"],
        "task_sha256": sha256_file(task_path),
        "config_sha256": sha256_file(config_path),
        "endpoint": endpoint,
        "architecture": "query_conditioned_compact_evidence_plus_qwen2_latent_replay",
        "relative_injection_depth": executed_depth,
        "registered_relative_injection_depth": registered_depth,
        "diagnostic_relative_injection_depth_override": diagnostic_depth_override,
        "injection_layer": layer,
        "calibration_samples": split,
        "test_samples": len(rows) - split,
        "foundation_nll": paired_summary(base_array),
        "arms": arms_payload,
        "fixed_state": "graph_small",
        "scale_state": scale_state,
        "memory_states": memory_states,
        "state_serialization": state_serialization,
        "rankings_sha256": hashlib.sha256(
            "\n".join(ranking_rows).encode()
        ).hexdigest(),
        "answer_and_support_labels_absent_from_reader": True,
        "state_is_shared_by_certified_and_latent_paths": True,
        "exact_nested_graph_state_token_prefixes": True,
        "native_context": True,
        "prompt_lengths": {
            name: {"minimum": min(values), "maximum": max(values)}
            for name, values in prompt_lengths.items()
        },
        "state_accounting": (
            "charged bytes = replay_tokens * hidden_size * 2 bfloat16 bytes "
            "+ replay_tokens * 8 int64 token-id bytes"
        ),
        "arrays_sha256": {
            name: array_sha256(values) for name, values in saved.items()
        },
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
        "arrays_path": str(arrays_path),
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
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--diagnostic-relative-injection-depth", type=float)
    args = parser.parse_args()
    run(
        args.config,
        args.data_root,
        args.endpoint,
        args.model_dir,
        args.output,
        args.arrays,
        args.diagnostic_relative_injection_depth,
    )


if __name__ == "__main__":
    main()
