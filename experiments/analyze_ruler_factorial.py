#!/usr/bin/env python3
"""Sample-weighted 2x2 foundation-by-memory analysis for RULER tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_ruler_niah import TASKS, bootstrap, sample_means, summary


def analyze(
    result_dir: Path,
    data_root: Path,
    tokenizer_path: str,
    tasks: list[str] | None = None,
    calibration_samples: int = 20,
    output_separator: str = ", ",
) -> dict[str, Any]:
    tasks = TASKS if tasks is None else tasks
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    lengths: dict[str, list[int]] = {}
    for task in tasks:
        rows = [json.loads(line) for line in (
            data_root / task / "validation.jsonl"
        ).read_text().splitlines()][calibration_samples:]
        lengths[task] = [
            len(tokenizer(" " + output_separator.join(row["outputs"]), add_special_tokens=False)["input_ids"])
            for row in rows
        ]

    endpoint_arrays = {
        "pythia-410m": np.load(result_dir / "ruler_factorial_410m.npz"),
        "pythia-2.8b": np.load(result_dir / "ruler_factorial_2p8b.npz"),
    }
    metadata = {
        "pythia-410m": json.loads((result_dir / "ruler_factorial_410m.json").read_text()),
        "pythia-2.8b": json.loads((result_dir / "ruler_factorial_2p8b.json").read_text()),
    }
    config_hashes = {value["config_sha256"] for value in metadata.values()}
    commits = {value["runtime"]["git_commit"] for value in metadata.values()}
    cells: dict[str, Any] = {}
    task_interactions: list[np.ndarray] = []
    all_memory_gates = True
    all_interaction_gates = True

    for task_index, task in enumerate(tasks):
        memory_effects: dict[str, np.ndarray] = {}
        task_payload: dict[str, Any] = {}
        for endpoint_index, endpoint in enumerate(("pythia-410m", "pythia-2.8b")):
            arrays = endpoint_arrays[endpoint]
            fixed = sample_means(
                arrays[f"{task}_ascent_fixed_latent_only_gain"], lengths[task]
            )
            rich = sample_means(
                arrays[f"{task}_ascent_scale_latent_only_gain"], lengths[task]
            )
            effect = rich - fixed
            memory_effects[endpoint] = effect
            normal = summary(effect)
            boot = bootstrap(effect, seed=161803 + 100 * task_index + endpoint_index)
            gate = normal["ci95_low"] > 0 and boot["ci95_low"] > 0
            all_memory_gates = all_memory_gates and gate
            task_payload[endpoint] = {
                "fixed_gain": summary(fixed),
                "rich_gain": summary(rich),
                "rich_minus_fixed": normal,
                "rich_minus_fixed_bootstrap": boot,
                "memory_gate_pass": gate,
            }
        interaction = memory_effects["pythia-2.8b"] - memory_effects["pythia-410m"]
        task_interactions.append(interaction)
        interaction_normal = summary(interaction)
        interaction_boot = bootstrap(interaction, seed=271828 + task_index)
        interaction_gate = (
            interaction_normal["ci95_low"] > 0 and interaction_boot["ci95_low"] > 0
        )
        all_interaction_gates = all_interaction_gates and interaction_gate
        task_payload["model_by_memory_interaction"] = interaction_normal
        task_payload["model_by_memory_interaction_bootstrap"] = interaction_boot
        task_payload["interaction_gate_pass"] = interaction_gate
        cells[task] = task_payload

    rng = np.random.default_rng(141421)
    hierarchical_draws = np.stack([
        values[rng.integers(0, values.size, size=(20_000, values.size))].mean(axis=1)
        for values in task_interactions
    ], axis=1).mean(axis=1)
    low, high = np.quantile(hierarchical_draws, [0.025, 0.975])
    macro_mean = float(np.mean([values.mean() for values in task_interactions]))
    macro = {
        "equal_task_interaction_mean_nats": macro_mean,
        "hierarchical_sample_bootstrap_ci95_low": float(low),
        "hierarchical_sample_bootstrap_ci95_high": float(high),
        "gate_pass": bool(low > 0),
    }
    passed = (
        all_memory_gates and all_interaction_gates and macro["gate_pass"]
        and len(config_hashes) == 1 and len(commits) == 1
    )
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "factorial_contrast": "(Rich64-Fixed16)_2.8B - (Rich64-Fixed16)_410M",
        "weighting": "Answer tokens averaged within sample; tasks tested separately and weighted equally in macro analysis.",
        "tasks": tasks,
        "config_hashes": sorted(config_hashes),
        "runtime_commits": sorted(commits),
        "cells": cells,
        "macro": macro,
        "all_14_memory_effect_gates_pass": all_memory_gates,
        "all_7_model_by_memory_interaction_gates_pass": all_interaction_gates,
        "metric_boundary": "Teacher-forced answer-token NLL, not official autoregressive exact match.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer-path", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--calibration-samples", type=int, default=20)
    parser.add_argument("--output-separator", default=", ")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(
        args.result_dir,
        args.data_root,
        args.tokenizer_path,
        tasks=args.tasks,
        calibration_samples=args.calibration_samples,
        output_separator=args.output_separator,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
