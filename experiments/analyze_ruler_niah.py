#!/usr/bin/env python3
"""Sample-weighted audit of independent-seed RULER NIAH confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer


TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3", "niah_multiquery",
)


def summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    se = float(array.std(ddof=1) / np.sqrt(array.size))
    return {
        "mean_nats_per_sample_token_average": mean,
        "standard_error": se,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
    }


def bootstrap(values: np.ndarray, *, seed: int, replicates: int = 20_000) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = array[rng.integers(0, array.size, size=(replicates, array.size))].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return {
        "replicates": replicates,
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def sample_means(token_values: np.ndarray, lengths: list[int]) -> np.ndarray:
    values = np.asarray(token_values, dtype=np.float64)
    if values.size != sum(lengths):
        raise RuntimeError(f"token count {values.size} does not match registered targets {sum(lengths)}")
    boundaries = np.cumsum([0, *lengths])
    return np.asarray([
        values[boundaries[index] : boundaries[index + 1]].mean()
        for index in range(len(lengths))
    ])


def analyze(result_dir: Path, data_root: Path, tokenizer_path: str) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    lengths: dict[str, list[int]] = {}
    for task in TASKS:
        rows = [
            json.loads(line)
            for line in (data_root / task / "validation.jsonl").read_text().splitlines()
        ][20:]
        lengths[task] = [
            len(tokenizer(" " + ", ".join(row["outputs"]), add_special_tokens=False)["input_ids"])
            for row in rows
        ]
    cells: dict[str, Any] = {}
    macro_latent: dict[str, list[np.ndarray]] = {"pythia-410m": [], "pythia-2.8b": []}
    macro_interaction: list[np.ndarray] = []
    all_latent_pass = True
    all_interaction_pass = True
    config_hashes: set[str] = set()
    commits: set[str] = set()

    for endpoint, stem in (("pythia-410m", "ruler_410m"), ("pythia-2.8b", "ruler_2p8b")):
        metadata = json.loads((result_dir / f"{stem}_confirm.json").read_text())
        arrays = np.load(result_dir / f"{stem}_confirm.npz")
        config_hashes.add(metadata["config_sha256"])
        commits.add(metadata["runtime"]["git_commit"])
        for task_index, task in enumerate(TASKS):
            latent = sample_means(
                arrays[f"{task}_ascent_scale_latent_only_gain"], lengths[task]
            )
            latent_summary = summary(latent)
            latent_bootstrap = bootstrap(latent, seed=271828 + task_index)
            latent_pass = latent_summary["ci95_low"] > 0 and latent_bootstrap["ci95_low"] > 0
            all_latent_pass = all_latent_pass and latent_pass
            macro_latent[endpoint].append(latent)
            cell: dict[str, Any] = {
                "samples": int(latent.size),
                "latent_only_scale_gain": latent_summary,
                "latent_only_scale_gain_bootstrap": latent_bootstrap,
                "latent_gate_pass": latent_pass,
            }
            if endpoint == "pythia-2.8b":
                fixed = sample_means(
                    arrays[f"{task}_ascent_fixed_latent_only_gain"], lengths[task]
                )
                interaction = latent - fixed
                interaction_summary = summary(interaction)
                interaction_bootstrap = bootstrap(
                    interaction, seed=314159 + task_index
                )
                interaction_pass = (
                    interaction_summary["ci95_low"] > 0
                    and interaction_bootstrap["ci95_low"] > 0
                )
                all_interaction_pass = all_interaction_pass and interaction_pass
                macro_interaction.append(interaction)
                cell["latent_only_scale_minus_fixed"] = interaction_summary
                cell["latent_only_scale_minus_fixed_bootstrap"] = interaction_bootstrap
                cell["interaction_gate_pass"] = interaction_pass
            cells[f"{endpoint}/{task}"] = cell

    macro: dict[str, Any] = {}
    rng = np.random.default_rng(161803)
    for endpoint, task_arrays in macro_latent.items():
        task_means = np.asarray([values.mean() for values in task_arrays])
        draws = np.stack([
            values[rng.integers(0, values.size, size=(20_000, values.size))].mean(axis=1)
            for values in task_arrays
        ], axis=1).mean(axis=1)
        low, high = np.quantile(draws, [0.025, 0.975])
        macro[endpoint] = {
            "equal_task_mean_nats": float(task_means.mean()),
            "hierarchical_sample_bootstrap_ci95_low": float(low),
            "hierarchical_sample_bootstrap_ci95_high": float(high),
        }
    task_means = np.asarray([values.mean() for values in macro_interaction])
    draws = np.stack([
        values[rng.integers(0, values.size, size=(20_000, values.size))].mean(axis=1)
        for values in macro_interaction
    ], axis=1).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    macro["pythia-2.8b_scale_minus_fixed"] = {
        "equal_task_mean_nats": float(task_means.mean()),
        "hierarchical_sample_bootstrap_ci95_low": float(low),
        "hierarchical_sample_bootstrap_ci95_high": float(high),
    }
    passed = all_latent_pass and all_interaction_pass and len(config_hashes) == 1 and len(commits) == 1
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "weighting": "Each answer is averaged over its tokens before every task-level test; macro results weight seven tasks equally.",
        "config_hashes": sorted(config_hashes),
        "runtime_commits": sorted(commits),
        "cells": cells,
        "macro": macro,
        "all_14_latent_task_endpoint_gates_pass": all_latent_pass,
        "all_7_large_scale_minus_fixed_gates_pass": all_interaction_pass,
        "metric_boundary": "Teacher-forced answer-token NLL, not official autoregressive exact match.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer-path", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(args.result_dir, args.data_root, args.tokenizer_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
