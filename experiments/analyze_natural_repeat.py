#!/usr/bin/env python3
"""Audit and aggregate preregistered natural-text ASCENT confirmations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    se = float(array.std(ddof=1) / np.sqrt(array.size))
    return {
        "mean_nats": mean,
        "standard_error": se,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
    }


def block_bootstrap(
    values: np.ndarray, *, block_size: int, seed: int, replicates: int = 10_000
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size % block_size:
        raise ValueError("registered test size must be divisible by every audit block size")
    blocks = array.reshape(-1, block_size).mean(axis=1)
    rng = np.random.default_rng(seed)
    sampled = blocks[rng.integers(0, blocks.size, size=(replicates, blocks.size))].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return {
        "block_size_events": block_size,
        "num_blocks": int(blocks.size),
        "replicates": replicates,
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def analyze(result_dir: Path) -> dict[str, Any]:
    names = {
        "wikitext103": {"pythia-410m": "wiki_410m", "pythia-2.8b": "wiki_2p8b"},
        "pg19": {"pythia-410m": "pg19_410m", "pythia-2.8b": "pg19_2p8b"},
    }
    cells: dict[str, Any] = {}
    pooled_gain: dict[str, list[np.ndarray]] = {name: [] for name in names["wikitext103"]}
    pooled_interaction: list[np.ndarray] = []
    config_hashes: set[str] = set()
    commits: set[str] = set()
    all_primary_pass = True
    all_block_pass = True
    epsilon = 1e-12

    for dataset, endpoints in names.items():
        event_hashes: set[str] = set()
        for endpoint, stem in endpoints.items():
            json_path = result_dir / f"{stem}_confirm.json"
            array_path = result_dir / f"{stem}_confirm.npz"
            metadata = json.loads(json_path.read_text())
            arrays = np.load(array_path)
            gain = arrays["ascent_scale_nll_gain"]
            interaction = gain - arrays["ascent_fixed_nll_gain"]
            base = np.clip(arrays["foundation_true_probability"], epsilon, 1.0)
            single_gains = {}
            for name, key in (
                ("certified_only", "certified_only_fused_true_probability"),
                ("latent_only", "latent_only_fused_true_probability"),
            ):
                single = np.clip(arrays[key], epsilon, 1.0)
                single_gains[name] = float((np.log(single) - np.log(base)).mean())
            best_single_name = max(single_gains, key=single_gains.get)
            primary = summary(gain)
            interaction_summary = summary(interaction)
            block_audits = [
                block_bootstrap(gain, block_size=size, seed=20260813 + size)
                for size in (64, 256, 1024)
            ]
            cell_pass = primary["ci95_low"] > 0.0
            block_pass = all(row["ci95_low"] > 0.0 for row in block_audits)
            if endpoint == "pythia-2.8b":
                cell_pass = cell_pass and interaction_summary["ci95_low"] > 0.0
                pooled_interaction.append(interaction)
            all_primary_pass = all_primary_pass and cell_pass
            all_block_pass = all_block_pass and block_pass
            pooled_gain[endpoint].append(gain)
            config_hashes.add(metadata["config_sha256"])
            commits.add(metadata["runtime"]["git_commit"])
            event_hashes.add(metadata["event_protocol"]["event_identity_sha256"])
            cells[f"{dataset}/{endpoint}"] = {
                "primary_ascent_scale_gain": primary,
                "scale_minus_fixed": interaction_summary,
                "raw_exact_accuracy": metadata["matched_raw_exact_retrieval"]["top1_accuracy"],
                "dual_minus_best_single_mean_nats": primary["mean_nats"] - single_gains[best_single_name],
                "best_single_path": best_single_name,
                "single_path_gains": single_gains,
                "block_bootstrap_sensitivity": block_audits,
                "primary_gate_pass": cell_pass,
                "block_sensitivity_pass": block_pass,
                "json_sha256": sha256_file(json_path),
                "arrays_sha256": sha256_file(array_path),
            }
        if len(event_hashes) != 1:
            raise RuntimeError(f"endpoint event identities differ for {dataset}")

    pooled = {
        endpoint: summary(np.concatenate(values)) for endpoint, values in pooled_gain.items()
    }
    pooled["pythia-2.8b_scale_minus_fixed"] = summary(np.concatenate(pooled_interaction))
    return {
        "schema_version": 1,
        "status": "passed" if all_primary_pass and all_block_pass else "failed",
        "config_hashes": sorted(config_hashes),
        "runtime_commits": sorted(commits),
        "endpoint_event_identity_match_within_dataset": True,
        "cells": cells,
        "equal_event_weight_pooled": pooled,
        "all_preregistered_primary_gates_pass": all_primary_pass,
        "all_block_bootstrap_sensitivity_gates_pass": all_block_pass,
        "scope_boundary": (
            "Results apply to causally retrieved repeated-context events, not unconditional "
            "whole-corpus perplexity. Absolute gain remains smaller at 2.8B than 410M."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(args.result_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
