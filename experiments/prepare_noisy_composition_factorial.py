#!/usr/bin/env python3
"""Freeze a post-training noisy-evidence composition benchmark before scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_reading(rng: np.random.Generator, value: int, labels: int, eta: float) -> int:
    if rng.random() >= eta:
        return value
    distractors = [candidate for candidate in range(labels) if candidate != value]
    return int(rng.choice(distractors))


def posterior(observations: list[int], labels: int, eta: float) -> np.ndarray:
    probabilities = np.full(labels, 1.0 / labels, dtype=np.float64)
    for observation in observations:
        likelihood = np.full(labels, eta / (labels - 1), dtype=np.float64)
        likelihood[observation] = 1.0 - eta
        probabilities *= likelihood
        probabilities /= probabilities.sum()
    return probabilities


def target_probability(
    first: np.ndarray, second: np.ndarray, target: int, labels: int
) -> float:
    probability = 0.0
    for left in range(labels):
        for right in range(labels):
            if (2 * left + right) % labels == target:
                probability += float(first[left] * second[right])
    return probability


def make_panel(
    *, seed: int, panel_index: int, rows: int, labels: int, eta: float, max_rounds: int
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    result: list[dict[str, Any]] = []
    for row_index in range(rows):
        first = int(rng.integers(labels))
        second = int(rng.integers(labels))
        first_observations = [
            sample_reading(rng, first, labels, eta) for _ in range(max_rounds)
        ]
        second_observations = [
            sample_reading(rng, second, labels, eta) for _ in range(max_rounds)
        ]
        episode_id = f"{int(rng.integers(0, 2**63, dtype=np.int64)):016x}"
        result.append(
            {
                "row_id": f"noisy_comp_p{panel_index:02d}_r{row_index:03d}",
                "episode_id": episode_id,
                "first_value": first,
                "second_value": second,
                "first_observations": first_observations,
                "second_observations": second_observations,
                "target": (2 * first + second) % labels,
            }
        )
    return result


def exact_nll(rows: list[dict[str, Any]], rounds: int, labels: int, eta: float) -> float:
    losses = []
    for row in rows:
        first = posterior(row["first_observations"][:rounds], labels, eta)
        second = posterior(row["second_observations"][:rounds], labels, eta)
        probability = target_probability(first, second, row["target"], labels)
        losses.append(-np.log(max(probability, np.finfo(np.float64).tiny)))
    return float(np.mean(losses))


def exact_sum_nll(
    rows: list[dict[str, Any]], rounds: int, labels: int, eta: float
) -> float:
    losses = []
    for row in rows:
        first = posterior(row["first_observations"][:rounds], labels, eta)
        second = posterior(row["second_observations"][:rounds], labels, eta)
        probability = 0.0
        for left in range(labels):
            for right in range(labels):
                if left + right == row["target"]:
                    probability += float(first[left] * second[right])
        losses.append(-np.log(max(probability, np.finfo(np.float64).tiny)))
    return float(np.mean(losses))


def freeze(output_root: Path, config_path: Path) -> dict[str, Any]:
    source = json.loads(
        (ROOT / "configs" / "babilong_falcon3_8k_raw_235_factorial.json").read_text()
    )
    labels = 8
    eta = 0.28
    rounds = [2, 3, 5]
    rows_per_panel = 120
    panel_names = [f"noisy_composition_panel_{index}" for index in range(1, 11)]
    panel_hashes: dict[str, str] = {}
    panel_paths: dict[str, str] = {}
    exact_by_panel: dict[str, dict[str, float]] = {}
    output_root.mkdir(parents=True, exist_ok=True)
    for index, panel in enumerate(panel_names, start=1):
        rows = make_panel(
            seed=2026081500 + index,
            panel_index=index,
            rows=rows_per_panel,
            labels=labels,
            eta=eta,
            max_rounds=max(rounds),
        )
        path = output_root / f"{panel}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        panel_hashes[panel] = sha256_file(path)
        panel_paths[panel] = str(path.relative_to(ROOT / "data"))
        exact_by_panel[panel] = {
            str(value): exact_nll(rows, value, labels, eta) for value in rounds
        }
        curve = [exact_by_panel[panel][str(value)] for value in rounds]
        if not all(high < low for low, high in zip(curve[:-1], curve[1:], strict=True)):
            raise RuntimeError(f"finite-panel exact information curve is not strict: {panel}")

    manifest_path = output_root / "MANIFEST.json"
    manifest = {
        "schema_version": 1,
        "generator": "prepare_noisy_composition_factorial.py",
        "seed_rule": "2026081500 + one_based_panel_index",
        "labels": labels,
        "crossover_probability": eta,
        "rounds": rounds,
        "rows_per_panel": rows_per_panel,
        "panel_sha256": panel_hashes,
        "exact_posterior_mean_nll_by_panel_and_rounds": exact_by_panel,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    endpoints = []
    batch_sizes = {
        "falcon3-1b-instruct": 32,
        "falcon3-3b-instruct": 16,
        "falcon3-7b-instruct": 8,
    }
    for endpoint in source["endpoints"]:
        copied = dict(endpoint)
        copied["batch_size"] = batch_sizes[copied["name"]]
        endpoints.append(copied)
    ratios = [
        slots / endpoint["model_parameters"]
        for slots, endpoint in zip(rounds, endpoints, strict=True)
    ]
    if not all(high < low for low, high in zip(ratios[:-1], ratios[1:], strict=True)):
        raise RuntimeError("co-scaled state/model ratios must strictly decrease")

    config = {
        "schema_version": 1,
        "experiment": "ascent_posttraining_noisy_composition_falcon3_factorial",
        "status": "prospective_development_frozen_before_any_model_score",
        "frozen_at": "2026-08-15T08:30:00+08:00",
        "construction": {
            "num_values": labels,
            "crossover_probability": eta,
            "channel": "independent categorical symmetric channel per register and round",
            "target_rule": "(2 * first_value + second_value) modulo 8",
            "fresh_information": "values and observations sampled after checkpoint selection",
        },
        "panels": panel_names,
        "development_panels": panel_names[:1],
        "confirmation_panels": panel_names[1:],
        "panel_path_by_name": panel_paths,
        "panel_sha256_by_name": panel_hashes,
        "data_manifest_sha256": sha256_file(manifest_path),
        "exact_posterior_mean_nll_by_panel_and_rounds": exact_by_panel,
        "evaluation_samples": rows_per_panel,
        "context_tokens": 2048,
        "max_new_tokens": 8,
        "runtime_dependencies": source["runtime_dependencies"],
        "endpoints": endpoints,
        "factorial": {
            "endpoints": [endpoint["name"] for endpoint in endpoints],
            "state_rounds": rounds,
            "co_scaled_cells": [
                f"{endpoint['name']}@rounds{slots}"
                for endpoint, slots in zip(endpoints, rounds, strict=True)
            ],
            "co_scaled_rounds_per_parameter": ratios,
            "strictly_decreasing_relative_state": True,
        },
        "execution": source["execution"],
        "development_gate": (
            "All score-free provenance, exact-channel, strict nonredundancy, nested-prefix, "
            "Foundation-invariance, and decreasing-relative-state checks pass; all three "
            "co-scaled gains, both adjacent co-scaled gain increments, and the 7B K5-minus-K3 "
            "gain increment are strictly positive. Interactions are reported but not required."
        ),
        "confirmation_gate": (
            "On nine untouched panels, every registered gain and increment has a positive "
            "two-sided t(8) 95% lower bound and positive Holm-adjusted one-sided direction; "
            "no panel, row, invalid output, regression, or outlier may be removed."
        ),
        "selection_and_stopping_rule": (
            "The synthetic post-training rows, channel, 2/3/5 law, prompts, endpoints, "
            "metrics, and gates are frozen before any model score. Only panel 1 is development. "
            "A failed development forbids every confirmation panel and forbids prompt, channel, "
            "state-law, parser, or gate tuning on visible rows."
        ),
        "interpretation": (
            "A proposition-matched neural readout benchmark. The exact posterior independently "
            "certifies that every append-only refinement contains target information; Falcon3 "
            "must infer two latent values from noisy ASCENT evidence and compose them. This is a "
            "synthetic mechanism test, not a public long-context leaderboard replacement."
        ),
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data" / "posttraining_noisy_composition",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "posttraining_noisy_composition_falcon3.json",
    )
    args = parser.parse_args()
    config = freeze(args.output_root, args.config)
    print(json.dumps({"config": str(args.config), "panels": len(config["panels"])}, indent=2))


if __name__ == "__main__":
    main()
