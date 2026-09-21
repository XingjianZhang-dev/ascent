#!/usr/bin/env python3
"""Freeze a fresh Qwen2.5 0.5B/3B/7B candidate-NLL factorial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.prepare_noisy_composition_factorial import (
    ROOT,
    exact_nll,
    make_panel,
    sha256_file,
)


def select_endpoint(config: dict, name: str) -> dict:
    matches = [row for row in config["endpoints"] if row["name"] == name]
    if len(matches) != 1:
        raise RuntimeError(f"endpoint lookup failed: {name}")
    return dict(matches[0])


def freeze(output_root: Path, config_path: Path) -> dict:
    small_source = json.loads(
        (ROOT / "configs" / "babilong_qwen2p5_canonical_coscale_16k_confirmatory.json").read_text()
    )
    large_source = json.loads(
        (ROOT / "configs" / "babilong_around7b_16k_extension_confirmatory.json").read_text()
    )
    runtime_source = json.loads(
        (ROOT / "configs" / "posttraining_noisy_composition_candidate_falcon3.json").read_text()
    )
    endpoint_names = [
        "qwen2p5-0p5b-instruct",
        "qwen2p5-3b-instruct",
        "qwen2p5-7b-instruct",
    ]
    endpoints = [
        select_endpoint(small_source, endpoint_names[0]),
        select_endpoint(small_source, endpoint_names[1]),
        select_endpoint(large_source, endpoint_names[2]),
    ]
    for endpoint, batch_size, state_slots in zip(
        endpoints, [32, 16, 8], [2, 3, 5], strict=True
    ):
        endpoint["batch_size"] = batch_size
        endpoint.setdefault("model_type", "qwen2")
        endpoint.setdefault("architecture", "Qwen2ForCausalLM")
        endpoint.setdefault("native_context_tokens", 32768)
        endpoint["state_fact_slots"] = state_slots

    labels = 8
    eta = 0.28
    rounds = [2, 3, 5]
    rows_per_panel = 512
    panels = [f"qwen_candidate_noisy_panel_{index}" for index in range(1, 11)]
    paths = {}
    hashes = {}
    exact_by_panel = {}
    output_root.mkdir(parents=True, exist_ok=True)
    for index, panel in enumerate(panels, start=1):
        rows = make_panel(
            seed=2026081700 + index,
            panel_index=index,
            rows=rows_per_panel,
            labels=labels,
            eta=eta,
            max_rounds=max(rounds),
        )
        for row_index, row in enumerate(rows):
            row["row_id"] = f"qwen_candidate_noisy_p{index:02d}_r{row_index:04d}"
        path = output_root / f"{panel}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        paths[panel] = str(path.relative_to(ROOT / "data"))
        hashes[panel] = sha256_file(path)
        exact_by_panel[panel] = {
            str(value): exact_nll(rows, value, labels, eta) for value in rounds
        }
        curve = [exact_by_panel[panel][str(value)] for value in rounds]
        if not all(high < low for low, high in zip(curve[:-1], curve[1:], strict=True)):
            raise RuntimeError(f"non-strict exact curve: {panel}")
    manifest_path = output_root / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generator": "prepare_noisy_composition_qwen_candidate.py",
                "seed_rule": "2026081700 + one_based_panel_index",
                "rows_per_panel": rows_per_panel,
                "labels": labels,
                "crossover_probability": eta,
                "rounds": rounds,
                "panel_sha256": hashes,
                "exact_posterior_mean_nll_by_panel_and_rounds": exact_by_panel,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    ratios = [
        state / endpoint["model_parameters"]
        for state, endpoint in zip(rounds, endpoints, strict=True)
    ]
    if not all(high < low for low, high in zip(ratios[:-1], ratios[1:], strict=True)):
        raise RuntimeError("relative state must decrease")
    config = {
        "schema_version": 1,
        "experiment": "ascent_posttraining_noisy_composition_qwen2p5_candidate_factorial",
        "status": "prospective_qwen_candidate_development_frozen_before_any_qwen_score",
        "frozen_at": "2026-08-15T09:45:00+08:00",
        "motivation": (
            "Falcon3 candidate-NLL development preserved positive state interactions and a positive "
            "7B K3-to-K5 gain, but failed its absolute co-scaled gate. Qwen2.5 is selected as an "
            "independent public instruction family with stronger arithmetic behavior. All rows are fresh."
        ),
        "construction": {
            "num_values": labels,
            "crossover_probability": eta,
            "channel": "independent categorical symmetric channel per register and round",
            "target_rule": "(2 * first_value + second_value) modulo 8",
            "fresh_information": "values and observations sampled after checkpoint selection",
        },
        "candidate_texts": [str(value) for value in range(labels)],
        "panels": panels,
        "development_panels": panels[:1],
        "confirmation_panels": panels[1:],
        "panel_path_by_name": paths,
        "panel_sha256_by_name": hashes,
        "data_manifest_path": "posttraining_noisy_composition_qwen_candidate/MANIFEST.json",
        "data_manifest_sha256": sha256_file(manifest_path),
        "exact_posterior_mean_nll_by_panel_and_rounds": exact_by_panel,
        "evaluation_samples": rows_per_panel,
        "context_tokens": 2048,
        "runtime_dependencies": runtime_source["runtime_dependencies"],
        "endpoints": endpoints,
        "factorial": {
            "endpoints": endpoint_names,
            "state_rounds": rounds,
            "co_scaled_cells": [
                f"{endpoint}@rounds{state}"
                for endpoint, state in zip(endpoint_names, rounds, strict=True)
            ],
            "co_scaled_rounds_per_parameter": ratios,
            "strictly_decreasing_relative_state": True,
        },
        "execution": {
            "node_by_endpoint": {endpoint: "node1" for endpoint in endpoint_names},
            "development_authorized": True,
            "confirmation_authorized_only_if_development_passes": True,
        },
        "primary_metric": "paired candidate-normalized answer NLL gain in nats",
        "require_model_by_state_interactions": True,
        "development_gate": (
            "All provenance, exact-channel, candidate-token, target-isolation, nesting, Foundation-"
            "invariance, and relative-state checks pass; all three diagonal gains, both diagonal "
            "increments, the 7B K5-minus-K3 gain, and both model-by-state interactions are positive."
        ),
        "confirmation_gate": (
            "Across all nine untouched panels, the three gains, two diagonal increments, 7B state "
            "increment, and two interactions each have positive two-sided t(8) 95% lower bounds."
        ),
        "selection_and_stopping_rule": (
            "The Qwen family, fresh rows, prompt, candidate head, 2/3/5 law, metrics, and gates are "
            "frozen before any Qwen candidate score. A failed panel-1 gate forbids confirmation and tuning."
        ),
        "interpretation": (
            "Cross-architecture neural evidence-use development on the proposition-matched fresh channel. "
            "It does not erase or relabel the failed Falcon development."
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
        default=ROOT / "data" / "posttraining_noisy_composition_qwen_candidate",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "posttraining_noisy_composition_qwen_candidate.json",
    )
    args = parser.parse_args()
    config = freeze(args.output_root, args.config)
    print(json.dumps({"config": str(args.config), "panels": len(config["panels"])}, indent=2))


if __name__ == "__main__":
    main()
