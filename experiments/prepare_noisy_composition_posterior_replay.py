#!/usr/bin/env python3
"""Freeze a dual-path ASCENT posterior-replay scale factorial."""

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
from experiments.prepare_noisy_composition_qwen_candidate import select_endpoint


def freeze(output_root: Path, config_path: Path) -> dict:
    canonical = json.loads(
        (ROOT / "configs" / "babilong_qwen2p5_canonical_coscale_16k_confirmatory.json").read_text()
    )
    around7b = json.loads(
        (ROOT / "configs" / "babilong_around7b_16k_extension_confirmatory.json").read_text()
    )
    runtime = json.loads(
        (ROOT / "configs" / "posttraining_noisy_composition_qwen_cot.json").read_text()
    )
    names = [
        "qwen2p5-1p5b-instruct",
        "qwen2p5-3b-instruct",
        "qwen2p5-7b-instruct",
    ]
    endpoints = [
        select_endpoint(canonical, names[0]),
        select_endpoint(canonical, names[1]),
        select_endpoint(around7b, names[2]),
    ]
    rounds = [2, 3, 5]
    for endpoint, batch_size, state_slots in zip(
        endpoints, [16, 8, 4], rounds, strict=True
    ):
        endpoint["batch_size"] = batch_size
        endpoint["state_fact_slots"] = state_slots
        endpoint.setdefault("model_type", "qwen2")
        endpoint.setdefault("architecture", "Qwen2ForCausalLM")
        endpoint.setdefault("native_context_tokens", 32768)
    ratios = [
        state / endpoint["model_parameters"]
        for state, endpoint in zip(rounds, endpoints, strict=True)
    ]
    if not all(high < low for low, high in zip(ratios[:-1], ratios[1:], strict=True)):
        raise RuntimeError("relative state must decrease")

    labels = 8
    eta = 0.28
    rows_per_panel = 256
    panels = [f"posterior_replay_panel_{index}" for index in range(1, 11)]
    paths = {}
    hashes = {}
    exact_by_panel = {}
    output_root.mkdir(parents=True, exist_ok=True)
    for index, panel in enumerate(panels, start=1):
        rows = make_panel(
            seed=2026081900 + index,
            panel_index=index,
            rows=rows_per_panel,
            labels=labels,
            eta=eta,
            max_rounds=max(rounds),
        )
        for row_index, row in enumerate(rows):
            row["row_id"] = f"posterior_replay_p{index:02d}_r{row_index:04d}"
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
                "generator": "prepare_noisy_composition_posterior_replay.py",
                "seed_rule": "2026081900 + one_based_panel_index",
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
    config = {
        "schema_version": 1,
        "experiment": "ascent_dual_path_posterior_replay_qwen2p5_factorial",
        "status": "prospective_posterior_replay_development_frozen_before_any_score",
        "frozen_at": "2026-08-15T11:00:00+08:00",
        "motivation": (
            "This interface directly instantiates ASCENT's proposed dual path: append-only noisy "
            "evidence is decoded by the exact certified posterior, while the frozen neural model "
            "performs only the registered composition from decoder MAP values. It is not a prompt "
            "repair of any visible panel and uses entirely fresh rows plus a 1.5B/3B/7B ladder."
        ),
        "construction": {
            "num_values": labels,
            "crossover_probability": eta,
            "channel": "independent categorical symmetric channel per register and round",
            "target_rule": "(2 * first_value + second_value) modulo 8",
            "fresh_information": "values and observations sampled after checkpoint selection",
        },
        "prompt_style": "posterior_replay",
        "panels": panels,
        "development_panels": panels[:1],
        "confirmation_panels": panels[1:],
        "panel_path_by_name": paths,
        "panel_sha256_by_name": hashes,
        "data_manifest_path": "posttraining_noisy_composition_posterior_replay/MANIFEST.json",
        "data_manifest_sha256": sha256_file(manifest_path),
        "exact_posterior_mean_nll_by_panel_and_rounds": exact_by_panel,
        "evaluation_samples": rows_per_panel,
        "context_tokens": 2048,
        "max_new_tokens": 16,
        "minimum_answer_coverage": 0.99,
        "runtime_dependencies": runtime["runtime_dependencies"],
        "endpoints": endpoints,
        "factorial": {
            "endpoints": names,
            "state_rounds": rounds,
            "co_scaled_cells": [
                f"{name}@rounds{state}"
                for name, state in zip(names, rounds, strict=True)
            ],
            "co_scaled_rounds_per_parameter": ratios,
            "strictly_decreasing_relative_state": True,
        },
        "execution": {
            "node_by_endpoint": {
                names[0]: "node2",
                names[1]: "node2",
                names[2]: "node1",
            },
            "development_authorized": True,
            "confirmation_authorized_only_if_development_passes": True,
        },
        "development_gate": (
            "All provenance, official-model, exact-channel, nested raw-prefix, exact-posterior, "
            "target-isolation, answer-coverage, Foundation-invariance, prompt-length, and decreasing-"
            "relative-state checks pass; all three diagonal gains, both adjacent diagonal gain "
            "increments, and the 7B K5-minus-K3 gain are positive. Interactions are reported."
        ),
        "confirmation_gate": (
            "Across nine untouched panels, all six registered accuracy estimands have positive "
            "two-sided t(8) 95% lower bounds and every cell has at least 99% parse coverage."
        ),
        "selection_and_stopping_rule": (
            "The architecture, fresh rows, 1.5B/3B/7B family, posterior formatter, 2/3/5 law, "
            "16-token decode, metrics, and gates are frozen before generation. Failure forbids confirmation."
        ),
        "interpretation": (
            "A direct dual-path ASCENT mechanism test: certified inference plus neural composition. "
            "The exact posterior remains the oracle boundary; neural accuracy is separately scored."
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
        default=ROOT / "data" / "posttraining_noisy_composition_posterior_replay",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "posttraining_noisy_composition_posterior_replay.json",
    )
    args = parser.parse_args()
    config = freeze(args.output_root, args.config)
    print(json.dumps({"config": str(args.config), "panels": len(config["panels"])}, indent=2))


if __name__ == "__main__":
    main()
