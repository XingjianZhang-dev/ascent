#!/usr/bin/env python3
"""Freeze fresh ordinary-sum posterior candidate panels before neural scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.prepare_noisy_composition_factorial import (
    exact_sum_nll,
    make_panel,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


def freeze(
    output_root: Path, config_path: Path, *, variant: str = "letter_labels"
) -> dict[str, Any]:
    output_root = output_root.resolve()
    config_path = config_path.resolve()
    if variant not in {"letter_labels", "numeric_sequences"}:
        raise ValueError(f"unknown sum candidate variant: {variant}")
    numeric = variant == "numeric_sequences"
    source = json.loads(
        (ROOT / "configs" / "posttraining_noisy_composition_map_sum.json").read_text()
    )
    labels = 8
    eta = 0.28
    rounds = [2, 3, 5]
    rows_per_panel = 256
    seed_base = 2026085000 if numeric else 2026084000
    panel_stem = "sum_numeric_candidate" if numeric else "sum_candidate"
    panels = [f"{panel_stem}_panel_{index}" for index in range(1, 11)]
    paths: dict[str, str] = {}
    hashes: dict[str, str] = {}
    exact: dict[str, dict[str, float]] = {}
    output_root.mkdir(parents=True, exist_ok=True)
    for index, panel in enumerate(panels, start=1):
        rows = make_panel(
            seed=seed_base + index,
            panel_index=index,
            rows=rows_per_panel,
            labels=labels,
            eta=eta,
            max_rounds=max(rounds),
        )
        for row_index, row in enumerate(rows):
            row["row_id"] = f"{panel_stem}_p{index:02d}_r{row_index:04d}"
            row["target"] = row["first_value"] + row["second_value"]
        path = output_root / f"{panel}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        paths[panel] = str(path.relative_to(ROOT / "data"))
        hashes[panel] = sha256_file(path)
        exact[panel] = {
            str(value): exact_sum_nll(rows, value, labels, eta) for value in rounds
        }
        curve = [exact[panel][str(value)] for value in rounds]
        if not all(high < low for low, high in zip(curve[:-1], curve[1:], strict=True)):
            raise RuntimeError(f"non-strict exact information curve: {panel}")

    manifest_path = output_root / "MANIFEST.json"
    manifest = {
        "schema_version": 1,
        "generator": "prepare_noisy_composition_sum_candidate.py",
        "seed_rule": f"{seed_base} + one_based_panel_index",
        "labels": labels,
        "target_classes": 15,
        "crossover_probability": eta,
        "rounds": rounds,
        "rows_per_panel": rows_per_panel,
        "panel_sha256": hashes,
        "exact_posterior_mean_nll_by_panel_and_rounds": exact,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    endpoints = source["endpoints"]
    ratios = [
        slots / endpoint["model_parameters"]
        for slots, endpoint in zip(rounds, endpoints, strict=True)
    ]
    if not all(high < low for low, high in zip(ratios[:-1], ratios[1:], strict=True)):
        raise RuntimeError("relative state ratio must strictly decrease")
    config = {
        "schema_version": 1,
        "experiment": (
            "ascent_sum_posterior_numeric_candidate_qwen2p5_factorial"
            if numeric
            else "ascent_sum_posterior_candidate_qwen2p5_factorial"
        ),
        "status": (
            "prospective_sum_numeric_candidate_development_frozen_before_any_score"
            if numeric
            else "prospective_sum_candidate_development_frozen_before_any_score"
        ),
        "frozen_at": (
            "2026-08-15T08:05:00+08:00"
            if numeric
            else "2026-08-15T07:45:00+08:00"
        ),
        "construction": {
            "num_values": labels,
            "crossover_probability": eta,
            "channel": "independent categorical symmetric channel per register and round",
            "target_rule": "first_value + second_value",
            "fresh_information": (
                "new values and observations after the letter-code interface audit"
                if numeric
                else "new values and observations after all earlier interface scores"
            ),
        },
        "target_classes": 15,
        "candidate_texts": (
            [f"{value:02d}" for value in range(15)]
            if numeric
            else list("ABCDEFGHIJKLMNO")
        ),
        "prompt_style": (
            "sum_posterior_candidate_numeric"
            if numeric
            else "sum_posterior_candidate_labels"
        ),
        "panels": panels,
        "development_panels": panels[:1],
        "confirmation_panels": panels[1:],
        "panel_path_by_name": paths,
        "panel_sha256_by_name": hashes,
        "data_manifest_path": str(manifest_path.relative_to(ROOT / "data")),
        "data_manifest_sha256": sha256_file(manifest_path),
        "exact_posterior_mean_nll_by_panel_and_rounds": exact,
        "evaluation_samples": rows_per_panel,
        "context_tokens": 2048,
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
        "require_model_by_state_interactions": True,
        "development_gate": (
            "All provenance, 15-way single-token candidate, exact-channel, nested-prefix, "
            "Foundation-invariance, nonredundancy, target-isolation, and relative-state "
            "checks pass; all three co-scaled NLL gains, both adjacent gain increments, "
            "the 7B K5-minus-K3 gain, and both model-by-state interactions are positive."
        ),
        "confirmation_gate": (
            "Across nine untouched panels, all eight registered NLL estimands have positive "
            "two-sided t(8) 95% lower bounds. A supplementary Bonferroni simultaneous family "
            "audit is reported and no panel, row, or outlier may be removed."
        ),
        "selection_and_stopping_rule": (
            "The two-token numeric scorer and fresh data are frozen after a score audit "
            "showed that arbitrary bare letter logits measure token-initial frequency. Only "
            "panel 1 may be scored for development; confirmation is forbidden unless the "
            "complete eight-estimand development gate passes without prompt or metric tuning."
            if numeric
            else
            "The complete-posterior candidate interface and fresh data are frozen after the "
            "ordinary-sum generation development failed only its parse-coverage gate. Only "
            "panel 1 may be scored for development; confirmation is forbidden unless the "
            "complete eight-estimand development gate passes without prompt or metric tuning."
        ),
        "interpretation": (
            "Candidate-normalized joint sequence NLL uses a fixed assistant prefix and 15 "
            "equal-length two-token numeric continuations 00 through 14. Exact boundary "
            "tokenization is audited at every row and endpoint."
            if numeric
            else
            "Candidate-normalized NLL removes generation parse censoring while retaining the "
            "full certified posterior interface that produced positive model-by-state "
            "interactions. Letter labels are equal-length, distinct single tokens at every "
            "registered Qwen2.5 endpoint and map fixedly to sums 0 through 14."
        ),
    }
    if numeric:
        config["scoring_prefix"] = "The ordinary integer sum as two digits is "
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data" / "posttraining_noisy_composition_sum_candidate",
    )
    parser.add_argument(
        "--variant",
        choices=("letter_labels", "numeric_sequences"),
        default="letter_labels",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "posttraining_noisy_composition_sum_candidate.json",
    )
    args = parser.parse_args()
    config = freeze(args.output_root, args.config, variant=args.variant)
    print(json.dumps({"config": str(args.config), "panels": len(config["panels"])}, indent=2))


if __name__ == "__main__":
    main()
