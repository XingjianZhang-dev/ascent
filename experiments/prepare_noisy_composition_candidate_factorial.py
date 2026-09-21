#!/usr/bin/env python3
"""Freeze fresh rows for the candidate-normalized noisy-composition readout."""

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


def freeze(output_root: Path, config_path: Path) -> dict:
    source = json.loads(
        (ROOT / "configs" / "posttraining_noisy_composition_falcon3.json").read_text()
    )
    labels = 8
    eta = 0.28
    rounds = [2, 3, 5]
    rows_per_panel = 512
    panel_names = [f"candidate_noisy_composition_panel_{index}" for index in range(1, 11)]
    panel_hashes = {}
    panel_paths = {}
    exact_by_panel = {}
    output_root.mkdir(parents=True, exist_ok=True)
    for index, panel in enumerate(panel_names, start=1):
        rows = make_panel(
            seed=2026081600 + index,
            panel_index=index,
            rows=rows_per_panel,
            labels=labels,
            eta=eta,
            max_rounds=max(rounds),
        )
        for row_index, row in enumerate(rows):
            row["row_id"] = f"candidate_noisy_comp_p{index:02d}_r{row_index:04d}"
        path = output_root / f"{panel}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        panel_hashes[panel] = sha256_file(path)
        panel_paths[panel] = str(path.relative_to(ROOT / "data"))
        exact_by_panel[panel] = {
            str(value): exact_nll(rows, value, labels, eta) for value in rounds
        }
        curve = [exact_by_panel[panel][str(value)] for value in rounds]
        if not all(high < low for low, high in zip(curve[:-1], curve[1:], strict=True)):
            raise RuntimeError(f"non-strict exact curve: {panel}")
    manifest_path = output_root / "MANIFEST.json"
    manifest = {
        "schema_version": 1,
        "generator": "prepare_noisy_composition_candidate_factorial.py",
        "seed_rule": "2026081600 + one_based_panel_index",
        "rows_per_panel": rows_per_panel,
        "labels": labels,
        "crossover_probability": eta,
        "rounds": rounds,
        "panel_sha256": panel_hashes,
        "exact_posterior_mean_nll_by_panel_and_rounds": exact_by_panel,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    config = {
        "schema_version": 1,
        "experiment": "ascent_posttraining_noisy_composition_candidate_factorial",
        "status": "prospective_candidate_nll_development_frozen_before_any_candidate_score",
        "frozen_at": "2026-08-15T09:10:00+08:00",
        "motivation": (
            "The prior free-generation development exposed an eight-token reasoning truncation "
            "for 1B/3B and an example-answer anchor for 7B. Its confirmation is forbidden. "
            "This independently frozen readout uses fresh rows, removes the worked example, and "
            "uses a proper candidate-normalized log-loss head so output length cannot censor the metric."
        ),
        "construction": source["construction"],
        "candidate_texts": [str(value) for value in range(labels)],
        "panels": panel_names,
        "development_panels": panel_names[:1],
        "confirmation_panels": panel_names[1:],
        "panel_path_by_name": panel_paths,
        "panel_sha256_by_name": panel_hashes,
        "data_manifest_sha256": sha256_file(manifest_path),
        "exact_posterior_mean_nll_by_panel_and_rounds": exact_by_panel,
        "evaluation_samples": rows_per_panel,
        "context_tokens": 2048,
        "runtime_dependencies": source["runtime_dependencies"],
        "endpoints": source["endpoints"],
        "factorial": source["factorial"],
        "execution": source["execution"],
        "primary_metric": "paired candidate-normalized answer NLL gain in nats",
        "development_gate": (
            "All provenance, exact-channel, single-token-candidate, target-isolation, nested-prefix, "
            "Foundation-invariance, and decreasing-relative-state checks pass; all three co-scaled "
            "NLL gains, both adjacent diagonal gain increments, and the 7B K5-minus-K3 gain are positive."
        ),
        "confirmation_gate": (
            "Across all nine untouched panels, each of the six registered NLL estimands has a positive "
            "two-sided t(8) 95% lower bound and a positive Holm-adjusted one-sided direction."
        ),
        "selection_and_stopping_rule": (
            "All rows, hashes, prompts, candidate head, channel, 2/3/5 law, endpoints, estimands, and "
            "gates are frozen before candidate scoring. Panel 1 alone is development. Any failed gate "
            "forbids the nine confirmation panels and any tuning on these rows."
        ),
        "interpretation": (
            "A proper-scoring neural evidence-use test matched to the formal post-training channel. "
            "It is a new readout on entirely fresh rows, not a relabeling of the failed free-generation panel."
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
        default=ROOT / "data" / "posttraining_noisy_composition_candidate",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "posttraining_noisy_composition_candidate_falcon3.json",
    )
    args = parser.parse_args()
    config = freeze(args.output_root, args.config)
    print(json.dumps({"config": str(args.config), "panels": len(config["panels"])}, indent=2))


if __name__ == "__main__":
    main()
