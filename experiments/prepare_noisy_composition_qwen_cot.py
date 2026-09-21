#!/usr/bin/env python3
"""Freeze fresh Qwen rows for a length-safe chain-of-thought readout."""

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
        (ROOT / "configs" / "posttraining_noisy_composition_qwen_candidate.json").read_text()
    )
    labels = 8
    eta = 0.28
    rounds = [2, 3, 5]
    rows_per_panel = 256
    panels = [f"qwen_cot_noisy_panel_{index}" for index in range(1, 11)]
    paths = {}
    hashes = {}
    exact_by_panel = {}
    output_root.mkdir(parents=True, exist_ok=True)
    for index, panel in enumerate(panels, start=1):
        rows = make_panel(
            seed=2026081800 + index,
            panel_index=index,
            rows=rows_per_panel,
            labels=labels,
            eta=eta,
            max_rounds=max(rounds),
        )
        for row_index, row in enumerate(rows):
            row["row_id"] = f"qwen_cot_noisy_p{index:02d}_r{row_index:04d}"
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
                "generator": "prepare_noisy_composition_qwen_cot.py",
                "seed_rule": "2026081800 + one_based_panel_index",
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
    endpoints = []
    for endpoint, batch_size in zip(source["endpoints"], [16, 8, 4], strict=True):
        copied = dict(endpoint)
        copied["batch_size"] = batch_size
        endpoints.append(copied)
    config = {
        "schema_version": 1,
        "experiment": "ascent_posttraining_noisy_composition_qwen2p5_cot_factorial",
        "status": "prospective_qwen_cot_development_frozen_before_any_cot_score",
        "frozen_at": "2026-08-15T10:20:00+08:00",
        "motivation": (
            "The candidate-next-token head produced positive state interactions and a large 7B gain, "
            "but small/mid endpoints did not use the evidence without deliberation. Its confirmation "
            "is forbidden. This new readout uses entirely fresh rows, removes all worked examples, "
            "allows 96 deterministic tokens for brief reasoning, and requires a FINAL marker."
        ),
        "construction": source["construction"],
        "prompt_style": "qwen_cot",
        "panels": panels,
        "development_panels": panels[:1],
        "confirmation_panels": panels[1:],
        "panel_path_by_name": paths,
        "panel_sha256_by_name": hashes,
        "data_manifest_path": "posttraining_noisy_composition_qwen_cot/MANIFEST.json",
        "data_manifest_sha256": sha256_file(manifest_path),
        "exact_posterior_mean_nll_by_panel_and_rounds": exact_by_panel,
        "evaluation_samples": rows_per_panel,
        "context_tokens": 2048,
        "max_new_tokens": 96,
        "minimum_final_marker_coverage": 0.95,
        "runtime_dependencies": source["runtime_dependencies"],
        "endpoints": endpoints,
        "factorial": source["factorial"],
        "execution": source["execution"],
        "development_gate": (
            "All provenance, exact-channel, target-isolation, nested-prefix, Foundation-invariance, "
            "prompt-length, FINAL-parser, and decreasing-relative-state checks pass; FINAL coverage "
            "is at least 95% in every cell; all three co-scaled gains, both diagonal gain increments, "
            "and the 7B K5-minus-K3 gain are positive. Interactions are reported."
        ),
        "confirmation_gate": (
            "Across nine untouched panels, all six registered accuracy estimands have positive two-sided "
            "t(8) 95% lower bounds and every cell has at least 95% FINAL coverage."
        ),
        "selection_and_stopping_rule": (
            "Family, fresh rows, prompts, 96-token budget, FINAL parser, channel, 2/3/5 law, metrics, "
            "and gates are frozen before generation. Panel 1 alone is development; failure forbids confirmation."
        ),
        "interpretation": (
            "A neural generative composition test on the registered fresh-information channel. It is "
            "separate from and cannot relabel either failed candidate-head development."
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
        default=ROOT / "data" / "posttraining_noisy_composition_qwen_cot",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "posttraining_noisy_composition_qwen_cot.json",
    )
    args = parser.parse_args()
    config = freeze(args.output_root, args.config)
    print(json.dumps({"config": str(args.config), "panels": len(config["panels"])}, indent=2))


if __name__ == "__main__":
    main()
