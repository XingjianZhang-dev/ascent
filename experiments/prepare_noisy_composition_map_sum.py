#!/usr/bin/env python3
"""Freeze compact certified-MAP neural-addition rows on fresh episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.prepare_noisy_composition_factorial import (
    ROOT,
    exact_sum_nll,
    make_panel,
    sha256_file,
)


def freeze(output_root: Path, config_path: Path) -> dict:
    source = json.loads(
        (ROOT / "configs" / "posttraining_noisy_composition_sum_replay.json").read_text()
    )
    labels = 8
    eta = 0.28
    rounds = [2, 3, 5]
    rows_per_panel = 256
    panels = [f"map_sum_panel_{index}" for index in range(1, 11)]
    paths = {}
    hashes = {}
    exact_by_panel = {}
    output_root.mkdir(parents=True, exist_ok=True)
    for index, panel in enumerate(panels, start=1):
        rows = make_panel(
            seed=2026082100 + index,
            panel_index=index,
            rows=rows_per_panel,
            labels=labels,
            eta=eta,
            max_rounds=max(rounds),
        )
        for row_index, row in enumerate(rows):
            row["row_id"] = f"map_sum_p{index:02d}_r{row_index:04d}"
            row["target"] = row["first_value"] + row["second_value"]
        path = output_root / f"{panel}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        paths[panel] = str(path.relative_to(ROOT / "data"))
        hashes[panel] = sha256_file(path)
        exact_by_panel[panel] = {
            str(value): exact_sum_nll(rows, value, labels, eta) for value in rounds
        }
        curve = [exact_by_panel[panel][str(value)] for value in rounds]
        if not all(high < low for low, high in zip(curve[:-1], curve[1:], strict=True)):
            raise RuntimeError(f"non-strict exact curve: {panel}")
    manifest_path = output_root / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generator": "prepare_noisy_composition_map_sum.py",
                "seed_rule": "2026082100 + one_based_panel_index",
                "rows_per_panel": rows_per_panel,
                "labels": labels,
                "crossover_probability": eta,
                "rounds": rounds,
                "target_rule": "first_value + second_value",
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
        "experiment": "ascent_compact_certified_map_sum_qwen2p5_factorial",
        "status": "prospective_map_sum_development_frozen_before_any_score",
        "frozen_at": "2026-08-15T12:00:00+08:00",
        "motivation": (
            "The full posterior text replay exposed every probability and distracted the neural "
            "composition path. The ASCENT proposal defines a decoder precisely so the expressive "
            "path can consume a compact decoded value. This interface passes only the certified MAP "
            "values to the frozen model, while the accounted persistent state remains the exact "
            "append-only 2/3/5 observation prefix. All episodes are fresh."
        ),
        "construction": source["construction"],
        "prompt_style": "map_sum_replay",
        "panels": panels,
        "development_panels": panels[:1],
        "confirmation_panels": panels[1:],
        "panel_path_by_name": paths,
        "panel_sha256_by_name": hashes,
        "data_manifest_path": "posttraining_noisy_composition_map_sum/MANIFEST.json",
        "data_manifest_sha256": sha256_file(manifest_path),
        "exact_posterior_mean_nll_by_panel_and_rounds": exact_by_panel,
        "evaluation_samples": rows_per_panel,
        "context_tokens": 2048,
        "max_new_tokens": 8,
        "minimum_answer_coverage": 0.99,
        "runtime_dependencies": source["runtime_dependencies"],
        "endpoints": source["endpoints"],
        "factorial": source["factorial"],
        "execution": source["execution"],
        "development_gate": (
            "All provenance, official-model, exact-channel, nested persistent-prefix, exact-posterior, "
            "target-isolation, answer-coverage, Foundation-invariance, prompt-length, and relative-state "
            "checks pass; every diagonal gain, both adjacent diagonal increments, and the 7B K5-minus-K3 "
            "gain are positive. Decoder-MAP transition counts and interactions are reported."
        ),
        "confirmation_gate": (
            "Across nine untouched panels, all six registered accuracy estimands have positive "
            "two-sided t(8) 95% lower bounds and every cell has at least 99% parse coverage."
        ),
        "selection_and_stopping_rule": (
            "The compact decoder contract, fresh rows, model ladder, 2/3/5 law, eight-token decode, "
            "metrics, and gates are frozen before generation. Failure forbids confirmation."
        ),
        "interpretation": (
            "A direct implementation of certified inference feeding expressive neural composition. "
            "The exact posterior and clean-value arithmetic oracle remain separately reportable bounds."
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
        default=ROOT / "data" / "posttraining_noisy_composition_map_sum",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "posttraining_noisy_composition_map_sum.json",
    )
    args = parser.parse_args()
    config = freeze(args.output_root, args.config)
    print(json.dumps({"config": str(args.config), "panels": len(config["panels"])}, indent=2))


if __name__ == "__main__":
    main()
