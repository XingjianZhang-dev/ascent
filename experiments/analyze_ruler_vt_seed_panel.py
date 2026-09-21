#!/usr/bin/env python3
"""Equal-seed hierarchical analysis of the frozen RULER-VT factorial panel."""

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

from experiments.analyze_ruler_niah import bootstrap, sample_means, summary


def hierarchical_bootstrap(
    values_by_seed: list[np.ndarray], *, seed: int, replicates: int = 20_000
) -> dict[str, Any]:
    """Resample data seeds, then samples within each selected seed."""
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected = rng.integers(0, len(values_by_seed), size=len(values_by_seed))
        seed_means = []
        for index in selected:
            values = values_by_seed[int(index)]
            sample = values[rng.integers(0, values.size, size=values.size)]
            seed_means.append(float(sample.mean()))
        draws[replicate] = float(np.mean(seed_means))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "equal_seed_mean": float(np.mean([values.mean() for values in values_by_seed])),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "replicates": replicates,
    }


def analyze(panel_root: Path, seeds: list[int], tokenizer_path: str) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    endpoint_effects: dict[str, list[np.ndarray]] = {
        "pythia-410m": [], "pythia-2.8b": []
    }
    interactions: list[np.ndarray] = []
    seed_results: dict[str, Any] = {}
    runtime_commits: set[str] = set()
    config_hashes: set[str] = set()

    for seed in seeds:
        root = panel_root / str(seed)
        metadata = {
            endpoint: json.loads((root / f"ruler_factorial_{suffix}.json").read_text())
            for endpoint, suffix in (("pythia-410m", "410m"), ("pythia-2.8b", "2p8b"))
        }
        arrays = {
            endpoint: np.load(root / f"ruler_factorial_{suffix}.npz")
            for endpoint, suffix in (("pythia-410m", "410m"), ("pythia-2.8b", "2p8b"))
        }
        calibration_samples = int(metadata["pythia-410m"]["tasks"]["vt"]["calibration_samples"])
        rows = [
            json.loads(line)
            for line in (root / "data" / "vt" / "validation.jsonl").read_text().splitlines()
        ][calibration_samples:]
        lengths = [
            len(tokenizer(" " + " ".join(row["outputs"]), add_special_tokens=False)["input_ids"])
            for row in rows
        ]
        effects: dict[str, np.ndarray] = {}
        seed_payload: dict[str, Any] = {}
        for endpoint in ("pythia-410m", "pythia-2.8b"):
            fixed = sample_means(
                arrays[endpoint]["vt_ascent_fixed_latent_only_gain"], lengths
            )
            rich = sample_means(
                arrays[endpoint]["vt_ascent_scale_latent_only_gain"], lengths
            )
            effect = rich - fixed
            effects[endpoint] = effect
            endpoint_effects[endpoint].append(effect)
            seed_payload[endpoint] = {
                "rich_minus_fixed": summary(effect),
                "rich_minus_fixed_bootstrap": bootstrap(effect, seed=seed + len(endpoint)),
            }
            runtime_commits.add(metadata[endpoint]["runtime"]["git_commit"])
            config_hashes.add(metadata[endpoint]["config_sha256"])
        interaction = effects["pythia-2.8b"] - effects["pythia-410m"]
        interactions.append(interaction)
        seed_payload["model_by_memory_interaction"] = summary(interaction)
        seed_payload["model_by_memory_interaction_bootstrap"] = bootstrap(
            interaction, seed=seed
        )
        seed_results[str(seed)] = seed_payload

    aggregate = {
        "pythia-410m_rich_minus_fixed": hierarchical_bootstrap(
            endpoint_effects["pythia-410m"], seed=410
        ),
        "pythia-2.8b_rich_minus_fixed": hierarchical_bootstrap(
            endpoint_effects["pythia-2.8b"], seed=2800
        ),
        "model_by_memory_interaction": hierarchical_bootstrap(
            interactions, seed=223607
        ),
    }
    gates = {
        name: values["ci95_low"] > 0 for name, values in aggregate.items()
    }
    return {
        "schema_version": 1,
        "status": "passed" if all(gates.values()) else "failed",
        "seeds": seeds,
        "seed_weighting": "Data seeds weighted equally; samples weighted equally within seed.",
        "seed_results": seed_results,
        "aggregate": aggregate,
        "gates": gates,
        "positive_interaction_seeds": int(sum(values.mean() > 0 for values in interactions)),
        "runtime_commits": sorted(runtime_commits),
        "provenance_note": "Multiple commits are allowed when the executed runner and memory implementation are content-identical; commit hashes remain reported for audit.",
        "config_hashes": sorted(config_hashes),
        "metric_boundary": "Teacher-forced answer-token NLL, not official autoregressive exact match.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--tokenizer-path", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(args.panel_root, args.seeds, args.tokenizer_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
