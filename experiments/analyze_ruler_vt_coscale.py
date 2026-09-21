#!/usr/bin/env python3
"""Analyze the three-endpoint co-scaled ASCENT RULER-VT gain curve."""

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


def endpoint_suffix(endpoint: str) -> str:
    return {
        "pythia-410m": "410m",
        "pythia-1b": "1b",
        "pythia-1.4b": "1p4b",
        "pythia-2.8b": "2p8b",
    }[endpoint]


def analyze(
    result_dir: Path,
    data_root: Path,
    tokenizer_path: str,
    middle_endpoint: str = "pythia-1.4b",
) -> dict[str, Any]:
    endpoints = (
        ("pythia-410m", endpoint_suffix("pythia-410m")),
        (middle_endpoint, endpoint_suffix(middle_endpoint)),
        ("pythia-2.8b", endpoint_suffix("pythia-2.8b")),
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    metadata = {
        endpoint: json.loads((result_dir / f"ruler_coscale_{suffix}.json").read_text())
        for endpoint, suffix in endpoints
    }
    arrays = {
        endpoint: np.load(result_dir / f"ruler_coscale_{suffix}.npz")
        for endpoint, suffix in endpoints
    }
    calibration_samples = int(metadata["pythia-410m"]["tasks"]["vt"]["calibration_samples"])
    rows = [
        json.loads(line)
        for line in (data_root / "vt" / "validation.jsonl").read_text().splitlines()
    ][calibration_samples:]
    lengths = [
        len(tokenizer(" " + " ".join(row["outputs"]), add_special_tokens=False)["input_ids"])
        for row in rows
    ]

    gains: dict[str, np.ndarray] = {}
    endpoint_results: dict[str, Any] = {}
    parameters = []
    budgets = []
    for endpoint, _ in endpoints:
        gains[endpoint] = sample_means(
            arrays[endpoint]["vt_ascent_scale_latent_only_gain"], lengths
        )
        endpoint_config = metadata[endpoint]["endpoint"]
        parameters.append(float(endpoint_config["model_parameters"]))
        budgets.append(int(endpoint_config["scale_replay_tokens"]))
        endpoint_results[endpoint] = {
            "model_parameters": int(endpoint_config["model_parameters"]),
            "state_replay_tokens": int(endpoint_config["scale_replay_tokens"]),
            "gain": summary(gains[endpoint]),
            "gain_bootstrap": bootstrap(gains[endpoint], seed=int(parameters[-1]) % 1_000_000),
        }

    adjacent: dict[str, Any] = {}
    adjacent_pass = True
    for index in range(2):
        left, right = endpoints[index][0], endpoints[index + 1][0]
        difference = gains[right] - gains[left]
        normal = summary(difference)
        boot = bootstrap(difference, seed=314159 + index)
        gate = normal["ci95_low"] > 0 and boot["ci95_low"] > 0
        adjacent_pass = adjacent_pass and gate
        adjacent[f"{right}_minus_{left}"] = {
            "gain_difference": normal,
            "gain_difference_bootstrap": boot,
            "gate_pass": gate,
        }

    x = np.log(np.asarray(parameters, dtype=np.float64))
    centered = x - x.mean()
    gain_matrix = np.stack([gains[endpoint] for endpoint, _ in endpoints], axis=1)
    sample_slopes = gain_matrix @ centered / float(centered @ centered)
    slope_normal = summary(sample_slopes)
    slope_boot = bootstrap(sample_slopes, seed=271828)
    endpoint_pass = all(
        values["gain"]["ci95_low"] > 0 and values["gain_bootstrap"]["ci95_low"] > 0
        for values in endpoint_results.values()
    )
    slope_pass = slope_normal["ci95_low"] > 0 and slope_boot["ci95_low"] > 0
    commits = {values["runtime"]["git_commit"] for values in metadata.values()}
    config_hashes = {values["config_sha256"] for values in metadata.values()}
    passed = endpoint_pass and adjacent_pass and slope_pass and len(commits) == 1 and len(config_hashes) == 1
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "estimand": "ASCENT-Scale latent-only NLL gain over the same-scale frozen foundation",
        "registered_state_replay_tokens": budgets,
        "endpoint_results": endpoint_results,
        "adjacent_differences": adjacent,
        "slope_nats_per_log_parameter": slope_normal,
        "slope_bootstrap": slope_boot,
        "gates": {"all_endpoint_gains_positive": endpoint_pass, "both_adjacent_differences_positive": adjacent_pass, "scale_slope_positive": slope_pass},
        "runtime_commits": sorted(commits),
        "config_hashes": sorted(config_hashes),
        "metric_boundary": "Teacher-forced answer-token NLL, not official autoregressive exact match.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer-path", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--middle-endpoint", default="pythia-1.4b")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(
        args.result_dir,
        args.data_root,
        args.tokenizer_path,
        middle_endpoint=args.middle_endpoint,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
