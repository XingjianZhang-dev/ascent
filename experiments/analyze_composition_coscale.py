#!/usr/bin/env python3
"""Analyze one three-scale two-fact ASCENT composition seed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_ruler_niah import bootstrap
from experiments.analyze_composition_factorial import summary


ENDPOINTS = (("pythia-410m", "410m"), ("pythia-1b", "1b"), ("pythia-2.8b", "2p8b"))


def analyze(result_dir: Path, seed: int | None = None) -> dict[str, Any]:
    prefix = f"seed{seed}_" if seed is not None else ""
    metadata = {
        endpoint: json.loads((result_dir / f"{prefix}composition_{suffix}.json").read_text())
        for endpoint, suffix in ENDPOINTS
    }
    arrays = {
        endpoint: np.load(result_dir / f"{prefix}composition_{suffix}.npz")
        for endpoint, suffix in ENDPOINTS
    }
    reference = metadata[ENDPOINTS[0][0]]
    invariant_fields = [
        "config_sha256", "registered_answers_sha256",
        "registered_full_signal_sha256", "identical_query_inputs_sha256",
    ]
    invariants = {
        field: all(payload[field] == reference[field] for payload in metadata.values())
        for field in invariant_fields
    }
    invariants.update({
        "answers_exact": all(
            np.array_equal(array["answers"], arrays[ENDPOINTS[0][0]]["answers"])
            for array in arrays.values()
        ),
        "clean": all(not payload["runtime"]["git_dirty"] for payload in metadata.values()),
        "same_commit": len({payload["runtime"]["git_commit"] for payload in metadata.values()}) == 1,
        "oracle_zero": all(payload["clean_kv_oracle_nll"] == 0.0 for payload in metadata.values()),
        "effective_seed": seed is None or all(
            payload["effective_seed"] == seed for payload in metadata.values()
        ),
    })
    if not all(invariants.values()):
        raise RuntimeError(f"composition co-scale invariants failed: {invariants}")

    gains = {
        endpoint: arrays[endpoint]["base_nll"] - arrays[endpoint]["rich_latent_fused_nll"]
        for endpoint, _ in ENDPOINTS
    }
    params = np.asarray(
        [metadata[endpoint]["endpoint"]["model_parameters"] for endpoint, _ in ENDPOINTS],
        dtype=np.float64,
    )
    x = np.log(params)
    centered = x - x.mean()
    matrix = np.stack([gains[endpoint] for endpoint, _ in ENDPOINTS], axis=1)
    slopes = matrix @ centered / float(centered @ centered)
    endpoint_results = {
        endpoint: {
            "model_parameters": int(metadata[endpoint]["endpoint"]["model_parameters"]),
            "rounds_per_fact": int(metadata[endpoint]["arms"]["rich"]["rounds_per_fact"]),
            "gain": summary(values),
            "gain_bootstrap": bootstrap(values, seed=int(params[index]) % 1_000_000),
        }
        for index, ((endpoint, _), values) in enumerate(zip(ENDPOINTS, gains.values(), strict=True))
    }
    adjacent: dict[str, Any] = {}
    for index in range(2):
        left, right = ENDPOINTS[index][0], ENDPOINTS[index + 1][0]
        values = gains[right] - gains[left]
        normal = summary(values)
        boot = bootstrap(values, seed=610001 + index)
        adjacent[f"{right}_minus_{left}"] = {
            "difference": normal,
            "bootstrap": boot,
            "gate_pass": normal["ci95_low"] > 0 and boot["ci95_low"] > 0,
        }
    slope = summary(slopes)
    slope_boot = bootstrap(slopes, seed=610003)
    gates = {
        "all_endpoint_gains_positive": all(
            row["gain"]["ci95_low"] > 0 and row["gain_bootstrap"]["ci95_low"] > 0
            for row in endpoint_results.values()
        ),
        "both_adjacent_differences_positive": all(row["gate_pass"] for row in adjacent.values()),
        "slope_positive": slope["ci95_low"] > 0 and slope_boot["ci95_low"] > 0,
    }
    return {
        "schema_version": 1,
        "status": "passed" if all(gates.values()) else "failed",
        "invariants": invariants,
        "endpoint_results": endpoint_results,
        "adjacent_differences": adjacent,
        "slope_nats_per_log_parameter": slope,
        "slope_bootstrap": slope_boot,
        "gates": gates,
        "strong_raw_boundary": "Clean-KV text and the zero-NLL clean-KV oracle remain separately reported; this curve does not claim raw-storage superiority.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    payload = analyze(args.result_dir, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
