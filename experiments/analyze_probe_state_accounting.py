#!/usr/bin/env python3
"""Produce exact ASCENT state and evidence-head accounting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ascent.state_accounting import (
    affine_head_parameters,
    latent_state_bytes,
    state_element_ratio,
)
from experiments.run_natural_repeat import sha256_file


def analyze(config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text())
    bytes_per_element = int(config["bytes_per_state_element"])
    classes = int(config["evidence_classes"])
    rows = []
    for endpoint in config["endpoints"]:
        tokens = int(endpoint["state_tokens"])
        width = int(endpoint["hidden_size"])
        parameters = int(endpoint["model_parameters"])
        state_bytes = latent_state_bytes(tokens, width, bytes_per_element)
        probe_parameters = affine_head_parameters(width, classes)
        ratio = state_element_ratio(tokens, width, parameters)
        rows.append(
            {
                **endpoint,
                "persistent_state_bytes": state_bytes,
                "persistent_state_kib": state_bytes / 1024.0,
                "state_elements_per_model_parameter": ratio,
                "evidence_head_parameters": probe_parameters,
                "evidence_head_to_model_parameter_ratio": probe_parameters
                / parameters,
            }
        )
    ratios = [row["state_elements_per_model_parameter"] for row in rows]
    return {
        "schema_version": 1,
        "config_sha256": sha256_file(config_path),
        "source_experiment_config": config["source_experiment_config"],
        "state_dtype": config["state_dtype"],
        "endpoints": rows,
        "gates": {
            "relative_state_ratio_strictly_decreases": all(
                left > right for left, right in zip(ratios, ratios[1:])
            ),
            "probe_is_below_one_basis_point_of_model": all(
                row["evidence_head_to_model_parameter_ratio"] < 1e-4
                for row in rows
            ),
        },
        "boundary": "Persistent dense bf16 latent payload and affine evidence-head parameters only; transient replay activations, beam KV work, latency, and optimizer state are excluded and require separate measurement.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
