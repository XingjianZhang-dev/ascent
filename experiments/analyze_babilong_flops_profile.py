#!/usr/bin/env python3
"""Audit the frozen three-endpoint BABILong supported-op FLOP profile."""

from __future__ import annotations

import argparse
import json
import statistics
import tarfile
from pathlib import Path
from typing import Any


ENDPOINTS = (
    "smollm2-135m-instruct",
    "smollm2-360m-instruct",
    "smollm2-1p7b-instruct",
)


def _reference_member(endpoint: str) -> str:
    node = "node2" if "360m" in endpoint else "node1"
    return (
        f"./confirmation/{node}/panel1_{endpoint}_scale_relevant.json"
    )


def _load_reference(archive: Path, endpoint: str) -> dict[str, Any]:
    with tarfile.open(archive, "r:gz") as stream:
        extracted = stream.extractfile(_reference_member(endpoint))
        if extracted is None:
            raise FileNotFoundError(_reference_member(endpoint))
        return json.load(extracted)


def analyze(reference_archive: Path, results_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    config_hashes: set[str] = set()
    commits: set[str] = set()
    for endpoint in ENDPOINTS:
        result = json.loads((results_root / f"{endpoint}.json").read_text())
        reference = _load_reference(reference_archive, endpoint)
        config_hashes.add(result["config"]["sha256"])
        commits.add(result["environment"]["git_commit"])
        reference_foundation = [
            row["foundation_location"] for row in reference["predictions"]
        ]
        reference_ascent = [
            row["ascent_location"] for row in reference["predictions"]
        ]
        flop_ratio = result["ratios"]["supported_operator_flops"]
        wall_ratio = result["ratios"]["wall_seconds"]
        rows.append(
            {
                "endpoint": endpoint,
                "foundation_supported_operator_tflops": result["foundation"][
                    "supported_operator_flops"
                ]
                / 1e12,
                "ascent_supported_operator_tflops": result["ascent"][
                    "supported_operator_flops"
                ]
                / 1e12,
                "supported_operator_flop_ratio": flop_ratio,
                "supported_operator_flop_reduction_factor": 1.0 / flop_ratio,
                "wall_time_ratio": wall_ratio,
                "profiled_wall_time_speedup": 1.0 / wall_ratio,
                "padded_prompt_token_ratio": result["ratios"][
                    "padded_prompt_tokens"
                ],
                "foundation_accuracy": result["foundation"]["accuracy"],
                "ascent_accuracy": result["ascent"]["accuracy"],
                "reference_foundation_accuracy": reference["foundation"]["mean"],
                "reference_ascent_accuracy": reference["ascent"]["mean"],
                "foundation_predictions_equal_reference": (
                    result["foundation"]["normalized_predictions"]
                    == reference_foundation
                ),
                "ascent_predictions_equal_reference": (
                    result["ascent"]["normalized_predictions"]
                    == reference_ascent
                ),
                "data_hash_equal_reference": (
                    result["data"]["sha256"] == reference["data"]["sha256"]
                ),
                "model_files_equal_reference": (
                    result["model_files"] == reference["model_files"]
                ),
                "worktree_clean": not result["environment"]["git_dirty"],
                "supported_operator_flops_positive": (
                    result["foundation"]["supported_operator_flops"] > 0
                    and result["ascent"]["supported_operator_flops"] > 0
                ),
                "flop_ratio_below_0p10": flop_ratio < 0.10,
                "strict_profiled_wall_ratio_below_0p10": wall_ratio < 0.10,
            }
        )
    gates = {
        "single_config_hash": len(config_hashes) == 1,
        "single_git_commit": len(commits) == 1,
        "all_worktrees_clean": all(row["worktree_clean"] for row in rows),
        "all_data_hashes_equal_reference": all(
            row["data_hash_equal_reference"] for row in rows
        ),
        "all_model_files_equal_reference": all(
            row["model_files_equal_reference"] for row in rows
        ),
        "all_predictions_equal_reference": all(
            row["foundation_predictions_equal_reference"]
            and row["ascent_predictions_equal_reference"]
            for row in rows
        ),
        "all_supported_operator_flops_positive": all(
            row["supported_operator_flops_positive"] for row in rows
        ),
        "all_flop_ratios_below_0p10": all(
            row["flop_ratio_below_0p10"] for row in rows
        ),
        "all_wall_ratios_below_0p10": all(
            row["strict_profiled_wall_ratio_below_0p10"] for row in rows
        ),
    }
    gates["frozen_profile_gate_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "endpoints": rows,
        "median_supported_operator_flop_ratio": statistics.median(
            row["supported_operator_flop_ratio"] for row in rows
        ),
        "median_wall_time_ratio": statistics.median(
            row["wall_time_ratio"] for row in rows
        ),
        "measurement_boundary": (
            "torch.profiler with_flops supported-operator lower bound; fused "
            "attention and unsupported elementwise formulas are not counted"
        ),
        "config_hashes": sorted(config_hashes),
        "git_commits": sorted(commits),
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-archive", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.reference_archive, args.results_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
