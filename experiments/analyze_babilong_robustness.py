#!/usr/bin/env python3
"""Analyze frozen BABILong fixed-state and negative-memory controls."""

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

from experiments.analyze_babilong_scale_panel import PANEL_T_95, mean_ci


ENDPOINTS = ("smollm2-135m", "smollm2-360m-node2", "smollm2-1p7b")
CONDITIONS = (
    "scale_relevant",
    "fixed_relevant",
    "scale_irrelevant",
    "scale_corrupted",
)


def result_path(root: Path, panel: int, condition: str, endpoint: str) -> Path:
    node = "node2" if "360m" in endpoint else "node1"
    return root / node / f"panel{panel}_{condition}_{endpoint}.json"


def analyze(root: Path) -> dict[str, Any]:
    results: dict[tuple[int, str, str], dict[str, Any]] = {}
    for panel in (1, 2, 3):
        for condition in CONDITIONS:
            for endpoint in ENDPOINTS:
                key = (panel, condition, endpoint)
                results[key] = json.loads(
                    result_path(root, panel, condition, endpoint).read_text()
                )

    commits = {result["environment"]["git_commit"] for result in results.values()}
    config_hashes = {result["config"]["sha256"] for result in results.values()}
    provenance = {
        "single_git_commit": len(commits) == 1,
        "git_commits": sorted(commits),
        "all_worktrees_clean": all(
            not result["environment"]["git_dirty"] for result in results.values()
        ),
        "single_config_hash": len(config_hashes) == 1,
        "config_hashes": sorted(config_hashes),
        "all_parser_accuracies_one": all(
            result["parser_accuracy"] == 1.0 for result in results.values()
        ),
    }

    foundation_invariant = True
    row_alignment = True
    for panel in (1, 2, 3):
        for endpoint in ENDPOINTS:
            rows = [results[(panel, condition, endpoint)] for condition in CONDITIONS]
            identifiers = [
                [prediction["row_id"] for prediction in result["predictions"]]
                for result in rows
            ]
            row_alignment &= all(value == identifiers[0] for value in identifiers[1:])
            outputs = [
                [prediction["foundation_output"] for prediction in result["predictions"]]
                for result in rows
            ]
            foundation_invariant &= all(value == outputs[0] for value in outputs[1:])
    provenance["row_alignment"] = row_alignment
    provenance["foundation_outputs_identical_across_conditions"] = foundation_invariant

    condition_gains: dict[str, np.ndarray] = {}
    condition_ascent: dict[str, np.ndarray] = {}
    summaries: dict[str, Any] = {}
    for condition in CONDITIONS:
        gains = np.asarray(
            [
                [
                    results[(panel, condition, endpoint)]["gain"]["mean"]
                    for endpoint in ENDPOINTS
                ]
                for panel in (1, 2, 3)
            ],
            dtype=np.float64,
        )
        ascent = np.asarray(
            [
                [
                    results[(panel, condition, endpoint)]["ascent"]["mean"]
                    for endpoint in ENDPOINTS
                ]
                for panel in (1, 2, 3)
            ],
            dtype=np.float64,
        )
        condition_gains[condition] = gains
        condition_ascent[condition] = ascent
        summaries[condition] = {
            "panel_gain_curves": gains.tolist(),
            "endpoint_gain_intervals": {
                endpoint: mean_ci(gains[:, index].tolist(), critical=PANEL_T_95)
                for index, endpoint in enumerate(ENDPOINTS)
            },
            "adjacent_gain_increment_intervals": {
                f"{ENDPOINTS[index]}_to_{ENDPOINTS[index + 1]}": mean_ci(
                    (gains[:, index + 1] - gains[:, index]).tolist(),
                    critical=PANEL_T_95,
                )
                for index in (0, 1)
            },
        }

    scale_increment = np.diff(condition_gains["scale_relevant"], axis=1)
    fixed_increment = np.diff(condition_gains["fixed_relevant"], axis=1)
    expansion_did = {
        f"{ENDPOINTS[index]}_to_{ENDPOINTS[index + 1]}": mean_ci(
            (scale_increment[:, index] - fixed_increment[:, index]).tolist(),
            critical=PANEL_T_95,
        )
        for index in (0, 1)
    }
    negative_contrasts: dict[str, Any] = {}
    for negative in ("scale_irrelevant", "scale_corrupted"):
        difference = condition_ascent["scale_relevant"] - condition_ascent[negative]
        negative_contrasts[negative] = {
            endpoint: mean_ci(difference[:, index].tolist(), critical=PANEL_T_95)
            for index, endpoint in enumerate(ENDPOINTS)
        }

    gates = {
        "provenance": all(
            value
            for key, value in provenance.items()
            if key not in {"git_commits", "config_hashes"}
        ),
        "scale_relevant_every_panel_strictly_monotone": bool(
            np.all(scale_increment > 0)
        ),
        "scale_expansion_did_lcbs_positive": all(
            interval["ci95_low"] > 0 for interval in expansion_did.values()
        ),
        "relevant_beats_irrelevant_all_lcbs": all(
            interval["ci95_low"] > 0
            for interval in negative_contrasts["scale_irrelevant"].values()
        ),
        "relevant_beats_corrupted_all_lcbs": all(
            interval["ci95_low"] > 0
            for interval in negative_contrasts["scale_corrupted"].values()
        ),
    }
    gates["robustness_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "endpoints": list(ENDPOINTS),
        "conditions": summaries,
        "scale_expansion_difference_in_differences": expansion_did,
        "relevant_accuracy_advantage_over_negative_memory": negative_contrasts,
        "provenance": provenance,
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
