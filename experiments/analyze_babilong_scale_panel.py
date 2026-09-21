#!/usr/bin/env python3
"""Audit the frozen three-panel BABILong scale-complementarity result."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ENDPOINTS = ("smollm2-135m", "smollm2-360m-node2", "smollm2-1p7b")
PARAMETERS = np.asarray((134515008, 361821120, 1711376384), dtype=np.float64)
PANEL_T_95 = 4.302652729911275  # two-sided 95%, df=2


def mean_ci(values: list[float], *, critical: float) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    standard_error = float(array.std(ddof=1) / math.sqrt(array.size))
    mean = float(array.mean())
    return {
        "values": [float(value) for value in array],
        "mean": mean,
        "standard_error": standard_error,
        "ci95_low": mean - critical * standard_error,
        "ci95_high": mean + critical * standard_error,
    }


def result_path(root: Path, panel: int, endpoint: str) -> Path:
    node = "node2" if "360m" in endpoint else "node1"
    return root / node / f"panel{panel}_{endpoint}.json"


def analyze(root: Path) -> dict[str, Any]:
    panels: list[list[dict[str, Any]]] = []
    for panel in (1, 2, 3):
        endpoint_results = [
            json.loads(result_path(root, panel, endpoint).read_text())
            for endpoint in ENDPOINTS
        ]
        row_ids = [
            [prediction["row_id"] for prediction in result["predictions"]]
            for result in endpoint_results
        ]
        if not all(ids == row_ids[0] for ids in row_ids[1:]):
            raise RuntimeError(f"panel {panel} row alignment mismatch")
        panels.append(endpoint_results)

    commits = {
        result["environment"]["git_commit"]
        for panel in panels
        for result in panel
    }
    config_hashes = {
        result["config"]["sha256"] for panel in panels for result in panel
    }
    data_hashes_by_panel = [
        {result["data"]["sha256"] for result in panel} for panel in panels
    ]
    provenance = {
        "single_git_commit": len(commits) == 1,
        "git_commits": sorted(commits),
        "all_worktrees_clean": all(
            not result["environment"]["git_dirty"]
            for panel in panels
            for result in panel
        ),
        "single_config_hash": len(config_hashes) == 1,
        "config_hashes": sorted(config_hashes),
        "one_data_hash_per_panel": all(
            len(hashes) == 1 for hashes in data_hashes_by_panel
        ),
        "disjoint_panel_hashes": len(
            {next(iter(hashes)) for hashes in data_hashes_by_panel}
        )
        == 3,
        "all_parser_accuracies_one": all(
            result["parser_accuracy"] == 1.0
            for panel in panels
            for result in panel
        ),
    }

    gains = np.asarray(
        [
            [result["gain"]["mean"] for result in panel]
            for panel in panels
        ],
        dtype=np.float64,
    )
    endpoint_summaries = {
        endpoint: mean_ci(gains[:, index].tolist(), critical=PANEL_T_95)
        for index, endpoint in enumerate(ENDPOINTS)
    }
    adjacent = {
        f"{ENDPOINTS[index]}_to_{ENDPOINTS[index + 1]}": mean_ci(
            (gains[:, index + 1] - gains[:, index]).tolist(),
            critical=PANEL_T_95,
        )
        for index in (0, 1)
    }
    log_parameters = np.log(PARAMETERS)
    slopes = [
        float(np.polyfit(log_parameters, panel_gains, 1)[0])
        for panel_gains in gains
    ]
    slope = mean_ci(slopes, critical=PANEL_T_95)

    panel_rows = []
    for panel_index, panel in enumerate(panels, start=1):
        panel_rows.append(
            {
                "panel": panel_index,
                "data_sha256": panel[0]["data"]["sha256"],
                "foundation": [result["foundation"]["mean"] for result in panel],
                "ascent": [result["ascent"]["mean"] for result in panel],
                "gain": gains[panel_index - 1].tolist(),
                "adjacent_gain_increments": np.diff(
                    gains[panel_index - 1]
                ).tolist(),
                "wins": [result["wins"] for result in panel],
                "regressions": [result["regressions"] for result in panel],
            }
        )

    gates = {
        "provenance": all(
            value
            for key, value in provenance.items()
            if key
            not in {"git_commits", "config_hashes"}
        ),
        "every_panel_strictly_monotone": bool(np.all(np.diff(gains, axis=1) > 0)),
        "all_endpoint_gain_lcbs_positive": all(
            summary["ci95_low"] > 0 for summary in endpoint_summaries.values()
        ),
        "both_adjacent_increment_lcbs_positive": all(
            summary["ci95_low"] > 0 for summary in adjacent.values()
        ),
        "log_parameter_slope_lcb_positive": slope["ci95_low"] > 0,
    }
    gates["confirmation_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "endpoints": list(ENDPOINTS),
        "parameters": PARAMETERS.astype(np.int64).tolist(),
        "panels": panel_rows,
        "endpoint_gain_panel_t_intervals": endpoint_summaries,
        "adjacent_gain_increment_panel_t_intervals": adjacent,
        "log_parameter_slope_panel_t_interval": slope,
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
