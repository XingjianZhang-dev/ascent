#!/usr/bin/env python3
"""Analyze frozen structured BABILong development or five-panel confirmation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ENDPOINTS = ("smollm2-135m", "smollm2-360m-node2", "smollm2-1p7b")
PARAMETERS = np.asarray((134515008, 361821120, 1711376384), dtype=np.float64)
T95_DF4 = 2.7764451051977987


def mean_ci(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    standard_error = float(array.std(ddof=1) / math.sqrt(array.size))
    return {
        "values": array.tolist(),
        "mean": mean,
        "standard_error": standard_error,
        "ci95_low": mean - T95_DF4 * standard_error,
        "ci95_high": mean + T95_DF4 * standard_error,
    }


def result_path(root: Path, phase: str, panel: int, endpoint: str) -> Path:
    node = "node2" if "360m" in endpoint else "node1"
    prefix = "development" if phase == "development" else f"panel{panel}"
    return root / node / f"{prefix}_{endpoint}.json"


def load_panels(root: Path, phase: str) -> list[list[dict[str, Any]]]:
    panel_numbers = (0,) if phase == "development" else (1, 2, 3, 4, 5)
    panels: list[list[dict[str, Any]]] = []
    for panel in panel_numbers:
        results = [
            json.loads(result_path(root, phase, panel, endpoint).read_text())
            for endpoint in ENDPOINTS
        ]
        row_ids = [
            [prediction["row_id"] for prediction in result["predictions"]]
            for result in results
        ]
        if not all(values == row_ids[0] for values in row_ids[1:]):
            raise RuntimeError(f"row alignment mismatch in panel {panel}")
        panels.append(results)
    return panels


def analyze(root: Path, phase: str) -> dict[str, Any]:
    panels = load_panels(root, phase)
    commits = {r["environment"]["git_commit"] for panel in panels for r in panel}
    config_hashes = {r["config"]["sha256"] for panel in panels for r in panel}
    data_hashes = [{r["data"]["sha256"] for r in panel} for panel in panels]
    provenance = {
        "single_git_commit": len(commits) == 1,
        "git_commits": sorted(commits),
        "all_worktrees_clean": all(
            not r["environment"]["git_dirty"] for panel in panels for r in panel
        ),
        "single_config_hash": len(config_hashes) == 1,
        "config_hashes": sorted(config_hashes),
        "one_data_hash_per_panel": all(len(values) == 1 for values in data_hashes),
        "disjoint_data_hashes": len({next(iter(values)) for values in data_hashes})
        == len(panels),
        "all_parser_accuracies_one": all(
            r["parser_accuracy"] == 1.0 for panel in panels for r in panel
        ),
        "all_primary_certified": all(
            r["condition"]["name"] == "primary_certified"
            and r["condition"]["memory_representation"] == "structured_inventory"
            and r["condition"]["readout_path"] == "certified_evidence"
            and not r["condition"]["diagnostic_no_promotion"]
            for panel in panels
            for r in panel
        ),
    }
    gains = np.asarray(
        [[result["gain"]["mean"] for result in panel] for panel in panels],
        dtype=np.float64,
    )
    remaining_error = np.asarray(
        [
            [result["remaining_error_elimination"] for result in panel]
            for panel in panels
        ],
        dtype=np.float64,
    )
    if not np.isfinite(remaining_error).all():
        raise RuntimeError("undefined remaining-error elimination")
    log_parameters = np.log(PARAMETERS)
    gain_slopes = [float(np.polyfit(log_parameters, values, 1)[0]) for values in gains]
    remaining_error_slopes = [
        float(np.polyfit(log_parameters, values, 1)[0]) for values in remaining_error
    ]
    panel_rows = [
        {
            "panel": index + 1,
            "data_sha256": panel[0]["data"]["sha256"],
            "foundation": [r["foundation"]["mean"] for r in panel],
            "ascent": [r["ascent"]["mean"] for r in panel],
            "absolute_gain": gains[index].tolist(),
            "remaining_error_elimination": remaining_error[index].tolist(),
            "absolute_gain_adjacent_increments": np.diff(gains[index]).tolist(),
            "remaining_error_adjacent_increments": np.diff(
                remaining_error[index]
            ).tolist(),
            "wins": [r["wins"] for r in panel],
            "regressions": [r["regressions"] for r in panel],
        }
        for index, panel in enumerate(panels)
    ]
    provenance_gate = all(
        value
        for key, value in provenance.items()
        if key not in {"git_commits", "config_hashes"}
    )
    if phase == "development":
        gates = {
            "provenance": provenance_gate,
            "all_absolute_gains_positive": bool(np.all(gains[0] > 0)),
            "remaining_error_strictly_monotone": bool(
                np.all(np.diff(remaining_error[0]) > 0)
            ),
            "remaining_error_log_parameter_slope_positive": (
                remaining_error_slopes[0] > 0
            ),
        }
        gates["development_pass"] = all(gates.values())
        return {
            "schema_version": 1,
            "phase": phase,
            "endpoints": list(ENDPOINTS),
            "parameters": PARAMETERS.astype(np.int64).tolist(),
            "panels": panel_rows,
            "absolute_gain_log_parameter_slope": gain_slopes[0],
            "remaining_error_log_parameter_slope": remaining_error_slopes[0],
            "provenance": provenance,
            "gates": gates,
        }

    absolute_gain_intervals = {
        endpoint: mean_ci(gains[:, index].tolist())
        for index, endpoint in enumerate(ENDPOINTS)
    }
    remaining_error_intervals = {
        endpoint: mean_ci(remaining_error[:, index].tolist())
        for index, endpoint in enumerate(ENDPOINTS)
    }
    absolute_adjacent = {
        f"{ENDPOINTS[index]}_to_{ENDPOINTS[index + 1]}": mean_ci(
            (gains[:, index + 1] - gains[:, index]).tolist()
        )
        for index in (0, 1)
    }
    remaining_error_adjacent = {
        f"{ENDPOINTS[index]}_to_{ENDPOINTS[index + 1]}": mean_ci(
            (remaining_error[:, index + 1] - remaining_error[:, index]).tolist()
        )
        for index in (0, 1)
    }
    gain_slope_interval = mean_ci(gain_slopes)
    remaining_error_slope_interval = mean_ci(remaining_error_slopes)
    gates = {
        "provenance": provenance_gate,
        "every_panel_remaining_error_strictly_monotone": bool(
            np.all(np.diff(remaining_error, axis=1) > 0)
        ),
        "all_absolute_gain_lcbs_positive": all(
            value["ci95_low"] > 0 for value in absolute_gain_intervals.values()
        ),
        "all_remaining_error_lcbs_positive": all(
            value["ci95_low"] > 0 for value in remaining_error_intervals.values()
        ),
        "both_remaining_error_adjacent_lcbs_positive": all(
            value["ci95_low"] > 0 for value in remaining_error_adjacent.values()
        ),
        "remaining_error_log_parameter_slope_lcb_positive": (
            remaining_error_slope_interval["ci95_low"] > 0
        ),
    }
    gates["confirmation_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "phase": phase,
        "endpoints": list(ENDPOINTS),
        "parameters": PARAMETERS.astype(np.int64).tolist(),
        "panels": panel_rows,
        "absolute_gain_panel_t_intervals": absolute_gain_intervals,
        "remaining_error_panel_t_intervals": remaining_error_intervals,
        "absolute_gain_adjacent_panel_t_intervals": absolute_adjacent,
        "remaining_error_adjacent_panel_t_intervals": remaining_error_adjacent,
        "absolute_gain_log_parameter_slope_panel_t_interval": gain_slope_interval,
        "remaining_error_log_parameter_slope_panel_t_interval": (
            remaining_error_slope_interval
        ),
        "provenance": provenance,
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("development", "confirmation"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.phase)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
