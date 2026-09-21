#!/usr/bin/env python3
"""Analyze frozen BABILong ASCENT neural and exact-byte controls."""

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


ENDPOINTS = (
    "smollm2-135m-instruct",
    "smollm2-360m-instruct",
    "smollm2-1p7b-instruct",
)
CONDITIONS = ("scale_relevant", "scale_matched_raw", "scale_bge_m3")
CONTROL_CONDITIONS = ("scale_matched_raw", "scale_bge_m3")
MEMORY_SOURCES = {
    "scale_relevant": "relevant",
    "scale_matched_raw": "matched_raw_event_fifo",
    "scale_bge_m3": "bge_m3_hybrid_rerank",
}
PARAMETERS = np.asarray((134515008, 361821120, 1711376384), dtype=np.float64)
EXPECTED_CACHE_HASHES = {
    1: "0b38858291f02715d758d5e5b0cefac341933e2996a3689a5fb1f16b92efb338",
    2: "9ab9e11288c138030d383a4514999057b92c032085d288c077cb58db73e0e936",
    3: "b7ecd848eba7776d0028f46d4568cd3c33addb7cfb4e01f28b19ae1638830099",
}


def path_for(root: Path, panel: int, endpoint: str, condition: str) -> Path:
    node = "node2" if endpoint == "smollm2-360m-instruct" else "node1"
    return root / node / f"panel{panel}_{endpoint}_{condition}.json"


def score_signature(result: dict[str, Any], arm: str) -> list[tuple[str, Any, float]]:
    return [
        (row["row_id"], row[f"{arm}_location"], row[f"{arm}_score"])
        for row in result["predictions"]
    ]


def analyze(root: Path) -> dict[str, Any]:
    cells = {
        (panel, endpoint, condition): json.loads(
            path_for(root, panel, endpoint, condition).read_text()
        )
        for panel in (1, 2, 3)
        for endpoint in ENDPOINTS
        for condition in CONDITIONS
    }
    results = list(cells.values())
    commits = {result["environment"]["git_commit"] for result in results}
    config_hashes = {result["config"]["sha256"] for result in results}
    provenance = {
        "single_git_commit": len(commits) == 1,
        "git_commits": sorted(commits),
        "all_worktrees_clean": all(
            not result["environment"]["git_dirty"] for result in results
        ),
        "single_config_hash": len(config_hashes) == 1,
        "config_hashes": sorted(config_hashes),
        "all_parser_accuracies_one": all(
            result["parser_accuracy"] == 1.0 for result in results
        ),
        "condition_sources_exact": all(
            cells[(panel, endpoint, condition)]["condition"]["memory_source"]
            == MEMORY_SOURCES[condition]
            for panel in (1, 2, 3)
            for endpoint in ENDPOINTS
            for condition in CONDITIONS
        ),
        "bge_cache_hashes_exact": all(
            cells[(panel, endpoint, "scale_bge_m3")]["retrieval_cache"]["sha256"]
            == EXPECTED_CACHE_HASHES[panel]
            for panel in (1, 2, 3)
            for endpoint in ENDPOINTS
        ),
    }

    panel_rows: list[dict[str, Any]] = []
    foundation_invariant = True
    unique_cap_rows: set[str] = set()
    for panel in (1, 2, 3):
        row_ids = {
            tuple(
                row["row_id"]
                for row in cells[(panel, endpoint, condition)]["predictions"]
            )
            for endpoint in ENDPOINTS
            for condition in CONDITIONS
        }
        if len(row_ids) != 1:
            raise RuntimeError(f"panel {panel} row alignment mismatch")
        relevant_gain: list[float] = []
        relevant_accuracy: list[float] = []
        controls = {condition: [] for condition in CONTROL_CONDITIONS}
        advantages = {condition: [] for condition in CONTROL_CONDITIONS}
        for endpoint in ENDPOINTS:
            relevant = cells[(panel, endpoint, "scale_relevant")]
            condition_results = [
                cells[(panel, endpoint, condition)] for condition in CONDITIONS
            ]
            signatures = [
                score_signature(result, "foundation")
                for result in condition_results
            ]
            foundation_invariant &= all(
                signature == signatures[0] for signature in signatures
            )
            relevant_gain.append(float(relevant["gain"]["mean"]))
            relevant_accuracy.append(float(relevant["ascent"]["mean"]))
            for condition in CONTROL_CONDITIONS:
                control_accuracy = float(
                    cells[(panel, endpoint, condition)]["ascent"]["mean"]
                )
                controls[condition].append(control_accuracy)
                advantages[condition].append(
                    float(relevant["ascent"]["mean"]) - control_accuracy
                )
            for result in condition_results:
                unique_cap_rows.update(
                    row["row_id"]
                    for row in result["predictions"]
                    if row["foundation_prompt_tokens"] == 8192
                )
        panel_rows.append(
            {
                "panel": panel,
                "data_sha256": cells[(panel, ENDPOINTS[0], CONDITIONS[0])]["data"][
                    "sha256"
                ],
                "relevant_accuracy": relevant_accuracy,
                "relevant_gain": relevant_gain,
                "control_accuracy": controls,
                "relevant_accuracy_advantage": advantages,
                "relevant_adjacent_gain_increments": [
                    relevant_gain[1] - relevant_gain[0],
                    relevant_gain[2] - relevant_gain[1],
                ],
            }
        )

    relevant_gain = np.asarray(
        [row["relevant_gain"] for row in panel_rows], dtype=np.float64
    )
    advantages = {
        condition: np.asarray(
            [row["relevant_accuracy_advantage"][condition] for row in panel_rows],
            dtype=np.float64,
        )
        for condition in CONTROL_CONDITIONS
    }
    advantage_intervals = {
        condition: {
            endpoint: mean_ci(values[:, index].tolist(), critical=PANEL_T_95)
            for index, endpoint in enumerate(ENDPOINTS)
        }
        for condition, values in advantages.items()
    }
    adjacent_intervals = {
        f"{ENDPOINTS[index]}_to_{ENDPOINTS[index + 1]}": mean_ci(
            (relevant_gain[:, index + 1] - relevant_gain[:, index]).tolist(),
            critical=PANEL_T_95,
        )
        for index in (0, 1)
    }
    slopes = [
        float(np.polyfit(np.log(PARAMETERS), row, 1)[0]) for row in relevant_gain
    ]
    slope_interval = mean_ci(slopes, critical=PANEL_T_95)

    task_contrasts: dict[str, Any] = {}
    for condition in CONTROL_CONDITIONS:
        task_contrasts[condition] = {}
        for endpoint in ENDPOINTS:
            task_contrasts[condition][endpoint] = {}
            for task in ("qa1", "qa2", "qa3"):
                values = [
                    cells[(panel, endpoint, "scale_relevant")]["by_task"][task][
                        "ascent"
                    ]["mean"]
                    - cells[(panel, endpoint, condition)]["by_task"][task][
                        "ascent"
                    ]["mean"]
                    for panel in (1, 2, 3)
                ]
                task_contrasts[condition][endpoint][task] = mean_ci(
                    values, critical=PANEL_T_95
                )

    state: dict[str, Any] = {}
    for endpoint in ENDPOINTS:
        state[endpoint] = {}
        for condition in CONDITIONS:
            rows = [
                cells[(panel, endpoint, condition)]["state"]
                for panel in (1, 2, 3)
            ]
            state[endpoint][condition] = {
                "mean_write_bytes_by_panel": [
                    float(row["write_state_payload_bytes"]["mean"]) for row in rows
                ],
                "mean_read_bytes_by_panel": [
                    float(row["read_state_payload_bytes"]["mean"]) for row in rows
                ],
            }

    all_advantage_lcbs_positive = all(
        interval["ci95_low"] > 0.0
        for control in advantage_intervals.values()
        for interval in control.values()
    )
    gates = {
        "provenance": all(
            value if isinstance(value, bool) else True for value in provenance.values()
        ),
        "foundation_predictions_identical_across_conditions": foundation_invariant,
        "relevant_curve_strictly_monotone_every_panel": all(
            np.all(np.diff(row) > 0.0) for row in relevant_gain
        ),
        "both_adjacent_gain_increment_lcbs_positive": all(
            interval["ci95_low"] > 0.0 for interval in adjacent_intervals.values()
        ),
        "log_parameter_slope_lcb_positive": slope_interval["ci95_low"] > 0.0,
        "relevant_beats_both_controls_at_all_endpoints_with_positive_lcbs": all_advantage_lcbs_positive,
    }
    return {
        "schema_version": 1,
        "endpoints": list(ENDPOINTS),
        "panels": panel_rows,
        "accuracy_advantage_panel_t_intervals": advantage_intervals,
        "relevant_adjacent_gain_increment_panel_t_intervals": adjacent_intervals,
        "relevant_log_parameter_slope_panel_t_interval": slope_interval,
        "task_stratified_accuracy_advantage": task_contrasts,
        "state_accounting": state,
        "unique_foundation_rows_at_8192_token_cap": len(unique_cap_rows),
        "provenance": provenance,
        "gates": gates,
        "confirmation_pass": all(gates.values()),
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
