#!/usr/bin/env python3
"""Audit frozen 8K ASCENT versus task-agnostic lexical retrieval controls."""

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


ENDPOINTS = ("smollm2-135m", "smollm2-360m", "smollm2-1p7b")
CONDITIONS = ("scale_relevant", "scale_generic_bm25")
PARAMETERS = np.asarray((134515008, 361821120, 1711376384), dtype=np.float64)


def path_for(root: Path, panel: int, endpoint: str, condition: str) -> Path:
    node = "node2" if endpoint == "smollm2-360m" else "node1"
    return root / node / f"panel{panel}_{endpoint}_{condition}.json"


def score_signature(result: dict[str, Any], arm: str) -> list[tuple[str, Any, float]]:
    return [
        (row["row_id"], row[f"{arm}_location"], row[f"{arm}_score"])
        for row in result["predictions"]
    ]


def analyze(root: Path) -> dict[str, Any]:
    cells: dict[tuple[int, str, str], dict[str, Any]] = {}
    for panel in (1, 2, 3):
        for endpoint in ENDPOINTS:
            for condition in CONDITIONS:
                cells[(panel, endpoint, condition)] = json.loads(
                    path_for(root, panel, endpoint, condition).read_text()
                )

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
            cells[(panel, endpoint, "scale_relevant")]["condition"]["memory_source"]
            == "relevant"
            and cells[(panel, endpoint, "scale_generic_bm25")]["condition"][
                "memory_source"
            ]
            == "generic_bm25"
            for panel in (1, 2, 3)
            for endpoint in ENDPOINTS
        ),
    }

    panel_rows: list[dict[str, Any]] = []
    foundation_invariant = True
    unique_cap_rows: set[str] = set()
    for panel in (1, 2, 3):
        row_ids = {
            tuple(row["row_id"] for row in cells[(panel, endpoint, condition)]["predictions"])
            for endpoint in ENDPOINTS
            for condition in CONDITIONS
        }
        if len(row_ids) != 1:
            raise RuntimeError(f"panel {panel} row alignment mismatch")
        relevant_gains: list[float] = []
        generic_gains: list[float] = []
        advantages: list[float] = []
        for endpoint in ENDPOINTS:
            relevant = cells[(panel, endpoint, "scale_relevant")]
            generic = cells[(panel, endpoint, "scale_generic_bm25")]
            foundation_invariant &= score_signature(
                relevant, "foundation"
            ) == score_signature(generic, "foundation")
            relevant_gains.append(float(relevant["gain"]["mean"]))
            generic_gains.append(float(generic["gain"]["mean"]))
            advantages.append(
                float(relevant["ascent"]["mean"] - generic["ascent"]["mean"])
            )
            for result in (relevant, generic):
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
                "relevant_gain": relevant_gains,
                "generic_gain": generic_gains,
                "relevant_accuracy_advantage_over_generic": advantages,
                "relevant_adjacent_gain_increments": [
                    relevant_gains[1] - relevant_gains[0],
                    relevant_gains[2] - relevant_gains[1],
                ],
            }
        )

    relevant_gain = np.asarray(
        [row["relevant_gain"] for row in panel_rows], dtype=np.float64
    )
    generic_gain = np.asarray(
        [row["generic_gain"] for row in panel_rows], dtype=np.float64
    )
    advantages = np.asarray(
        [row["relevant_accuracy_advantage_over_generic"] for row in panel_rows],
        dtype=np.float64,
    )
    advantage_intervals = {
        endpoint: mean_ci(advantages[:, index].tolist(), critical=PANEL_T_95)
        for index, endpoint in enumerate(ENDPOINTS)
    }
    relevant_adjacent = {
        f"{ENDPOINTS[index]}_to_{ENDPOINTS[index + 1]}": mean_ci(
            (relevant_gain[:, index + 1] - relevant_gain[:, index]).tolist(),
            critical=PANEL_T_95,
        )
        for index in (0, 1)
    }
    generic_gain_intervals = {
        endpoint: mean_ci(generic_gain[:, index].tolist(), critical=PANEL_T_95)
        for index, endpoint in enumerate(ENDPOINTS)
    }
    slopes = [
        float(np.polyfit(np.log(PARAMETERS), row, 1)[0]) for row in relevant_gain
    ]
    relevant_slope = mean_ci(slopes, critical=PANEL_T_95)

    task_contrasts: dict[str, dict[str, Any]] = {}
    for endpoint in ENDPOINTS:
        task_contrasts[endpoint] = {}
        for task in ("qa1", "qa2", "qa3"):
            values = []
            for panel in (1, 2, 3):
                relevant = cells[(panel, endpoint, "scale_relevant")]
                generic = cells[(panel, endpoint, "scale_generic_bm25")]
                values.append(
                    relevant["by_task"][task]["ascent"]["mean"]
                    - generic["by_task"][task]["ascent"]["mean"]
                )
            task_contrasts[endpoint][task] = mean_ci(
                values, critical=PANEL_T_95
            )

    state = {}
    for endpoint in ENDPOINTS:
        relevant_states = [
            cells[(panel, endpoint, "scale_relevant")]["state"]
            for panel in (1, 2, 3)
        ]
        generic_states = [
            cells[(panel, endpoint, "scale_generic_bm25")]["state"]
            for panel in (1, 2, 3)
        ]

        def across_panels(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
            values = [float(row[key]["mean"]) for row in rows]
            return {"panel_values": values, "mean": float(np.mean(values))}

        state[endpoint] = {
            "relevant_write_bytes": across_panels(
                relevant_states, "write_state_payload_bytes"
            ),
            "generic_full_corpus_utf8_bytes_lower_bound": across_panels(
                generic_states, "write_state_payload_bytes"
            ),
            "relevant_read_bytes": across_panels(
                relevant_states, "read_state_payload_bytes"
            ),
            "generic_read_bytes": across_panels(
                generic_states, "read_state_payload_bytes"
            ),
        }

    gates = {
        "provenance": all(
            value if isinstance(value, bool) else True for value in provenance.values()
        ),
        "foundation_predictions_identical_between_conditions": foundation_invariant,
        "relevant_curve_strictly_monotone_every_panel": all(
            np.all(np.diff(row) > 0.0) for row in relevant_gain
        ),
        "relevant_both_adjacent_increment_lcbs_positive": all(
            value["ci95_low"] > 0.0 for value in relevant_adjacent.values()
        ),
        "relevant_log_parameter_slope_lcb_positive": relevant_slope["ci95_low"]
        > 0.0,
        "relevant_beats_generic_at_every_endpoint_with_positive_lcb": all(
            value["ci95_low"] > 0.0 for value in advantage_intervals.values()
        ),
    }
    return {
        "schema_version": 1,
        "endpoints": list(ENDPOINTS),
        "panels": panel_rows,
        "relevant_accuracy_advantage_over_generic_panel_t_intervals": advantage_intervals,
        "generic_gain_panel_t_intervals": generic_gain_intervals,
        "relevant_adjacent_gain_increment_panel_t_intervals": relevant_adjacent,
        "relevant_log_parameter_slope_panel_t_interval": relevant_slope,
        "task_stratified_relevant_accuracy_advantage": task_contrasts,
        "state_accounting": state,
        "unique_foundation_rows_at_8192_token_cap": len(unique_cap_rows),
        "provenance": provenance,
        "gates": gates,
        "control_pass": all(gates.values()),
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
