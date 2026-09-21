#!/usr/bin/env python3
"""Audit the corrected generative factorial across two frozen references.

The immediately preceding factorial is the decode/score reference.  The older
diagonal archive predates inventory-task support and is the state-byte reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_babilong_prompt import extract_task_answer


ENDPOINTS = (
    ("smollm2-135m-instruct", "node1", 1),
    ("smollm2-360m-instruct", "node2", 2),
    ("smollm2-1p7b-instruct", "node1", 3),
)
PANELS = (1, 2, 3)
NEW_COMMIT = "693c41455d3a431a6da6880d64eb791ec56a6592"
NEW_CONFIG_SHA256 = "5085ca195269ae38830d8967cd0e0ae3a308764b1a83cd03266229343fc546a6"
REFERENCE_COMMIT = "072adbd014ce4123c8eba284ea7ab635e912afde"
REFERENCE_CONFIG_SHA256 = "d6987b5780b12ee4e39666536805c133408c6055a2defee57bbda080196c405c"
T95_DF2 = 4.302652729911275


PREDICTION_FIELDS = (
    "row_id",
    "task",
    "target",
    "foundation_output",
    "foundation_location",
    "foundation_score",
    "ascent_output",
    "ascent_location",
    "ascent_score",
    "supporting_fact_count",
    "retained_fact_count",
    "retained_facts",
    "foundation_prompt_tokens",
    "ascent_prompt_tokens",
)
RESULT_FIELDS = (
    "parser_accuracy",
    "foundation",
    "ascent",
    "gain",
    "wins",
    "regressions",
)


def remaining_error_elimination(result: dict[str, Any]) -> float:
    """Derive RWE from primitive metrics for old schemas that did not store it."""
    foundation = float(result["foundation"]["mean"])
    gain = float(result["gain"]["mean"])
    return gain / (1.0 - foundation)


def primitive_by_task(result: dict[str, Any]) -> dict[str, Any]:
    """Project nested task summaries across the pre-/post-RWE schemas."""
    return {
        task: {field: summary[field] for field in ("foundation", "ascent", "gain")}
        for task, summary in result["by_task"].items()
    }


def parsed_answers_match_outputs(result: dict[str, Any]) -> bool:
    return all(
        prediction["foundation_answer"]
        == extract_task_answer(prediction["foundation_output"], prediction["task"])
        and prediction["ascent_answer"]
        == extract_task_answer(prediction["ascent_output"], prediction["task"])
        for prediction in result["predictions"]
    )


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def clustered_interval(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    standard_error = float(array.std(ddof=1) / np.sqrt(array.size))
    return {
        "mean": mean,
        "standard_error": standard_error,
        "ci95_low": mean - T95_DF2 * standard_error,
        "ci95_high": mean + T95_DF2 * standard_error,
    }


def projected_predictions(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {field: prediction.get(field) for field in PREDICTION_FIELDS}
        for prediction in result["predictions"]
    ]


def analyze(
    new_root: Path,
    previous_root: Path,
    reference_root: Path,
    factorial_analysis: dict[str, Any],
) -> dict[str, Any]:
    all_cell_comparisons: list[dict[str, Any]] = []
    all_cells_exact = True
    for endpoint, node, _ in ENDPOINTS:
        for slots in (1, 2, 3):
            for panel in PANELS:
                new = load(
                    new_root
                    / node
                    / f"generative_slots_{slots}_panel{panel}_{endpoint}.json"
                )
                previous = load(
                    previous_root
                    / node
                    / f"generative_slots_{slots}_panel{panel}_{endpoint}.json"
                )
                checks = {
                    "new_commit": new["environment"]["git_commit"] == NEW_COMMIT,
                    "new_clean": not new["environment"]["git_dirty"],
                    "new_config": new["config"]["sha256"] == NEW_CONFIG_SHA256,
                    "new_audit_status": new["status"]
                    == "posthoc_code_audit_correction_rerun_no_independence_claim",
                    "previous_commit": previous["environment"]["git_commit"]
                    == "e1843cd518d67ccee3592e5a13210b1fc7ebbeeb",
                    "previous_clean": not previous["environment"]["git_dirty"],
                    "previous_config": previous["config"]["sha256"]
                    == NEW_CONFIG_SHA256,
                    "data_sha256_exact": new["data"]["sha256"]
                    == previous["data"]["sha256"],
                    "model_files_exact": new["model_files"]
                    == previous["model_files"],
                    "decode_and_score_predictions_exact": projected_predictions(new)
                    == projected_predictions(previous),
                    "result_fields_exact": all(
                        new[field] == previous[field] for field in RESULT_FIELDS
                    ),
                    "by_task_primitives_exact": primitive_by_task(new)
                    == primitive_by_task(previous),
                    "new_parsed_answers_match_outputs": parsed_answers_match_outputs(new),
                    "new_rwe_matches_primitives": bool(
                        np.isclose(
                            new["remaining_error_elimination"],
                            remaining_error_elimination(new),
                            atol=1e-15,
                        )
                    ),
                    "new_fact_slots": new["condition"]["fact_slots"] == slots,
                    "previous_fact_slots": previous["condition"]["fact_slots"]
                    == slots,
                }
                exact = all(checks.values())
                all_cells_exact &= exact
                all_cell_comparisons.append(
                    {
                        "endpoint": endpoint,
                        "panel": panel,
                        "slots": slots,
                        "exact": exact,
                        "checks": checks,
                    }
                )

    diagonal_comparisons: list[dict[str, Any]] = []
    all_diagonal_payloads_exact = True
    for endpoint, node, slots in ENDPOINTS:
        for panel in PANELS:
            new = load(
                new_root
                / node
                / f"generative_slots_{slots}_panel{panel}_{endpoint}.json"
            )
            reference = load(
                reference_root
                / node
                / f"panel{panel}_{endpoint}_scale_relevant.json"
            )
            checks = {
                "reference_commit": reference["environment"]["git_commit"]
                == REFERENCE_COMMIT,
                "reference_clean": not reference["environment"]["git_dirty"],
                "reference_config": reference["config"]["sha256"]
                == REFERENCE_CONFIG_SHA256,
                "data_sha256_exact": new["data"]["sha256"]
                == reference["data"]["sha256"],
                "model_files_exact": new["model_files"] == reference["model_files"],
                "row_task_target_exact": all(
                    (left["row_id"], left["task"], left["target"])
                    == (right["row_id"], right["task"], right["target"])
                    for left, right in zip(
                        new["predictions"], reference["predictions"], strict=True
                    )
                ),
                "persistent_payload_bytes_exact": all(
                    left["persistent_payload_bytes"]
                    == right["persistent_payload_bytes"]
                    for left, right in zip(
                        new["predictions"], reference["predictions"], strict=True
                    )
                ),
                "new_fact_slots": new["condition"]["fact_slots"] == slots,
                "reference_fact_slots": reference["condition"]["fact_slots"]
                == slots,
            }
            exact = all(checks.values())
            all_diagonal_payloads_exact &= exact
            diagonal_comparisons.append(
                {
                    "endpoint": endpoint,
                    "panel": panel,
                    "slots": slots,
                    "exact": exact,
                    "checks": checks,
                }
            )

    interactions = factorial_analysis[
        "registered_foundation_by_state_interactions"
    ]
    independent_intervals = {
        key: clustered_interval(interval["values"])
        for key, interval in interactions.items()
    }
    interval_exact = bool(all(
        all(
            np.isclose(independent_intervals[key][field], interval[field], atol=1e-15)
            for field in ("mean", "standard_error", "ci95_low", "ci95_high")
        )
        for key, interval in interactions.items()
    ))
    return {
        "schema_version": 1,
        "all_cell_decode_score_comparisons": all_cell_comparisons,
        "all_27_decode_and_score_runs_exact_to_previous_factorial": all_cells_exact,
        "diagonal_state_byte_comparisons": diagonal_comparisons,
        "all_nine_diagonal_state_payloads_exact_to_pre_inventory_reference": all_diagonal_payloads_exact,
        "independent_df2_intervals": independent_intervals,
        "factorial_interval_fields_exact": interval_exact,
        "factorial_provenance_gate": factorial_analysis["gates"][
            "provenance_and_alignment"
        ],
        "audit_pass": all_cells_exact
        and all_diagonal_payloads_exact
        and interval_exact
        and factorial_analysis["gates"]["provenance_and_alignment"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--factorial-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.new_root,
        args.previous_root,
        args.reference_root,
        load(args.factorial_analysis),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
