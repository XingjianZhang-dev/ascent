#!/usr/bin/env python3
"""Validate and summarize the frozen BABILong systems repetitions."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


EXPECTED_COMMIT = "ef2322d093679b3b3bbb1f1ed3f090c7ae046424"
EXPECTED_CONFIG_HASH = "f2075a7bf6d4bf31ddfc6da7fd57eecabac931d7cdd68d502a02ce670e6b2798"
EXPECTED_ORDERS = ("foundation_first", "ascent_first")
EXPECTED_REPETITIONS = 3


def span(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def prediction_signature(run: dict[str, Any]) -> list[tuple[Any, ...]]:
    return [
        (
            row["row_id"],
            row["foundation_location"],
            row["foundation_score"],
            row["ascent_location"],
            row["ascent_score"],
        )
        for row in run["predictions"]
    ]


def summarize_group(runs: list[dict[str, Any]]) -> dict[str, Any]:
    systems = [run["systems"] for run in runs]
    return {
        "runs": len(runs),
        "foundation_decode_seconds": span(
            [row["foundation_decode"]["seconds"] for row in systems]
        ),
        "ascent_decode_seconds": span(
            [row["ascent_decode"]["seconds"] for row in systems]
        ),
        "ascent_to_foundation_decode_time_ratio": span(
            [row["ascent_to_foundation_decode_time_ratio"] for row in systems]
        ),
        "ascent_to_foundation_prompt_token_ratio": span(
            [row["ascent_to_foundation_prompt_token_ratio"] for row in systems]
        ),
        "retrieval_microseconds_per_sample": span(
            [row["retrieval_microseconds_per_sample"] for row in systems]
        ),
        "foundation_incremental_peak_bytes": span(
            [row["foundation_decode"]["incremental_peak_bytes"] for row in systems]
        ),
        "ascent_incremental_peak_bytes": span(
            [row["ascent_decode"]["incremental_peak_bytes"] for row in systems]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths: list[Path] = []
    for source in args.inputs:
        paths.extend(sorted(source.glob("*.json")) if source.is_dir() else [source])
    runs = [json.loads(path.read_text()) for path in paths]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors: list[str] = []
    for path, run in zip(paths, runs, strict=True):
        endpoint = run["endpoint"]["name"]
        grouped[endpoint].append(run)
        if run["config"]["sha256"] != EXPECTED_CONFIG_HASH:
            errors.append(f"{path}: config hash mismatch")
        if run["environment"]["git_commit"] != EXPECTED_COMMIT:
            errors.append(f"{path}: systems commit mismatch")
        if run["environment"]["git_dirty"]:
            errors.append(f"{path}: dirty worktree")
        if run["parser_accuracy"] != 1.0:
            errors.append(f"{path}: parser mismatch")

    endpoint_summaries: dict[str, Any] = {}
    state_rows: list[tuple[int, float, float, float]] = []
    accuracy_invariant = True
    order_direction_agrees = True
    every_retrieval_below_1ms = True
    for endpoint, endpoint_runs in sorted(
        grouped.items(), key=lambda item: item[1][0]["endpoint"]["model_parameters"]
    ):
        if len(endpoint_runs) != len(EXPECTED_ORDERS) * EXPECTED_REPETITIONS:
            errors.append(f"{endpoint}: expected 6 runs, found {len(endpoint_runs)}")
        by_order: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for run in endpoint_runs:
            by_order[run["systems"]["decode_order"]].append(run)
        for order in EXPECTED_ORDERS:
            if len(by_order[order]) != EXPECTED_REPETITIONS:
                errors.append(
                    f"{endpoint}/{order}: expected 3 runs, found {len(by_order[order])}"
                )

        reference = prediction_signature(endpoint_runs[0])
        same_predictions = all(
            prediction_signature(run) == reference for run in endpoint_runs[1:]
        )
        same_scores = all(
            (run["foundation"]["mean"], run["ascent"]["mean"], run["gain"]["mean"])
            == (
                endpoint_runs[0]["foundation"]["mean"],
                endpoint_runs[0]["ascent"]["mean"],
                endpoint_runs[0]["gain"]["mean"],
            )
            for run in endpoint_runs[1:]
        )
        accuracy_invariant &= same_predictions and same_scores

        order_summaries = {
            order: summarize_group(by_order[order])
            for order in EXPECTED_ORDERS
            if by_order[order]
        }
        order_direction = all(
            row["ascent_to_foundation_decode_time_ratio"]["median"] < 1.0
            and row["ascent_incremental_peak_bytes"]["median"]
            < row["foundation_incremental_peak_bytes"]["median"]
            for row in order_summaries.values()
        )
        order_direction_agrees &= order_direction
        retrieval_pass = all(
            run["systems"]["retrieval_microseconds_per_sample"] < 1000.0
            for run in endpoint_runs
        )
        every_retrieval_below_1ms &= retrieval_pass

        exemplar = endpoint_runs[0]
        state = exemplar["state"]
        model_bytes = float(exemplar["endpoint"]["model_artifact"]["bytes"])
        total_state = float(state["mean_total_external_state_bytes"])
        state_rows.append(
            (
                int(exemplar["endpoint"]["model_parameters"]),
                float(state["write_state_payload_bytes"]["mean"]),
                float(state["read_state_payload_bytes"]["mean"]),
                total_state / model_bytes,
            )
        )
        endpoint_summaries[endpoint] = {
            "accuracy": {
                "foundation": exemplar["foundation"]["mean"],
                "ascent": exemplar["ascent"]["mean"],
                "gain": exemplar["gain"]["mean"],
                "predictions_identical_across_repetitions": same_predictions,
            },
            "state": {
                **state,
                "mean_total_external_state_to_model_artifact_byte_ratio": total_state
                / model_bytes,
            },
            "all_repetitions": summarize_group(endpoint_runs),
            "by_decode_order": order_summaries,
            "order_direction_pass": order_direction,
            "every_retrieval_repetition_below_1ms": retrieval_pass,
        }

    state_rows.sort()
    fixed_write_state = len({row[1] for row in state_rows}) <= 1
    read_state_strictly_grows = all(
        left[2] < right[2] for left, right in zip(state_rows, state_rows[1:])
    )
    state_ratio_strictly_decreases = all(
        left[3] > right[3] for left, right in zip(state_rows, state_rows[1:])
    )
    complete = len(grouped) == 3 and not errors
    gates = {
        "complete_six_repetitions_per_endpoint": complete,
        "accuracy_and_predictions_invariant": accuracy_invariant,
        "both_decode_orders_agree_in_direction": order_direction_agrees,
        "every_retrieval_repetition_below_1ms": every_retrieval_below_1ms,
        "write_state_fixed_across_endpoints": fixed_write_state,
        "query_conditioned_read_state_strictly_grows": read_state_strictly_grows,
        "external_state_to_model_byte_ratio_strictly_decreases": state_ratio_strictly_decreases,
    }
    output = {
        "schema_version": 1,
        "source_files": [str(path) for path in paths],
        "validation_errors": errors,
        "endpoints": endpoint_summaries,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "retrieval_gate_interpretation": "Strict: every fresh-process repetition must be below 1000 microseconds/sample; medians alone do not pass this gate.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
