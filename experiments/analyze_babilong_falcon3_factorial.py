#!/usr/bin/env python3
"""Analyze the prospectively frozen Falcon3 1B/3B/7B BABILong factorial."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


T95_DF8 = 2.306004135204166
T_ONE_SIDED_BONFERRONI_DF8_M6 = 3.015761836887169


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_model_files(endpoint: dict[str, Any]) -> list[dict[str, Any]]:
    singular = endpoint.get("model_artifact")
    plural = endpoint.get("model_artifacts")
    if (singular is None) == (plural is None):
        raise RuntimeError("endpoint must freeze exactly one weight-manifest field")
    return [singular] if singular is not None else list(plural)


def paired(left: list[float], right: list[float]) -> list[float]:
    return (
        np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    ).tolist()


def mean_interval(values: list[float], critical: float | None) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("an estimand requires at least one panel")
    mean = float(array.mean())
    if critical is None:
        return {
            "values": array.tolist(),
            "mean": mean,
            "standard_error": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    if array.size != 9:
        raise ValueError("confirmation requires exactly nine panel values")
    standard_error = float(array.std(ddof=1) / np.sqrt(array.size))
    return {
        "values": array.tolist(),
        "mean": mean,
        "standard_error": standard_error,
        "ci95_low": mean - critical * standard_error,
        "ci95_high": mean + critical * standard_error,
    }


def values(
    results: dict[tuple[str, int], list[dict[str, Any]]],
    endpoint: str,
    slots: int,
    field: str,
) -> list[float]:
    return [float(row[field]["mean"]) for row in results[(endpoint, slots)]]


def collect(
    root: Path,
    config_path: Path,
    phase: str,
    run_commit: str,
) -> tuple[
    dict[tuple[str, int], list[dict[str, Any]]],
    list[str],
    dict[str, Any],
]:
    config = load(config_path)
    config_sha256 = sha256_file(config_path)
    panel_key = "development_panels" if phase == "development" else "confirmation_panels"
    panels = list(config[panel_key])
    if phase == "development" and len(panels) != 1:
        raise RuntimeError("development must contain exactly one frozen panel")
    if phase == "confirmation" and len(panels) != 9:
        raise RuntimeError("confirmation must contain exactly nine frozen panels")
    endpoints = {row["name"]: row for row in config["endpoints"]}
    endpoint_order = list(config["factorial"]["endpoints"])
    state_slots = [int(value) for value in config["factorial"]["state_fact_slots"]]
    node_by_endpoint = config["execution"]["node_by_endpoint"]
    results: dict[tuple[str, int], list[dict[str, Any]]] = {}
    checks: list[bool] = []
    for endpoint in endpoint_order:
        node = node_by_endpoint[endpoint]
        for slots in state_slots:
            rows = [
                load(
                    root
                    / phase
                    / node
                    / f"generative_slots_{slots}_{panel}_{endpoint}.json"
                )
                for panel in panels
            ]
            results[(endpoint, slots)] = rows
            for panel, row in zip(panels, rows, strict=True):
                checks.extend(
                    (
                        row["endpoint"] == endpoints[endpoint],
                        row["condition"]["name"] == f"generative_slots_{slots}",
                        row["condition"]["fact_slots"] == slots,
                        row["condition"]["memory_source"] == "relevant",
                        row["condition"]["memory_representation"] == "event_facts",
                        row["condition"]["readout_path"] == "foundation_generation",
                        not row["condition"]["diagnostic_no_promotion"],
                        row["config"]["sha256"] == config_sha256,
                        row["data"]["sha256"]
                        == config["panel_sha256_by_name"][panel],
                        row["model_files"] == expected_model_files(endpoints[endpoint]),
                        row["status"] == config["status"],
                        row["parser_accuracy"] == 1.0,
                        len(row["predictions"]) == config["evaluation_samples"],
                        not row["environment"]["git_dirty"],
                        row["environment"]["git_commit"] == run_commit,
                        row["data"]["path"].endswith(f"{panel}.jsonl"),
                    )
                )

    def foundation_projection(row: dict[str, Any]) -> list[tuple[Any, ...]]:
        return [
            (
                prediction["row_id"],
                prediction["task"],
                prediction["target"],
                prediction["foundation_output"],
                prediction["foundation_answer"],
                prediction["foundation_score"],
                prediction["foundation_prompt_tokens"],
            )
            for prediction in row["predictions"]
        ]

    foundation_equal = True
    nested_state = True
    refinement_rows = {
        endpoint: {
            f"slots_{low}_to_{high}": 0
            for low, high in zip(state_slots[:-1], state_slots[1:], strict=True)
        }
        for endpoint in endpoint_order
    }
    for endpoint in endpoint_order:
        for panel_index in range(len(panels)):
            rows = [results[(endpoint, slots)][panel_index] for slots in state_slots]
            foundation_equal &= all(
                foundation_projection(row) == foundation_projection(rows[0])
                for row in rows[1:]
            )
            for transition_index, (low, high) in enumerate(
                zip(rows[:-1], rows[1:], strict=True)
            ):
                transition = (
                    f"slots_{state_slots[transition_index]}_to_"
                    f"{state_slots[transition_index + 1]}"
                )
                for low_prediction, high_prediction in zip(
                    low["predictions"], high["predictions"], strict=True
                ):
                    low_facts = low_prediction["retained_facts"]
                    high_facts = high_prediction["retained_facts"]
                    nested_state &= len(low_facts) <= len(high_facts)
                    nested_state &= high_facts[-len(low_facts) :] == low_facts
                    refinement_rows[endpoint][transition] += low_facts != high_facts

    panel_alignment = True
    for panel_index in range(len(panels)):
        identities = [
            [
                (prediction["row_id"], prediction["task"], prediction["target"])
                for prediction in rows[panel_index]["predictions"]
            ]
            for rows in results.values()
        ]
        panel_alignment &= all(identity == identities[0] for identity in identities[1:])

    state_ratios = [
        slots / int(endpoints[endpoint]["model_parameters"])
        for endpoint, slots in zip(endpoint_order, state_slots, strict=True)
    ]
    decreasing_state_ratio = all(
        high < low for low, high in zip(state_ratios[:-1], state_ratios[1:], strict=True)
    )
    nonredundant_refinement = all(
        count > 0
        for endpoint_counts in refinement_rows.values()
        for count in endpoint_counts.values()
    )
    provenance = {
        "config_sha256": config_sha256,
        "run_commit": run_commit,
        "all_cell_checks_pass": all(checks),
        "foundation_equal_across_state_sizes": foundation_equal,
        "all_cells_row_task_target_aligned": panel_alignment,
        "exact_nested_state_suffixes": nested_state,
        "nonredundant_rows_by_endpoint_and_transition": refinement_rows,
        "every_registered_transition_is_nonredundant": nonredundant_refinement,
        "co_scaled_state_slots_per_parameter": state_ratios,
        "co_scaled_state_ratio_strictly_decreases": decreasing_state_ratio,
    }
    provenance["provenance_and_alignment_pass"] = all(
        (
            all(checks),
            foundation_equal,
            panel_alignment,
            nested_state,
            nonredundant_refinement,
            decreasing_state_ratio,
        )
    )
    return results, panels, provenance


def registered_estimand_values(
    results: dict[tuple[str, int], list[dict[str, Any]]],
    endpoints: list[str],
    state_slots: list[int],
) -> dict[str, list[float]]:
    estimands: dict[str, list[float]] = {}
    diagonal = [
        values(results, endpoint, slots, "gain")
        for endpoint, slots in zip(endpoints, state_slots, strict=True)
    ]
    for index, (low_endpoint, high_endpoint) in enumerate(
        zip(endpoints[:-1], endpoints[1:], strict=True)
    ):
        low_slots, high_slots = state_slots[index], state_slots[index + 1]
        prefix = "first" if index == 0 else "second"
        interaction = paired(
            paired(
                values(results, high_endpoint, high_slots, "ascent"),
                values(results, high_endpoint, low_slots, "ascent"),
            ),
            paired(
                values(results, low_endpoint, high_slots, "ascent"),
                values(results, low_endpoint, low_slots, "ascent"),
            ),
        )
        diagonal_increment = paired(diagonal[index + 1], diagonal[index])
        memory_only = paired(
            values(results, low_endpoint, high_slots, "gain"),
            values(results, low_endpoint, low_slots, "gain"),
        )
        estimands[f"{prefix}_registered_interaction"] = interaction
        estimands[f"{prefix}_diagonal_gain_increment"] = diagonal_increment
        estimands[f"{prefix}_coscale_excess"] = paired(
            diagonal_increment, memory_only
        )
    return estimands


def analyze(
    config_path: Path, root: Path, phase: str, run_commit: str
) -> dict[str, Any]:
    if phase not in {"development", "confirmation"}:
        raise ValueError("phase must be development or confirmation")
    config = load(config_path)
    endpoints = list(config["factorial"]["endpoints"])
    state_slots = [int(value) for value in config["factorial"]["state_fact_slots"]]
    results, panels, provenance = collect(root, config_path, phase, run_commit)
    critical = None if phase == "development" else T95_DF8
    cells = {
        f"{endpoint}__slots_{slots}": {
            "foundation_accuracy": mean_interval(
                values(results, endpoint, slots, "foundation"), critical
            ),
            "ascent_accuracy": mean_interval(
                values(results, endpoint, slots, "ascent"), critical
            ),
            "absolute_gain": mean_interval(
                values(results, endpoint, slots, "gain"), critical
            ),
        }
        for endpoint in endpoints
        for slots in state_slots
    }
    raw_estimands = registered_estimand_values(results, endpoints, state_slots)
    estimands = {
        key: mean_interval(value, critical) for key, value in raw_estimands.items()
    }
    diagonal_keys = [
        f"{endpoint}__slots_{slots}"
        for endpoint, slots in zip(endpoints, state_slots, strict=True)
    ]
    diagonal_gains_positive = all(
        cells[key]["absolute_gain"]["mean"] > 0.0 for key in diagonal_keys
    )
    if phase == "development":
        directional = {key: value["mean"] > 0.0 for key, value in estimands.items()}
        gates = {
            "provenance": provenance["provenance_and_alignment_pass"],
            "all_diagonal_gains_positive": diagonal_gains_positive,
            "all_six_registered_estimand_means_positive": all(directional.values()),
        }
        gates["development_pass"] = all(gates.values())
        familywise = None
        familywise_positive = None
    else:
        directional = {
            key: value["ci95_low"] > 0.0 for key, value in estimands.items()
        }
        familywise = {
            key: mean_interval(value, T_ONE_SIDED_BONFERRONI_DF8_M6)
            for key, value in raw_estimands.items()
        }
        familywise_positive = {
            key: value["ci95_low"] > 0.0 for key, value in familywise.items()
        }
        gates = {
            "provenance": provenance["provenance_and_alignment_pass"],
            "all_diagonal_gain_lcbs_positive": all(
                cells[key]["absolute_gain"]["ci95_low"] > 0.0
                for key in diagonal_keys
            ),
            "all_six_registered_lcbs_positive": all(directional.values()),
            "all_six_bonferroni_familywise_lcbs_positive": all(
                familywise_positive.values()
            ),
        }
        gates["confirmation_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "phase": phase,
        "panels": panels,
        "endpoints": endpoints,
        "state_fact_slots": state_slots,
        "provenance": provenance,
        "cells": cells,
        "registered_estimands": estimands,
        "registered_directions_positive": directional,
        "familywise_directional_estimands": familywise,
        "familywise_directions_positive": familywise_positive,
        "gates": gates,
        "interpretation": (
            "A prospectively frozen same-family Falcon3 neural-generation factorial. "
            "Confirmation is authorized only by an untouched development pass."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("development", "confirmation"), required=True
    )
    parser.add_argument("--run-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.config, args.root, args.phase, args.run_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
