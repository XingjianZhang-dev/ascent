#!/usr/bin/env python3
"""Audit and analyze the frozen noisy-composition scale factorial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.prepare_noisy_composition_factorial import sha256_file


T95_DF8 = 2.306004135204166
# Two-sided familywise alpha=.05 for the six registered estimands:
# t.ppf(1 - .05 / (2 * 6), df=8).  This supplementary interval is fixed
# before inspecting the confirmation-panel results and does not replace the
# prospectively registered per-estimand 95% gate.
T95_BONFERRONI_DF8_M6 = 3.478879189965178


def paired(left: list[float], right: list[float]) -> list[float]:
    return (np.asarray(left) - np.asarray(right)).tolist()


def interval(
    values: list[float], phase: str, critical_value: float = T95_DF8
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if phase == "development":
        return {"values": array.tolist(), "mean": mean, "ci95_low": None, "ci95_high": None}
    if array.size != 9:
        raise RuntimeError("confirmation requires nine panels")
    se = float(array.std(ddof=1) / np.sqrt(array.size))
    return {
        "values": array.tolist(),
        "mean": mean,
        "standard_error": se,
        "ci95_low": mean - critical_value * se,
        "ci95_high": mean + critical_value * se,
    }


def expected_model_files(endpoint: dict[str, Any]) -> list[dict[str, Any]]:
    singular = endpoint.get("model_artifact")
    plural = endpoint.get("model_artifacts")
    if (singular is None) == (plural is None):
        raise RuntimeError("endpoint must freeze exactly one artifact field")
    return [singular] if singular is not None else list(plural)


def analyze(config_path: Path, root: Path, phase: str, run_commit: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    config_sha = sha256_file(config_path)
    panel_key = "development_panels" if phase == "development" else "confirmation_panels"
    panels = config[panel_key]
    if len(panels) != (1 if phase == "development" else 9):
        raise RuntimeError("wrong phase panel count")
    endpoints = {endpoint["name"]: endpoint for endpoint in config["endpoints"]}
    endpoint_order = config["factorial"]["endpoints"]
    rounds_order = config["factorial"]["state_rounds"]
    node_by_endpoint = config["execution"]["node_by_endpoint"]
    cells: dict[tuple[str, int], list[dict[str, Any]]] = {}
    checks: list[bool] = []
    for endpoint_name in endpoint_order:
        node = node_by_endpoint[endpoint_name]
        endpoint = endpoints[endpoint_name]
        for rounds in rounds_order:
            rows = []
            for panel in panels:
                path = (
                    root
                    / phase
                    / node
                    / f"rounds_{rounds}_{panel}_{endpoint_name}.json"
                )
                result = json.loads(path.read_text())
                rows.append(result)
                checks.extend(
                    [
                        result["status"] == config["status"],
                        result["endpoint"] == endpoint,
                        result["rounds"] == rounds,
                        result["panel"] == panel,
                        result["config_sha256"] == config_sha,
                        result["data_sha256"] == config["panel_sha256_by_name"][panel],
                        result["model_files"] == expected_model_files(endpoint),
                        result["model_identity"]["loaded_parameters"]
                        == endpoint["model_parameters"],
                        result["preflight"]["pass"],
                        len(result["predictions"]) == config["evaluation_samples"],
                        not result["environment"]["git_dirty"],
                        result["environment"]["git_commit"] == run_commit,
                        not result["environment"]["audit_rerun"],
                    ]
                )
                minimum_final_coverage = config.get("minimum_final_marker_coverage")
                if minimum_final_coverage is not None:
                    checks.extend(
                        [
                            result["final_marker_coverage"]["foundation"]
                            >= minimum_final_coverage,
                            result["final_marker_coverage"]["ascent"]
                            >= minimum_final_coverage,
                        ]
                    )
                minimum_answer_coverage = config.get("minimum_answer_coverage")
                if minimum_answer_coverage is not None:
                    checks.extend(
                        [
                            result["answer_coverage"]["foundation"]
                            >= minimum_answer_coverage,
                            result["answer_coverage"]["ascent"]
                            >= minimum_answer_coverage,
                        ]
                    )
            cells[(endpoint_name, rounds)] = rows

    foundation_equal = True
    row_alignment = True
    nested = True
    transition_changes = {
        endpoint: {f"rounds_{low}_to_{high}": 0 for low, high in zip(rounds_order[:-1], rounds_order[1:], strict=True)}
        for endpoint in endpoint_order
    }
    for endpoint in endpoint_order:
        for panel_index in range(len(panels)):
            rows = [cells[(endpoint, rounds)][panel_index] for rounds in rounds_order]
            foundation_projection = [
                [
                    (
                        prediction["row_id"],
                        prediction["episode_id"],
                        prediction["target"],
                        prediction["foundation_output"],
                        prediction["foundation_answer"],
                        prediction["foundation_score"],
                        prediction["foundation_prompt_tokens"],
                    )
                    for prediction in row["predictions"]
                ]
                for row in rows
            ]
            foundation_equal &= all(value == foundation_projection[0] for value in foundation_projection[1:])
            identities = [
                [(prediction["row_id"], prediction["target"]) for prediction in row["predictions"]]
                for row in rows
            ]
            row_alignment &= all(value == identities[0] for value in identities[1:])
            for index, (low_result, high_result) in enumerate(zip(rows[:-1], rows[1:], strict=True)):
                transition = f"rounds_{rounds_order[index]}_to_{rounds_order[index + 1]}"
                for low_prediction, high_prediction in zip(
                    low_result["predictions"], high_result["predictions"], strict=True
                ):
                    low_first = low_prediction["retained_first_observations"]
                    high_first = high_prediction["retained_first_observations"]
                    low_second = low_prediction["retained_second_observations"]
                    high_second = high_prediction["retained_second_observations"]
                    nested &= high_first[: len(low_first)] == low_first
                    nested &= high_second[: len(low_second)] == low_second
                    changed = low_first != high_first or low_second != high_second
                    transition_changes[endpoint][transition] += int(changed)

    for panel_index in range(len(panels)):
        identities = []
        for endpoint in endpoint_order:
            for rounds in rounds_order:
                identities.append(
                    [
                        (prediction["row_id"], prediction["target"])
                        for prediction in cells[(endpoint, rounds)][panel_index]["predictions"]
                    ]
                )
        row_alignment &= all(value == identities[0] for value in identities[1:])
    nonredundant = all(
        count == config["evaluation_samples"] * len(panels)
        for values in transition_changes.values()
        for count in values.values()
    )
    ratios = [
        rounds / endpoints[endpoint]["model_parameters"]
        for endpoint, rounds in zip(endpoint_order, rounds_order, strict=True)
    ]
    decreasing_ratio = all(high < low for low, high in zip(ratios[:-1], ratios[1:], strict=True))
    exact_information = True
    for panel in panels:
        curve = [
            config["exact_posterior_mean_nll_by_panel_and_rounds"][panel][str(rounds)]
            for rounds in rounds_order
        ]
        exact_information &= all(high < low for low, high in zip(curve[:-1], curve[1:], strict=True))

    def metric(endpoint: str, rounds: int, field: str) -> list[float]:
        return [float(row[field]["mean"]) for row in cells[(endpoint, rounds)]]

    diagonal_gains = [
        metric(endpoint, rounds, "gain")
        for endpoint, rounds in zip(endpoint_order, rounds_order, strict=True)
    ]
    diagonal_names = [
        f"gain_{endpoint}_k{rounds}"
        for endpoint, rounds in zip(endpoint_order, rounds_order, strict=True)
    ]
    estimand_values: dict[str, list[float]] = {
        diagonal_names[0]: diagonal_gains[0],
        diagonal_names[1]: diagonal_gains[1],
        diagonal_names[2]: diagonal_gains[2],
        "first_diagonal_gain_increment": paired(diagonal_gains[1], diagonal_gains[0]),
        "second_diagonal_gain_increment": paired(diagonal_gains[2], diagonal_gains[1]),
        "7b_k5_minus_k3_gain": paired(
            metric(endpoint_order[2], rounds_order[2], "gain"),
            metric(endpoint_order[2], rounds_order[1], "gain"),
        ),
    }
    for index, (low_endpoint, high_endpoint) in enumerate(zip(endpoint_order[:-1], endpoint_order[1:], strict=True)):
        low_rounds = rounds_order[index]
        high_rounds = rounds_order[index + 1]
        prefix = "first" if index == 0 else "second"
        estimand_values[f"{prefix}_model_by_state_interaction"] = paired(
            paired(metric(high_endpoint, high_rounds, "ascent"), metric(high_endpoint, low_rounds, "ascent")),
            paired(metric(low_endpoint, high_rounds, "ascent"), metric(low_endpoint, low_rounds, "ascent")),
        )
    estimands = {name: interval(values, phase) for name, values in estimand_values.items()}
    registered = [
        *diagonal_names,
        "first_diagonal_gain_increment",
        "second_diagonal_gain_increment",
        "7b_k5_minus_k3_gain",
    ]
    familywise_registered_estimands = {
        name: interval(
            estimand_values[name], phase, critical_value=T95_BONFERRONI_DF8_M6
        )
        for name in registered
    }
    provenance = {
        "all_cell_checks_pass": all(checks),
        "foundation_equal_across_state_sizes": foundation_equal,
        "all_cells_row_target_aligned": row_alignment,
        "exact_nested_observation_prefixes": nested,
        "nonredundant_rows_by_endpoint_and_transition": transition_changes,
        "every_transition_changes_every_row": nonredundant,
        "exact_posterior_strictly_improves_every_transition_every_panel": exact_information,
        "co_scaled_rounds_per_parameter": ratios,
        "co_scaled_ratio_strictly_decreases": decreasing_ratio,
    }
    provenance["pass"] = all(
        [all(checks), foundation_equal, row_alignment, nested, nonredundant, exact_information, decreasing_ratio]
    )
    if phase == "development":
        performance_pass = all(estimands[name]["mean"] > 0 for name in registered)
        familywise_performance_pass = None
    else:
        performance_pass = all(estimands[name]["ci95_low"] > 0 for name in registered)
        familywise_performance_pass = all(
            familywise_registered_estimands[name]["ci95_low"] > 0
            for name in registered
        )
    payload = {
        "schema_version": 1,
        "phase": phase,
        "config_sha256": config_sha,
        "run_commit": run_commit,
        "provenance": provenance,
        "cells": {
            f"{endpoint}@k{rounds}": {
                "foundation": interval(metric(endpoint, rounds, "foundation"), phase),
                "ascent": interval(metric(endpoint, rounds, "ascent"), phase),
                "gain": interval(metric(endpoint, rounds, "gain"), phase),
            }
            for endpoint in endpoint_order
            for rounds in rounds_order
        },
        "estimands": estimands,
        "supplementary_bonferroni_familywise_estimands": familywise_registered_estimands,
        "supplementary_bonferroni_familywise_performance_pass": familywise_performance_pass,
        "registered_performance_pass": performance_pass,
        "gate_pass": provenance["pass"] and performance_pass,
        "confirmation_authorized": phase == "development" and provenance["pass"] and performance_pass,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--run-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.config, args.root, args.phase, args.run_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gate_pass": result["gate_pass"], "estimands": result["estimands"]}, indent=2))


if __name__ == "__main__":
    main()
