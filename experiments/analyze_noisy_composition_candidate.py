#!/usr/bin/env python3
"""Analyze the candidate-normalized noisy-composition factorial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.prepare_noisy_composition_factorial import posterior, sha256_file


T95_DF8 = 2.306004135204166
T95_BONFERRONI_DF8 = {
    6: 3.478879189965178,
    8: 3.676567780902183,
}


def paired(left: list[float], right: list[float]) -> list[float]:
    return (np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)).tolist()


def interval(
    values: list[float], phase: str, critical_value: float = T95_DF8
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    result: dict[str, Any] = {"values": array.tolist(), "mean": float(array.mean())}
    if phase == "development":
        result.update({"standard_error": None, "ci95_low": None, "ci95_high": None})
        return result
    if array.size != 9:
        raise RuntimeError("confirmation requires nine panels")
    se = float(array.std(ddof=1) / np.sqrt(array.size))
    result.update(
        {
            "standard_error": se,
            "ci95_low": result["mean"] - critical_value * se,
            "ci95_high": result["mean"] + critical_value * se,
        }
    )
    return result


def expected_model_files(endpoint: dict[str, Any]) -> list[dict[str, Any]]:
    if "model_artifact" in endpoint:
        return [endpoint["model_artifact"]]
    return endpoint["model_artifacts"]


def analyze(config_path: Path, root: Path, phase: str, run_commit: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    config_sha = sha256_file(config_path)
    panels = config["development_panels" if phase == "development" else "confirmation_panels"]
    if len(panels) != (1 if phase == "development" else 9):
        raise RuntimeError("wrong panel count")
    endpoint_order = config["factorial"]["endpoints"]
    rounds_order = config["factorial"]["state_rounds"]
    endpoints = {row["name"]: row for row in config["endpoints"]}
    node_by_endpoint = config["execution"]["node_by_endpoint"]
    cells: dict[tuple[str, int], list[dict[str, Any]]] = {}
    checks = []
    for endpoint in endpoint_order:
        for rounds in rounds_order:
            results = []
            for panel in panels:
                path = (
                    root
                    / phase
                    / node_by_endpoint[endpoint]
                    / f"rounds_{rounds}_{panel}_{endpoint}.json"
                )
                row = json.loads(path.read_text())
                results.append(row)
                checks.extend(
                    [
                        row["status"] == config["status"],
                        row["config_sha256"] == config_sha,
                        row["data_sha256"] == config["panel_sha256_by_name"][panel],
                        row["panel"] == panel,
                        row["endpoint"] == endpoints[endpoint],
                        row["rounds"] == rounds,
                        row["model_files"] == expected_model_files(endpoints[endpoint]),
                        row["model_identity"]["loaded_parameters"]
                        == endpoints[endpoint]["model_parameters"],
                        row["preflight"]["pass"],
                        row["preflight"].get(
                            "candidate_support_valid",
                            row["preflight"].get(
                                "distinct_single_token_candidates", False
                            ),
                        ),
                        row["preflight"].get("candidate_boundary_stable", True),
                        not row["preflight"]["generation_invoked"],
                        len(row["predictions"]) == config["evaluation_samples"],
                        not row["environment"]["git_dirty"],
                        row["environment"]["git_commit"] == run_commit,
                        not row["environment"]["audit_rerun"],
                    ]
                )
            cells[(endpoint, rounds)] = results

    foundation_equal = True
    aligned = True
    nested = True
    transition_counts = {
        endpoint: {
            f"rounds_{low}_to_{high}": 0
            for low, high in zip(rounds_order[:-1], rounds_order[1:], strict=True)
        }
        for endpoint in endpoint_order
    }
    for endpoint in endpoint_order:
        for panel_index in range(len(panels)):
            results = [cells[(endpoint, rounds)][panel_index] for rounds in rounds_order]
            projections = [
                [
                    (
                        prediction["row_id"],
                        prediction["episode_id"],
                        prediction["target"],
                        prediction["foundation_probabilities"],
                        prediction["foundation_nll"],
                    )
                    for prediction in result["predictions"]
                ]
                for result in results
            ]
            foundation_equal &= all(value == projections[0] for value in projections[1:])
            for index, (low, high) in enumerate(zip(results[:-1], results[1:], strict=True)):
                transition = f"rounds_{rounds_order[index]}_to_{rounds_order[index + 1]}"
                for low_prediction, high_prediction in zip(
                    low["predictions"], high["predictions"], strict=True
                ):
                    aligned &= (
                        low_prediction["row_id"], low_prediction["target"]
                    ) == (high_prediction["row_id"], high_prediction["target"])
                    low_first = low_prediction["retained_first_observations"]
                    high_first = high_prediction["retained_first_observations"]
                    low_second = low_prediction["retained_second_observations"]
                    high_second = high_prediction["retained_second_observations"]
                    nested &= high_first[: len(low_first)] == low_first
                    nested &= high_second[: len(low_second)] == low_second
                    transition_counts[endpoint][transition] += int(
                        low_first != high_first or low_second != high_second
                    )
    for panel_index in range(len(panels)):
        identities = [
            [
                (prediction["row_id"], prediction["target"])
                for prediction in cells[(endpoint, rounds)][panel_index]["predictions"]
            ]
            for endpoint in endpoint_order
            for rounds in rounds_order
        ]
        aligned &= all(value == identities[0] for value in identities[1:])

    nonredundant = all(
        value == config["evaluation_samples"] * len(panels)
        for counts in transition_counts.values()
        for value in counts.values()
    )
    exact_information = all(
        all(
            high < low
            for low, high in zip(curve[:-1], curve[1:], strict=True)
        )
        for curve in [
            [
                config["exact_posterior_mean_nll_by_panel_and_rounds"][panel][str(rounds)]
                for rounds in rounds_order
            ]
            for panel in panels
        ]
    )
    ratios = [
        rounds / endpoints[endpoint]["model_parameters"]
        for endpoint, rounds in zip(endpoint_order, rounds_order, strict=True)
    ]
    decreasing_ratio = all(high < low for low, high in zip(ratios[:-1], ratios[1:], strict=True))

    def metric(endpoint: str, rounds: int, name: str) -> list[float]:
        return [float(result[name]["mean"]) for result in cells[(endpoint, rounds)]]

    diagonal = [
        metric(endpoint, rounds, "gain_nll")
        for endpoint, rounds in zip(endpoint_order, rounds_order, strict=True)
    ]
    diagonal_names = [
        f"gain_{endpoint}_k{rounds}"
        for endpoint, rounds in zip(endpoint_order, rounds_order, strict=True)
    ]
    values = {
        diagonal_names[0]: diagonal[0],
        diagonal_names[1]: diagonal[1],
        diagonal_names[2]: diagonal[2],
        "first_diagonal_gain_increment": paired(diagonal[1], diagonal[0]),
        "second_diagonal_gain_increment": paired(diagonal[2], diagonal[1]),
        "7b_k5_minus_k3_gain": paired(
            metric(endpoint_order[2], rounds_order[2], "gain_nll"),
            metric(endpoint_order[2], rounds_order[1], "gain_nll"),
        ),
    }
    values["first_model_by_state_interaction"] = paired(
        paired(metric(endpoint_order[1], rounds_order[1], "gain_nll"), metric(endpoint_order[1], rounds_order[0], "gain_nll")),
        paired(metric(endpoint_order[0], rounds_order[1], "gain_nll"), metric(endpoint_order[0], rounds_order[0], "gain_nll")),
    )
    values["second_model_by_state_interaction"] = paired(
        paired(metric(endpoint_order[2], rounds_order[2], "gain_nll"), metric(endpoint_order[2], rounds_order[1], "gain_nll")),
        paired(metric(endpoint_order[1], rounds_order[2], "gain_nll"), metric(endpoint_order[1], rounds_order[1], "gain_nll")),
    )
    estimands = {name: interval(value, phase) for name, value in values.items()}
    registered = [
        *diagonal_names,
        "first_diagonal_gain_increment",
        "second_diagonal_gain_increment",
        "7b_k5_minus_k3_gain",
    ]
    if config.get("require_model_by_state_interactions", False):
        registered.extend(
            [
                "first_model_by_state_interaction",
                "second_model_by_state_interaction",
            ]
        )
    familywise_registered_estimands = {
        name: interval(
            values[name],
            phase,
            critical_value=T95_BONFERRONI_DF8[len(registered)],
        )
        for name in registered
    }
    provenance = {
        "all_cell_checks_pass": all(checks),
        "foundation_probabilities_identical_across_state_sizes": foundation_equal,
        "all_cells_row_target_aligned": aligned,
        "exact_nested_observation_prefixes": nested,
        "nonredundant_rows_by_endpoint_and_transition": transition_counts,
        "every_transition_changes_every_row": nonredundant,
        "exact_posterior_strictly_improves_every_transition_every_panel": exact_information,
        "co_scaled_rounds_per_parameter": ratios,
        "co_scaled_ratio_strictly_decreases": decreasing_ratio,
    }
    provenance["pass"] = all(
        [all(checks), foundation_equal, aligned, nested, nonredundant, exact_information, decreasing_ratio]
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
    result = {
        "schema_version": 1,
        "phase": phase,
        "config_sha256": config_sha,
        "run_commit": run_commit,
        "provenance": provenance,
        "cells": {
            f"{endpoint}@k{rounds}": {
                "foundation_nll": interval(metric(endpoint, rounds, "foundation_nll"), phase),
                "ascent_nll": interval(metric(endpoint, rounds, "ascent_nll"), phase),
                "gain_nll": interval(metric(endpoint, rounds, "gain_nll"), phase),
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
    return result




def exact_sum_row_nll(
    first_observations: list[int], second_observations: list[int], target: int, labels: int, eta: float
) -> float:
    """Exact Bayesian posterior NLL of ``target = first + second`` for one row."""
    first = posterior(first_observations, labels, eta)
    second = posterior(second_observations, labels, eta)
    probability = 0.0
    for left in range(labels):
        for right in range(labels):
            if left + right == target:
                probability += float(first[left] * second[right])
    return float(-np.log(max(probability, np.finfo(np.float64).tiny)))


def emit_row_audit(config_path: Path, root: Path, phase: str, run_commit: str, output: Path) -> dict[str, Any]:
    """Write one CSV line per (endpoint, transition, panel, row): the 13,824 row checks.

    Each line records the two by-construction checks that the analysis aggregates
    (``nested_prefix``: the higher-K retained observations extend the lower-K ones;
    ``observations_changed``: the transition added at least one observation) and,
    as information rather than as gates, the exact Bayesian posterior NLL and the
    model's ASCENT NLL before and after the transition. Per-row exact-posterior
    improvement is *not* guaranteed under a noisy channel and is not claimed; the
    registered exact-posterior check is at the panel-mean level.
    """
    import csv

    config = json.loads(config_path.read_text())
    labels = int(config["construction"]["num_values"])
    eta = float(config["construction"]["crossover_probability"])
    endpoints = {row["name"]: row for row in config["endpoints"]}
    endpoint_order = [row["name"] for row in config["endpoints"]]
    rounds_order = list(config["factorial"]["state_rounds"])
    panels = config[f"{phase}_panels"]
    node_by_endpoint = config["execution"]["node_by_endpoint"]
    fieldnames = [
        "endpoint", "panel", "row_id", "episode_id", "target", "transition", "k_low", "k_high",
        "retained_first_low", "retained_first_high", "retained_second_low", "retained_second_high",
        "nested_prefix", "observations_changed", "row_check_pass",
        "exact_posterior_nll_low", "exact_posterior_nll_high", "exact_posterior_delta",
        "model_ascent_nll_low", "model_ascent_nll_high", "model_ascent_delta",
        "model_foundation_nll", "foundation_nll_identical_across_k",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"rows": 0, "row_check_pass": 0, "exact_posterior_improved": 0, "model_ascent_improved": 0}
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for endpoint in endpoint_order:
            for panel in panels:
                results = {}
                for rounds in rounds_order:
                    path = root / phase / node_by_endpoint[endpoint] / f"rounds_{rounds}_{panel}_{endpoint}.json"
                    results[rounds] = json.loads(path.read_text())
                    if results[rounds]["environment"]["git_commit"] != run_commit:
                        raise RuntimeError(f"{path}: unexpected commit")
                for low_k, high_k in zip(rounds_order[:-1], rounds_order[1:], strict=True):
                    for low, high in zip(results[low_k]["predictions"], results[high_k]["predictions"], strict=True):
                        if (low["row_id"], low["target"]) != (high["row_id"], high["target"]):
                            raise RuntimeError("row misalignment across state sizes")
                        lf, hf = low["retained_first_observations"], high["retained_first_observations"]
                        ls, hs = low["retained_second_observations"], high["retained_second_observations"]
                        nested = hf[: len(lf)] == lf and hs[: len(ls)] == ls
                        changed = lf != hf or ls != hs
                        exact_low = exact_sum_row_nll(lf, ls, int(low["target"]), labels, eta)
                        exact_high = exact_sum_row_nll(hf, hs, int(high["target"]), labels, eta)
                        row = {
                            "endpoint": endpoint, "panel": panel, "row_id": low["row_id"], "episode_id": low["episode_id"],
                            "target": low["target"], "transition": f"rounds_{low_k}_to_{high_k}", "k_low": low_k, "k_high": high_k,
                            "retained_first_low": " ".join(map(str, lf)), "retained_first_high": " ".join(map(str, hf)),
                            "retained_second_low": " ".join(map(str, ls)), "retained_second_high": " ".join(map(str, hs)),
                            "nested_prefix": nested, "observations_changed": changed, "row_check_pass": nested and changed,
                            "exact_posterior_nll_low": repr(exact_low), "exact_posterior_nll_high": repr(exact_high),
                            "exact_posterior_delta": repr(exact_low - exact_high),
                            "model_ascent_nll_low": repr(low["ascent_nll"]), "model_ascent_nll_high": repr(high["ascent_nll"]),
                            "model_ascent_delta": repr(low["ascent_nll"] - high["ascent_nll"]),
                            "model_foundation_nll": repr(low["foundation_nll"]),
                            "foundation_nll_identical_across_k": low["foundation_nll"] == high["foundation_nll"],
                        }
                        writer.writerow(row)
                        counts["rows"] += 1
                        counts["row_check_pass"] += int(nested and changed)
                        counts["exact_posterior_improved"] += int(exact_low > exact_high)
                        counts["model_ascent_improved"] += int(low["ascent_nll"] > high["ascent_nll"])
    summary = {
        "output": str(output), "sha256": sha256_file(output), **counts,
        "expected_rows": len(endpoint_order) * (len(rounds_order) - 1) * len(panels) * int(config["evaluation_samples"]),
        "note": "row_check_pass counts the registered by-construction checks (nesting and non-redundancy); the two 'improved' counts are informational per-row tallies, not registered gates",
    }
    summary["all_row_checks_pass"] = summary["row_check_pass"] == summary["expected_rows"] == summary["rows"]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--run-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--emit-row-audit",
        type=Path,
        help="also write the flat per-row transition audit CSV (13,824 lines for the confirmation phase) to this new path",
    )
    args = parser.parse_args()
    if args.emit_row_audit is not None:
        summary = emit_row_audit(args.config, args.root, args.phase, args.run_commit, args.emit_row_audit)
        print(json.dumps({"row_audit": summary}, indent=2))
    result = analyze(args.config, args.root, args.phase, args.run_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gate_pass": result["gate_pass"], "estimands": result["estimands"]}, indent=2))


if __name__ == "__main__":
    main()
