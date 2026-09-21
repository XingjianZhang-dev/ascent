#!/usr/bin/env python3
"""Render mechanism and around-7B paper tables from immutable analyses."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


MECHANISM_ROWS = (
    ("qwen2p5-1p5b-instruct", 2, "1.5B"),
    ("qwen2p5-3b-instruct", 3, "3B"),
    ("qwen2p5-7b-instruct", 5, "7B"),
)

AROUND_7B_ROWS = (
    ("qwen2p5-7b-instruct", "Qwen2.5-7B"),
    ("mistral-7b-instruct-v0p3", "Mistral-7B-v0.3"),
    ("falcon3-7b-instruct", "Falcon3-7B"),
    ("granite-3p3-8b-instruct", "Granite-3.3-8B"),
    ("qwen3-8b-nonthinking", "Qwen3-8B"),
)

SYSTEM_ROWS = (
    ("smollm2-135m-instruct", "135M"),
    ("smollm2-360m-instruct", "360M"),
    ("smollm2-1p7b-instruct", "1.7B"),
)


def decimal(value: float, places: int = 3) -> str:
    quantum = Decimal(1).scaleb(-places)
    rendered = format(
        Decimal(f"{value:.12f}").quantize(quantum, rounding=ROUND_HALF_UP),
        "f",
    )
    if rendered.startswith("0."):
        return rendered[1:]
    if rendered.startswith("-0."):
        return "-" + rendered[2:]
    return rendered


def scientific(value: float, places: int = 3) -> str:
    coefficient, exponent = f"{value:.{places}e}".split("e")
    return rf"${coefficient}\times10^{{{int(exponent)}}}$"


def render_mechanism(analysis: dict) -> str:
    required_true = (
        "gate_pass",
        "registered_performance_pass",
        "supplementary_bonferroni_familywise_performance_pass",
    )
    if not all(analysis.get(key) is True for key in required_true):
        raise RuntimeError("refusing to render a failed mechanism confirmation")
    if analysis.get("provenance", {}).get("pass") is not True:
        raise RuntimeError("refusing to render mechanism results with failed provenance")

    estimates = analysis["estimands"]
    simultaneous = analysis["supplementary_bonferroni_familywise_estimands"]
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Reader & State $K$ & Gain (nats) & Simultaneous 95\% CI \\",
        r"\midrule",
    ]
    for endpoint, state_size, label in MECHANISM_ROWS:
        key = f"gain_{endpoint}_k{state_size}"
        point = estimates[key]
        family = simultaneous[key]
        lines.append(
            f"{label} & {state_size} & {decimal(point['mean'])} & "
            f"[{decimal(family['ci95_low'])},{decimal(family['ci95_high'])}] \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            _contrast_row(r"Model$\times$state: 1.5B$\to$3B", simultaneous["first_model_by_state_interaction"]),
            _contrast_row(r"Model$\times$state: 3B$\to$7B", simultaneous["second_model_by_state_interaction"]),
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    return "\n".join(lines)


def _contrast_row(label: str, estimate: dict) -> str:
    return (
        f"{label} & -- & {decimal(estimate['mean'])} & "
        f"[{decimal(estimate['ci95_low'])},{decimal(estimate['ci95_high'])}] \\\\"
    )


def render_around7b(analysis: dict) -> str:
    retention = analysis.get("retention_audit", {})
    if not all(
        retention.get(key) is True
        for key in (
            "all_models_retained",
            "all_panels_and_tasks_retained",
            "invalid_outputs_and_regressions_retained",
        )
    ):
        raise RuntimeError("refusing to render an incomplete around-7B analysis")

    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Reader & Foundation & \ascent{} & Gain [95\% CI] \\",
        r"\midrule",
    ]
    endpoints = analysis["endpoint_results"]
    for endpoint, label in AROUND_7B_ROWS:
        primary = endpoints[endpoint]["primary"]
        gain = primary["gain"]
        lines.append(
            f"{label} & {decimal(primary['foundation']['mean'], 4)} & "
            f"{decimal(primary['ascent']['mean'], 4)} & "
            f"{decimal(gain['mean'], 4)} "
            f"[{decimal(gain['ci95_low'], 4)},{decimal(gain['ci95_high'], 4)}] \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def render_systems(systems: dict, flops: dict) -> str:
    system_gates = systems.get("gates", {})
    required_system_gates = (
        "complete_six_repetitions_per_endpoint",
        "accuracy_and_predictions_invariant",
        "both_decode_orders_agree_in_direction",
        "write_state_fixed_across_endpoints",
        "query_conditioned_read_state_strictly_grows",
        "external_state_to_model_byte_ratio_strictly_decreases",
    )
    if systems.get("validation_errors") or not all(
        system_gates.get(key) is True for key in required_system_gates
    ):
        raise RuntimeError("refusing to render invalid systems measurements")

    flop_gates = flops.get("gates", {})
    required_flop_gates = (
        "single_config_hash",
        "single_git_commit",
        "all_worktrees_clean",
        "all_data_hashes_equal_reference",
        "all_model_files_equal_reference",
        "all_predictions_equal_reference",
        "all_supported_operator_flops_positive",
        "all_flop_ratios_below_0p10",
    )
    if not all(flop_gates.get(key) is True for key in required_flop_gates):
        raise RuntimeError("refusing to render invalid FLOP measurements")

    flop_by_endpoint = {row["endpoint"]: row for row in flops["endpoints"]}
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Reader & State bytes & State/model & Decode ratio & FLOP reduction \\",
        r"\midrule",
    ]
    for endpoint, label in SYSTEM_ROWS:
        system = systems["endpoints"][endpoint]
        profile = flop_by_endpoint[endpoint]
        lines.append(
            f"{label} & {decimal(system['state']['mean_total_external_state_bytes'], 1)} & "
            f"{scientific(system['state']['mean_total_external_state_to_model_artifact_byte_ratio'])} & "
            f"{decimal(system['all_repetitions']['ascent_to_foundation_decode_time_ratio']['median'], 3)} & "
            f"{decimal(profile['supported_operator_flop_reduction_factor'], 1)}$\\times$ \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--around7b", type=Path, required=True)
    parser.add_argument("--mechanism-output", type=Path, required=True)
    parser.add_argument("--around7b-output", type=Path, required=True)
    parser.add_argument("--systems", type=Path, required=True)
    parser.add_argument("--flops", type=Path, required=True)
    parser.add_argument("--systems-output", type=Path, required=True)
    args = parser.parse_args()

    mechanism = json.loads(args.mechanism.read_text())
    around7b = json.loads(args.around7b.read_text())
    systems = json.loads(args.systems.read_text())
    flops = json.loads(args.flops.read_text())
    outputs = (
        (args.mechanism_output, render_mechanism(mechanism)),
        (args.around7b_output, render_around7b(around7b)),
        (args.systems_output, render_systems(systems, flops)),
    )
    for output, content in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content)


if __name__ == "__main__":
    main()
