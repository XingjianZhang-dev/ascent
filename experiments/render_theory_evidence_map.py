#!/usr/bin/env python3
"""Render the manuscript's theory-to-evidence correspondence table.

The table is generated exclusively from immutable analysis JSON.  It fails
closed if any promoted confirmation or provenance gate is absent.
"""

from __future__ import annotations

import argparse
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


QWEN_ENDPOINTS = (
    "qwen2p5-0p5b-instruct",
    "qwen2p5-1p5b-instruct",
    "qwen2p5-3b-instruct",
)

SYSTEM_ENDPOINTS = (
    "smollm2-135m-instruct",
    "smollm2-360m-instruct",
    "smollm2-1p7b-instruct",
)


def decimal(value: float, places: int = 3, leading_zero: bool = False) -> str:
    quantum = Decimal(1).scaleb(-places)
    rendered = format(
        Decimal(f"{value:.12f}").quantize(quantum, rounding=ROUND_HALF_UP),
        "f",
    )
    if not leading_zero and rendered.startswith("0."):
        return rendered[1:]
    if not leading_zero and rendered.startswith("-0."):
        return "-" + rendered[2:]
    return rendered


def scientific(value: float, places: int = 3) -> str:
    coefficient, exponent = f"{value:.{places}e}".split("e")
    return rf"{coefficient}\!\times\!10^{{{int(exponent)}}}"


def interval(item: dict, places: int = 3) -> str:
    return (
        f"{decimal(item['mean'], places)} "
        f"[{decimal(item['ci95_low'], places)},{decimal(item['ci95_high'], places)}]"
    )


def render(official: dict, mechanism: dict, around7b: dict, systems: dict) -> str:
    if official.get("primary_gate_pass") is not True:
        raise RuntimeError("official confirmation gate is not positive")
    if not all(
        mechanism.get(key) is True
        for key in (
            "gate_pass",
            "registered_performance_pass",
            "supplementary_bonferroni_familywise_performance_pass",
        )
    ):
        raise RuntimeError("mechanism confirmation gate is not positive")
    provenance = mechanism.get("provenance", {})
    required_provenance = (
        "all_cell_checks_pass",
        "all_cells_row_target_aligned",
        "co_scaled_ratio_strictly_decreases",
        "every_transition_changes_every_row",
        "exact_nested_observation_prefixes",
        "exact_posterior_strictly_improves_every_transition_every_panel",
        "foundation_probabilities_identical_across_state_sizes",
        "pass",
    )
    if not all(provenance.get(key) is True for key in required_provenance):
        raise RuntimeError("mechanism provenance or oracle audit failed")
    retention = around7b.get("retention_audit", {})
    if not all(
        retention.get(key) is True
        for key in (
            "all_models_retained",
            "all_panels_and_tasks_retained",
            "invalid_outputs_and_regressions_retained",
        )
    ):
        raise RuntimeError("large-model retention audit failed")
    system_gates = systems.get("gates", {})
    if not all(
        system_gates.get(key) is True
        for key in (
            "complete_six_repetitions_per_endpoint",
            "accuracy_and_predictions_invariant",
            "write_state_fixed_across_endpoints",
            "query_conditioned_read_state_strictly_grows",
            "external_state_to_model_byte_ratio_strictly_decreases",
        )
    ):
        raise RuntimeError("systems evidence gate failed")

    simultaneous = mechanism["supplementary_bonferroni_familywise_estimands"]
    diagonal = [
        simultaneous["gain_qwen2p5-1p5b-instruct_k2"]["mean"],
        simultaneous["gain_qwen2p5-3b-instruct_k3"]["mean"],
        simultaneous["gain_qwen2p5-7b-instruct_k5"]["mean"],
    ]
    official_gains = [
        official["endpoint_results"][endpoint]["primary"]["gain"]["mean"] * 100
        for endpoint in QWEN_ENDPOINTS
    ]
    state_bytes = [
        systems["endpoints"][endpoint]["state"]["mean_total_external_state_bytes"]
        for endpoint in SYSTEM_ENDPOINTS
    ]
    state_ratios = [
        systems["endpoints"][endpoint]["state"]
        ["mean_total_external_state_to_model_artifact_byte_ratio"]
        for endpoint in SYSTEM_ENDPOINTS
    ]
    changed = sum(
        count
        for endpoint in provenance["nonredundant_rows_by_endpoint_and_transition"].values()
        for count in endpoint.values()
    )

    lines = [
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{.15\textwidth}>{\raggedright\arraybackslash}p{.25\textwidth}>{\raggedright\arraybackslash}p{.33\textwidth}>{\raggedright\arraybackslash}p{.12\textwidth}}",
        r"\toprule",
        r"Theoretical claim & Empirical quantity & Quantitative result & Conclusion \\",
        r"\midrule",
        (
            r"Refinement value (Prop.~\ref{prop:information-value}) & "
            r"Exact-posterior loss under each nested $K$ transition & "
            f"{changed:,} of {changed:,} endpoint--transition row checks: each larger state extends the smaller "
            r"one and adds an observation (true by construction); panel-mean exact-posterior NLL decreases on every "
            r"transition in every panel. & Design check\footnotemark[1] \\"
        ),
        (
            r"Co-scaled complementarity (Eq.~\ref{eq:scale-complementarity}) & "
            r"Diagonal gain and two adjacent increments & "
            f"{decimal(diagonal[0])}$\\to${decimal(diagonal[1])}$\\to$"
            f"{decimal(diagonal[2])} nats; increments "
            f"{interval(simultaneous['first_diagonal_gain_increment'])} and "
            f"{interval(simultaneous['second_diagonal_gain_increment'])}. & Established \\\\"
        ),
        (
            r"Finite-reader interaction (Prop.~\ref{prop:finite-reader}) & "
            r"Two crossed interaction contrasts & "
            f"{interval(simultaneous['first_model_by_state_interaction'])} and "
            f"{interval(simultaneous['second_model_by_state_interaction'])}; "
            r"simultaneous 95\% CIs (Bonferroni). & Identified \\"
        ),
        (
            r"Task-level consequence & Official 16K generation accuracy gain & "
            f"{decimal(official_gains[0], 2)}$\\to${decimal(official_gains[1], 2)}$\\to$"
            f"{decimal(official_gains[2], 2)} points; both directional "
            r"contrasts remain significant after Holm correction. & Established \\"
        ),
        (
            r"Relative footprint (Prop.~\ref{prop:footprint}) & "
            r"External-state bytes and state/model ratio & "
            f"{decimal(state_bytes[0], 1, True)}$\\to${decimal(state_bytes[1], 1, True)}"
            f"$\\to${decimal(state_bytes[2], 1, True)} bytes while the ratio falls "
            f"${scientific(state_ratios[0])}\\to{scientific(state_ratios[1])}"
            f"\\to{scientific(state_ratios[2])}$. & Established \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--around7b", type=Path, required=True)
    parser.add_argument("--systems", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    content = render(
        json.loads(args.official.read_text()),
        json.loads(args.mechanism.read_text()),
        json.loads(args.around7b.read_text()),
        json.loads(args.systems.read_text()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)


if __name__ == "__main__":
    main()
