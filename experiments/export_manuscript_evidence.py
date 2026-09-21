#!/usr/bin/env python3
"""Validate immutable analyses and export plot-ready manuscript evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


QWEN = (
    ("qwen2p5-0p5b-instruct", 0.5, "0.5B"),
    ("qwen2p5-1p5b-instruct", 1.5, "1.5B"),
    ("qwen2p5-3b-instruct", 3.0, "3B"),
)
FACTORIAL_MODELS = (
    ("qwen2p5-1p5b-instruct", 1.5, "1.5B"),
    ("qwen2p5-3b-instruct", 3.0, "3B"),
    ("qwen2p5-7b-instruct", 7.0, "7B"),
)
AROUND7B = (
    ("qwen2p5-7b-instruct", "Qwen2.5-7B"),
    ("mistral-7b-instruct-v0p3", "Mistral-7B-v0.3"),
    ("falcon3-7b-instruct", "Falcon3-7B"),
    ("granite-3p3-8b-instruct", "Granite-3.3-8B"),
    ("qwen3-8b-nonthinking", "Qwen3-8B"),
)
SYSTEMS = (
    ("smollm2-135m-instruct", "135M"),
    ("smollm2-360m-instruct", "360M"),
    ("smollm2-1p7b-instruct", "1.7B"),
)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def exact_mean(values: list[float], expected: float, label: str) -> None:
    observed = math.fsum(values) / len(values)
    if not math.isclose(observed, expected, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError(f"{label}: summary mean {expected} != recomputed {observed}")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--qrag", type=Path, required=True)
    parser.add_argument("--factorial", type=Path, required=True)
    parser.add_argument("--around7b", type=Path, required=True)
    parser.add_argument("--systems", type=Path, required=True)
    parser.add_argument("--flops", type=Path, required=True)
    parser.add_argument("--cross-node", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    official, semantic, qrag = map(load, (args.official, args.semantic, args.qrag))
    factorial, around7b = map(load, (args.factorial, args.around7b))
    systems, flops, cross_node = map(load, (args.systems, args.flops, args.cross_node))

    if not official.get("primary_gate_pass") or not semantic.get("primary_gate_pass"):
        raise RuntimeError("prospective Qwen confirmation gate failed")
    if not (
        factorial.get("gate_pass")
        and factorial.get("registered_performance_pass")
        and factorial.get("supplementary_bonferroni_familywise_performance_pass")
        and factorial.get("provenance", {}).get("pass")
    ):
        raise RuntimeError("full-factorial mechanism gate failed")
    retention = around7b.get("retention_audit", {})
    if not all(
        retention.get(key)
        for key in (
            "all_models_retained",
            "all_panels_and_tasks_retained",
            "invalid_outputs_and_regressions_retained",
        )
    ):
        raise RuntimeError("around-7B retention gate failed")
    if not cross_node.get("gates", {}).get("cross_node_replication_pass"):
        raise RuntimeError("cross-node reproduction gate failed")

    primary_rows: list[dict[str, object]] = []
    for study, analysis, contrast in (
        ("Official 16K", official, "ASCENT - Foundation"),
        ("Semantic holdout", semantic, "ASCENT - Foundation"),
    ):
        for endpoint, scale, scale_label in QWEN:
            result = analysis["endpoint_results"][endpoint]["primary"]["gain"]
            values = result["panel_values"]
            if len(values) != 10:
                raise RuntimeError(f"{study} {endpoint}: expected ten panel clusters")
            exact_mean(values, result["mean"], f"{study} {endpoint}")
            for panel, value in enumerate(values, 1):
                primary_rows.append(
                    {"study": study, "contrast": contrast, "scale_b": scale,
                     "scale_label": scale_label, "panel": panel, "value": value}
                )
    for endpoint, scale, scale_label in QWEN:
        result = qrag["endpoint_results"][endpoint]["overall"]["ascent_minus_qrag"]
        values = result["panel_values"]
        if len(values) != 10:
            raise RuntimeError(f"Q-RAG {endpoint}: expected ten panel clusters")
        exact_mean(values, result["mean"], f"Q-RAG {endpoint}")
        for panel, value in enumerate(values, 1):
            primary_rows.append(
                {"study": "Direct peer", "contrast": "ASCENT - Q-RAG",
                 "scale_b": scale, "scale_label": scale_label,
                 "panel": panel, "value": value}
            )

    factorial_rows: list[dict[str, object]] = []
    for endpoint, scale, scale_label in FACTORIAL_MODELS:
        for slots in (2, 3, 5):
            result = factorial["cells"][f"{endpoint}@k{slots}"]["gain_nll"]
            values = result["values"]
            if len(values) != 9:
                raise RuntimeError(f"{endpoint}@k{slots}: expected nine panels")
            exact_mean(values, result["mean"], f"{endpoint}@k{slots}")
            for panel, value in enumerate(values, 1):
                factorial_rows.append(
                    {"model": scale_label, "model_b": scale, "slots": slots,
                     "panel": panel, "gain_nll": value}
                )

    interaction_rows: list[dict[str, object]] = []
    simultaneous = factorial["supplementary_bonferroni_familywise_estimands"]
    for key, label in (
        ("first_diagonal_gain_increment", "Diagonal: 1.5B to 3B"),
        ("second_diagonal_gain_increment", "Diagonal: 3B to 7B"),
        ("first_model_by_state_interaction", "Interaction: 1.5B to 3B"),
        ("second_model_by_state_interaction", "Interaction: 3B to 7B"),
        ("7b_k5_minus_k3_gain", "7B refinement: K3 to K5"),
    ):
        value = simultaneous[key]
        interaction_rows.append(
            {"estimand": label, "mean": value["mean"],
             "ci_low": value["ci95_low"], "ci_high": value["ci95_high"]}
        )

    breadth_rows: list[dict[str, object]] = []
    for endpoint, label in AROUND7B:
        primary = around7b["endpoint_results"][endpoint]["primary"]
        gain = primary["gain"]
        values = gain["panel_values"]
        exact_mean(values, gain["mean"], endpoint)
        for panel, value in enumerate(values, 1):
            breadth_rows.append(
                {"model": label, "panel": panel, "gain": value,
                 "foundation": primary["foundation"]["panel_values"][panel - 1],
                 "ascent": primary["ascent"]["panel_values"][panel - 1]}
            )

    system_rows: list[dict[str, object]] = []
    flop_by_endpoint = {row["endpoint"]: row for row in flops["endpoints"]}
    for endpoint, label in SYSTEMS:
        system = systems["endpoints"][endpoint]
        profile = flop_by_endpoint[endpoint]
        system_rows.append(
            {"model": label,
             "state_bytes": system["state"]["mean_total_external_state_bytes"],
             "state_model_ratio": system["state"]["mean_total_external_state_to_model_artifact_byte_ratio"],
             "decode_ratio": system["all_repetitions"]["ascent_to_foundation_decode_time_ratio"]["median"],
             "flop_reduction": profile["supported_operator_flop_reduction_factor"]}
        )

    out = args.output_dir
    write_csv(out / "primary_panel.csv", primary_rows)
    write_csv(out / "factorial_panel.csv", factorial_rows)
    write_csv(out / "interaction_intervals.csv", interaction_rows)
    write_csv(out / "around7b_panel.csv", breadth_rows)
    write_csv(out / "systems.csv", system_rows)
    profile = {
        "primary": {"rows": len(primary_rows), "panels_per_cell": 10,
                    "missing": 0, "range": [min(r["value"] for r in primary_rows), max(r["value"] for r in primary_rows)]},
        "factorial": {"rows": len(factorial_rows), "panels_per_cell": 9,
                      "missing": 0, "range": [min(r["gain_nll"] for r in factorial_rows), max(r["gain_nll"] for r in factorial_rows)]},
        "around7b": {"rows": len(breadth_rows), "panels_per_cell": 10,
                     "missing": 0, "range": [min(r["gain"] for r in breadth_rows), max(r["gain"] for r in breadth_rows)]},
        "systems": {"rows": len(system_rows), "missing": 0},
        "gates": {"official": True, "semantic": True, "factorial": True,
                  "retention": True, "cross_node": True},
    }
    (out / "evidence_profile.json").write_text(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
