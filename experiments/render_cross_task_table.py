#!/usr/bin/env python3
"""Render the cross-task confirmation matrix from immutable analyses."""

from __future__ import annotations

import argparse
import json
import tarfile
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


def decimal(value: float, places: int = 4) -> str:
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


def interval(item: dict) -> str:
    return (
        f"{decimal(item['mean'])} "
        f"[{decimal(item['ci95_low'])},{decimal(item['ci95_high'])}]"
    )


def load_single_analysis(archive: Path) -> dict:
    with tarfile.open(archive, "r:gz") as handle:
        members = [
            member
            for member in handle.getmembers()
            if member.isfile() and member.name.endswith("/analysis.json")
        ]
        if len(members) != 1:
            raise RuntimeError(
                f"expected one analysis.json in {archive}, found {len(members)}"
            )
        stream = handle.extractfile(members[0])
        if stream is None:
            raise RuntimeError(f"could not read {members[0].name}")
        return json.load(stream)


def validate_multiquery(analysis: dict) -> None:
    required = (
        "all_adjacent_cluster_lcbs_positive",
        "confirmation_audit_passed",
        "every_seed_adjacent_difference_positive",
        "every_seed_endpoint_gain_positive",
        "slope_cluster_lcb_positive",
    )
    if not all(analysis.get("gates", {}).get(key) is True for key in required):
        raise RuntimeError("multi-query confirmation gate failed")
    audit_gates = analysis.get("confirmation_audit", {}).get("gates", {})
    if not all(value is True for value in audit_gates.values()):
        raise RuntimeError("multi-query provenance audit failed")


def validate_cwe(analysis: dict, require_full_confirmation: bool) -> None:
    gates = analysis.get("gates", {})
    common = (
        "single_config_hash",
        "single_git_commit",
        "exact_registered_data_hashes",
        "all_worktrees_clean",
        "all_parsers_exact",
        "all_native_lengths_safe",
        "every_seed_has_positive_endpoint_gains",
        "every_seed_has_positive_adjacent_differences",
        "all_endpoint_gain_t_lcbs_positive",
        "slope_t_lcb_positive",
        "no_scale_regressions",
        "relative_state_cost_strictly_decreases",
    )
    if not all(gates.get(key) is True for key in common):
        raise RuntimeError("CWE provenance or directional gate failed")
    if require_full_confirmation:
        if gates.get("both_adjacent_t_lcbs_positive") is not True:
            raise RuntimeError("expected full CWE confirmation")
    elif gates.get("both_adjacent_t_lcbs_positive") is not False:
        raise RuntimeError("expected the retained CWE upper-adjacent boundary")


def curve(items: list[dict]) -> str:
    return "$\\to$".join(decimal(item["mean"]) for item in items)


def multiquery_row(
    task: str,
    family: str,
    context: str,
    analysis: dict,
    endpoint_order: tuple[str, str, str],
    contrast_order: tuple[str, str],
) -> str:
    validate_multiquery(analysis)
    endpoints = [analysis["endpoint_gain_clusters"][key] for key in endpoint_order]
    contrasts = [
        analysis["adjacent_gain_difference_clusters"][key]
        for key in contrast_order
    ]
    return (
        f"{task} & {family} / {context} & {curve(endpoints)} & "
        f"{interval(contrasts[0])}; {interval(contrasts[1])} & Two positive increments \\\\"
    )


def cwe_row(family: str, context: str, analysis: dict, confirmed: bool) -> str:
    validate_cwe(analysis, confirmed)
    endpoints = list(analysis["endpoint_gain_t_intervals"].values())
    contrasts = analysis["adjacent_gain_difference_t_intervals"]
    status = "Two positive increments" if confirmed else "Positive scaling slope"
    return (
        f"Common-word aggregation & {family} / {context} & {curve(endpoints)} & "
        f"{interval(contrasts[0])}; {interval(contrasts[1])} & {status} \\\\"
    )


def render(qwen_multiquery: dict, smol_multiquery: dict, smol_cwe: dict, qwen_cwe: dict) -> str:
    lines = [
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{.18\textwidth}>{\raggedright\arraybackslash}p{.15\textwidth}>{\raggedright\arraybackslash}p{.20\textwidth}>{\raggedright\arraybackslash}p{.28\textwidth}>{\raggedright\arraybackslash}p{.11\textwidth}}",
        r"\toprule",
        r"Official RULER task & Reader family / context & Gain curve & Adjacent increments [95\% CI] & Scaling pattern \\",
        r"\midrule",
        multiquery_row(
            "Multi-query retrieval",
            "Qwen2.5",
            "16K",
            qwen_multiquery,
            ("qwen2p5-0p5b", "qwen2p5-1p5b", "qwen2p5-3b"),
            ("0p5b_to_1p5b", "1p5b_to_3b"),
        ),
        multiquery_row(
            "Multi-query retrieval",
            "SmolLM2",
            "4K",
            smol_multiquery,
            ("smollm2-135m", "smollm2-360m", "smollm2-1p7b"),
            (
                "smollm2-135m_to_smollm2-360m",
                "smollm2-360m_to_smollm2-1p7b",
            ),
        ),
        cwe_row("SmolLM2", "8K", smol_cwe, True),
        cwe_row("Qwen2.5", "16K", qwen_cwe, False),
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-multiquery-archive", type=Path, required=True)
    parser.add_argument("--smol-multiquery-archive", type=Path, required=True)
    parser.add_argument("--smol-cwe", type=Path, required=True)
    parser.add_argument("--qwen-cwe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    content = render(
        load_single_analysis(args.qwen_multiquery_archive),
        load_single_analysis(args.smol_multiquery_archive),
        json.loads(args.smol_cwe.read_text()),
        json.loads(args.qwen_cwe.read_text()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)


if __name__ == "__main__":
    main()
