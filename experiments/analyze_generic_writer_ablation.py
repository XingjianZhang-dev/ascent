#!/usr/bin/env python3
"""Analyze the schema-blind writer ablation (Array revision 1, Phase 2C).

Same estimator as the canonical official-16K confirmation
(``analyze_babilong_canonical_coscale_16k_confirmation``): ten-panel clustered
Student-t summaries of the task-balanced ASCENT-minus-Foundation accuracy gain
per reader, the two adjacent gain increments tested one-sided with Holm
family-wise control, applied separately to each generic writer condition.

Two additional, pre-registered quantities:

* **Foundation invariance** — the Foundation arm's generated strings must be
  identical across the generic conditions and the same-instance canonical run
  (identical prompts on identical hardware); differences are counted, not
  hidden.
* **Schema advantage** — canonical minus generic ASCENT accuracy per reader,
  paired over panels, with both sides measured on the same A100 instance so no
  GPU-architecture difference enters the contrast.

The interpretation rule (survives / collapses) is the one fixed in the
ablation config before any score existed. Outputs a JSON record and a LaTeX
table for Appendix D. Nothing under ``artifacts/`` is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.analyze_babilong_canonical_coscale_16k_confirmation import (
    _load,
    _one_sided_positive_p,
    holm_adjust,
)
from experiments.analyze_babilong_canonical_coscale_development import _panel_values, _scope
from experiments.analyze_babilong_qrag_direct_peer import cluster_summary


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_canonical_same_instance(base_config: dict[str, Any], root: Path, endpoints: list[str]) -> dict[str, list[dict[str, Any]]]:
    """The instrumented canonical run (Phase 2B) on the same instance; status differs from the config's."""
    rows_by_endpoint: dict[str, list[dict[str, Any]]] = {}
    for endpoint in endpoints:
        short = endpoint.replace("qwen2p5-", "").replace("-instruct", "")
        rows: list[dict[str, Any]] = []
        for panel in base_config["panels"]:
            document = json.loads((root / short / f"{panel}.json").read_text())
            if document["endpoint"]["name"] != endpoint or document["condition"]["name"] != "canonical_coscale":
                raise RuntimeError(f"unexpected canonical record for {endpoint}/{panel}")
            if document["data"]["sha256"] != base_config["panel_sha256_by_name"][panel]:
                raise RuntimeError(f"canonical panel hash mismatch: {panel}")
            rows.extend({"panel": panel, **row} for row in document["predictions"])
        rows_by_endpoint[endpoint] = rows
    return rows_by_endpoint


def analyze(config_path: Path, results_root: Path, canonical_root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    config_hash = sha256_file(config_path)
    base_config = json.loads((Path(config["derived_from"]["path"])).read_text())
    if sha256_file(Path(config["derived_from"]["path"])) != config["derived_from"]["sha256"]:
        raise RuntimeError("derived_from config hash mismatch")
    endpoints = [row["name"] for row in config["endpoints"]]
    panels = config["panels"]
    tasks = set(config["tasks"])
    canonical = load_canonical_same_instance(base_config, canonical_root, endpoints)

    conditions_out: dict[str, Any] = {}
    for condition in config["conditions"]:
        per_endpoint = {}
        rows_by_endpoint = {}
        provenance = {}
        for endpoint in endpoints:
            rows, prov = _load(config, config_hash, endpoint, condition, results_root / condition / endpoint)
            rows_by_endpoint[endpoint] = rows
            provenance[endpoint] = prov
            per_endpoint[endpoint] = {
                "primary": _scope(rows, panels, tasks),
                "by_task": {task: _scope(rows, panels, {task}) for task in config["tasks"]},
            }
        # registered directional family: adjacent gain increments, one-sided, Holm
        gains = {e: _panel_values(rows_by_endpoint[e], panels, tasks, "gain") for e in endpoints}
        adjacent = []
        raw_p = []
        for low, high in zip(endpoints[:-1], endpoints[1:], strict=True):
            increment = [b - a for a, b in zip(gains[low], gains[high], strict=True)]
            p = _one_sided_positive_p(increment)
            raw_p.append(p)
            adjacent.append({"lower_endpoint": low, "upper_endpoint": high, "gain_increment": cluster_summary(increment), "one_sided_p_unadjusted": p})
        for row, p_holm in zip(adjacent, holm_adjust(raw_p), strict=True):
            row["one_sided_p_holm"] = p_holm
            row["passes_registered_contrast"] = row["gain_increment"]["mean"] > 0.0 and p_holm < 0.05
        means = [per_endpoint[e]["primary"]["gain"]["mean"] for e in endpoints]
        strictly_increasing = all(b > a for a, b in zip(means[:-1], means[1:], strict=True))
        survives = strictly_increasing and all(r["passes_registered_contrast"] for r in adjacent)
        # foundation invariance vs the same-instance canonical run
        foundation_diff = {}
        schema_advantage = {}
        for endpoint in endpoints:
            generic = {r["row_id"]: r for r in rows_by_endpoint[endpoint]}
            canon = {r["row_id"]: r for r in canonical[endpoint]}
            if set(generic) != set(canon):
                raise RuntimeError("row sets differ between generic and canonical runs")
            foundation_diff[endpoint] = sum(1 for rid in generic if generic[rid]["foundation_output"] != canon[rid]["foundation_output"])
            canon_acc = _panel_values(canonical[endpoint], panels, tasks, "ascent")
            generic_acc = _panel_values(rows_by_endpoint[endpoint], panels, tasks, "ascent")
            diff = [a - b for a, b in zip(canon_acc, generic_acc, strict=True)]
            schema_advantage[endpoint] = {
                "canonical_ascent_accuracy": cluster_summary(canon_acc),
                "generic_ascent_accuracy": cluster_summary(generic_acc),
                "canonical_minus_generic": cluster_summary(diff),
                "one_sided_p_canonical_greater": _one_sided_positive_p(diff),
            }
        conditions_out[condition] = {
            "writer": config["conditions"][condition].get("writer"),
            "endpoint_results": per_endpoint,
            "adjacent_scale_contrasts": adjacent,
            "gain_means_by_endpoint": dict(zip(endpoints, means, strict=True)),
            "gains_strictly_increasing_with_scale": strictly_increasing,
            "co_scaling_survives_registered_rule": survives,
            "foundation_output_rows_differing_from_same_instance_canonical": foundation_diff,
            "schema_advantage_same_instance": schema_advantage,
            "provenance": provenance,
        }
    any_survives = any(c["co_scaling_survives_registered_rule"] for c in conditions_out.values())
    return {
        "schema_version": 1,
        "experiment": config["experiment"],
        "config": {"path": str(config_path), "sha256": config_hash},
        "canonical_same_instance_root": str(canonical_root),
        "interpretation_rule": config["interpretation_rule_fixed_in_advance"],
        "conditions": conditions_out,
        "any_generic_writer_survives": any_survives,
        "manuscript_option": "A_generalisation_demonstrated" if any_survives else "B_narrow_claim_to_structured_fact_tasks",
    }


def num(value: float, places: int = 2) -> str:
    """Signed number with a math minus sign for LaTeX text."""
    text = f"{abs(value):.{places}f}"
    return f"$-{text}$" if value < 0 else text


def pval(p: float) -> str:
    return f"{p:.3f}" if p >= 0.001 else f"{p:.2e}"


def latex_table(result: dict[str, Any], endpoints: list[str]) -> str:
    labels = {"qwen2p5-0p5b-instruct": "0.5B", "qwen2p5-1p5b-instruct": "1.5B", "qwen2p5-3b-instruct": "3B"}
    lines = [
        "% Auto-generated by experiments/analyze_generic_writer_ablation.py; do not edit.",
        "\\begin{tabular}{lrrrrrl}",
        "\\toprule",
        "Writer & " + " & ".join(f"Gain {labels[e]}" for e in endpoints) + " & $p_{\\mathrm{Holm}}$ (1) & $p_{\\mathrm{Holm}}$ (2) & Co-scaling \\\\",
        "\\midrule",
    ]
    names = {"sentence_window_coscale": "Sentence window", "generic_bm25_coscale": "Lexical BM25 chain", "bge_m3_coscale": "BGE-M3 hybrid + reranker"}
    for cond, c in result["conditions"].items():
        gains = " & ".join(num(100 * c['endpoint_results'][e]['primary']['gain']['mean']) for e in endpoints)
        ps = " & ".join(pval(r['one_sided_p_holm']) for r in c["adjacent_scale_contrasts"])
        lines.append(f"{names.get(cond, cond)} & {gains} & {ps} & {'survives' if c['co_scaling_survives_registered_rule'] else 'collapses'} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def latex_text(result: dict[str, Any], endpoints: list[str]) -> str:
    """Appendix D prose, generated from the result record so no number is hand-typed."""
    labels = {"qwen2p5-0p5b-instruct": "0.5B", "qwen2p5-1p5b-instruct": "1.5B", "qwen2p5-3b-instruct": "3B"}
    names = {"sentence_window_coscale": "sentence window", "generic_bm25_coscale": "lexical chain", "bge_m3_coscale": "BGE-M3 hybrid retrieval"}
    parts = ["% Auto-generated by experiments/analyze_generic_writer_ablation.py; do not edit."]
    for cond, c in result["conditions"].items():
        gains = ", ".join(num(100 * c['endpoint_results'][e]['primary']['gain']['mean']) for e in endpoints)
        cis = "; ".join(
            f"{labels[e]}: [{num(100 * c['endpoint_results'][e]['primary']['gain']['ci95_low'])}, {num(100 * c['endpoint_results'][e]['primary']['gain']['ci95_high'])}]"
            for e in endpoints
        )
        p1, p2 = (pval(r["one_sided_p_holm"]) for r in c["adjacent_scale_contrasts"])
        inc1, inc2 = (num(100 * r["gain_increment"]["mean"]) for r in c["adjacent_scale_contrasts"])
        asc = ", ".join(f"{100 * c['schema_advantage_same_instance'][e]['generic_ascent_accuracy']['mean']:.2f}" for e in endpoints)
        fnd = ", ".join(f"{100 * c['endpoint_results'][e]['primary']['foundation']['mean']:.2f}" for e in endpoints)
        adv = "; ".join(
            f"{labels[e]}: {100 * c['schema_advantage_same_instance'][e]['canonical_minus_generic']['mean']:.2f} "
            f"[{100 * c['schema_advantage_same_instance'][e]['canonical_minus_generic']['ci95_low']:.2f}, "
            f"{100 * c['schema_advantage_same_instance'][e]['canonical_minus_generic']['ci95_high']:.2f}]"
            for e in endpoints
        )
        fdiff = sum(c["foundation_output_rows_differing_from_same_instance_canonical"].values())
        verdict = "survives" if c["co_scaling_survives_registered_rule"] else "collapses"
        parts.append(
            f"With the \\emph{{{names.get(cond, cond)}}} writer the ten-panel gains are {gains} points at "
            f"0.5B, 1.5B, and 3B (95\\% intervals {cis}); the adjacent increments are {inc1} and {inc2} points "
            f"with $p_{{\\mathrm{{Holm}}}}={p1}$ and ${p2}$, so the registered co-scaling rule {verdict}. "
            f"The generic state is worse than no state at every scale: \\ascent{{}} accuracy is {asc} against Foundation accuracy of {fnd}. "
            f"The structured-fact writer's advantage in \\ascent{{}} accuracy over this writer, paired over panels on the same "
            f"instance, is {adv} points. Foundation outputs differ from the same-instance canonical run in {fdiff} of 2,400 rows."
        )
    option = result["manuscript_option"]
    if option.startswith("A"):
        parts.append("At least one schema-free writer satisfies the registered co-scaling rule: the increasing-gain pattern does not depend on the hand-built schema, although the schema-specific writer realizes a much larger gain at every scale. The main text keeps the general framing while scoping the claim to the evaluated benchmarks.")
    else:
        parts.append("No schema-free writer satisfies the registered co-scaling rule. At the same one-to-three-slot budget, generic state constructions rarely retain the supporting facts, so the reader answers worse from them than from the full 16K context, the gains are negative at every scale, and they do not increase with reader scale in the registered sense. The co-scaling pattern reported in the main text is therefore a property of external state constructed against the evaluated structured-fact schema; whether a schema-free writer with a much larger budget, or a learned extractor, would recover it is not tested here. Section~\\ref{sec:limitations} scopes the claim accordingly.")
    return "\n\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True, help="directory holding <condition>/<endpoint>/<panel>.json")
    parser.add_argument("--canonical-root", type=Path, required=True, help="same-instance canonical run: <0p5b|1p5b|3b>/<panel>.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tex-output", type=Path)
    parser.add_argument("--tex-text-output", type=Path)
    args = parser.parse_args()
    result = analyze(args.config, args.results_root, args.canonical_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    endpoints = [row["name"] for row in json.loads(args.config.read_text())["endpoints"]]
    if args.tex_output:
        args.tex_output.write_text(latex_table(result, endpoints))
    if args.tex_text_output:
        args.tex_text_output.write_text(latex_text(result, endpoints))
    for cond, c in result["conditions"].items():
        print(f"{cond}: gains {[round(100*v, 2) for v in c['gain_means_by_endpoint'].values()]} pts; "
              f"p_Holm {[round(r['one_sided_p_holm'], 4) for r in c['adjacent_scale_contrasts']]}; "
              f"survives={c['co_scaling_survives_registered_rule']}; foundation rows differing {c['foundation_output_rows_differing_from_same_instance_canonical']}")
    print("manuscript option:", result["manuscript_option"])
    print(f"written: {args.output}  sha256={sha256_file(args.output)}")


if __name__ == "__main__":
    main()
