#!/usr/bin/env python3
"""Build the canonical artifact payload for the ASCENT TNNLS readiness report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ENDPOINT_LABELS = {
    "smollm2-135m": "135M",
    "smollm2-360m-node2": "360M",
    "smollm2-1p7b": "1.7B",
}
CONTROL_LABELS = {
    "fixed_one_slot": "Fixed one-slot state",
    "irrelevant_structured": "Irrelevant structured state",
    "matched_raw_event_fifo": "Byte-matched raw FIFO",
    "generic_bm25": "Generic BM25",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def build(
    confirmation: dict[str, Any],
    controls: dict[str, Any],
    certified_factorial: dict[str, Any],
    generative_factorial: dict[str, Any],
    generative_audit: dict[str, Any],
    generative_fresh: dict[str, Any],
    generative_focused: dict[str, Any],
    generated_at: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    accuracy_rows: list[dict[str, Any]] = []
    for endpoint in confirmation["endpoints"]:
        primary = controls["conditions"]["fixed_one_slot"][endpoint][
            "primary_accuracy"
        ]
        accuracy_rows.append(
            {
                "endpoint": ENDPOINT_LABELS[endpoint],
                "endpoint_id": endpoint,
                "method": "ASCENT certified co-scale",
                "accuracy": primary["mean"],
                "ci95_low": primary["ci95_low"],
                "ci95_high": primary["ci95_high"],
            }
        )
        for condition, label in CONTROL_LABELS.items():
            summary = controls["conditions"][condition][endpoint]["control_accuracy"]
            accuracy_rows.append(
                {
                    "endpoint": ENDPOINT_LABELS[endpoint],
                    "endpoint_id": endpoint,
                    "method": label,
                    "accuracy": summary["mean"],
                    "ci95_low": summary["ci95_low"],
                    "ci95_high": summary["ci95_high"],
                }
            )

    absolute = confirmation["absolute_gain_panel_t_intervals"]
    adjacent = confirmation["absolute_gain_adjacent_panel_t_intervals"]
    slope = confirmation["remaining_error_log_parameter_slope_panel_t_interval"]
    generative_interactions = list(
        generative_factorial[
            "registered_foundation_by_state_interactions"
        ].values()
    )
    generative_excess = list(
        generative_factorial["coscale_increment_minus_memory_only"].values()
    )
    focused = generative_focused["primary_estimands"]
    headline = [
        {
            "largest_scale_gain": absolute["smollm2-1p7b"]["mean"],
            "largest_scale_gain_lcb": absolute["smollm2-1p7b"]["ci95_low"],
            "remaining_error_slope_lcb": slope["ci95_low"],
            "first_adjacent_gain_lcb": adjacent[
                "smollm2-135m_to_smollm2-360m-node2"
            ]["ci95_low"],
            "second_adjacent_gain_lcb": adjacent[
                "smollm2-360m-node2_to_smollm2-1p7b"
            ]["ci95_low"],
            "generative_first_interaction": generative_interactions[0]["mean"],
            "generative_first_interaction_lcb": generative_interactions[0]["ci95_low"],
            "generative_second_interaction": generative_interactions[1]["mean"],
            "generative_second_interaction_lcb": generative_interactions[1]["ci95_low"],
            "high_scale_coscale_excess_lcb": generative_excess[1]["ci95_low"],
            "focused_first_coscale_excess_lcb": focused[
                "first_coscale_excess"
            ]["ci95_low"],
            "focused_second_interaction": focused[
                "second_registered_interaction"
            ]["mean"],
            "focused_second_interaction_lcb": focused[
                "second_registered_interaction"
            ]["ci95_low"],
            "focused_upper_diagonal_lcb": focused[
                "upper_diagonal_gain_increment"
            ]["ci95_low"],
            "focused_upper_coscale_lcb": focused[
                "upper_coscale_excess"
            ]["ci95_low"],
        }
    ]

    readiness_rows = [
        {
            "priority": 1,
            "gate": "Five-panel certified QA7/QA8 scale confirmation",
            "status": "Passed",
            "evidence": "Gain 47.5% → 69.4% → 90.6%; both adjacent 95% LCBs > 0",
            "next_action": "Preserve immutable artifacts",
        },
        {
            "priority": 2,
            "gate": "Fixed-state causal contrast",
            "status": "Passed",
            "evidence": "Co-scale minus one-slot rises 0% → 25% → 50%; both interaction LCBs = 25%",
            "next_action": "Complete the full 3×3 matrix",
        },
        {
            "priority": 3,
            "gate": "Irrelevant-state and BM25 controls",
            "status": "Passed",
            "evidence": "Primary-minus-control LCB is positive at every endpoint",
            "next_action": "Retain as preregistered controls",
        },
        {
            "priority": 4,
            "gate": "Byte-matched raw FIFO separation",
            "status": "Partial",
            "evidence": "Primary is noninferior on every panel; medium LCB > 0, small/large LCBs cross 0 near ceiling",
            "next_action": "Report the strong raw boundary; add a non-saturated task",
        },
        {
            "priority": 5,
            "gate": "Certified model-scale × state-scale factorial",
            "status": "Boundary established",
            "evidence": "State effects are +25 points per step, but fixed-state model effects and model×state interactions are exactly 0",
            "next_action": "Do not use the certified path alone as model-complementarity proof",
        },
        {
            "priority": 6,
            "gate": "Generative QA1–QA3 factorial and focused confirmation",
            "status": "Passed with retained near miss",
            "evidence": "Fresh diagonal gains rise 29.2%→53.0%→65.3%; the original five-panel gate missed two LCBs, then a frozen ten-panel confirmation passed all four focused LCBs (+3.38%, +2.31%, +12.21%, +12.97%)",
            "next_action": "Preserve both the five-panel near miss and ten-panel confirmation",
        },
        {
            "priority": 7,
            "gate": "Independent task and model-family replication",
            "status": "Passed for the scoped claim",
            "evidence": "Official RULER multiquery and CWE curves reproduce across Qwen2.5 and SmolLM2; RULER-VT and composition add independent positive interactions",
            "next_action": "Add same-task Qwen generative evidence if feasible, without making it a post-hoc blocker",
        },
        {
            "priority": 8,
            "gate": "Internal high-readiness promotion threshold",
            "status": "Reached",
            "evidence": "Prospective scale curves, interaction contrasts, controls, cross-family tasks, systems accounting, and provenance now jointly pass the scoped internal gate",
            "next_action": "Begin the double-anonymous TNNLS manuscript; do not imply an acceptance guarantee",
        },
    ]

    portfolio_rows = [
        {
            "family": "Official RULER multiquery",
            "models": "Qwen2.5 + SmolLM2",
            "result": "Qwen gain 4.4%→12.1%→31.9%; Smol gain 10.4%→21.7%→52.3%",
            "strength": "Both untouched three-seed curves pass; 194/0 and 255/0 wins/regressions",
        },
        {
            "family": "Official RULER CWE aggregation",
            "models": "SmolLM2 + Qwen2.5",
            "result": "Smol 26.3%→43.8%→95.6%; Qwen RWE 25.8%→69.2%→100%",
            "strength": "Strong cross-architecture aggregation replication with ceiling-aware gate",
        },
        {
            "family": "Official BABILong QA1–QA3",
            "models": "SmolLM2 135M/360M/1.7B",
            "result": "Latest three gain curves: 20.8%→51.7%→70.8%, 27.5%→50.0%→70.0%, 24.2%→48.3%→60.0%",
            "strength": "Repeated 4K/8K generative confirmations; BGE and exact-byte FIFO controls pass",
        },
        {
            "family": "RULER-VT causal interaction",
            "models": "Pythia 410M/2.8B",
            "result": "Foundation×state interaction +0.0203, bootstrap 95% CI [0.0110, 0.0295]",
            "strength": "Direct evidence that a larger Foundation extracts more value on one interface",
        },
        {
            "family": "Composition mechanism",
            "models": "Pythia 410M/2.8B",
            "result": "Cross-model interaction +0.1252 nats [0.1190, 0.1313]",
            "strength": "Independent constructive composition result; raw exact storage remains stronger",
        },
        {
            "family": "Systems and reproducibility",
            "models": "Two RTX PRO 6000 nodes",
            "result": "State 589–641 bytes; supported-op FLOPs 78–84× below Foundation; cross-node predictions exact",
            "strength": "State/FLOP/provenance gates pass; strict profiler-wall gate remains partial",
        },
    ]

    synthesis = {
        "generated_at": generated_at,
        "accuracy_by_condition": accuracy_rows,
        "headline_metrics": headline,
        "readiness_gates": readiness_rows,
        "positive_evidence_portfolio": portfolio_rows,
        "input_gates": confirmation["gates"],
        "control_findings": controls["findings"],
        "certified_factorial_gates": certified_factorial["gates"],
        "generative_factorial_gates": generative_factorial["gates"],
        "generative_code_audit_pass": generative_audit["audit_pass"],
        "generative_fresh_gates": generative_fresh["fresh_gates"],
        "generative_combined_gates": generative_fresh["combined_gates"],
        "generative_focused_pass": generative_focused[
            "focused_confirmation_primary_pass"
        ],
    }

    sources = [
        {
            "id": "headline_metrics_sql",
            "label": "Frozen confirmation headline metrics",
            "path": "reports/ascent_tnnls_readiness_2026-08-14/report_data.json",
        },
        {
            "id": "condition_accuracy_sql",
            "label": "Frozen QA7/QA8 condition accuracy synthesis",
            "path": "reports/ascent_tnnls_readiness_2026-08-14/report_data.json",
        },
        {
            "id": "readiness_gates_sql",
            "label": "ASCENT promotion-gate synthesis",
            "path": "reports/ascent_tnnls_readiness_2026-08-14/report_data.json",
        },
        {
            "id": "positive_portfolio_sql",
            "label": "ASCENT validated positive-evidence portfolio",
            "path": "reports/ascent_tnnls_readiness_2026-08-14/report_data.json",
        },
        {
            "id": "readiness_audit",
            "label": "ASCENT TNNLS readiness audit",
            "path": "docs/TNNLS_READINESS_AUDIT.md",
        },
    ]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "ASCENT — TNNLS 证据就绪度审计",
            "description": "截至 2026-08-14 的冻结实验结果、因果控制、剩余硬门槛与下一步。",
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "core_metrics",
                    "description": "五个未触碰确认面板上的核心统计量。",
                    "dataset": "headline_metrics",
                    "sourceId": "headline_metrics_sql",
                    "metrics": [
                        {"label": "1.7B absolute gain", "field": "largest_scale_gain", "format": "percent"},
                        {"label": "1.7B gain 95% LCB", "field": "largest_scale_gain_lcb", "format": "percent"},
                        {"label": "RWE slope 95% LCB", "field": "remaining_error_slope_lcb", "format": "number"},
                        {"label": "Adjacent gain LCB (low→mid)", "field": "first_adjacent_gain_lcb", "format": "percent"},
                        {"label": "Adjacent gain LCB (mid→high)", "field": "second_adjacent_gain_lcb", "format": "percent"},
                        {"label": "Generative interaction (low→mid)", "field": "generative_first_interaction", "format": "percent"},
                        {"label": "Generative interaction LCB (low→mid)", "field": "generative_first_interaction_lcb", "format": "percent"},
                        {"label": "Generative interaction (mid→high)", "field": "generative_second_interaction", "format": "percent"},
                        {"label": "Generative interaction LCB (mid→high)", "field": "generative_second_interaction_lcb", "format": "percent"},
                        {"label": "High-scale co-scale excess LCB", "field": "high_scale_coscale_excess_lcb", "format": "percent"},
                        {"label": "Focused first co-scale LCB", "field": "focused_first_coscale_excess_lcb", "format": "percent"},
                        {"label": "Focused second interaction", "field": "focused_second_interaction", "format": "percent"},
                        {"label": "Focused second interaction LCB", "field": "focused_second_interaction_lcb", "format": "percent"},
                        {"label": "Focused upper diagonal LCB", "field": "focused_upper_diagonal_lcb", "format": "percent"},
                        {"label": "Focused upper co-scale LCB", "field": "focused_upper_coscale_lcb", "format": "percent"},
                    ],
                }
            ],
            "charts": [
                {
                    "id": "condition_accuracy",
                    "title": "Certified co-scale accuracy rises while fixed and irrelevant controls do not",
                    "subtitle": "Five-panel means; accuracy axis begins at zero. Raw FIFO is intentionally retained as a strong boundary.",
                    "headerMarkdown": "The chart compares the registered **certified ASCENT path** with four frozen controls.",
                    "type": "bar",
                    "dataset": "accuracy_by_condition",
                    "sourceId": "condition_accuracy_sql",
                    "valueFormat": "percent",
                    "encodings": {
                        "x": {"field": "endpoint", "type": "nominal", "label": "SmolLM2 model scale"},
                        "y": {"field": "accuracy", "type": "quantitative", "label": "Exact accuracy", "scale": {"zero": True}},
                        "color": {"field": "method", "type": "nominal", "label": "Condition"},
                        "tooltip": [
                            {"field": "method", "type": "nominal", "label": "Condition"},
                            {"field": "accuracy", "type": "quantitative", "label": "Accuracy", "format": "percent"},
                            {"field": "ci95_low", "type": "quantitative", "label": "95% LCB", "format": "percent"},
                            {"field": "ci95_high", "type": "quantitative", "label": "95% UCB", "format": "percent"},
                        ],
                    },
                }
            ],
            "tables": [
                {
                    "id": "readiness_table",
                    "title": "Promotion-gate status",
                    "subtitle": "Passed means the frozen artifact directly proves the stated gate; partial/missing cells are not promoted.",
                    "dataset": "readiness_gates",
                    "sourceId": "readiness_gates_sql",
                    "defaultSort": {"field": "priority", "direction": "asc"},
                    "columns": [
                        {"field": "priority", "label": "Order", "format": "number"},
                        {"field": "gate", "label": "Gate", "type": "text"},
                        {"field": "status", "label": "Status", "type": "text"},
                        {"field": "evidence", "label": "Evidence", "type": "text"},
                        {"field": "next_action", "label": "Next action", "type": "text"},
                    ],
                }
                ,
                {
                    "id": "portfolio_table",
                    "title": "Validated positive-evidence portfolio",
                    "subtitle": "Only frozen or independently confirmed results are included; failed boundaries remain in the readiness audit.",
                    "dataset": "positive_evidence_portfolio",
                    "sourceId": "positive_portfolio_sql",
                    "columns": [
                        {"field": "family", "label": "Task / mechanism", "type": "text"},
                        {"field": "models", "label": "Models", "type": "text"},
                        {"field": "result", "label": "Positive result", "type": "text"},
                        {"field": "strength", "label": "Why it matters", "type": "text"},
                    ],
                }
            ],
            "sources": sources,
            "blocks": [
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "body": (
                        "## Technical summary\n\n"
                        "The core certified scale-complementarity result is now statistically strong, reproducible, and control-backed: across five disjoint public QA7/QA8 panels, ASCENT absolute gain increases from **47.5% to 69.4% to 90.6%**, with both adjacent panel-clustered 95% lower bounds above zero. Fixed one-slot, irrelevant-state, and BM25 controls rule out several simple explanations.\n\n"
                        "The internal high-readiness gate is now **reached**. A five-panel official QA1–QA3 factorial first produced a strong co-scaled gain curve but left two fresh-only lower bounds below zero; that near miss remains unchanged. A separately frozen ten-panel, 1,200-row confirmation then passed all four focused df=9 contrasts, including the formerly marginal second model×state interaction (**+4.33 points [95% CI +2.31,+6.36]**). Together with cross-family official RULER replication, controls, systems evidence, and provenance, this authorizes formal manuscript writing. It does not guarantee TNNLS acceptance."
                    ),
                },
                {"id": "metrics", "type": "metric-strip", "cardIds": ["core_metrics"]},
                {
                    "id": "key_finding",
                    "type": "markdown",
                    "body": (
                        "## Co-scaled state produces a large monotone curve\n\n"
                        "At 135M/360M/1.7B, certified ASCENT accuracy is 50%/75%/100%. The one-slot state remains at 50% for every model, while irrelevant state remains near 34%. BM25 improves with scale but stays below ASCENT. Byte-matched raw FIFO is much stronger and nearly saturates at 1.7B; its honest inclusion limits any claim of universal storage superiority."
                    ),
                },
                {"id": "chart", "type": "chart", "chartId": "condition_accuracy"},
                {
                    "id": "portfolio_intro",
                    "type": "markdown",
                    "body": (
                        "## The broader positive portfolio is already substantial\n\n"
                        "ASCENT is not resting on one certified panel: official RULER and BABILong tasks provide repeated generative scale curves, two autoregressive Transformer families reproduce the long-context pattern, causal interaction tests pass on RULER-VT and composition, and the primary artifacts have clean provenance plus exact cross-node prediction replication."
                    ),
                },
                {"id": "portfolio", "type": "table", "tableId": "portfolio_table"},
                {
                    "id": "scope",
                    "type": "markdown",
                    "body": (
                        "## Scope, data, and metrics\n\n"
                        "The evaluated mechanism is public BABILong-style QA7/QA8 aggregation at an 8,192-token context using three frozen SmolLM2 endpoints and five 32-row panels. State sizes on the co-scaled diagonal are 1/2/8 retained query-relevant events. **Absolute gain** is exact ASCENT accuracy minus unchanged Foundation accuracy. **Remaining-error elimination (RWE)** is `(ASCENT − Foundation) / (1 − Foundation)` and was frozen before confirmation to handle the 100% ceiling without pretending that accuracy can exceed 100%."
                    ),
                },
                {
                    "id": "design",
                    "type": "markdown",
                    "body": (
                        "## Experimental design and validation\n\n"
                        "Public source questions and noise backgrounds are disjoint across registered confirmation panels. Configuration, data, model revisions, git commit, parser accuracy, and clean-worktree state are recorded in every artifact. Foundation generation is unchanged across state sizes. Student-t intervals cluster by panels, not rows. After the 27-cell historical audit, a five-panel 45-cell full factorial and a disjoint ten-panel 50-cell focused confirmation were frozen before scoring. The latter uses all remaining official hash-ordered 40-row/task panels, with 1,200/1,200 target-blind parser reconstruction and no panel/task/outlier deletion."
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## What the current result does not establish\n\n"
                        "The certified 3×3 factorial directly confirms that its +25-point state increments are model-independent, so it is not used alone as model-complementarity proof. The original five-panel generative primary gate remains a near miss even though its combined eight-panel support gate passed. The ten-panel follow-up is an independent confirmation of the marginal estimands, not a retroactive relabeling. Raw FIFO is close to ceiling at 1.7B, while HotpotQA 7B, multivalue, and overwrite remain transparent negative boundaries."
                    ),
                },
                {"id": "table", "type": "table", "tableId": "readiness_table"},
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## Highest-value next steps\n\n"
                        "1. Begin the formal double-anonymous TNNLS manuscript around the scoped scale-complementarity claim.\n"
                        "2. Generate the primary tables/figures directly from immutable analysis JSON and archive hashes.\n"
                        "3. Add same-task Qwen generative replication if it remains non-saturated and feasible; preserve any failure.\n"
                        "4. Keep raw exact storage, HotpotQA 7B, multivalue, overwrite, and profiler-wall boundaries in limitations.\n"
                        "5. Run a final leakage, citation, anonymization, and reproducibility audit before submission."
                    ),
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": (
                        "## Further questions\n\n"
                        "Can a non-saturated aggregation task separate ASCENT from raw FIFO at the large endpoint? Does the same co-scaled law reproduce on Qwen under identical frozen state sizes? Can a learned generative decoder recover the certified posterior without reintroducing the QA8 person-name failure mode?"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "accuracy_by_condition": accuracy_rows,
                "headline_metrics": headline,
                "readiness_gates": readiness_rows,
                "positive_evidence_portfolio": portfolio_rows,
            },
            "accessIssues": [],
        },
        "sources": [
            {
                "id": "headline_metrics_sql",
                "query": {
                    "engine": "duckdb",
                    "language": "sql",
                    "sql": "SELECT item.* FROM read_json_auto('reports/ascent_tnnls_readiness_2026-08-14/report_data.json') AS r, UNNEST(r.headline_metrics) AS t(item)",
                    "description": "Loads the reviewed headline metrics produced by the reproducible Python synthesis.",
                    "tables_used": [
                        "reports/ascent_tnnls_readiness_2026-08-14/report_data.json",
                    ],
                    "metric_definitions": {
                        "absolute_gain": "Exact ASCENT accuracy minus exact Foundation accuracy.",
                        "remaining_error_elimination": "(ASCENT accuracy - Foundation accuracy) / (1 - Foundation accuracy).",
                        "confidence_interval": "Two-sided 95% Student-t interval clustered over five panels (df=4).",
                    },
                    "executed_at": generated_at,
                },
            },
            {
                "id": "condition_accuracy_sql",
                "query": {
                    "engine": "duckdb",
                    "language": "sql",
                    "sql": "SELECT item.* FROM read_json_auto('reports/ascent_tnnls_readiness_2026-08-14/report_data.json') AS r, UNNEST(r.accuracy_by_condition) AS t(item)",
                    "description": "Loads all five registered methods at all three endpoints; no condition is filtered.",
                    "tables_used": ["reports/ascent_tnnls_readiness_2026-08-14/report_data.json"],
                    "filters": ["All five frozen panels", "All three endpoints", "All primary/control conditions"],
                    "metric_definitions": {"accuracy": "Mean exact accuracy over five frozen 32-row panels."},
                    "executed_at": generated_at,
                },
            },
            {
                "id": "readiness_gates_sql",
                "query": {
                    "engine": "duckdb",
                    "language": "sql",
                    "sql": "SELECT item.* FROM read_json_auto('reports/ascent_tnnls_readiness_2026-08-14/report_data.json') AS r, UNNEST(r.readiness_gates) AS t(item) ORDER BY item.priority",
                    "description": "Loads the complete promotion-gate audit in its prespecified priority order.",
                    "tables_used": ["reports/ascent_tnnls_readiness_2026-08-14/report_data.json"],
                    "filters": ["No readiness gate omitted"],
                    "executed_at": generated_at,
                },
            },
            {
                "id": "positive_portfolio_sql",
                "query": {
                    "engine": "duckdb",
                    "language": "sql",
                    "sql": "SELECT item.* FROM read_json_auto('reports/ascent_tnnls_readiness_2026-08-14/report_data.json') AS r, UNNEST(r.positive_evidence_portfolio) AS t(item)",
                    "description": "Loads the validated positive-evidence portfolio summarized from the current readiness audit.",
                    "tables_used": [
                        "reports/ascent_tnnls_readiness_2026-08-14/report_data.json",
                        "docs/TNNLS_READINESS_AUDIT.md",
                    ],
                    "filters": ["Frozen or independently confirmed positive results only"],
                    "executed_at": generated_at,
                },
            },
            {
                "id": "readiness_audit",
                "query": {
                    "engine": "local-markdown",
                    "description": "Current evidence matrix and non-negotiable promotion gates.",
                    "executed_at": generated_at,
                },
            },
        ],
        "package_info": {
            "originUrl": "artifact://ascent-tnnls-readiness-2026-08-14",
            "controls": {"edit": False, "refresh": False},
        },
    }
    return synthesis, artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--certified-factorial", type=Path, required=True)
    parser.add_argument("--generative-factorial", type=Path, required=True)
    parser.add_argument("--generative-audit", type=Path, required=True)
    parser.add_argument("--generative-fresh", type=Path, required=True)
    parser.add_argument("--generative-focused", type=Path, required=True)
    parser.add_argument("--data-output", type=Path, required=True)
    parser.add_argument("--artifact-output", type=Path, required=True)
    args = parser.parse_args()
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    synthesis, artifact = build(
        load(args.confirmation),
        load(args.controls),
        load(args.certified_factorial),
        load(args.generative_factorial),
        load(args.generative_audit),
        load(args.generative_fresh),
        load(args.generative_focused),
        generated_at,
    )
    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    args.artifact_output.parent.mkdir(parents=True, exist_ok=True)
    args.data_output.write_text(json.dumps(synthesis, indent=2) + "\n")
    args.artifact_output.write_text(json.dumps(artifact, indent=2) + "\n")


if __name__ == "__main__":
    main()
