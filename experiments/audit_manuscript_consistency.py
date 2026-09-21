#!/usr/bin/env python3
"""Audit every promoted manuscript number and the complete reference graph.

The audit is intentionally independent of the table renderers.  It recomputes
means, standard errors, Student-t intervals, and the registered eight-estimand
Bonferroni intervals directly from panel/seed values, reconciles the plot-data
exports with their immutable analyses, checks promoted narrative claims across
the manuscript and submission documents, and verifies the bibliography graph.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import bibtexparser
from scipy.stats import t

from experiments.build_verified_bibliography import render_entry
from experiments.render_cross_task_table import load_single_analysis


TOLERANCE = 1e-9


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0, abs_tol=TOLERANCE)


def citation_keys(tex: str) -> set[str]:
    keys: set[str] = set()
    for group in re.findall(r"\\cite[a-zA-Z]*\{([^}]+)\}", tex):
        keys.update(key.strip() for key in group.split(","))
    return keys


def reference_keys(tex: str) -> list[str]:
    return re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", tex)


def labels(tex: str) -> list[str]:
    return re.findall(r"\\label\{([^}]+)\}", tex)


def summary_values(item: dict[str, Any]) -> list[float] | None:
    for key in ("values", "panel_values", "seed_values"):
        values = item.get(key)
        if isinstance(values, list) and values:
            return [float(value) for value in values]
    return None


def audit_summaries(
    payload: Any,
    path: str,
    failures: list[str],
) -> int:
    """Recursively recompute every summary carrying its underlying clusters."""
    count = 0
    if isinstance(payload, dict):
        values = summary_values(payload)
        required = {"mean", "standard_error", "ci95_low", "ci95_high"}
        if values is not None and required <= payload.keys():
            count += 1
            n = len(values)
            mean = math.fsum(values) / n
            se = statistics.stdev(values) / math.sqrt(n) if n > 1 else 0.0
            alpha = (
                0.05 / 8
                if "supplementary_bonferroni_familywise_estimands" in path
                else 0.05
            )
            critical = float(t.ppf(1 - alpha / 2, n - 1)) if se else 0.0
            expected = {
                "mean": mean,
                "standard_error": se,
                "ci95_low": mean - critical * se,
                "ci95_high": mean + critical * se,
            }
            for key, value in expected.items():
                if not close(payload[key], value):
                    failures.append(
                        f"{path}.{key}: stored={payload[key]!r}, recomputed={value!r}"
                    )
            if "degrees_of_freedom" in payload and payload["degrees_of_freedom"] != n - 1:
                failures.append(
                    f"{path}.degrees_of_freedom: stored={payload['degrees_of_freedom']}, expected={n - 1}"
                )
        for key, value in payload.items():
            count += audit_summaries(value, f"{path}.{key}", failures)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            count += audit_summaries(value, f"{path}[{index}]", failures)
    return count


def cwe_raw_vectors(analysis: dict[str, Any]) -> dict[str, list[float]]:
    endpoint_order = [cell["endpoint"] for cell in analysis["seeds"][0]["cells"]]
    vectors = {
        f"endpoint:{endpoint}": [
            next(cell["scale_gain"] for cell in seed["cells"] if cell["endpoint"] == endpoint)
            for seed in analysis["seeds"]
        ]
        for endpoint in endpoint_order
    }
    for index in range(2):
        vectors[f"adjacent:{index}"] = [
            seed["adjacent_gain_differences"][index] for seed in analysis["seeds"]
        ]
    vectors["slope"] = [seed["slope_on_log_parameters"] for seed in analysis["seeds"]]
    return vectors


def audit_cwe(analysis: dict[str, Any], name: str, failures: list[str]) -> int:
    vectors = cwe_raw_vectors(analysis)
    summaries: dict[str, dict[str, Any]] = {
        **{
            f"endpoint:{key}": value
            for key, value in analysis["endpoint_gain_t_intervals"].items()
        },
        **{
            f"adjacent:{index}": value
            for index, value in enumerate(analysis["adjacent_gain_difference_t_intervals"])
        },
        "slope": analysis["slope_t_interval"],
    }
    for key, values in vectors.items():
        item = summaries[key]
        n = len(values)
        mean = math.fsum(values) / n
        se = statistics.stdev(values) / math.sqrt(n)
        critical = float(t.ppf(0.975, n - 1))
        expected = (mean, se, mean - critical * se, mean + critical * se)
        observed = (
            item["mean"], item["standard_error"], item["ci95_low"], item["ci95_high"]
        )
        if not all(close(left, right) for left, right in zip(observed, expected, strict=True)):
            failures.append(f"{name}.{key}: seed-level t summary mismatch")
    return len(vectors)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def audit_plot_data(
    paper: Path,
    official: dict[str, Any],
    semantic: dict[str, Any],
    qrag: dict[str, Any],
    mechanism: dict[str, Any],
    around7b: dict[str, Any],
    systems: dict[str, Any],
    flops: dict[str, Any],
    failures: list[str],
) -> int:
    data = paper / "data/evidence"
    comparisons = 0

    primary = csv_rows(data / "primary_panel.csv")
    expected_primary: list[tuple[str, str, float, int, float]] = []
    endpoints = (
        ("qwen2p5-0p5b-instruct", 0.5),
        ("qwen2p5-1p5b-instruct", 1.5),
        ("qwen2p5-3b-instruct", 3.0),
    )
    for study, payload in (("Official 16K", official), ("Semantic holdout", semantic)):
        for endpoint, scale in endpoints:
            values = payload["endpoint_results"][endpoint]["primary"]["gain"]["panel_values"]
            expected_primary.extend(
                (study, "ASCENT - Foundation", scale, index, value)
                for index, value in enumerate(values, 1)
            )
    for endpoint, scale in endpoints:
        values = qrag["endpoint_results"][endpoint]["overall"]["ascent_minus_qrag"]["panel_values"]
        expected_primary.extend(
            ("Direct peer", "ASCENT - Q-RAG", scale, index, value)
            for index, value in enumerate(values, 1)
        )
    observed_primary = [
        (row["study"], row["contrast"], float(row["scale_b"]), int(row["panel"]), float(row["value"]))
        for row in primary
    ]
    comparisons += len(expected_primary)
    if observed_primary != expected_primary:
        failures.append("paper/data/evidence/primary_panel.csv does not match source analyses")

    factorial = csv_rows(data / "factorial_panel.csv")
    expected_factorial: list[tuple[str, int, int, float]] = []
    for endpoint, label in (
        ("qwen2p5-1p5b-instruct", "1.5B"),
        ("qwen2p5-3b-instruct", "3B"),
        ("qwen2p5-7b-instruct", "7B"),
    ):
        for slots in (2, 3, 5):
            values = mechanism["cells"][f"{endpoint}@k{slots}"]["gain_nll"]["values"]
            expected_factorial.extend(
                (label, slots, index, value) for index, value in enumerate(values, 1)
            )
    observed_factorial = [
        (row["model"], int(row["slots"]), int(row["panel"]), float(row["gain_nll"]))
        for row in factorial
    ]
    comparisons += len(expected_factorial)
    if observed_factorial != expected_factorial:
        failures.append("paper/data/evidence/factorial_panel.csv does not match source analysis")

    breadth = csv_rows(data / "around7b_panel.csv")
    expected_breadth: list[tuple[str, int, float, float, float]] = []
    for endpoint, label in (
        ("qwen2p5-7b-instruct", "Qwen2.5-7B"),
        ("mistral-7b-instruct-v0p3", "Mistral-7B-v0.3"),
        ("falcon3-7b-instruct", "Falcon3-7B"),
        ("granite-3p3-8b-instruct", "Granite-3.3-8B"),
        ("qwen3-8b-nonthinking", "Qwen3-8B"),
    ):
        primary_result = around7b["endpoint_results"][endpoint]["primary"]
        expected_breadth.extend(
            (
                label,
                index,
                primary_result["gain"]["panel_values"][index - 1],
                primary_result["foundation"]["panel_values"][index - 1],
                primary_result["ascent"]["panel_values"][index - 1],
            )
            for index in range(1, 11)
        )
    observed_breadth = [
        (row["model"], int(row["panel"]), float(row["gain"]), float(row["foundation"]), float(row["ascent"]))
        for row in breadth
    ]
    comparisons += len(expected_breadth)
    if observed_breadth != expected_breadth:
        failures.append("paper/data/evidence/around7b_panel.csv does not match source analysis")

    intervals = csv_rows(data / "interaction_intervals.csv")
    keys = (
        "first_diagonal_gain_increment",
        "second_diagonal_gain_increment",
        "first_model_by_state_interaction",
        "second_model_by_state_interaction",
        "7b_k5_minus_k3_gain",
    )
    source_intervals = mechanism["supplementary_bonferroni_familywise_estimands"]
    comparisons += len(keys)
    for row, key in zip(intervals, keys, strict=True):
        item = source_intervals[key]
        if not all(
            close(float(row[field]), item[source])
            for field, source in (("mean", "mean"), ("ci_low", "ci95_low"), ("ci_high", "ci95_high"))
        ):
            failures.append(f"interaction_intervals.csv mismatch at {key}")

    system_rows = csv_rows(data / "systems.csv")
    flop_by_endpoint = {row["endpoint"]: row for row in flops["endpoints"]}
    for row, endpoint in zip(
        system_rows,
        ("smollm2-135m-instruct", "smollm2-360m-instruct", "smollm2-1p7b-instruct"),
        strict=True,
    ):
        state = systems["endpoints"][endpoint]
        profile = flop_by_endpoint[endpoint]
        expected = (
            state["state"]["mean_total_external_state_bytes"],
            state["state"]["mean_total_external_state_to_model_artifact_byte_ratio"],
            state["all_repetitions"]["ascent_to_foundation_decode_time_ratio"]["median"],
            profile["supported_operator_flop_reduction_factor"],
        )
        observed = tuple(float(row[key]) for key in ("state_bytes", "state_model_ratio", "decode_ratio", "flop_reduction"))
        comparisons += 4
        if not all(close(left, right) for left, right in zip(observed, expected, strict=True)):
            failures.append(f"systems.csv mismatch at {endpoint}")
    return comparisons


def audit_raw_systems(root: Path, systems: dict[str, Any], flops: dict[str, Any], failures: list[str]) -> int:
    comparisons = 0
    raw_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in systems["source_files"]:
        run = load(root / source)
        raw_groups[run["endpoint"]["name"]].append(run)
    for endpoint, runs in raw_groups.items():
        summary = systems["endpoints"][endpoint]
        ratios = [run["systems"]["ascent_to_foundation_decode_time_ratio"] for run in runs]
        observed = summary["all_repetitions"]["ascent_to_foundation_decode_time_ratio"]["median"]
        comparisons += 1
        if not close(observed, statistics.median(ratios)):
            failures.append(f"systems analysis median mismatch at {endpoint}")
        exemplar = runs[0]
        state_bytes = exemplar["state"]["mean_total_external_state_bytes"]
        model_bytes = exemplar["endpoint"]["model_artifact"]["bytes"]
        comparisons += 2
        if not close(summary["state"]["mean_total_external_state_bytes"], state_bytes):
            failures.append(f"systems state-byte mismatch at {endpoint}")
        if not close(summary["state"]["mean_total_external_state_to_model_artifact_byte_ratio"], state_bytes / model_bytes):
            failures.append(f"systems state/model mismatch at {endpoint}")

    flop_root = root / "artifacts/remote_results/flops_e49eae9/flops_e49eae9"
    for row in flops["endpoints"]:
        raw = load(flop_root / f"{row['endpoint']}.json")
        expected_ratio = raw["ascent"]["supported_operator_flops"] / raw["foundation"]["supported_operator_flops"]
        comparisons += 2
        if not close(row["supported_operator_flop_ratio"], expected_ratio):
            failures.append(f"FLOP ratio mismatch at {row['endpoint']}")
        if not close(row["supported_operator_flop_reduction_factor"], 1 / expected_ratio):
            failures.append(f"FLOP reduction mismatch at {row['endpoint']}")
    return comparisons


def audit_references(paper: Path, tex: str, failures: list[str]) -> dict[str, int]:
    bib_paths = (paper / "references.bib", paper / "references_verified.bib")
    entries: list[dict[str, str]] = []
    for path in bib_paths:
        parsed = bibtexparser.loads(path.read_text())
        for entry in parsed.entries:
            entry["_source"] = path.name
            entries.append(entry)

    cited = citation_keys(tex)
    defined = {entry["ID"] for entry in entries}
    if not cited <= defined:
        failures.append(f"undefined citation keys: {sorted(cited - defined)}")

    bbl = (paper / "build/main.bbl").read_text()
    compiled = set(re.findall(r"\\bibitem(?:\[[^]]*\])?\{([^}]+)\}", bbl))
    if compiled != cited:
        failures.append(
            f"compiled bibliography mismatch: missing={sorted(cited - compiled)}, extra={sorted(compiled - cited)}"
        )

    for field in ("ID", "doi", "url", "title"):
        groups: dict[str, list[str]] = defaultdict(list)
        for entry in entries:
            value = entry.get(field, "").strip().lower()
            if value:
                groups[value].append(entry["ID"])
        duplicates = {value: keys for value, keys in groups.items() if len(keys) > 1}
        if duplicates:
            failures.append(f"duplicate bibliography {field}: {duplicates}")

    for entry in entries:
        required = {"author", "title", "year"}
        if not required <= entry.keys() or not (entry.get("journal") or entry.get("booktitle") or entry.get("howpublished")):
            failures.append(f"incomplete bibliography entry: {entry['ID']}")
        if not re.fullmatch(r"(?:19|20)\d{2}", entry.get("year", "")):
            failures.append(f"invalid bibliography year: {entry['ID']}={entry.get('year')!r}")

    manifest = load(paper / "data/reference_verification.json")
    records = manifest.get("records", [])
    verified_entries = bibtexparser.loads((paper / "references_verified.bib").read_text()).entries
    manifest_keys = [record["key"] for record in records]
    verified_keys = [entry["ID"] for entry in verified_entries]
    if manifest.get("count") != len(records) or len(set(manifest_keys)) != len(records):
        failures.append("reference verification manifest count/key uniqueness mismatch")
    if set(manifest_keys) != set(verified_keys):
        failures.append("reference verification manifest and generated BibTeX key sets differ")
    regenerated = "\n\n".join(render_entry(record) for record in records) + "\n"
    if regenerated != (paper / "references_verified.bib").read_text():
        failures.append("references_verified.bib is not an exact rendering of its metadata manifest")

    return {
        "cited_keys": len(cited),
        "defined_keys": len(defined),
        "compiled_entries": len(compiled),
        "verified_manifest_records": len(records),
        "uncited_defined_entries": len(defined - cited),
    }


def audit_cross_references(tex: str, failures: list[str]) -> dict[str, int]:
    label_list = labels(tex)
    refs = reference_keys(tex)
    duplicate_labels = sorted(key for key, count in Counter(label_list).items() if count > 1)
    undefined_refs = sorted(set(refs) - set(label_list))
    if duplicate_labels:
        failures.append(f"duplicate LaTeX labels: {duplicate_labels}")
    if undefined_refs:
        failures.append(f"undefined LaTeX references: {undefined_refs}")
    return {
        "labels": len(label_list),
        "cross_references": len(refs),
        "duplicate_labels": len(duplicate_labels),
        "undefined_cross_references": len(undefined_refs),
    }


def audit_revision1_claims(root: Path, failures: list[str]) -> int:
    """Revision-1 sentences whose numbers must resolve to a released record."""
    import csv

    paper = root / "paper"
    main = normalized((paper / "main.tex").read_text())
    appendix = normalized((paper / "appendix_revision1.tex").read_text())
    checks = 0

    def expect(document: str, text: str, claim: str) -> None:
        nonlocal checks
        checks += 1
        if normalized(claim) not in text:
            failures.append(f"revision-1 claim missing or drifted in {document}: {claim}")

    # 7B accuracy regression in the abstract, from the around-7B analysis
    fourth = load(root / "artifacts/around7b_formal/analysis.json")["qwen_fourth_scale"]["gain_increment"]
    from decimal import ROUND_HALF_UP, Decimal

    def pts(value: float) -> str:
        # panel means are exact decimals (multiples of 1/800); round the exact value half-up
        return str(Decimal(repr(round(abs(100 * value), 8))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

    expected = f"$-{pts(fourth['mean'])}$ points [$-{pts(fourth['ci95_low'])}$, $-{pts(fourth['ci95_high'])}$]"
    expect("main", main, expected)
    # power appendix numbers, from the power analysis record
    power = load(root / "reports/PANEL_POWER_ANALYSIS.json")
    second = [c for s_ in power["primary_studies"] if s_["study"] == "Official 16K" for c in s_["adjacent_increments"]][1]
    first = [c for s_ in power["primary_studies"] if s_["study"] == "Official 16K" for c in s_["adjacent_increments"]][0]
    expect("appendix", appendix, f"the MDE at 80\\% power is {second['mde_points_at_alpha_smallest_power_target']:.2f} points")
    expect("appendix", appendix, f"({first['mde_points_at_alpha_smallest_power_target']:.2f} on the first's)")
    expect("appendix", appendix, f"detected at power $\\approx{second['achieved_power_at_alpha_smallest']:.2f}$".replace("0.", "."))
    expect("appendix", appendix, f"($t_9={first['t']:.1f}$)")
    expect("appendix", appendix, f"SE {first['se_points']:.2f} points gives $t_9={first['t']:.1f}$")
    expect("main", main, f"power of about .{int(round(100 * second['achieved_power_at_alpha_smallest'])):02d}")
    # 13,824 row audit tallies, from the released CSV
    rows = list(csv.DictReader((root / "reports/row_audits/FACTORIAL_TRANSITION_ROW_AUDIT_13824.csv").open()))
    improved = sum(1 for r in rows if float(r["exact_posterior_delta"]) > 0)
    model_improved = sum(1 for r in rows if float(r["model_ascent_delta"]) > 0)
    if len(rows) != 13824 or not all(r["row_check_pass"] == "True" for r in rows):
        failures.append("row audit CSV does not hold 13,824 passing row checks")
    expect("main", main, f"improves on {improved:,} of the 13,824 transitions")
    expect("appendix", appendix, f"improves on\n{improved:,} of 13,824 transitions and the reader's NLL on {model_improved:,}")
    # cross-architecture sensitivity, from the comparison record
    cross = load(root / "artifacts_revision/crossnode_2026-09/CROSSARCH_COMPARISON.json")
    max_pts = max(abs(v["abs_diff"]) for r in cross["results"] if r["family"] == "babilong" for k, v in r["ladder"]["3_panel_precision"]["statistics"].items() if k == "gain.mean")
    max_nats = max(abs(v["abs_diff"]) for r in cross["results"] if r["family"] == "factorial" for k, v in r["ladder"]["3_panel_precision"]["statistics"].items() if k == "gain_nll.mean")
    if max_nats > 0.015:
        failures.append(f"cross-architecture factorial shift changed: {max_nats:.4f} nats")
    expect("main", main, "approximately 0.01 nats")
    full = load(root / "artifacts_revision/target_blindness_2026-09/TARGET_BLINDNESS_AUDIT.json")["cross_architecture"]
    changed = full["rows_with_changed_score_total"]
    changed_f = sum(c["rows_with_changed_score"]["foundation"] for c in full["cells"])
    changed_a = sum(c["rows_with_changed_score"]["ascent"] for c in full["cells"])
    unchanged_cells = sum(1 for c in full["cells"] if abs(c["gain_shift"]) < 1e-12)
    expect("main", main, f"{changed} of {2 * full['rows_total']:,} scored outputs changed ({changed_f} Foundation,\n{changed_a} \\ascent{{}})")
    expect("main", main, f"up to {full['max_abs_gain_shift_points']:.2f} accuracy points")
    if unchanged_cells != 20:
        failures.append(f"unchanged-cell count is {unchanged_cells}, manuscript says twenty")
    stats = full["study_statistics"]
    def trio(study: str, key: str) -> str:
        return "/".join(f"{100 * v:.2f}" for v in stats[study][key].values())
    expect("main", main, f"official 16K: {trio('official16k', 'endpoint_gain_reference')} $\\to$ {trio('official16k', 'endpoint_gain_a100')} points")
    expect("main", main, f"semantic holdout: {trio('semantic_holdout', 'endpoint_gain_reference')} $\\to$ {trio('semantic_holdout', 'endpoint_gain_a100')}")
    mean_shift = max(abs(stats[s_]["endpoint_gain_a100"][e] - stats[s_]["endpoint_gain_reference"][e]) for s_ in stats for e in stats[s_]["endpoint_gain_reference"])
    expect("main", main, f"moved by at most\n{100 * mean_shift:.2f} point")
    off2 = stats["official16k"]["adjacent_contrasts_a100"][1]; sem2 = stats["semantic_holdout"]["adjacent_contrasts_a100"][1]
    expect("main", main, f"second official increment {100 * off2['increment']:.2f} points,\n$p_{{\\mathrm{{Holm}}}}={off2['p_holm']:.4f}$")
    expect("main", main, f"semantic holdout {100 * sem2['increment']:.2f} points,\n$p_{{\\mathrm{{Holm}}}}={sem2['p_holm']:.3f}$")
    if not all(c["passes"] for s_ in stats for c in stats[s_]["adjacent_contrasts_a100"]):
        failures.append("a registered contrast fails on the A100 rerun; manuscript says all pass")
    if cross["all_binary64_identical"] or cross["all_scientific_fields_and_predictions_equal"]:
        failures.append("cross-architecture record no longer shows the reported non-identity")
    # schema-blind ablation (Phase 2C): the §7 outcome sentence must match the record
    ablation = root / "artifacts_revision/schema_blind_2026-09/GENERIC_WRITER_ABLATION.json"
    if ablation.exists():
        rec = load(ablation)
        if rec["any_generic_writer_survives"]:
            failures.append("a generic writer survives the registered rule; §7 says none does")
        gains = [100 * c["endpoint_results"][e]["primary"]["gain"]["mean"] for c in rec["conditions"].values() for e in c["endpoint_results"]]
        if max(gains) >= 0:
            failures.append("a generic-writer gain is non-negative; §7 says the generic state is worse than no state at every scale")
        lo, hi = min(gains), max(gains)
        expect("main", main, f"(gains of $-{abs(round(hi))}$ to $-{abs(round(lo))}$ points)")
        adv = {e: [100 * c["schema_advantage_same_instance"][e]["canonical_minus_generic"]["mean"] for c in rec["conditions"].values()] for e in next(iter(rec["conditions"].values()))["schema_advantage_same_instance"]}
        small = adv["qwen2p5-0p5b-instruct"]; large = adv["qwen2p5-1p5b-instruct"] + adv["qwen2p5-3b-instruct"]
        expect("main", main, f"is {round(sum(small) / len(small))} points at 0.5B and {round(min(large))}--{round(max(large))}\npoints at 1.5B and 3B")
        if any(sum(c["foundation_output_rows_differing_from_same_instance_canonical"].values()) for c in rec["conditions"].values()):
            failures.append("Foundation outputs differ across ablation conditions; Appendix D says they are identical")
    else:
        failures.append("schema-blind ablation record missing (Phase 2C not finalized)")
    # preregistration timeline count
    timeline = load(root / "reports/PREREGISTRATION_TIMELINE.json")
    scored = [c for c in timeline["configs"] if c["retained_records_with_this_config_hash"]]
    if any(c["freeze_precedes_earliest_score"] is False for c in scored):
        failures.append("a configuration was scored before its freeze commit")
    expect("main", main, f"for all {len(scored)} frozen configurations with retained scores")
    # target-blindness rerun summary (Phase 2B), when present
    tb = root / "artifacts_revision/target_blindness_2026-09/TARGET_BLINDNESS_AUDIT.json"
    if tb.exists():
        record = load(tb)
        if not record.get("all_rows_target_blind") or not record.get("same_instance_controls_identical"):
            failures.append("target-blindness audit record does not support the manuscript statement")
        expect("main", main, f"All {record['rows_total']:,} rows of the official 16K and semantic-holdout studies")
    else:
        failures.append("target-blindness audit record missing (Phase 2B not finalized)")
    return checks


def audit_promoted_claims(paper: Path, failures: list[str]) -> int:
    documents = {
        "main": normalized((paper / "main.tex").read_text()),
        "appendix": normalized((paper / "appendix.tex").read_text()),
        "highlights": normalized((paper / "submission/Highlights.txt").read_text()),
        "cover": normalized((paper / "submission/Cover_Letter.md").read_text()),
    }
    claims = (
        ("main", "10.50, 78.88, and 82.63 percentage points at 0.5B, 1.5B, and 3B"),
        ("main", "log-likelihood gains of 0.388, 2.099, and 7.406 nats at 1.5B, 3B, and 7B"),
        ("main", "Five public 7B--8B readers obtain 66.75--85.63 point gains"),
        ("main", "68.38 points ($p_{\\mathrm{Holm}}=3.43\\times10^{-11}$) and 3.75 points ($p_{\\mathrm{Holm}}=0.0255$)"),
        ("main", "73.75 points ($p_{\\mathrm{Holm}}=1.65\\times10^{-13}$) and 3.50 points ($p_{\\mathrm{Holm}}=0.02897$)"),
        ("main", "a 15.38-point difference with 95\\% CI [13.69,17.06]"),
        ("main", "adjacent increments are 1.711 [1.540,1.882] and 5.307 [5.053,5.561]"),
        ("main", "interactions are positive: .781 [.539,1.023] and .968 [.671,1.264]"),
        ("main", "all 13,824 endpoint--transition row checks"),
        ("main", "Qwen2.5-7B gains 66.75 points, Qwen3-8B 72.00, Mistral-7B 80.75, Falcon3-7B 80.88, and Granite-3.3-8B 85.63"),
        ("main", "gains increase from .2628 to .4378 to .9561"),
        ("main", "Qwen2.5 at 16K also has increasing means, .2417 to .6144 to .7100"),
        ("main", "both Qwen2.5 (.0438 to .1208 to .3188) and SmolLM2 (.1042 to .2167 to .5229)"),
        ("main", "leads by 22.75 points [.1885,.2666] at 1.5B and by 26.25 points [.2307,.2943] at 3B"),
        ("main", "median isolated decode-time ratios are .4130, .4247, and .2791"),
        ("main", "Fixed write-plus-read state occupies 589.0, 624.6, and 640.6 bytes"),
        ("main", "83.7$\\times$, 79.4$\\times$, and 78.2$\\times$ fewer FLOPs"),
        ("main", "increment of $-.1588$ [$-.2055,-.1120$]"),
        ("main", "upper adjacent interval is .0956 [$-.0043$,.1954]"),
        ("main", "wall-time ratios of .9008, .8651, and .7262"),
        ("appendix", "gains are .2417, .5000, and .6694; their respective 95\\% intervals are [.1589,.3245], [.4586,.5414], and [.5197,.8192]"),
        ("appendix", "Both diagonal increments are positive: .2583 [.1488,.3679] and .1694 [.0554,.2835]"),
        ("appendix", "same-refinement interactions are .0417 [$-.0618$,.1452] and .0333 [$-.0081$,.0747]"),
        ("appendix", "co-scaled gains are .4750, .6938, and .9063"),
        ("appendix", "adjacent increments are .2188 [.1462,.2913] and .2125 [.1549,.2701]"),
        ("appendix", "Wall ratios of .9008, .8651, and .7262"),
        ("highlights", "Log-loss gains increase 19.1-fold from 0.388 to 7.406 nats."),
        ("highlights", "Five public 7B–8B readers gain 66.75–85.63 accuracy points."),
        ("cover", "increase from 0.388 to 2.099 to 7.406 nats"),
        ("cover", "10.50, 78.88, and 82.63 percentage-point gains at 0.5B, 1.5B, and 3B"),
        ("cover", "public 7B–8B readers gain 66.75–85.63 points"),
    )
    for document, claim in claims:
        if normalized(claim) not in documents[document]:
            failures.append(f"promoted claim missing or drifted in {document}: {claim}")
    return len(claims)


def audit(root: Path) -> dict[str, Any]:
    paper = root / "paper"
    main = (paper / "main.tex").read_text()
    appendix = (paper / "appendix.tex").read_text()
    appendix_revision1 = (paper / "appendix_revision1.tex").read_text()
    tex = main + "\n" + appendix + "\n" + appendix_revision1
    failures: list[str] = []

    paths = {
        "official": root / "artifacts/remote_results/babilong_qwen_canonical_16k_confirmation_4fd25e9/analysis.json",
        "semantic": root / "artifacts/remote_results/babilong_qwen_canonical_semantic_holdout_b7480e6/analysis.json",
        "mechanism": root / "artifacts/sum_numeric_candidate/confirmation_analysis.json",
        "around7b": root / "artifacts/around7b_formal/analysis.json",
        "qrag": root / "artifacts/remote_results/babilong_qrag_direct_peer_89a8280/analysis.json",
        "systems": root / "artifacts/remote_results/babilong_4k_systems/analysis.json",
        "flops": root / "artifacts/remote_results/flops_e49eae9/analysis.json",
        "generative_factorial": root / "artifacts/remote_results/babilong_8k_generative_factorial_audit_693c414/factorial_analysis.json",
        "certified_factorial": root / "artifacts/remote_results/babilong_qa78_certified_factorial_ab4222a/factorial_analysis.json",
        "smol_cwe": root / "artifacts/remote_results/cwe_confirm_c749f24/analysis.json",
        "qwen_cwe": root / "artifacts/remote_results/qwen_cwe_confirm_cd8352f/analysis.json",
    }
    analyses = {name: load(path) for name, path in paths.items()}
    analyses["qwen_multiquery"] = load_single_analysis(
        root / "artifacts/remote_results/qwen_niah_fullctx_16k_multiquery_width_linear_confirmation_372xxx.tar.gz"
    )
    analyses["smol_multiquery"] = load_single_analysis(
        root / "artifacts/remote_results/smollm2_4k_multiquery_width075_confirm_376101_376202_376303.tar.gz"
    )

    recalculated = 0
    for name in (
        "official", "semantic", "mechanism", "around7b", "qrag",
        "generative_factorial", "certified_factorial", "qwen_multiquery", "smol_multiquery",
    ):
        recalculated += audit_summaries(analyses[name], name, failures)
    recalculated += audit_cwe(analyses["smol_cwe"], "smol_cwe", failures)
    recalculated += audit_cwe(analyses["qwen_cwe"], "qwen_cwe", failures)

    plot_comparisons = audit_plot_data(
        paper,
        analyses["official"], analyses["semantic"], analyses["qrag"],
        analyses["mechanism"], analyses["around7b"], analyses["systems"],
        analyses["flops"], failures,
    )
    raw_system_comparisons = audit_raw_systems(
        root, analyses["systems"], analyses["flops"], failures
    )
    promoted_claims = audit_promoted_claims(paper, failures)
    reference_counts = audit_references(paper, tex, failures)
    revision1_claims = audit_revision1_claims(root, failures)
    cross_reference_counts = audit_cross_references(tex, failures)

    log = (paper / "build/main.log").read_text()
    if re.search(r"undefined references|undefined citation|Citation .* undefined|Reference .* undefined", log, re.IGNORECASE):
        failures.append("compiled LaTeX log contains undefined citation/reference diagnostics")
    if "multiply defined" in log.lower():
        failures.append("compiled LaTeX log contains multiply-defined labels")

    return {
        "pass": not failures,
        "confidence": "high" if not failures else "low",
        "independently_recalculated_statistical_summaries": recalculated,
        "plot_data_value_comparisons": plot_comparisons,
        "raw_system_value_comparisons": raw_system_comparisons,
        "promoted_narrative_claims_checked": promoted_claims,
        "revision1_claims_checked": revision1_claims,
        "references": reference_counts,
        "cross_references": cross_reference_counts,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = audit(args.root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
