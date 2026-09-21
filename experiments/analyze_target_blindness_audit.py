#!/usr/bin/env python3
"""Analyze the instrumented target-blindness rerun (Phase 2B, Array revision 1).

Three questions, each answered from the records under
``artifacts_revision/target_blindness_2026-09/``:

1. **Per-row target-blindness.** Every one of the rerun rows must record zero
   blocked target accesses, no target read before the reveal, only
   ``input``/``question``/``task`` read before the reveal, and a passing
   three-link state-provenance check. Failures are listed, not summarized away.
2. **Instrumentation changed nothing.** For the six pre-registered same-instance
   control pairs (uninstrumented vs instrumented runner, same A100, same
   environment) every scientific field, prediction row and stored float must be
   identical (``compare_crossarch_reproduction`` ladder, all four levels).
3. **Cross-architecture sensitivity, all panels.** The 60 instrumented cells are
   compared with their retained Blackwell references at the ladder's four
   levels, and the ten-panel study statistics (endpoint gains, adjacent
   increments, Holm p) are recomputed from the A100 records with the canonical
   estimator so that the manuscript can state how the printed numbers move.

Writes ``TARGET_BLINDNESS_AUDIT.json`` next to the records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.analyze_babilong_canonical_coscale_16k_confirmation import _one_sided_positive_p, holm_adjust
from experiments.analyze_babilong_canonical_coscale_development import _panel_values, _scope
from experiments.analyze_babilong_qrag_direct_peer import cluster_summary
from experiments.compare_crossarch_reproduction import compare_records

ALLOWED_KEYS = {"input", "question", "task"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_rows(record: dict[str, Any], path: str) -> list[str]:
    problems = []
    if not record.get("preflight", {}).get("target_values_not_rendered_by_prompt_builder"):
        problems.append(f"{path}: preflight.target_values_not_rendered_by_prompt_builder is not true")
    for row in record["predictions"]:
        tb = row.get("target_blindness")
        if tb is None:
            problems.append(f"{path}:{row['row_id'][:8]}: no target_blindness block")
            continue
        if tb["blocked_target_access_attempts"] != 0:
            problems.append(f"{path}:{row['row_id'][:8]}: {tb['blocked_target_access_attempts']} blocked target accesses")
        if tb["target_read_before_reveal"]:
            problems.append(f"{path}:{row['row_id'][:8]}: target read before reveal")
        extra = set(tb["keys_read_before_reveal"]) - ALLOWED_KEYS
        if extra:
            problems.append(f"{path}:{row['row_id'][:8]}: unexpected keys read before reveal {sorted(extra)}")
        if tb["revealed_phase"] != "parser_audit_after_state_and_prompt_construction":
            problems.append(f"{path}:{row['row_id'][:8]}: reveal phase {tb['revealed_phase']!r}")
        if tb["state_provenance"].get("state_derived_from_input_only") is not True:
            problems.append(f"{path}:{row['row_id'][:8]}: provenance {tb['state_provenance']}")
    return problems


def study_statistics(rows_by_endpoint: dict[str, list[dict]], panels: list[str], tasks: set[str]) -> dict[str, Any]:
    endpoints = list(rows_by_endpoint)
    out = {"endpoint_gain": {e: _scope(rows_by_endpoint[e], panels, tasks)["gain"] for e in endpoints}}
    gains = {e: _panel_values(rows_by_endpoint[e], panels, tasks, "gain") for e in endpoints}
    adjacent, raw = [], []
    for low, high in zip(endpoints[:-1], endpoints[1:], strict=True):
        inc = [b - a for a, b in zip(gains[low], gains[high], strict=True)]
        p = _one_sided_positive_p(inc)
        raw.append(p)
        adjacent.append({"lower_endpoint": low, "upper_endpoint": high, "gain_increment": cluster_summary(inc), "one_sided_p_unadjusted": p})
    for a, ph in zip(adjacent, holm_adjust(raw), strict=True):
        a["one_sided_p_holm"] = ph
        a["passes_registered_contrast"] = a["gain_increment"]["mean"] > 0 and ph < 0.05
    out["adjacent_scale_contrasts"] = adjacent
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts_revision/target_blindness_2026-09"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root
    pre = json.loads((root / "PREREGISTRATION.json").read_text())
    output = args.output or root / "TARGET_BLINDNESS_AUDIT.json"

    # 1. per-row blindness
    problems: list[str] = []
    rows_total = 0
    cells_seen = 0
    for cell in pre["cells"]:
        path = Path(cell["output"])
        if not path.exists():
            problems.append(f"missing cell record: {path}")
            continue
        record = json.loads(path.read_text())
        cells_seen += 1
        rows_total += len(record["predictions"])
        problems += audit_rows(record, str(path))

    # 2. same-instance controls (instrumented vs uninstrumented)
    controls = []
    for ctl in pre["controls"]:
        a, b = Path(ctl["output"]), Path(ctl["compared_with"])
        if not (a.exists() and b.exists()):
            controls.append({"id": ctl["id"], "status": "missing", "uninstrumented": str(a), "instrumented": str(b)})
            continue
        result = compare_records(a, b)
        controls.append({"id": ctl["id"], "status": "compared", "verdicts": result["verdicts"],
                         "rows_differing_per_field": result["ladder"]["1_discrete"]["rows_differing_per_field"],
                         "uninstrumented": {"path": str(a), "sha256": sha256_file(a)}, "instrumented": {"path": str(b), "sha256": sha256_file(b)}})
    controls_identical = bool(controls) and all(c["status"] == "compared" and all(c["verdicts"].values()) for c in controls)

    # 3. cross-architecture sensitivity on all 60 cells + recomputed study statistics on the A100
    sensitivity = []
    per_study_rows: dict[str, dict[str, list[dict]]] = {}
    per_study_panels: dict[str, list[str]] = {}
    for cell in pre["cells"]:
        path = Path(cell["output"])
        if not path.exists():
            continue
        result = compare_records(Path(cell["reference"]["path"]), path)
        disc = result["ladder"]["1_discrete"]
        stats = result["ladder"]["3_panel_precision"]["statistics"]
        sensitivity.append({
            "id": cell["id"], "study": cell["study"], "endpoint": cell["endpoint"],
            "rows_with_changed_score": {arm: disc["rows_differing_per_field"].get(f"{arm}_score", 0) for arm in ("foundation", "ascent")},
            "rows_with_changed_answer": {arm: disc["rows_differing_per_field"].get(f"{arm}_answer", 0) for arm in ("foundation", "ascent")},
            "gain_reference": stats["gain.mean"]["reference"], "gain_a100": stats["gain.mean"]["rerun"], "gain_shift": stats["gain.mean"]["rerun"] - stats["gain.mean"]["reference"],
            "binary64_identical": result["verdicts"]["binary64_identical"],
        })
        record = json.loads(path.read_text())
        panel = record["data"]["path"].split("/")[-1].removesuffix(".jsonl")
        per_study_rows.setdefault(cell["study"], {}).setdefault(cell["endpoint"], []).extend({"panel": panel, **r} for r in record["predictions"])
        per_study_panels.setdefault(cell["study"], [])
        if panel not in per_study_panels[cell["study"]]:
            per_study_panels[cell["study"]].append(panel)
    study_stats = {}
    for study, rows_by_endpoint in per_study_rows.items():
        panels = sorted(per_study_panels[study], key=lambda p: int(p.rsplit("_", 1)[1]))
        a100 = study_statistics(rows_by_endpoint, panels, {"qa2", "qa3"})
        ref_path = {"official16k": "artifacts/remote_results/babilong_qwen_canonical_16k_confirmation_4fd25e9/analysis.json",
                    "semantic_holdout": "artifacts/remote_results/babilong_qwen_canonical_semantic_holdout_b7480e6/analysis.json"}[study]
        ref = json.loads(Path(ref_path).read_text())
        study_stats[study] = {
            "panels": len(panels),
            "endpoint_gain_reference": {e: ref["endpoint_results"][e]["primary"]["gain"]["mean"] for e in rows_by_endpoint},
            "endpoint_gain_a100": {e: a100["endpoint_gain"][e]["mean"] for e in rows_by_endpoint},
            "endpoint_gain_a100_ci95": {e: [a100["endpoint_gain"][e]["ci95_low"], a100["endpoint_gain"][e]["ci95_high"]] for e in rows_by_endpoint},
            "adjacent_contrasts_reference": [{"increment": c["gain_increment"]["mean"], "p_holm": c["one_sided_p_holm"]} for c in ref["adjacent_scale_contrasts"]],
            "adjacent_contrasts_a100": [{"increment": c["gain_increment"]["mean"], "p_holm": c["one_sided_p_holm"], "passes": c["passes_registered_contrast"]} for c in a100["adjacent_scale_contrasts"]],
            "registered_directional_family_passes_on_a100": all(c["passes_registered_contrast"] for c in a100["adjacent_scale_contrasts"]),
        }
    total_changed = sum(sum(s["rows_with_changed_score"].values()) for s in sensitivity)
    summary = {
        "schema_version": 1,
        "preregistration_sha256": sha256_file(root / "PREREGISTRATION.json"),
        "cells_expected": len(pre["cells"]), "cells_present": cells_seen,
        "rows_total": rows_total, "row_problems": problems,
        "all_rows_target_blind": cells_seen == len(pre["cells"]) and not problems,
        "same_instance_controls": controls, "same_instance_controls_identical": controls_identical,
        "cross_architecture": {
            "cells": sensitivity,
            "rows_with_changed_score_total": total_changed,
            "rows_total": rows_total,
            "max_abs_gain_shift_points": 100 * max(abs(s["gain_shift"]) for s in sensitivity) if sensitivity else None,
            "cells_binary64_identical": sum(1 for s in sensitivity if s["binary64_identical"]),
            "study_statistics": study_stats,
        },
    }
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("cross_architecture", "same_instance_controls")}, indent=2))
    for c in controls:
        print("control", c["id"], c["status"], c.get("verdicts"))
    print(f"cross-architecture: {total_changed} of {rows_total} scored outputs changed; max |gain shift| {summary['cross_architecture']['max_abs_gain_shift_points']:.2f} pts; binary64-identical cells {summary['cross_architecture']['cells_binary64_identical']}/{len(sensitivity)}")
    for study, st in study_stats.items():
        print(study, "gains ref", {k: round(100 * v, 2) for k, v in st["endpoint_gain_reference"].items()}, "a100", {k: round(100 * v, 2) for k, v in st["endpoint_gain_a100"].items()},
              "p_holm ref", [round(c["p_holm"], 5) for c in st["adjacent_contrasts_reference"]], "a100", [round(c["p_holm"], 5) for c in st["adjacent_contrasts_a100"]])
    print(f"written: {output}  sha256={sha256_file(output)}")


if __name__ == "__main__":
    main()
