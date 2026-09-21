#!/usr/bin/env python3
"""Account for the three around-7B result files excluded from the formal family.

Phase 0 (Array revision 1), question Q4. ``audit_formal_validation_records.py``
skips any path containing a ``diagnostics`` component, so the formal family
holds 70 files while ``artifacts/around7b_formal/`` holds 73 result JSONs. This
script identifies the three excluded files, states what each is, and checks
them field by field and row by row against the formal record for the same
endpoint and panel. Nothing under ``artifacts/`` is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DIAGNOSTIC_DIR = Path("artifacts/around7b_formal/diagnostics/ascent_first_batch1")
FORMAL_PAIRS = {
    "qwen2p5-7b-panel1.json": "artifacts/around7b_formal/qwen2p5-7b/canonical_slots_4/canonical_16k_confirmation_panel_1.json",
    "mistral-7b-panel1.json": "artifacts/around7b_formal/mistral-7b/canonical_slots_4/canonical_16k_confirmation_panel_1.json",
    "qwen3-8b-panel1.json": "artifacts/around7b_formal/qwen3-8b/canonical_slots_4/canonical_16k_confirmation_panel_1.json",
}
SCIENTIFIC_FIELDS = (
    "endpoint",
    "config",
    "data",
    "model_files",
    "model_identity",
    "parser_accuracy",
    "foundation",
    "ascent",
    "gain",
    "remaining_error_elimination",
    "wins",
    "regressions",
    "by_task",
    "state",
    "predictions",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/AROUND7B_DIAGNOSTICS_EXCLUSION_AUDIT_2026-09-20.json"),
    )
    args = parser.parse_args()
    root = args.root

    all_results = sorted(
        p for p in (root / "artifacts/around7b_formal").rglob("*.json") if p.name != "analysis.json"
    )
    excluded = sorted(p for p in all_results if "diagnostics" in p.parts)
    formal = [p for p in all_results if "diagnostics" not in p.parts]
    if len(excluded) != 3 or len(formal) != 70:
        raise RuntimeError(f"expected 3 excluded + 70 formal files, found {len(excluded)} + {len(formal)}")

    files = []
    for diag_path in excluded:
        formal_path = root / FORMAL_PAIRS[diag_path.name]
        diag = json.loads(diag_path.read_text())
        form = json.loads(formal_path.read_text())
        diag_condition = {k: v for k, v in diag["condition"].items() if k != "diagnostic_no_promotion"}
        form_condition = {k: v for k, v in form["condition"].items() if k != "diagnostic_no_promotion"}
        field_equal = {f: canonical(diag[f]) == canonical(form[f]) for f in SCIENTIFIC_FIELDS}
        field_equal["condition_excluding_diagnostic_flag"] = canonical(diag_condition) == canonical(form_condition)
        rows_differing = sum(
            1 for a, b in zip(diag["predictions"], form["predictions"], strict=True) if canonical(a) != canonical(b)
        )
        files.append(
            {
                "excluded_path": str(diag_path.relative_to(root)),
                "excluded_sha256": sha256_file(diag_path),
                "status": diag["status"],
                "diagnostic_no_promotion": diag["condition"]["diagnostic_no_promotion"],
                "endpoint": diag["endpoint"]["name"],
                "panel": diag["data"]["path"],
                "rows": len(diag["predictions"]),
                "what_it_is": (
                    "post-hoc decode-order diagnostic: identical frozen config, panel, "
                    "checkpoint and batch size 1, but the two arms decoded in the order "
                    f"'{diag['systems']['decode_order']}' instead of the frozen "
                    f"'{form['systems']['decode_order']}'"
                ),
                "decode_order": {"excluded": diag["systems"]["decode_order"], "formal": form["systems"]["decode_order"]},
                "git": {
                    "excluded": (diag["environment"]["git_commit"], diag["environment"]["git_dirty"]),
                    "formal": (form["environment"]["git_commit"], form["environment"]["git_dirty"]),
                },
                "started_utc": {"excluded": diag["started_utc"], "formal": form["started_utc"]},
                "formal_counterpart": str(formal_path.relative_to(root)),
                "formal_counterpart_sha256": sha256_file(formal_path),
                "scientific_fields_equal": field_equal,
                "all_scientific_fields_equal": all(field_equal.values()),
                "prediction_rows_differing": rows_differing,
                "is_partial_or_abandoned_run": False,
                "any_reported_number_depends_on_it": False,
            }
        )

    result = {
        "question": "Q4: the three around-7B result files excluded from the 70-file formal family",
        "formal_family_files": len(formal),
        "excluded_files": len(excluded),
        "exclusion_rule": "audit_formal_validation_records.py skips paths with a 'diagnostics' component",
        "all_excluded_files_reproduce_their_formal_counterpart": all(f["all_scientific_fields_equal"] for f in files),
        "files": files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for f in files:
        print(
            f"{f['excluded_path']}: {f['endpoint']} decode_order={f['decode_order']['excluded']} "
            f"rows={f['rows']} all_equal={f['all_scientific_fields_equal']} rows_differing={f['prediction_rows_differing']}"
        )
    print(f"written: {args.output}  sha256={sha256_file(args.output)}")


if __name__ == "__main__":
    main()
