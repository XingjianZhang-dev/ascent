#!/usr/bin/env python3
"""Compare the Qwen3-8B third-instance audit rerun against the formal records.

Phase 0 (Array revision 1), questions Q1/Q2, escalation level L2: a CPU-only,
field-by-field, row-by-row diff of the ten retained
``artifacts/array_third_node_audit/qwen3-8b-panel{N}-audit-rerun.json`` records
against the ten formal
``artifacts/around7b_formal/qwen3-8b/canonical_slots_4/canonical_16k_confirmation_panel_{N}.json``
records that feed the reported Qwen3-8B number.

The comparison target is fixed here, before any rerun: every retained
scientific field and every prediction row must be equal. Provenance fields
(timestamps, wall time, status string, git commit, dirty flag, systems timing,
``diagnostic_no_promotion``) are reported but are not equality criteria.

Nothing under ``artifacts/`` is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCIENTIFIC_TOP_LEVEL_FIELDS = (
    "schema_version",
    "experiment",
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
    "retrieval_cache",
)

# ``condition`` is scientific except for the ``diagnostic_no_promotion`` flag,
# which is a bookkeeping marker set on audit reruns so they cannot be promoted.
CONDITION_PROVENANCE_KEYS = ("diagnostic_no_promotion",)

PROVENANCE_FIELDS = (
    "status",
    "started_utc",
    "completed_utc",
    "wall_seconds",
    "end_to_end_seconds",
    "systems",
    "environment",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compare_panel(formal_path: Path, audit_path: Path) -> dict[str, Any]:
    formal = json.loads(formal_path.read_text())
    audit = json.loads(audit_path.read_text())

    top_level = {
        field: canonical(formal.get(field)) == canonical(audit.get(field))
        for field in SCIENTIFIC_TOP_LEVEL_FIELDS
    }
    formal_condition = {
        k: v for k, v in formal["condition"].items() if k not in CONDITION_PROVENANCE_KEYS
    }
    audit_condition = {
        k: v for k, v in audit["condition"].items() if k not in CONDITION_PROVENANCE_KEYS
    }
    top_level["condition_excluding_provenance_keys"] = canonical(formal_condition) == canonical(
        audit_condition
    )

    formal_rows = formal["predictions"]
    audit_rows = audit["predictions"]
    if len(formal_rows) != len(audit_rows):
        raise RuntimeError(
            f"row count differs: {formal_path.name}={len(formal_rows)} "
            f"{audit_path.name}={len(audit_rows)}"
        )
    row_keys = sorted(set().union(*(set(r) for r in formal_rows + audit_rows)))
    per_field_mismatch_counts = {key: 0 for key in row_keys}
    mismatching_row_ids: list[str] = []
    for formal_row, audit_row in zip(formal_rows, audit_rows, strict=True):
        row_equal = True
        for key in row_keys:
            if canonical(formal_row.get(key)) != canonical(audit_row.get(key)):
                per_field_mismatch_counts[key] += 1
                row_equal = False
        if not row_equal:
            mismatching_row_ids.append(formal_row.get("row_id"))
    row_ids_same_order = all(
        f.get("row_id") == a.get("row_id") for f, a in zip(formal_rows, audit_rows, strict=True)
    )
    whole_predictions_equal = canonical(formal_rows) == canonical(audit_rows)

    provenance = {
        field: {"formal": formal.get(field), "audit": audit.get(field)}
        for field in PROVENANCE_FIELDS
    }
    provenance["condition.diagnostic_no_promotion"] = {
        "formal": formal["condition"].get("diagnostic_no_promotion"),
        "audit": audit["condition"].get("diagnostic_no_promotion"),
    }

    return {
        "formal_path": str(formal_path),
        "formal_sha256": sha256_file(formal_path),
        "audit_path": str(audit_path),
        "audit_sha256": sha256_file(audit_path),
        "rows": len(formal_rows),
        "row_ids_same_order": row_ids_same_order,
        "top_level_scientific_fields_equal": top_level,
        "all_top_level_scientific_fields_equal": all(top_level.values()),
        "predictions_equal": whole_predictions_equal,
        "prediction_field_mismatch_counts": {
            k: v for k, v in per_field_mismatch_counts.items() if v
        },
        "mismatching_row_ids": mismatching_row_ids,
        "provenance_not_compared": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/THIRD_NODE_AUDIT_ROW_DIFF_2026-09-20.json"),
    )
    args = parser.parse_args()

    formal_dir = args.root / "artifacts/around7b_formal/qwen3-8b/canonical_slots_4"
    audit_dir = args.root / "artifacts/array_third_node_audit"
    panels = []
    for n in range(1, 11):
        panels.append(
            compare_panel(
                formal_dir / f"canonical_16k_confirmation_panel_{n}.json",
                audit_dir / f"qwen3-8b-panel{n}-audit-rerun.json",
            )
        )

    total_rows = sum(p["rows"] for p in panels)
    all_equal = all(
        p["all_top_level_scientific_fields_equal"] and p["predictions_equal"]
        for p in panels
    )
    commits = {
        "formal": sorted({p["provenance_not_compared"]["environment"]["formal"]["git_commit"] for p in panels}),
        "audit": sorted({p["provenance_not_compared"]["environment"]["audit"]["git_commit"] for p in panels}),
    }
    dirty = {
        "formal": sorted({p["provenance_not_compared"]["environment"]["formal"]["git_dirty"] for p in panels}),
        "audit": sorted({p["provenance_not_compared"]["environment"]["audit"]["git_dirty"] for p in panels}),
    }
    statuses = {
        "formal": sorted({p["provenance_not_compared"]["status"]["formal"] for p in panels}),
        "audit": sorted({p["provenance_not_compared"]["status"]["audit"] for p in panels}),
    }
    result = {
        "comparison": "qwen3-8b formal canonical_slots_4 vs array_third_node_audit rerun",
        "escalation_level": "L2_recomputation_cpu_only",
        "panels": 10,
        "total_rows": total_rows,
        "all_scientific_fields_and_predictions_equal": all_equal,
        "status": "identical" if all_equal else "DIFFERENT",
        "git_commits": commits,
        "git_dirty": dirty,
        "status_strings": statuses,
        "per_panel": panels,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_panel"}, indent=2))
    for p in panels:
        print(
            f"panel {Path(p['formal_path']).stem[-2:].lstrip('_')}: rows={p['rows']} "
            f"top_level={p['all_top_level_scientific_fields_equal']} "
            f"predictions={p['predictions_equal']} "
            f"mismatch_fields={p['prediction_field_mismatch_counts'] or '{}'}"
        )
    print(f"written: {args.output}  sha256={sha256_file(args.output)}")


if __name__ == "__main__":
    main()
