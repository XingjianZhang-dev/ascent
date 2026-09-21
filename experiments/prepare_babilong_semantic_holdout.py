#!/usr/bin/env python3
"""Freeze BABILong panels after score-blind semantic-overlap exclusion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ascent.babilong_memory import extract_babilong_events
from experiments.prepare_babilong_panel import canonical_bytes, sha256_file, stable_rows


def semantic_fingerprint(task: str, row: dict[str, Any]) -> str:
    events = [
        (event.kind, event.person, event.object_name, event.location, event.recipient)
        for event in extract_babilong_events(row["input"])
    ]
    payload = [task, row["question"].strip().lower(), events]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--exclusion-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--panels", type=int, default=10)
    parser.add_argument("--rows-per-task", type=int, default=40)
    parser.add_argument("--tasks", nargs="+", default=["qa2", "qa3"])
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    selected_by_task: dict[str, list[dict[str, Any]]] = {}
    audit: dict[str, Any] = {}
    required = args.panels * args.rows_per_task
    for task in args.tasks:
        candidate_path = args.candidate_root / f"{task}.jsonl"
        exclusion_path = args.exclusion_root / f"{task}.jsonl"
        excluded = {
            semantic_fingerprint(task, row) for row in stable_rows(exclusion_path)
        }
        candidates = stable_rows(candidate_path)
        retained = [
            row
            for row in candidates
            if semantic_fingerprint(task, row) not in excluded
        ]
        if len(retained) < required:
            raise RuntimeError(f"too few semantic-holdout rows for {task}")
        selected_by_task[task] = retained[:required]
        audit[task] = {
            "candidate_path": str(candidate_path),
            "candidate_sha256": sha256_file(candidate_path),
            "candidate_rows": len(candidates),
            "exclusion_path": str(exclusion_path),
            "exclusion_sha256": sha256_file(exclusion_path),
            "exclusion_rows": len(excluded),
            "overlap_rows_excluded": len(candidates) - len(retained),
            "eligible_rows": len(retained),
        }

    panels: list[dict[str, Any]] = []
    for panel_index in range(args.panels):
        name = f"canonical_train_8k_semantic_panel_{panel_index + 1}"
        rows: list[dict[str, Any]] = []
        start = panel_index * args.rows_per_task
        stop = start + args.rows_per_task
        for task in args.tasks:
            for row in selected_by_task[task][start:stop]:
                rows.append(
                    {
                        "row_id": hashlib.sha256(canonical_bytes(row)).hexdigest(),
                        "task": task,
                        "input": row["input"],
                        "question": row["question"],
                        "target": row["target"].strip().lower(),
                    }
                )
        output = args.output_root / f"{name}.jsonl"
        output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
        panels.append(
            {
                "panel": name,
                "path": output.name,
                "start_in_filtered_stable_hash_order": start,
                "rows_per_task": args.rows_per_task,
                "tasks": args.tasks,
                "rows": len(rows),
                "sha256": sha256_file(output),
            }
        )
    manifest = {
        "selection": (
            "Exclude any candidate whose task, normalized question, and parsed "
            "event sequence matches the pinned official test pool; then take "
            "disjoint blocks in SHA-256 canonical-row order. No target value or "
            "model score enters filtering or ordering."
        ),
        "semantic_overlap_after_selection": 0,
        "source_audit": audit,
        "panels": panels,
    }
    output = args.output_root / "MANIFEST.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
