#!/usr/bin/env python3
"""Audit target-blind BGE retrieval caches against parser-derived support facts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ascent.babilong_memory import read_babilong
from experiments.run_natural_repeat import sha256_file


def audit_rows(
    data_rows: list[dict[str, Any]],
    cache_rows: list[dict[str, Any]],
    *,
    slot_counts: tuple[int, ...] = (1, 2, 3),
) -> dict[str, Any]:
    cache_by_id = {row["row_id"]: row for row in cache_rows}
    if len(cache_by_id) != len(cache_rows):
        raise RuntimeError("duplicate row_id in retrieval cache")
    if set(cache_by_id) != {row["row_id"] for row in data_rows}:
        raise RuntimeError("retrieval cache row IDs do not match data")

    summaries: dict[str, Any] = {}
    for slots in slot_counts:
        by_task: dict[str, list[tuple[bool, float]]] = defaultdict(list)
        rows: list[dict[str, Any]] = []
        for row in data_rows:
            read = read_babilong(row["input"], row["question"])
            selected = [
                item["passage"]
                for item in cache_by_id[row["row_id"]]["ranked_passages"][:slots]
            ]
            found = [
                any(fact.source in passage for passage in selected)
                for fact in read.facts
            ]
            exact = all(found)
            recall = sum(found) / len(found)
            by_task[read.task].append((exact, recall))
            rows.append(
                {
                    "row_id": row["row_id"],
                    "task": read.task,
                    "supporting_fact_count": len(found),
                    "support_found": found,
                    "exact_support_recall": exact,
                    "support_fact_recall": recall,
                }
            )
        summaries[str(slots)] = {
            "rows": len(rows),
            "exact_support_recall": sum(item["exact_support_recall"] for item in rows)
            / len(rows),
            "support_fact_recall": sum(item["support_fact_recall"] for item in rows)
            / len(rows),
            "by_task": {
                task: {
                    "rows": len(values),
                    "exact_support_recall": sum(exact for exact, _ in values)
                    / len(values),
                    "support_fact_recall": sum(recall for _, recall in values)
                    / len(values),
                }
                for task, values in sorted(by_task.items())
            },
            "row_diagnostics": rows,
        }
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data_rows = [json.loads(line) for line in args.data.read_text().splitlines()]
    cache_document = json.loads(args.cache.read_text())
    if cache_document["data_sha256"] != sha256_file(args.data):
        raise RuntimeError("retrieval cache data hash mismatch")
    result = {
        "schema_version": 1,
        "data": str(args.data),
        "data_sha256": sha256_file(args.data),
        "cache": str(args.cache),
        "cache_sha256": sha256_file(args.cache),
        "protocol_sha256": cache_document["protocol_sha256"],
        "definition": "A support fact is recalled only when its exact parser-derived source event occurs in a selected generic passage.",
        "slots": audit_rows(data_rows, cache_document["rows"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for slots, summary in result["slots"].items():
        print(
            f"slots={slots} exact={summary['exact_support_recall']:.4f} "
            f"fact_recall={summary['support_fact_recall']:.4f}"
        )


if __name__ == "__main__":
    main()
