#!/usr/bin/env python3
"""Verify that cached greedy decoding reproduces no-cache RULER outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FIELDS = (
    "outputs",
    "foundation_prediction",
    "ascent_prediction",
    "foundation_score",
    "ascent_score",
    "ascent_evidence_values",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(reference_path: Path, cached_path: Path) -> dict[str, Any]:
    reference = json.loads(reference_path.read_text())
    cached = json.loads(cached_path.read_text())
    reference_rows = {int(row["index"]): row for row in reference["predictions"]}
    comparisons = []
    for cached_row in cached["predictions"]:
        index = int(cached_row["index"])
        reference_row = reference_rows.get(index)
        field_matches = {
            field: reference_row is not None
            and reference_row.get(field) == cached_row.get(field)
            for field in FIELDS
        }
        comparisons.append(
            {
                "index": index,
                "field_matches": field_matches,
                "all_fields_match": all(field_matches.values()),
            }
        )
    gates = {
        "cached_runtime_flag_enabled": cached["runtime"].get(
            "foundation_decode_cache"
        )
        is True,
        "reference_runtime_flag_disabled": reference["runtime"].get(
            "foundation_decode_cache", False
        )
        is False,
        "same_endpoint": cached["endpoint"]["repo_id"]
        == reference["endpoint"]["repo_id"],
        "same_task_sha256": cached["task_sha256"] == reference["task_sha256"],
        "nonempty_comparison": bool(comparisons),
        "every_compared_prediction_identical": all(
            row["all_fields_match"] for row in comparisons
        ),
    }
    return {
        "schema_version": 1,
        "status": "passed" if all(gates.values()) else "failed",
        "reference_path": str(reference_path),
        "reference_sha256": sha256_file(reference_path),
        "cached_path": str(cached_path),
        "cached_sha256": sha256_file(cached_path),
        "compared_rows": len(comparisons),
        "comparisons": comparisons,
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cached", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.cached)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["status"] != "passed":
        raise SystemExit("foundation cache equivalence failed")


if __name__ == "__main__":
    main()
