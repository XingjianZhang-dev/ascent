#!/usr/bin/env python3
"""Compare the frozen 7B/K=5 factorial rerun across two physical nodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


SCIENTIFIC_FIELDS = (
    "config_sha256",
    "data_sha256",
    "panel",
    "rounds",
    "endpoint",
    "model_files",
    "model_identity",
    "preflight",
    "foundation_nll",
    "ascent_nll",
    "gain_nll",
    "foundation_accuracy",
    "ascent_accuracy",
    "predictions",
)

ROW_FLOAT_FIELDS = ("foundation_nll", "ascent_nll", "gain_nll")
ROW_VECTOR_FIELDS = ("foundation_probabilities", "ascent_probabilities")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary64_equal(left: float, right: float) -> bool:
    return struct.pack(">d", float(left)) == struct.pack(">d", float(right))


def compare(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = json.loads(left_path.read_text())
    right = json.loads(right_path.read_text())
    field_equality = {field: left[field] == right[field] for field in SCIENTIFIC_FIELDS}

    left_rows = left["predictions"]
    right_rows = right["predictions"]
    if len(left_rows) != len(right_rows):
        raise RuntimeError("cross-node prediction row counts differ")

    row_ids_match = all(
        left_row["row_id"] == right_row["row_id"]
        for left_row, right_row in zip(left_rows, right_rows, strict=True)
    )
    scalar_binary64_equal = all(
        binary64_equal(left_row[field], right_row[field])
        for left_row, right_row in zip(left_rows, right_rows, strict=True)
        for field in ROW_FLOAT_FIELDS
    )
    vector_binary64_equal = all(
        len(left_row[field]) == len(right_row[field])
        and all(
            binary64_equal(left_value, right_value)
            for left_value, right_value in zip(
                left_row[field], right_row[field], strict=True
            )
        )
        for left_row, right_row in zip(left_rows, right_rows, strict=True)
        for field in ROW_VECTOR_FIELDS
    )
    probability_values_per_arm = sum(
        len(row["foundation_probabilities"]) for row in left_rows
    )
    vector_lengths = sorted(
        {
            len(row[field])
            for row in left_rows
            for field in ROW_VECTOR_FIELDS
        }
    )
    audit_pass = (
        all(field_equality.values())
        and row_ids_match
        and scalar_binary64_equal
        and vector_binary64_equal
        and vector_lengths == [15]
    )
    return {
        "schema_version": 1,
        "audit": "qwen2p5_7b_k5_factorial_cross_node_exact_comparison",
        "status": "passed" if audit_pass else "failed",
        "left": {"path": str(left_path), "sha256": sha256_file(left_path)},
        "right": {"path": str(right_path), "sha256": sha256_file(right_path)},
        "identity": {
            "panel": left["panel"],
            "rounds": left["rounds"],
            "endpoint": left["endpoint"]["name"],
            "config_sha256": left["config_sha256"],
            "data_sha256": left["data_sha256"],
            "left_hostname": left["environment"]["hostname"],
            "right_hostname": right["environment"]["hostname"],
            "left_git_commit": left["environment"]["git_commit"],
            "right_git_commit": right["environment"]["git_commit"],
        },
        "counts": {
            "prediction_rows": len(left_rows),
            "probability_vectors_per_arm": len(left_rows),
            "probability_vectors_both_arms": 2 * len(left_rows),
            "probability_vector_lengths": vector_lengths,
            "candidate_probabilities_per_row_per_arm": len(
                left_rows[0]["foundation_probabilities"]
            ),
            "probability_values_per_arm": probability_values_per_arm,
            "probability_values_both_arms": 2 * probability_values_per_arm,
            "row_level_nll_values": len(left_rows) * len(ROW_FLOAT_FIELDS),
        },
        "comparison": {
            "scientific_field_equality": field_equality,
            "row_ids_and_order_equal": row_ids_match,
            "serialized_json_scientific_records_equal": left["predictions"]
            == right["predictions"],
            "probability_vectors_binary64_equal_after_json_parse": vector_binary64_equal,
            "row_nll_binary64_equal_after_json_parse": scalar_binary64_equal,
            "audit_pass": audit_pass,
        },
        "exact_definition": (
            "The two saved JSON records have identical scientific fields. After JSON "
            "parsing, every stored probability and row-level NLL has the same IEEE-754 "
            "binary64 byte representation. Runtime-only fields such as timestamps, "
            "hostnames, audit_rerun, and wall time are intentionally excluded."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.left, args.right)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    raise SystemExit(0 if result["comparison"]["audit_pass"] else 1)


if __name__ == "__main__":
    main()
