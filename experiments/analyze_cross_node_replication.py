#!/usr/bin/env python3
"""Compare a frozen BABILong endpoint rerun with its archived reference."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any


REFERENCE_TEMPLATE = (
    "./confirmation/node2/"
    "panel{panel}_smollm2-360m-instruct_scale_relevant.json"
)
REPLICATION_TEMPLATE = (
    "panel{panel}_smollm2-360m-instruct_scale_relevant.json"
)


def _load_archive_json(archive: Path, member: str) -> dict[str, Any]:
    with tarfile.open(archive, "r:gz") as stream:
        extracted = stream.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(member)
        return json.load(extracted)


def _prediction_signature(result: dict[str, Any]) -> list[dict[str, Any]]:
    keys = (
        "row_id",
        "task",
        "target",
        "foundation_output",
        "foundation_location",
        "foundation_score",
        "ascent_output",
        "ascent_location",
        "ascent_score",
        "supporting_fact_count",
        "retained_fact_count",
        "retained_facts",
        "foundation_prompt_tokens",
        "ascent_prompt_tokens",
    )
    return [{key: row[key] for key in keys} for row in result["predictions"]]


def analyze(reference_archive: Path, replication_root: Path) -> dict[str, Any]:
    panels: list[dict[str, Any]] = []
    for panel in (1, 2, 3):
        reference = _load_archive_json(
            reference_archive, REFERENCE_TEMPLATE.format(panel=panel)
        )
        replication_path = replication_root / REPLICATION_TEMPLATE.format(
            panel=panel
        )
        replication = json.loads(replication_path.read_text())
        metric_keys = ("foundation", "ascent", "gain", "wins", "regressions")
        metrics_equal = all(
            replication[key] == reference[key] for key in metric_keys
        )
        predictions_equal = (
            _prediction_signature(replication) == _prediction_signature(reference)
        )
        panels.append(
            {
                "panel": panel,
                "reference_data_sha256": reference["data"]["sha256"],
                "replication_data_sha256": replication["data"]["sha256"],
                "reference_git_commit": reference["environment"]["git_commit"],
                "replication_git_commit": replication["environment"]["git_commit"],
                "replication_worktree_clean": not replication["environment"][
                    "git_dirty"
                ],
                "model_files_equal": (
                    replication["model_files"] == reference["model_files"]
                ),
                "metrics_equal": metrics_equal,
                "predictions_and_token_counts_equal": predictions_equal,
                "reference_gain": reference["gain"]["mean"],
                "replication_gain": replication["gain"]["mean"],
                "reference_decode_ratio": reference["systems"][
                    "ascent_to_foundation_decode_time_ratio"
                ],
                "replication_decode_ratio": replication["systems"][
                    "ascent_to_foundation_decode_time_ratio"
                ],
            }
        )
    gates = {
        "all_panel_hashes_equal": all(
            row["reference_data_sha256"] == row["replication_data_sha256"]
            for row in panels
        ),
        "all_replication_worktrees_clean": all(
            row["replication_worktree_clean"] for row in panels
        ),
        "all_model_files_equal": all(row["model_files_equal"] for row in panels),
        "all_metrics_equal": all(row["metrics_equal"] for row in panels),
        "all_predictions_and_token_counts_equal": all(
            row["predictions_and_token_counts_equal"] for row in panels
        ),
    }
    gates["cross_node_replication_pass"] = all(gates.values())
    return {"schema_version": 1, "panels": panels, "gates": gates}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-archive", type=Path, required=True)
    parser.add_argument("--replication-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.reference_archive, args.replication_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
