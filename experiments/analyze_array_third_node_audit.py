#!/usr/bin/env python3
"""Verify the posthoc third-node Qwen3-8B audit against formal outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


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
    "retrieval_cache",
    "predictions",
)

CONDITION_FIELDS = (
    "name",
    "memory_source",
    "memory_representation",
    "neutral_answer_format",
    "readout_path",
    "fact_slots",
    "frozen_batch_size",
    "effective_batch_size",
)

RUNTIME_FIELDS = (
    "python",
    "torch",
    "transformers",
    "tokenizers",
    "sentencepiece",
    "safetensors",
    "cuda_device",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--rerun-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    panels = []
    total_predictions = 0
    total_foundation_correct = 0
    total_ascent_correct = 0
    total_wins = 0
    total_regressions = 0

    for panel in range(1, 11):
        original_path = args.original_dir / f"canonical_16k_confirmation_panel_{panel}.json"
        rerun_path = args.rerun_dir / f"qwen3-8b-panel{panel}-audit-rerun.json"
        original = json.loads(original_path.read_text(encoding="utf-8"))
        rerun = json.loads(rerun_path.read_text(encoding="utf-8"))

        unequal = [
            field for field in SCIENTIFIC_FIELDS
            if original.get(field) != rerun.get(field)
        ]
        if unequal:
            raise RuntimeError(f"panel {panel} scientific mismatch: {unequal}")
        for field in CONDITION_FIELDS:
            if original["condition"].get(field) != rerun["condition"].get(field):
                raise RuntimeError(f"panel {panel} condition mismatch: {field}")
        if original["condition"].get("diagnostic_no_promotion") is not False:
            raise RuntimeError(f"panel {panel} original result lacks promotion status")
        if rerun["condition"].get("diagnostic_no_promotion") is not True:
            raise RuntimeError(f"panel {panel} rerun lacks audit-only status")
        for field in RUNTIME_FIELDS:
            if original["environment"].get(field) != rerun["environment"].get(field):
                raise RuntimeError(f"panel {panel} runtime mismatch: {field}")

        rows = len(rerun["predictions"])
        total_predictions += rows
        total_foundation_correct += sum(
            int(row["foundation_score"]) for row in rerun["predictions"]
        )
        total_ascent_correct += sum(
            int(row["ascent_score"]) for row in rerun["predictions"]
        )
        total_wins += rerun["wins"]
        total_regressions += rerun["regressions"]
        panels.append(
            {
                "panel": panel,
                "rows": rows,
                "foundation_accuracy": rerun["foundation"]["mean"],
                "ascent_accuracy": rerun["ascent"]["mean"],
                "gain": rerun["gain"]["mean"],
                "wins": rerun["wins"],
                "regressions": rerun["regressions"],
                "original_sha256": sha256(original_path),
                "rerun_sha256": sha256(rerun_path),
                "prediction_rows_exact": True,
                "scientific_fields_exact": True,
            }
        )

    if total_predictions != 800:
        raise RuntimeError(f"expected 800 predictions, found {total_predictions}")

    result = {
        "schema_version": 1,
        "experiment": "array_qwen3_8b_third_node_posthoc_audit",
        "status": "audit_rerun_ineligible_for_independence_or_promotion_claims",
        "completed_utc_date": "2026-08-23",
        "endpoint": "qwen3-8b-nonthinking",
        "panels": panels,
        "summary": {
            "panels": len(panels),
            "predictions": total_predictions,
            "all_prediction_rows_exact": True,
            "all_scientific_fields_exact": True,
            "foundation_accuracy": total_foundation_correct / total_predictions,
            "ascent_accuracy": total_ascent_correct / total_predictions,
            "gain": (total_ascent_correct - total_foundation_correct) / total_predictions,
            "wins": total_wins,
            "regressions": total_regressions,
        },
        "interpretation": (
            "The audit establishes deterministic reproduction on a third physical "
            "node. It is posthoc and is not treated as an independent confirmation "
            "panel or added to the prespecified inferential family."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
