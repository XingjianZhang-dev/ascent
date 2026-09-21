#!/usr/bin/env python3
"""Build disjoint QA7/QA8 panels with guaranteed State-3/State-7 semantic refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ascent.babilong_memory import read_babilong
from experiments.prepare_babilong_structured_aggregation import (
    canonical_bytes,
    inject_facts,
    parse_babi_questions,
    projected_answer,
    remove_recognized_events,
    sha256_bytes,
    sha256_file,
)


PANEL_NAMES = (
    "headroom_development",
    "headroom_confirmation_1",
    "headroom_confirmation_2",
    "headroom_confirmation_3",
    "headroom_confirmation_4",
    "headroom_confirmation_5",
)
ROWS_PER_TASK = 12
SMALL_SLOTS = 3
LARGE_SLOTS = 7


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line]
    document = json.loads(path.read_text())
    if not isinstance(document, list):
        raise RuntimeError(f"noise source must be a JSON list or JSONL: {path}")
    return document


def load_noise_documents(paths: list[Path]) -> list[dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    for path in paths:
        for source_index, row in enumerate(_read_rows(path)):
            text = remove_recognized_events(row["input"])
            text_hash = sha256_bytes(text.encode("utf-8"))
            documents.setdefault(
                text_hash,
                {
                    "text": text,
                    "noise_sha256": text_hash,
                    "source_file_sha256": sha256_file(path),
                    "source_index": source_index,
                },
            )
    return [documents[key] for key in sorted(documents)]


def excluded_source_ids(paths: list[Path]) -> set[str]:
    return {
        row["source_row_id"]
        for path in paths
        for row in _read_rows(path)
        if "source_row_id" in row
    }


def semantic_refinement(sample: dict[str, Any]) -> tuple[str, str] | None:
    read = read_babilong(" ".join(sample["facts"]), sample["question"])
    small = projected_answer(read, sample["task"], SMALL_SLOTS)
    large = projected_answer(read, sample["task"], LARGE_SLOTS)
    full = projected_answer(read, sample["task"], len(read.facts))
    if small == large or large != full:
        return None
    return small, large


def build_panels(
    task_sources: dict[str, Path],
    noise_sources: list[Path],
    exclude_sources: list[Path],
    output_root: Path,
) -> dict[str, Any]:
    excluded = excluded_source_ids(exclude_sources)
    required_per_task = len(PANEL_NAMES) * ROWS_PER_TASK
    selected: dict[str, list[dict[str, Any]]] = {}
    source_hashes: dict[str, str] = {}
    for task, path in task_sources.items():
        source_hashes[task] = sha256_file(path)
        eligible: dict[str, dict[str, Any]] = {}
        for sample in parse_babi_questions(path, task):
            if sample["source_row_id"] in excluded:
                continue
            refinement = semantic_refinement(sample)
            if refinement is None:
                continue
            eligible.setdefault(sample["selection_id"], sample)
        ordered = sorted(eligible.values(), key=lambda row: row["selection_id"])
        if len(ordered) < required_per_task:
            raise RuntimeError(
                f"insufficient nonredundant {task} rows: {len(ordered)} < {required_per_task}"
            )
        selected[task] = ordered[:required_per_task]

    noise = load_noise_documents(noise_sources)
    required_noise = len(PANEL_NAMES) * ROWS_PER_TASK * len(task_sources)
    if len(noise) < required_noise:
        raise RuntimeError(f"insufficient unique noise: {len(noise)} < {required_noise}")

    output_root.mkdir(parents=True, exist_ok=True)
    panel_manifests: list[dict[str, Any]] = []
    used_source_ids: set[str] = set()
    for panel_index, panel_name in enumerate(PANEL_NAMES):
        panel_rows: list[dict[str, Any]] = []
        for task_index, task in enumerate(sorted(task_sources)):
            start = panel_index * ROWS_PER_TASK
            stop = start + ROWS_PER_TASK
            for row_index, source in enumerate(selected[task][start:stop]):
                if source["source_row_id"] in used_source_ids:
                    raise RuntimeError("source row reused across panels")
                used_source_ids.add(source["source_row_id"])
                noise_index = (
                    task_index * len(PANEL_NAMES) * ROWS_PER_TASK
                    + panel_index * ROWS_PER_TASK
                    + row_index
                )
                noise_row = noise[noise_index]
                input_text = inject_facts(noise_row["text"], source["facts"])
                read = read_babilong(input_text, source["question"])
                small = projected_answer(read, task, SMALL_SLOTS)
                large = projected_answer(read, task, LARGE_SLOTS)
                full = projected_answer(read, task, len(read.facts))
                if small == large or large != full:
                    raise RuntimeError("injected row lost registered semantic refinement")
                if read.answer != source["target"]:
                    raise RuntimeError("target-blind parser disagrees with public target")
                row_spec = {
                    "task": task,
                    "source_row_id": source["source_row_id"],
                    "selection_id": source["selection_id"],
                    "noise_sha256": noise_row["noise_sha256"],
                    "small_slots": SMALL_SLOTS,
                    "large_slots": LARGE_SLOTS,
                }
                panel_rows.append(
                    {
                        "row_id": sha256_bytes(canonical_bytes(row_spec)),
                        "task": task,
                        "input": input_text,
                        "question": source["question"],
                        "target": source["target"],
                        "supporting_fact_count": len(source["reference_numbers"]),
                        "refinement_category": "nonredundant_state3_to_state7",
                        "source_row_id": source["source_row_id"],
                        "selection_id": source["selection_id"],
                        "noise_sha256": noise_row["noise_sha256"],
                    }
                )
        panel_rows.sort(key=lambda row: (row["task"], row["row_id"]))
        path = output_root / f"{panel_name}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in panel_rows)
        )
        panel_manifests.append(
            {
                "panel": panel_name,
                "path": path.name,
                "rows": len(panel_rows),
                "rows_per_task": ROWS_PER_TASK,
            "state3_to_state7_semantic_changes": len(panel_rows),
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "status": "prospective_before_any_qwen_nonredundant_headroom_score",
        "official_babi_split": "tasks_1-20_v1-2/en-10k train",
        "task_source_sha256": source_hashes,
        "noise_source_sha256": {str(path): sha256_file(path) for path in noise_sources},
        "excluded_panel_sha256": {
            str(path): sha256_file(path) for path in exclude_sources
        },
        "excluded_source_rows": len(excluded),
        "unique_clean_noise_documents": len(noise),
        "selection_rule": (
            "Exclude every previously materialized source_row_id, require the target-blind "
            "State-3 projected answer to differ from State-7, require State-7 to equal the "
            "full-event projection, sort by target-excluded selection_id, and take the first "
            f"{required_per_task} rows per task without model scores or reference-answer values."
        ),
        "small_slots": SMALL_SLOTS,
        "large_slots": LARGE_SLOTS,
        "panels": panel_manifests,
    }
    manifest_path = output_root / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa7-source", type=Path, required=True)
    parser.add_argument("--qa8-source", type=Path, required=True)
    parser.add_argument("--noise-source", type=Path, action="append", required=True)
    parser.add_argument("--exclude-panel", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_panels(
                {"qa7": args.qa7_source, "qa8": args.qa8_source},
                args.noise_source,
                args.exclude_panel,
                args.output_root,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
