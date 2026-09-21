#!/usr/bin/env python3
"""Freeze stratified QA7/QA8 panels from public BABILong/bAbI sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ascent.babilong_memory import (
    extract_babilong_events,
    project_babilong_inventory,
    read_babilong,
)


PANEL_NAMES = (
    "structured_development",
    "structured_confirmation_1",
    "structured_confirmation_2",
    "structured_confirmation_3",
    "structured_confirmation_4",
    "structured_confirmation_5",
)
REFINEMENT_QUOTAS = {
    "needs_medium_state": 4,
    "needs_large_state": 4,
    "multi_event_general": 8,
}
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_babi_questions(path: Path, task: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    facts_by_number: dict[int, str] = {}
    for raw_line in path.read_text().splitlines():
        line_number_text, content = raw_line.split(" ", 1)
        line_number = int(line_number_text)
        if line_number == 1:
            facts_by_number = {}
        if "\t" not in content:
            facts_by_number[line_number] = content
            continue
        question, target, references_text = content.split("\t")
        references = tuple(int(value) for value in references_text.split())
        facts = tuple(facts_by_number[index] for index in sorted(facts_by_number))
        source = {
            "task": task,
            "facts": facts,
            "question": question.strip(),
            "target": target.strip().lower(),
            "reference_numbers": references,
            "reference_facts": tuple(facts_by_number[index] for index in references),
        }
        source["selection_id"] = sha256_bytes(
            canonical_bytes(
                {
                    "task": task,
                    "facts": facts,
                    "question": question.strip(),
                    "reference_numbers": references,
                    "reference_facts": source["reference_facts"],
                }
            )
        )
        source["source_row_id"] = sha256_bytes(canonical_bytes(source))
        samples.append(source)
    return samples


def remove_recognized_events(input_text: str) -> str:
    cleaned = input_text
    for event in reversed(extract_babilong_events(input_text)):
        start = event.character_position
        stop = start + len(event.source)
        cleaned = cleaned[:start] + " " * (stop - start) + cleaned[stop:]
    if extract_babilong_events(cleaned):
        raise RuntimeError("recognized BABILong event remained in noise text")
    return re.sub(r"[ \t]+", " ", cleaned).strip()


def load_noise_documents(paths: list[Path]) -> list[dict[str, str]]:
    documents: dict[str, dict[str, str]] = {}
    for path in paths:
        rows = json.loads(path.read_text())
        for source_index, row in enumerate(rows):
            text = remove_recognized_events(row["input"])
            text_hash = sha256_bytes(text.encode("utf-8"))
            documents.setdefault(
                text_hash,
                {
                    "text": text,
                    "noise_sha256": text_hash,
                    "source_file_sha256": sha256_file(path),
                    "source_index": str(source_index),
                },
            )
    return [documents[key] for key in sorted(documents)]


def projected_answer(read: Any, task: str, fact_slots: int) -> str:
    projection = project_babilong_inventory(read.facts[-fact_slots:], read.question)
    return projection.count_word if task == "qa7" else projection.inventory_text


def refinement_category(sample: dict[str, Any]) -> str | None:
    read = read_babilong(" ".join(sample["facts"]), sample["question"])
    if read.supporting_fact_count < 2:
        return None
    one = projected_answer(read, sample["task"], 1)
    two = projected_answer(read, sample["task"], 2)
    full = projected_answer(read, sample["task"], 8)
    if two != full:
        return "needs_large_state"
    if one != two:
        return "needs_medium_state"
    return "multi_event_general"


def inject_facts(noise_text: str, facts: tuple[str, ...]) -> str:
    sentences = [
        sentence for sentence in SENTENCE_BOUNDARY.split(noise_text) if sentence
    ]
    if len(sentences) <= len(facts):
        raise RuntimeError("noise document has too few sentence slots")
    insertions: defaultdict[int, list[str]] = defaultdict(list)
    for fact_index, fact in enumerate(facts, start=1):
        position = round(fact_index * len(sentences) / (len(facts) + 1))
        insertions[position].append(fact)
    output: list[str] = []
    for sentence_index, sentence in enumerate(sentences):
        output.extend(insertions[sentence_index])
        output.append(sentence)
    output.extend(insertions[len(sentences)])
    return " ".join(output)


def build_panels(
    task_sources: dict[str, Path],
    noise_sources: list[Path],
    output_root: Path,
    *,
    panel_names: tuple[str, ...] = PANEL_NAMES,
    official_babi_split: str = "tasks_1-20_v1-2/en-10k test",
    selection_panel_offset: int = 0,
) -> dict[str, Any]:
    if selection_panel_offset < 0:
        raise ValueError("selection_panel_offset must be nonnegative")
    noise_documents = load_noise_documents(noise_sources)
    noise_documents_per_task = len(panel_names) * 16
    required_noise_documents = 2 * noise_documents_per_task
    if len(noise_documents) < required_noise_documents:
        raise RuntimeError(
            f"at least {required_noise_documents} unique public noise documents "
            "are required"
        )
    selected_by_task: dict[str, list[list[dict[str, Any]]]] = {}
    task_source_hashes: dict[str, str] = {}
    for task, source_path in task_sources.items():
        task_source_hashes[task] = sha256_file(source_path)
        category_rows: defaultdict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for sample in parse_babi_questions(source_path, task):
            category = refinement_category(sample)
            if category is not None:
                sample["refinement_category"] = category
                category_rows[category].setdefault(sample["selection_id"], sample)
        categories = {
            category: sorted(values.values(), key=lambda row: row["selection_id"])
            for category, values in category_rows.items()
        }
        panels: list[list[dict[str, Any]]] = []
        for panel_index in range(len(panel_names)):
            panel_rows: list[dict[str, Any]] = []
            for category, quota in REFINEMENT_QUOTAS.items():
                start = (selection_panel_offset + panel_index) * quota
                stop = start + quota
                if len(categories[category]) < stop:
                    raise RuntimeError(f"insufficient {task} rows in {category}")
                panel_rows.extend(categories[category][start:stop])
            panel_rows.sort(key=lambda row: row["source_row_id"])
            panels.append(panel_rows)
        selected_by_task[task] = panels

    output_root.mkdir(parents=True, exist_ok=True)
    panel_manifests: list[dict[str, Any]] = []
    for panel_index, panel_name in enumerate(panel_names):
        panel_rows: list[dict[str, Any]] = []
        for task_index, task in enumerate(sorted(task_sources)):
            for row_index, source in enumerate(selected_by_task[task][panel_index]):
                noise_index = (
                    task_index * noise_documents_per_task + panel_index * 16 + row_index
                )
                noise = noise_documents[noise_index]
                input_text = inject_facts(noise["text"], source["facts"])
                read = read_babilong(input_text, source["question"])
                if read.answer != source["target"]:
                    raise RuntimeError(
                        "target-blind parser disagrees with public target"
                    )
                if read.supporting_fact_count != len(source["reference_numbers"]):
                    raise RuntimeError("parser/reference supporting-fact mismatch")
                row_spec = {
                    "task": task,
                    "source_row_id": source["source_row_id"],
                    "selection_id": source["selection_id"],
                    "noise_sha256": noise["noise_sha256"],
                    "injection_rule": "chronological_facts_at_even_sentence_quantiles",
                }
                panel_rows.append(
                    {
                        "row_id": sha256_bytes(canonical_bytes(row_spec)),
                        "task": task,
                        "input": input_text,
                        "question": source["question"],
                        "target": source["target"],
                        "supporting_fact_count": len(source["reference_numbers"]),
                        "refinement_category": source["refinement_category"],
                        "source_row_id": source["source_row_id"],
                        "selection_id": source["selection_id"],
                        "noise_sha256": noise["noise_sha256"],
                    }
                )
        output_path = output_root / f"{panel_name}.jsonl"
        output_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in panel_rows)
        )
        panel_manifests.append(
            {
                "panel": panel_name,
                "path": output_path.name,
                "rows": len(panel_rows),
                "rows_per_task": 16,
                "rows_per_task_by_refinement_category": REFINEMENT_QUOTAS,
                "sha256": sha256_file(output_path),
            }
        )
    manifest = {
        "schema_version": 1,
        "status": "prospective_before_any_structured_readout_model_score",
        "official_babilong_repository_commit": "7a6efee29f5cac03c3c410e6799c80fd2ffe3610",
        "official_babi_split": official_babi_split,
        "task_source_sha256": task_source_hashes,
        "noise_source_sha256": {str(path): sha256_file(path) for path in noise_sources},
        "unique_clean_noise_documents": len(noise_documents),
        "selection_panel_offset": selection_panel_offset,
        "selection_rule": (
            "Using only retained event streams, assign public bAbI questions "
            "with at least two relevant events to needs-medium-state when the "
            "one-slot and two-slot projections differ, needs-large-state when "
            "the two-slot and full projections differ, or multi-event-general. "
            "Within each task/category, sort by target-excluded canonical-row "
            "SHA-256 and allocate fixed 4/4/8 quotas per panel. Model scores and "
            "reference-answer values do not enter selection."
        ),
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--panel-name",
        action="append",
        help="ordered output panel name; repeat once per panel",
    )
    parser.add_argument(
        "--official-babi-split",
        default="tasks_1-20_v1-2/en-10k test",
    )
    parser.add_argument("--selection-panel-offset", type=int, default=0)
    args = parser.parse_args()
    manifest = build_panels(
        {"qa7": args.qa7_source, "qa8": args.qa8_source},
        args.noise_source,
        args.output_root,
        panel_names=tuple(args.panel_name) if args.panel_name else PANEL_NAMES,
        official_babi_split=args.official_babi_split,
        selection_panel_offset=args.selection_panel_offset,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
