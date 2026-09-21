#!/usr/bin/env python3
"""Freeze disjoint BABILong panels from official JSON or JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_bytes(row: dict[str, Any]) -> bytes:
    return json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")


def stable_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        document = json.loads(path.read_text())
        if not isinstance(document, list):
            raise ValueError(f"expected a JSON row list: {path}")
        rows = document
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
    return sorted(rows, key=lambda row: hashlib.sha256(canonical_bytes(row)).digest())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_panel(
    source_root: Path,
    output_root: Path,
    *,
    panel_name: str,
    start: int,
    rows_per_task: int,
    tasks: tuple[str, ...] = ("qa1", "qa2", "qa3"),
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"{panel_name}.jsonl"
    selected: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for task in tasks:
        candidates = (source_root / f"{task}.jsonl", source_root / f"{task}.json")
        source = next((path for path in candidates if path.exists()), None)
        if source is None:
            raise FileNotFoundError(f"no JSON/JSONL source found for {task}")
        source_hashes[task] = sha256_file(source)
        rows = stable_rows(source)
        stop = start + rows_per_task
        if stop > len(rows):
            raise ValueError(f"{task} has only {len(rows)} rows")
        for row in rows[start:stop]:
            selected.append(
                {
                    "row_id": hashlib.sha256(canonical_bytes(row)).hexdigest(),
                    "task": task,
                    "input": row["input"],
                    "question": row["question"],
                    "target": row["target"].strip().lower(),
                }
            )
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
    )
    return {
        "panel": panel_name,
        "path": output.name,
        "start_in_stable_hash_order": start,
        "rows_per_task": rows_per_task,
        "tasks": list(tasks),
        "rows": len(selected),
        "sha256": sha256_file(output),
        "source_sha256": source_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=(
            "primary",
            "robustness",
            "neural_control",
            "generative_fresh",
            "generative_focused_confirmation",
            "canonical_16k_confirmation",
            "canonical_train_8k_confirmation",
            "aggregation",
        ),
        default="primary",
    )
    args = parser.parse_args()
    if args.profile == "primary":
        specifications = (
            ("development", 0, 20),
            ("confirmation_seed_1", 20, 40),
            ("confirmation_seed_2", 60, 40),
            ("confirmation_seed_3", 100, 40),
        )
    elif args.profile == "robustness":
        specifications = (
            ("robustness_panel_1", 140, 40),
            ("robustness_panel_2", 180, 40),
            ("robustness_panel_3", 220, 40),
        )
    elif args.profile == "neural_control":
        specifications = (
            ("neural_control_panel_1", 260, 40),
            ("neural_control_panel_2", 300, 40),
            ("neural_control_panel_3", 340, 40),
        )
    elif args.profile == "generative_fresh":
        specifications = tuple(
            (f"generative_fresh_panel_{index}", 380 + 40 * (index - 1), 40)
            for index in range(1, 6)
        )
    elif args.profile == "generative_focused_confirmation":
        specifications = tuple(
            (
                f"generative_focused_confirmation_panel_{index}",
                580 + 40 * (index - 1),
                40,
            )
            for index in range(1, 11)
        )
    elif args.profile == "canonical_16k_confirmation":
        specifications = tuple(
            (
                f"canonical_16k_confirmation_panel_{index}",
                40 * (index - 1),
                40,
            )
            for index in range(1, 11)
        )
    elif args.profile == "canonical_train_8k_confirmation":
        specifications = tuple(
            (
                f"canonical_train_8k_confirmation_panel_{index}",
                40 * (index - 1),
                40,
            )
            for index in range(1, 11)
        )
    else:
        specifications = (
            ("aggregation_development", 0, 20),
            ("aggregation_confirmation_1", 20, 20),
            ("aggregation_confirmation_2", 40, 20),
            ("aggregation_confirmation_3", 60, 20),
        )
    if args.profile == "aggregation":
        tasks = ("qa7", "qa8")
    elif args.profile in {
        "canonical_16k_confirmation",
        "canonical_train_8k_confirmation",
    }:
        tasks = ("qa2", "qa3")
    else:
        tasks = ("qa1", "qa2", "qa3")
    manifests = [
        write_panel(
            args.source_root,
            args.output_root,
            panel_name=name,
            start=start,
            rows_per_task=rows,
            tasks=tasks,
        )
        for name, start, rows in specifications
    ]
    manifest_document: dict[str, Any] = {"panels": manifests}
    source_manifest = args.source_root / "SOURCE_MANIFEST.json"
    if source_manifest.exists():
        manifest_document["official_source"] = json.loads(source_manifest.read_text())
    manifest_path = args.output_root / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest_document, indent=2) + "\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
