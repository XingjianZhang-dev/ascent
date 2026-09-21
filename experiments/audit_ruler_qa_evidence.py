#!/usr/bin/env python3
"""Score-free structural and native-length audit for frozen RULER QA evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ascent.ruler_qa_memory import read_ruler_qa_graph, serialize_documents
from experiments.run_natural_repeat import sha256_file
from experiments.run_ruler_qa_evidence import evidence_content


def audit(
    config_path: Path, data_root: Path, endpoint_name: str, model_dir: Path
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    config = json.loads(config_path.read_text())
    endpoint = {row["name"]: row for row in config["endpoints"]}[endpoint_name]
    task_path = data_root / config["task_name"] / "validation.jsonl"
    if sha256_file(task_path) != config["task_sha256"]:
        raise RuntimeError("RULER QA task hash mismatch")
    rows = [json.loads(line) for line in task_path.read_text().splitlines()]
    if len(rows) != config["samples_per_task"]:
        raise RuntimeError("unexpected row count")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    fixed_documents = int(config["fixed_documents"])
    scale_documents = int(endpoint["scale_documents"])
    context_limit = int(config["query_context_tokens"])
    lengths: dict[str, list[int]] = {
        name: [] for name in ("foundation", "fixed", "scale", "query_only", "irrelevant")
    }
    document_counts: list[int] = []
    ranking_rows: list[str] = []
    graph_prefix_changes = {2: 0, 3: 0, 4: 0}
    for row in rows:
        read = read_ruler_qa_graph(
            row["input"], seed_documents=int(config["bm25_seed_documents"])
        )
        document_counts.append(len(read.documents))
        graph_numbers = [document.number for document in read.graph_ranked]
        query_numbers = [document.number for document in read.query_only_ranked]
        ranking_rows.append(",".join(map(str, graph_numbers)))
        for count in graph_prefix_changes:
            graph_prefix_changes[count] += int(
                graph_numbers[:count] != query_numbers[:count]
            )
        sources = {
            "fixed": serialize_documents(read.graph_ranked[:fixed_documents]),
            "scale": serialize_documents(read.graph_ranked[:scale_documents]),
            "query_only": serialize_documents(read.query_only_ranked[:scale_documents]),
            "irrelevant": serialize_documents(read.irrelevant_ranked[:scale_documents]),
        }
        for name in lengths:
            content = row["input"] if name == "foundation" else row["input"] + evidence_content(sources[name])
            user_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                tokenize=True,
            )
            prefix_ids = tokenizer(row["answer_prefix"], add_special_tokens=False)[
                "input_ids"
            ]
            lengths[name].append(len(user_ids) + len(prefix_ids))
    maximum = max(max(values) for values in lengths.values())
    if maximum > context_limit:
        raise RuntimeError("frozen native context limit exceeded")
    return {
        "schema_version": 1,
        "score_free": True,
        "answer_and_support_labels_read": False,
        "config_sha256": sha256_file(config_path),
        "task_sha256": sha256_file(task_path),
        "endpoint": endpoint,
        "rows": len(rows),
        "document_count_range": [min(document_counts), max(document_counts)],
        "graph_prefix_change_rows_vs_query_only": graph_prefix_changes,
        "rankings_sha256": hashlib.sha256("\n".join(ranking_rows).encode()).hexdigest(),
        "prompt_token_ranges": {
            name: [min(values), max(values)] for name, values in lengths.items()
        },
        "native_context_limit": context_limit,
        "maximum_prompt_tokens": maximum,
        "native_untruncated": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(args.config, args.data_root, args.endpoint, args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
