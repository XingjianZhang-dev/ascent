#!/usr/bin/env python3
"""Score-free audit for the frozen RULER QA dual-path ASCENT panel."""

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

from ascent.ruler_qa_memory import encode_ruler_qa_state, read_ruler_qa_graph
from experiments.run_natural_repeat import sha256_file
from experiments.run_ruler_qa_dual_path_nll import compact_evidence_content


def audit(
    config_path: Path, data_root: Path, endpoint_name: str, model_dir: Path
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    config = json.loads(config_path.read_text())
    endpoint = {row["name"]: row for row in config["endpoints"]}[endpoint_name]
    task_path = data_root / config["task_name"] / "validation.jsonl"
    if sha256_file(task_path) != config["task_sha256"]:
        raise RuntimeError("task hash mismatch")
    rows = [json.loads(line) for line in task_path.read_text().splitlines()]
    if len(rows) != int(config["samples_per_task"]):
        raise RuntimeError("unexpected task row count")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    states = {
        row["name"]: {
            "documents": int(row["documents"]),
            "replay_tokens": int(row["replay_tokens"]),
        }
        for row in config["memory_states"]
    }
    scale_state = endpoint["scale_state"]
    scale_spec = states[scale_state]
    state_serialization = str(config.get("state_serialization", "document_sketches"))
    arm_names = (*states, "query_only", "irrelevant")
    lengths: dict[str, list[int]] = {
        "foundation": [],
        **{f"{name}_certified": [] for name in arm_names},
    }
    actual_tokens: dict[str, list[int]] = {name: [] for name in arm_names}
    token_hashes: dict[str, list[str]] = {name: [] for name in arm_names}
    ranking_rows: list[str] = []
    document_counts: list[int] = []
    graph_changes = {2: 0, 3: 0, 4: 0}

    def prompt_length(content: str, answer_prefix: str) -> int:
        user_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
        )
        prefix_ids = tokenizer(answer_prefix, add_special_tokens=False)["input_ids"]
        return len(user_ids) + len(prefix_ids)

    for row in rows:
        read = read_ruler_qa_graph(
            row["input"], seed_documents=int(config["bm25_seed_documents"])
        )
        document_counts.append(len(read.documents))
        graph_numbers = [doc.number for doc in read.graph_ranked]
        query_numbers = [doc.number for doc in read.query_only_ranked]
        ranking_rows.append(",".join(map(str, graph_numbers)))
        for count in graph_changes:
            graph_changes[count] += int(
                graph_numbers[:count] != query_numbers[:count]
            )
        documents = {
            **{
                name: read.graph_ranked[: spec["documents"]]
                for name, spec in states.items()
            },
            "query_only": read.query_only_ranked[: scale_spec["documents"]],
            "irrelevant": read.irrelevant_ranked[: scale_spec["documents"]],
        }
        state_ids: dict[str, list[int]] = {}
        for name in arm_names:
            budget = (
                states[name]["replay_tokens"]
                if name in states
                else scale_spec["replay_tokens"]
            )
            ids = encode_ruler_qa_state(
                tokenizer,
                documents[name],
                read.question,
                mode=state_serialization,
            )[:budget]
            if not ids:
                raise RuntimeError("empty ASCENT state")
            state_ids[name] = ids
            state_text = tokenizer.decode(ids, skip_special_tokens=True).strip()
            lengths[f"{name}_certified"].append(
                prompt_length(
                    compact_evidence_content(state_text, read.question),
                    row["answer_prefix"],
                )
            )
            actual_tokens[name].append(len(ids))
            token_hashes[name].append(
                hashlib.sha256(bytes().join(int(value).to_bytes(8, "little") for value in ids)).hexdigest()
            )
        for smaller, larger in zip(states, tuple(states)[1:]):
            if state_ids[larger][: len(state_ids[smaller])] != state_ids[smaller]:
                raise RuntimeError("graph state token prefixes are not nested")
        lengths["foundation"].append(
            prompt_length(row["input"], row["answer_prefix"])
        )
    maximum = max(max(values) for values in lengths.values())
    if maximum > int(config["query_context_tokens"]):
        raise RuntimeError("frozen native context limit exceeded")
    return {
        "schema_version": 1,
        "score_free": True,
        "answer_outputs_read": False,
        "support_labels_read": False,
        "config_sha256": sha256_file(config_path),
        "task_sha256": sha256_file(task_path),
        "endpoint": endpoint,
        "rows": len(rows),
        "document_count_range": [min(document_counts), max(document_counts)],
        "graph_prefix_change_rows_vs_query_only": graph_changes,
        "rankings_sha256": hashlib.sha256("\n".join(ranking_rows).encode()).hexdigest(),
        "prompt_token_ranges": {
            name: [min(values), max(values)] for name, values in lengths.items()
        },
        "state_token_ranges": {
            name: [min(values), max(values)] for name, values in actual_tokens.items()
        },
        "mean_state_tokens": {
            name: sum(values) / len(values) for name, values in actual_tokens.items()
        },
        "state_token_ids_sha256": {
            name: hashlib.sha256("\n".join(values).encode()).hexdigest()
            for name, values in token_hashes.items()
        },
        "exact_nested_graph_state_token_prefixes": True,
        "state_serialization": state_serialization,
        "native_context_limit": int(config["query_context_tokens"]),
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
