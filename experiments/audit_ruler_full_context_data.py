#!/usr/bin/env python3
"""Audit official full-context RULER data before confirmatory execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ascent.ruler_memory import read_ruler_niah, read_ruler_niah_all_values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_ids(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(value) for value in ids]


def audit_task(
    path: Path, tokenizer: Any, reader_name: str = "niah"
) -> dict[str, Any]:
    readers = {
        "niah": read_ruler_niah,
        "niah_all_values": read_ruler_niah_all_values,
    }
    if reader_name not in readers:
        raise ValueError(f"unsupported reader: {reader_name}")
    memory_reader = readers[reader_name]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    parser_exact = 0
    chat_lengths = []
    source_lengths = []
    indices = []
    for row in rows:
        indices.append(int(row["index"]))
        read = memory_reader(row["input"], memory_slots=256)
        parsed_values = [str(value) for value in read.values]
        expected_values = [str(value) for value in row["outputs"]]
        # Official RULER ``string_match_all`` checks whether every reference
        # occurs in the prediction and does not impose reference order.  The
        # generator also shuffles multivalue needles independently of the
        # stored reference list, so require exact multiset equality here.
        if reader_name == "niah_all_values":
            parser_matches = Counter(parsed_values) == Counter(expected_values)
        else:
            parser_matches = parsed_values == expected_values
        if parser_matches:
            parser_exact += 1
        user_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["input"]}],
            add_generation_prompt=True,
            tokenize=True,
        )
        if hasattr(user_ids, "tolist"):
            user_ids = user_ids.tolist()
        if user_ids and isinstance(user_ids[0], list):
            user_ids = user_ids[0]
        prefix_ids = token_ids(tokenizer, row.get("answer_prefix", ""))
        chat_lengths.append(len(user_ids) + len(prefix_ids))
        source_text = "\n".join(fact.source for fact in read.facts)
        source_lengths.append(len(token_ids(tokenizer, source_text)))
    index_counts = Counter(indices)
    duplicate_index_occurrences = {
        str(index): count for index, count in index_counts.items() if count > 1
    }
    canonical_row_hashes = [
        hashlib.sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for row in rows
    ]
    input_hashes = [
        hashlib.sha256(row["input"].encode()).hexdigest() for row in rows
    ]
    gates = {
        "nonempty": bool(rows),
        "unique_full_rows": len(set(canonical_row_hashes)) == len(rows),
        "unique_inputs": len(set(input_hashes)) == len(rows),
        "parser_exact_every_row": parser_exact == len(rows),
        "four_outputs_every_row": all(len(row["outputs"]) == 4 for row in rows),
    }
    return {
        "validation_sha256": sha256_file(path),
        "parser_match_semantics": (
            "exact_multiset_official_string_match_all"
            if reader_name == "niah_all_values"
            else "exact_ordered"
        ),
        "rows": len(rows),
        "parser_exact_rows": parser_exact,
        "duplicate_index_occurrences": duplicate_index_occurrences,
        "chat_token_range": [min(chat_lengths), max(chat_lengths)],
        "selected_source_token_range": [min(source_lengths), max(source_lengths)],
        "gates": gates,
        "status": "passed" if all(gates.values()) else "failed",
    }


def parse_seed_root(value: str) -> tuple[int, Path]:
    seed, separator, root = value.partition("=")
    if not separator or not seed.isdigit() or not root:
        raise argparse.ArgumentTypeError("expected SEED=/absolute/data/root")
    return int(seed), Path(root)


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-root", action="append", type=parse_seed_root, required=True)
    parser.add_argument("--task-name", default="niah_multiquery")
    parser.add_argument(
        "--reader", choices=("niah", "niah_all_values"), default="niah"
    )
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_dir, local_files_only=True
    )
    seeds = {}
    for seed, root in args.seed_root:
        seeds[str(seed)] = audit_task(
            root / args.task_name / "validation.jsonl", tokenizer, args.reader
        )
    result = {
        "schema_version": 1,
        "task_name": args.task_name,
        "reader": args.reader,
        "tokenizer_dir": str(args.tokenizer_dir),
        "seeds": seeds,
        "status": "passed"
        if seeds and all(row["status"] == "passed" for row in seeds.values())
        else "failed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["status"] != "passed":
        raise SystemExit("full-context data audit failed")


if __name__ == "__main__":
    main()
