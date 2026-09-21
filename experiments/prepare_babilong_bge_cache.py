#!/usr/bin/env python3
"""Build frozen target-blind BGE-M3 hybrid+rereanking caches for BABILong."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ascent.generic_retrieval import lexical_tokens, split_passages
from experiments.run_natural_repeat import git_value, sha256_file
from experiments.run_ruler_niah import verify_model_artifact


def bm25_scores(passages: list[str], query: str) -> np.ndarray:
    documents = [Counter(lexical_tokens(passage)) for passage in passages]
    document_frequency = Counter(token for document in documents for token in document)
    query_terms = Counter(lexical_tokens(query))
    document_count = len(documents)
    scores = np.zeros(document_count, dtype=np.float64)
    for index, document in enumerate(documents):
        for token, query_frequency in query_terms.items():
            term_frequency = document.get(token, 0)
            if not term_frequency:
                continue
            df = document_frequency[token]
            inverse_frequency = math.log(
                1.0 + (document_count - df + 0.5) / (df + 0.5)
            )
            scores[index] += (
                query_frequency
                * inverse_frequency
                * term_frequency
                / (term_frequency + 1.2)
            )
    return scores


def ranks_descending(scores: np.ndarray) -> np.ndarray:
    # Stable sort followed by explicit recency preference for exact ties.
    indices = sorted(
        range(scores.size), key=lambda index: (float(scores[index]), index), reverse=True
    )
    ranks = np.empty(scores.size, dtype=np.int64)
    for rank, index in enumerate(indices, start=1):
        ranks[index] = rank
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--embedding-model-dir", type=Path, required=True)
    parser.add_argument("--reranker-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as functional
    import transformers
    from transformers import (
        AutoModel,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    protocol: dict[str, Any] = json.loads(args.protocol.read_text())
    if sha256_file(args.data) not in protocol["allowed_data_sha256"]:
        raise RuntimeError("data hash absent from frozen retrieval protocol")
    rows = [json.loads(line) for line in args.data.read_text().splitlines()]
    if any("target" not in row for row in rows):
        raise RuntimeError("unexpected BABILong row schema")
    embedding_files = verify_model_artifact(
        args.embedding_model_dir, protocol["embedding_model"]["model_artifact"]
    )
    reranker_files = verify_model_artifact(
        args.reranker_model_dir, protocol["reranker_model"]["model_artifact"]
    )
    torch.manual_seed(int(protocol["seed"]))
    np.random.seed(int(protocol["seed"]))

    embedding_tokenizer = AutoTokenizer.from_pretrained(
        args.embedding_model_dir, local_files_only=True
    )
    embedding_model = AutoModel.from_pretrained(
        args.embedding_model_dir, local_files_only=True, dtype=torch.bfloat16
    ).to("cuda").eval()
    reranker_tokenizer = AutoTokenizer.from_pretrained(
        args.reranker_model_dir, local_files_only=True
    )
    reranker_model = AutoModelForSequenceClassification.from_pretrained(
        args.reranker_model_dir, local_files_only=True, dtype=torch.bfloat16
    ).to("cuda").eval()
    for model in (embedding_model, reranker_model):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    embedding_batch_size = int(protocol["embedding_batch_size"])
    reranker_batch_size = int(protocol["reranker_batch_size"])
    passage_max_tokens = int(protocol["passage_max_tokens"])
    reranker_max_tokens = int(protocol["reranker_max_tokens"])
    candidate_count = int(protocol["candidate_count"])
    output_passages = int(protocol["output_passages"])
    rrf_constant = int(protocol["rrf_constant"])

    def dense_encode(texts: list[str], *, max_tokens: int) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), embedding_batch_size):
            encoded = embedding_tokenizer(
                texts[start : start + embedding_batch_size],
                padding=True,
                truncation=True,
                max_length=max_tokens,
                return_tensors="pt",
            ).to("cuda")
            with torch.inference_mode():
                hidden = embedding_model(**encoded).last_hidden_state[:, 0]
                normalized = functional.normalize(hidden.float(), p=2, dim=1)
            outputs.append(normalized.cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def rerank(query: str, passages: list[str]) -> np.ndarray:
        scores: list[np.ndarray] = []
        for start in range(0, len(passages), reranker_batch_size):
            pairs = [
                [query, passage]
                for passage in passages[start : start + reranker_batch_size]
            ]
            encoded = reranker_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=reranker_max_tokens,
                return_tensors="pt",
            ).to("cuda")
            with torch.inference_mode():
                logits = reranker_model(**encoded, return_dict=True).logits.view(-1)
            scores.append(logits.float().cpu().numpy())
        return np.concatenate(scores)

    cache_rows: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc).isoformat()
    for row_index, row in enumerate(rows):
        positioned = split_passages(row["input"])
        positions = [position for position, _ in positioned]
        passages = [passage for _, passage in positioned]
        passage_vectors = dense_encode(passages, max_tokens=passage_max_tokens)
        selected: list[int] = []
        ranked_passages: list[dict[str, Any]] = []
        expanded_query = row["question"]
        for rank in range(1, output_passages + 1):
            query_vector = dense_encode(
                [expanded_query], max_tokens=passage_max_tokens
            )[0]
            dense_scores = passage_vectors @ query_vector
            lexical_scores = bm25_scores(passages, expanded_query)
            dense_ranks = ranks_descending(dense_scores)
            lexical_ranks = ranks_descending(lexical_scores)
            rrf_scores = 1.0 / (rrf_constant + dense_ranks) + 1.0 / (
                rrf_constant + lexical_ranks
            )
            candidate_indices = [
                index
                for index in sorted(
                    range(len(passages)),
                    key=lambda index: (float(rrf_scores[index]), index),
                    reverse=True,
                )
                if index not in selected
            ][:candidate_count]
            candidate_scores = rerank(
                expanded_query, [passages[index] for index in candidate_indices]
            )
            best_offset = max(
                range(len(candidate_indices)),
                key=lambda offset: (
                    float(candidate_scores[offset]),
                    float(rrf_scores[candidate_indices[offset]]),
                    candidate_indices[offset],
                ),
            )
            selected_index = candidate_indices[best_offset]
            selected.append(selected_index)
            ranked_passages.append(
                {
                    "rank": rank,
                    "passage": passages[selected_index],
                    "character_position": positions[selected_index],
                    "reranker_score": float(candidate_scores[best_offset]),
                    "rrf_score": float(rrf_scores[selected_index]),
                    "dense_score": float(dense_scores[selected_index]),
                    "bm25_score": float(lexical_scores[selected_index]),
                }
            )
            expanded_query = expanded_query + "\n" + passages[selected_index]
        cache_rows.append(
            {
                "row_id": row["row_id"],
                "ranked_passages": ranked_passages,
                "sentence_count": len(passages),
                "index_payload_bytes_lower_bound": len(
                    row["input"].encode("utf-8")
                )
                + len(passages) * int(protocol["embedding_dimensions"]) * 2,
            }
        )
        print(f"{row_index + 1}/{len(rows)} {row['row_id'][:12]}", flush=True)

    result = {
        "schema_version": 1,
        "protocol": str(args.protocol),
        "protocol_sha256": sha256_file(args.protocol),
        "data": str(args.data),
        "data_sha256": sha256_file(args.data),
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_model": {
            "repo_id": protocol["embedding_model"]["repo_id"],
            "revision": protocol["embedding_model"]["revision"],
            "files": embedding_files,
        },
        "reranker_model": {
            "repo_id": protocol["reranker_model"]["repo_id"],
            "revision": protocol["reranker_model"]["revision"],
            "files": reranker_files,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
            "git_commit": git_value(["rev-parse", "HEAD"]),
            "git_dirty": bool(git_value(["status", "--porcelain"])),
        },
        "rows": cache_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
