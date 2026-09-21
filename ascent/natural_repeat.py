"""Causal bounded-memory event extraction from natural token streams."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


TokenArray = NDArray[np.uint16]


@dataclass(frozen=True)
class NaturalRepeatEvents:
    query_contexts: TokenArray
    source_fragments: TokenArray
    targets: TokenArray
    retrieved_targets: TokenArray
    query_positions: NDArray[np.int64]
    source_positions: NDArray[np.int64]
    hits_seen: int
    tokens_scanned: int


@dataclass(frozen=True)
class _Record:
    target: int
    source_fragment: tuple[int, ...]
    target_position: int


def persistent_state_bytes(
    *, memory_slots: int, key_length: int, source_tokens: int, token_bytes: int = 2
) -> int:
    """Return the explicit packed-state accounting used by the benchmark.

    Each occupied slot contains key and source token IDs, one target token ID,
    one signed 64-bit timestamp, and one validity byte. Container overhead is
    excluded and the packed representation is reported separately from model
    parameters and transient replay activations.
    """
    if min(memory_slots, key_length, source_tokens, token_bytes) <= 0:
        raise ValueError("state dimensions must be positive")
    per_slot = (key_length + source_tokens + 1) * token_bytes + 8 + 1
    return memory_slots * per_slot


def collect_natural_repeat_events(
    tokens: TokenArray,
    *,
    event_offset: int,
    num_events: int,
    key_length: int,
    context_length: int,
    source_tokens: int,
    min_distance: int,
    memory_slots: int,
    eos_token_id: int | None,
) -> NaturalRepeatEvents:
    """Collect read-before-write exact-key repeats using a bounded LRU store.

    The query is always the current ``context_length``-token prefix and never
    contains its target. A source fragment ends at the earlier occurrence's
    target. Memory is cleared at document boundaries when ``eos_token_id`` is
    supplied, preventing cross-document retrieval.
    """
    stream = np.asarray(tokens)
    if stream.ndim != 1 or not np.issubdtype(stream.dtype, np.integer):
        raise ValueError("tokens must be a one-dimensional integer array")
    if event_offset < 0 or num_events <= 0:
        raise ValueError("event_offset must be nonnegative and num_events positive")
    if key_length <= 0 or source_tokens <= 0 or memory_slots <= 0:
        raise ValueError("memory dimensions must be positive")
    if context_length < key_length or min_distance <= 0:
        raise ValueError("context must cover the key and min_distance must be positive")

    needed = event_offset + num_events
    memory: OrderedDict[tuple[int, ...], _Record] = OrderedDict()
    contexts: list[NDArray[np.integer]] = []
    fragments: list[tuple[int, ...]] = []
    targets: list[int] = []
    retrieved: list[int] = []
    query_positions: list[int] = []
    source_positions: list[int] = []
    document_start = 0
    hits_seen = 0
    final_position = 0

    for position in range(context_length, stream.size):
        final_position = position + 1
        if eos_token_id is not None and int(stream[position - 1]) == eos_token_id:
            document_start = position
            memory.clear()
        if position - document_start < max(context_length, source_tokens - 1):
            continue

        key = tuple(int(value) for value in stream[position - key_length : position])
        previous = memory.get(key)
        if previous is not None:
            memory.move_to_end(key)
            if position - previous.target_position >= min_distance:
                if hits_seen >= event_offset:
                    contexts.append(stream[position - context_length : position].copy())
                    fragments.append(previous.source_fragment)
                    targets.append(int(stream[position]))
                    retrieved.append(previous.target)
                    query_positions.append(position)
                    source_positions.append(previous.target_position)
                hits_seen += 1

        source_start = position - source_tokens + 1
        fragment = tuple(int(value) for value in stream[source_start : position + 1])
        memory[key] = _Record(int(stream[position]), fragment, position)
        memory.move_to_end(key)
        while len(memory) > memory_slots:
            memory.popitem(last=False)
        if hits_seen >= needed:
            break

    if len(contexts) != num_events:
        raise RuntimeError(
            f"only {len(contexts)} requested events were collected after skipping "
            f"{event_offset}; scanned {final_position} of {stream.size} tokens"
        )
    return NaturalRepeatEvents(
        query_contexts=np.asarray(contexts, dtype=np.uint16),
        source_fragments=np.asarray(fragments, dtype=np.uint16),
        targets=np.asarray(targets, dtype=np.uint16),
        retrieved_targets=np.asarray(retrieved, dtype=np.uint16),
        query_positions=np.asarray(query_positions, dtype=np.int64),
        source_positions=np.asarray(source_positions, dtype=np.int64),
        hits_seen=hits_seen,
        tokens_scanned=final_position,
    )
