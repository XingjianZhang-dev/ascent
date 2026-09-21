"""Causal bounded ASCENT memory for the official RULER NIAH format."""

from __future__ import annotations

import re
from collections import OrderedDict, deque
from dataclasses import dataclass


FACT_PATTERN = re.compile(
    r"One of the special magic (?:numbers|uuids) for (.+?) is: ([^.\n]+)\."
)


@dataclass(frozen=True)
class RulerFact:
    key: str
    value: str
    source: str
    character_position: int


@dataclass(frozen=True)
class RulerRead:
    question: str
    facts: tuple[RulerFact, ...]
    values: tuple[str, ...]
    persistent_payload_bytes: int


def read_ruler_niah(input_text: str, *, memory_slots: int = 256) -> RulerRead:
    """Write facts in stream order, then retrieve only keys named by the query.

    The final nonempty line is treated as the query. It is never used while
    facts are written. The memory is bounded and uses deterministic LRU
    overwrite semantics for duplicate keys.
    """
    if memory_slots <= 0:
        raise ValueError("memory_slots must be positive")
    lines = [line for line in input_text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("input_text must contain a query")
    question = lines[-1]
    query_start = input_text.rfind(question)
    memory: OrderedDict[str, RulerFact] = OrderedDict()
    for match in FACT_PATTERN.finditer(input_text[:query_start]):
        key, value = match.group(1), match.group(2)
        fact = RulerFact(key, value, match.group(0), match.start())
        memory[key] = fact
        memory.move_to_end(key)
        while len(memory) > memory_slots:
            memory.popitem(last=False)
    selected = [fact for key, fact in memory.items() if key in question]
    selected.sort(key=lambda fact: question.index(fact.key))
    payload_bytes = sum(
        len(fact.key.encode("utf-8"))
        + len(fact.value.encode("utf-8"))
        + len(fact.source.encode("utf-8"))
        + 8
        for fact in memory.values()
    )
    return RulerRead(
        question=question,
        facts=tuple(selected),
        values=tuple(fact.value for fact in selected),
        persistent_payload_bytes=payload_bytes,
    )


def read_ruler_niah_all_values(
    input_text: str, *, memory_slots: int = 256
) -> RulerRead:
    """Retain a bounded append-only event log and return every queried value.

    Unlike ``read_ruler_niah``, this reader intentionally does not collapse
    repeated keys to their latest version. It is the registered semantics for
    RULER's ``niah_multivalue`` task, where all values for one key are targets.
    """
    if memory_slots <= 0:
        raise ValueError("memory_slots must be positive")
    lines = [line for line in input_text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("input_text must contain a query")
    question = lines[-1]
    query_start = input_text.rfind(question)
    memory: deque[RulerFact] = deque(maxlen=memory_slots)
    for match in FACT_PATTERN.finditer(input_text[:query_start]):
        key, value = match.group(1), match.group(2)
        memory.append(RulerFact(key, value, match.group(0), match.start()))
    selected = [fact for fact in memory if fact.key in question]
    selected.sort(key=lambda fact: (question.index(fact.key), fact.character_position))
    payload_bytes = sum(
        len(fact.key.encode("utf-8"))
        + len(fact.value.encode("utf-8"))
        + len(fact.source.encode("utf-8"))
        + 8
        for fact in memory
    )
    return RulerRead(
        question=question,
        facts=tuple(selected),
        values=tuple(fact.value for fact in selected),
        persistent_payload_bytes=payload_bytes,
    )
