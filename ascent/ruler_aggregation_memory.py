"""Bounded causal frequency state for official RULER aggregation tasks."""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass


FWE_CONTEXT_START = (
    "Find the three most frequently appeared coded words. "
)
CWE_CONTEXT_START = "Memorize the ones that appear most often.\n"
QUESTION_MARKER = "\nQuestion:"
CODED_WORD_PATTERN = re.compile(r"(?<![a-z])[a-z]{6}(?![a-z])")
NUMBERED_WORD_PATTERN = re.compile(
    r"\d+\.\s+(.*?)(?=\s+\d+\.\s+|$)"
)


@dataclass(frozen=True)
class FrequencyEntry:
    word: str
    count: int
    first_character_position: int


@dataclass(frozen=True)
class FrequencyRead:
    entries: tuple[FrequencyEntry, ...]
    top_words: tuple[str, ...]
    persistent_payload_bytes: int


def read_ruler_frequency(
    input_text: str, *, memory_slots: int = 256
) -> FrequencyRead:
    """Count only the coded-word stream before the held-out final question.

    The official FWE generator emits six-letter lowercase codes between a
    fixed task instruction and the final question.  The query and answer are
    therefore excluded from every write.  ``memory_slots`` bounds the exact
    counter table; exceeding it is an explicit protocol failure rather than an
    unreported eviction that could depend on the target.
    """
    if memory_slots <= 0:
        raise ValueError("memory_slots must be positive")
    start = input_text.find(FWE_CONTEXT_START)
    if start < 0:
        raise ValueError("official FWE context marker is missing")
    start += len(FWE_CONTEXT_START)
    end = input_text.rfind(QUESTION_MARKER)
    if end <= start:
        raise ValueError("official FWE final question marker is missing")

    counts: OrderedDict[str, list[int]] = OrderedDict()
    for match in CODED_WORD_PATTERN.finditer(input_text, start, end):
        word = match.group(0)
        if word not in counts:
            if len(counts) >= memory_slots:
                raise RuntimeError("FWE exact counter exceeded the frozen slot bound")
            counts[word] = [0, match.start()]
        counts[word][0] += 1

    entries = tuple(
        FrequencyEntry(word, count, position)
        for word, (count, position) in counts.items()
    )
    ranked = sorted(entries, key=lambda row: (-row.count, row.first_character_position))
    payload_bytes = sum(
        len(row.word.encode("utf-8")) + 8 + 8 for row in entries
    )
    return FrequencyRead(
        entries=entries,
        top_words=tuple(row.word for row in ranked),
        persistent_payload_bytes=payload_bytes,
    )


def read_ruler_common_words(
    input_text: str, *, memory_slots: int = 512
) -> FrequencyRead:
    """Count the official CWE numbered list before the final question."""
    if memory_slots <= 0:
        raise ValueError("memory_slots must be positive")
    start = input_text.find(CWE_CONTEXT_START)
    if start < 0:
        raise ValueError("official CWE context marker is missing")
    start += len(CWE_CONTEXT_START)
    end = input_text.rfind(QUESTION_MARKER)
    if end <= start:
        raise ValueError("official CWE final question marker is missing")

    counts: OrderedDict[str, list[int]] = OrderedDict()
    for match in NUMBERED_WORD_PATTERN.finditer(input_text, start, end):
        word = match.group(1).strip()
        if word not in counts:
            if len(counts) >= memory_slots:
                raise RuntimeError("CWE exact counter exceeded the frozen slot bound")
            counts[word] = [0, match.start(1)]
        counts[word][0] += 1
    entries = tuple(
        FrequencyEntry(word, count, position)
        for word, (count, position) in counts.items()
    )
    ranked = sorted(entries, key=lambda row: (-row.count, row.first_character_position))
    payload_bytes = sum(
        len(row.word.encode("utf-8")) + 8 + 8 for row in entries
    )
    return FrequencyRead(
        entries=entries,
        top_words=tuple(row.word for row in ranked),
        persistent_payload_bytes=payload_bytes,
    )
