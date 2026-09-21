"""Strong, target-blind controls for the official BABILong experiments."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ascent.babilong_memory import extract_babilong_events
from ascent.generic_retrieval import retrieve_lexical_chain


@dataclass(frozen=True)
class MatchedRawRead:
    passages: tuple[str, ...]
    cached_events: tuple[str, ...]
    payload_bytes: int
    budget_bytes: int


def matched_raw_event_fifo(
    input_text: str,
    question: str,
    *,
    count: int,
    budget_bytes: int,
) -> MatchedRawRead:
    """Store a raw FIFO over the same query-blind detected event stream.

    The semantic payload cost is exact UTF-8 source bytes, matching ASCENT's
    semantic-payload accounting.  Writes never inspect the question.  At read
    time, the generic lexical chain selects at most ``count`` cached events.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    if budget_bytes <= 0:
        raise ValueError("budget_bytes must be positive")
    cache: deque[tuple[str, int]] = deque()
    payload_bytes = 0
    for event in extract_babilong_events(input_text):
        source_bytes = len(event.source.encode("utf-8"))
        if source_bytes > budget_bytes:
            continue
        cache.append((event.source, source_bytes))
        payload_bytes += source_bytes
        while cache and payload_bytes > budget_bytes:
            _, removed_bytes = cache.popleft()
            payload_bytes -= removed_bytes
    cached_events = tuple(source for source, _ in cache)
    retrieval = retrieve_lexical_chain(
        " ".join(cached_events), question, count=count
    )
    return MatchedRawRead(
        passages=retrieval.passages,
        cached_events=cached_events,
        payload_bytes=payload_bytes,
        budget_bytes=budget_bytes,
    )
