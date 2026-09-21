"""Minimal causal/versioned episodic state used by ASCENT invariant tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class _Event:
    key: int
    value: int
    timestamp: int
    version: int
    refinements: NDArray[np.uint8]


class CausalEpisodicMemory:
    """Versioned event store whose reads are strictly earlier than the query."""

    def __init__(self) -> None:
        self._events: list[_Event] = []
        self._versions: dict[int, int] = {}

    def write(
        self,
        *,
        key: int,
        value: int,
        timestamp: int,
        refinements: NDArray[np.uint8],
    ) -> None:
        data = np.asarray(refinements, dtype=np.uint8)
        if data.ndim != 2 or data.size == 0 or np.any(data > 1):
            raise ValueError("refinements must be a non-empty binary [rounds, bits] array")
        version = self._versions.get(key, 0) + 1
        self._versions[key] = version
        self._events.append(
            _Event(
                key=key,
                value=value,
                timestamp=timestamp,
                version=version,
                refinements=data.copy(),
            )
        )

    def read(self, *, key: int, query_timestamp: int) -> tuple[int, int] | None:
        candidates = [
            event
            for event in self._events
            if event.key == key and event.timestamp < query_timestamp
        ]
        if not candidates:
            return None
        event = max(candidates, key=lambda item: (item.timestamp, item.version))
        return event.value, event.version

    def signal(
        self, *, key: int, query_timestamp: int, rounds: int
    ) -> NDArray[np.uint8] | None:
        candidates = [
            event
            for event in self._events
            if event.key == key and event.timestamp < query_timestamp
        ]
        if not candidates:
            return None
        event = max(candidates, key=lambda item: (item.timestamp, item.version))
        if rounds < 1 or rounds > event.refinements.shape[0]:
            raise ValueError("requested rounds are outside the stored refinement range")
        return event.refinements[:rounds].copy()

