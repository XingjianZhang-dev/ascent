"""Causal bounded ASCENT memory for official RULER variable tracking."""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass


ASSIGNMENT_PATTERN = re.compile(
    r"^VAR[ \t]+([A-Z]+)[ \t]*=[ \t]*(?:VAR[ \t]+([A-Z]+)|(\d+))[ \t]*$",
    re.MULTILINE,
)
TARGET_PATTERN = re.compile(r"assigned the value\s+(\d+)")
TASK_MARKER = "Memorize and track the chain(s) of variable assignment"


@dataclass(frozen=True)
class VariableAssignment:
    variable: str
    parent: str | None
    scalar: str | None
    source: str
    character_position: int


@dataclass(frozen=True)
class VariableTrackingRead:
    question: str
    facts: tuple[VariableAssignment, ...]
    values: tuple[str, ...]
    persistent_payload_bytes: int


def read_ruler_vt(input_text: str, *, memory_slots: int = 256) -> VariableTrackingRead:
    """Store assignments in stream order and retrieve the queried value chain.

    RULER prompts may contain solved demonstrations before the final instance.
    Only the final instance is written to memory. The final question is held out
    until after all writes, and duplicate variables follow bounded LRU overwrite
    semantics.
    """
    if memory_slots <= 0:
        raise ValueError("memory_slots must be positive")
    marker_start = input_text.rfind(TASK_MARKER)
    segment = input_text[marker_start:] if marker_start >= 0 else input_text
    lines = [line for line in segment.splitlines() if line.strip()]
    if not lines:
        raise ValueError("input_text must contain a query")
    question = lines[-1]
    target_match = TARGET_PATTERN.search(question)
    if target_match is None:
        raise ValueError("variable-tracking query does not name a scalar value")
    target = target_match.group(1)
    query_start = segment.rfind(question)

    memory: OrderedDict[str, VariableAssignment] = OrderedDict()
    for match in ASSIGNMENT_PATTERN.finditer(segment[:query_start]):
        variable, parent, scalar = match.groups()
        assignment = VariableAssignment(
            variable=variable,
            parent=parent,
            scalar=scalar,
            source=match.group(0),
            character_position=match.start(),
        )
        memory[variable] = assignment
        memory.move_to_end(variable)
        while len(memory) > memory_slots:
            memory.popitem(last=False)

    def resolved_scalar(variable: str) -> str | None:
        visited: set[str] = set()
        current = variable
        while current in memory and current not in visited:
            visited.add(current)
            assignment = memory[current]
            if assignment.scalar is not None:
                return assignment.scalar
            if assignment.parent is None:
                return None
            current = assignment.parent
        return None

    selected = [
        assignment
        for variable, assignment in memory.items()
        if resolved_scalar(variable) == target
    ]
    selected.sort(key=lambda assignment: assignment.character_position)
    payload_bytes = sum(
        len(assignment.variable.encode("utf-8"))
        + len((assignment.parent or "").encode("utf-8"))
        + len((assignment.scalar or "").encode("utf-8"))
        + len(assignment.source.encode("utf-8"))
        + 8
        for assignment in memory.values()
    )
    return VariableTrackingRead(
        question=question,
        facts=tuple(selected),
        values=tuple(assignment.variable for assignment in selected),
        persistent_payload_bytes=payload_bytes,
    )
