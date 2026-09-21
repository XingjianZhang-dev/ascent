"""Per-row target-blindness instrumentation for the BABILong write path.

Array revision 1, item C-08. Kept separate from :mod:`ascent.babilong_memory` so
that the static invariant "the writer module never names the target field"
(``tests/test_babilong_target_blind_static.py``) remains meaningful: the writer
cannot read the target because it is never given the row, and this module
makes that structural fact machine-checkable per row.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ascent.babilong_memory import BabiFact


class TargetAccessError(RuntimeError):
    """Raised when the write/readout/prompt path reads a blinded row field."""


class TargetBlindRow(Mapping[str, Any]):
    """Read-only view of a panel row whose target is inaccessible until revealed.

    The writer, readout, and prompt builder receive this view instead of the raw
    row. Reading a blinded key before :meth:`reveal` raises
    :class:`TargetAccessError`; every key read before the reveal is recorded so
    that each result row can carry a machine-checkable record of what the
    state-construction path saw. Iteration (``dict(row)``, ``{**row}``) before the
    reveal also raises, because it would read the blinded key.
    """

    __slots__ = ("_row", "_blinded", "_revealed_phase", "keys_read_before_reveal", "blocked_attempts")

    def __init__(self, row: Mapping[str, Any], blinded: tuple[str, ...] = ("target",)) -> None:
        self._row = row
        self._blinded = frozenset(blinded)
        self._revealed_phase: str | None = None
        self.keys_read_before_reveal: set[str] = set()
        self.blocked_attempts = 0

    def __getitem__(self, key: str) -> Any:
        if self._revealed_phase is None:
            if key in self._blinded:
                self.blocked_attempts += 1
                raise TargetAccessError(f"{key!r} is blinded until reveal()")
            self.keys_read_before_reveal.add(key)
        return self._row[key]

    def __iter__(self):
        for key in self._row:
            if self._revealed_phase is None and key in self._blinded:
                self.blocked_attempts += 1
                raise TargetAccessError(f"iteration exposes blinded key {key!r} before reveal()")
            yield key

    def __len__(self) -> int:
        return len(self._row)

    def reveal(self, phase: str) -> None:
        """Make the blinded keys readable; ``phase`` names where in the runner this happened."""
        if self._revealed_phase is not None:
            raise RuntimeError("reveal() called twice")
        self._revealed_phase = phase

    @property
    def revealed_phase(self) -> str | None:
        return self._revealed_phase

    def record(self) -> dict[str, Any]:
        return {
            "blinded_keys": sorted(self._blinded),
            "keys_read_before_reveal": sorted(self.keys_read_before_reveal),
            "blocked_target_access_attempts": self.blocked_attempts,
            "revealed_phase": self._revealed_phase,
            "target_read_before_reveal": bool(self.keys_read_before_reveal & self._blinded),
        }


def canonical_event_template(fact: BabiFact) -> str:
    """The canonical rendering of one parsed event, from its own fields only."""
    if fact.kind == "move":
        return f"{fact.person} moved to the {fact.location}."
    if fact.kind == "pickup":
        return f"{fact.person} picked up the {fact.object_name}."
    if fact.kind == "drop":
        return f"{fact.person} dropped the {fact.object_name}."
    if fact.kind == "transfer":
        return f"{fact.person} gave the {fact.object_name} to {fact.recipient}."
    raise ValueError(f"unknown event kind {fact.kind!r}")


def fact_provenance_record(
    input_text: str,
    facts: tuple[BabiFact, ...],
    rendered_sources: list[str],
) -> dict[str, Any]:
    """Machine-checkable record that retained state derives from the input alone.

    Three links are checked for every retained fact: (1) ``fact.source`` is the
    verbatim input substring at ``fact.character_position``; (2) every parsed
    entity field (person, location, object, recipient) occurs in that source
    sentence; (3) the string actually placed in the prompt is either the raw
    source sentence or the canonical template filled from the fact's own fields.
    """
    if len(facts) != len(rendered_sources):
        raise ValueError("facts and rendered sources differ in length")
    verbatim = [
        input_text[fact.character_position : fact.character_position + len(fact.source)] == fact.source
        for fact in facts
    ]
    fields_in_source = [
        all(
            value in fact.source.lower()
            for value in (fact.person, fact.location, fact.object_name, fact.recipient)
            if value is not None
        )
        for fact in facts
    ]
    rendered_from_fact = [
        rendered == fact.source or rendered == canonical_event_template(fact)
        for fact, rendered in zip(facts, rendered_sources, strict=True)
    ]
    return {
        "retained_facts": len(facts),
        "all_sources_verbatim_at_recorded_position": all(verbatim),
        "all_parsed_fields_present_in_source_sentence": all(fields_in_source),
        "all_rendered_strings_derived_from_own_fact": all(rendered_from_fact),
        "state_derived_from_input_only": all(verbatim) and all(fields_in_source) and all(rendered_from_fact),
    }
