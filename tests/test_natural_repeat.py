import numpy as np
import pytest

from ascent.natural_repeat import collect_natural_repeat_events, persistent_state_bytes


def test_events_are_causal_read_before_write_and_aligned() -> None:
    block = np.array([7, 8, 9, 10, 11, 12], dtype=np.uint16)
    tokens = np.tile(block, 5)
    events = collect_natural_repeat_events(
        tokens,
        event_offset=0,
        num_events=4,
        key_length=2,
        context_length=4,
        source_tokens=3,
        min_distance=4,
        memory_slots=16,
        eos_token_id=None,
    )
    assert np.all(events.source_positions < events.query_positions)
    assert np.all(events.query_positions - events.source_positions >= 4)
    for context, source, target, query_position, source_position in zip(
        events.query_contexts,
        events.source_fragments,
        events.targets,
        events.query_positions,
        events.source_positions,
        strict=True,
    ):
        assert np.array_equal(context, tokens[query_position - 4 : query_position])
        assert np.array_equal(source, tokens[source_position - 2 : source_position + 1])
        assert target == tokens[query_position]


def test_document_boundary_prevents_cross_document_hit() -> None:
    tokens = np.array([1, 2, 3, 4, 0, 1, 2, 3, 4, 0], dtype=np.uint16)
    with pytest.raises(RuntimeError, match="only 0 requested events"):
        collect_natural_repeat_events(
            tokens,
            event_offset=0,
            num_events=1,
            key_length=2,
            context_length=2,
            source_tokens=2,
            min_distance=1,
            memory_slots=16,
            eos_token_id=0,
        )


def test_packed_state_accounting_is_monotone() -> None:
    small = persistent_state_bytes(memory_slots=8, key_length=2, source_tokens=4)
    large = persistent_state_bytes(memory_slots=8, key_length=2, source_tokens=8)
    assert small == 8 * ((2 + 4 + 1) * 2 + 8 + 1)
    assert large > small
