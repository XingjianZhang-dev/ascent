import numpy as np

from ascent.episodic_memory import CausalEpisodicMemory


def test_read_before_write_at_same_timestamp() -> None:
    memory = CausalEpisodicMemory()
    bits = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    memory.write(key=3, value=7, timestamp=10, refinements=bits)
    assert memory.read(key=3, query_timestamp=10) is None
    assert memory.read(key=3, query_timestamp=11) == (7, 1)


def test_overwrite_is_versioned_and_causal() -> None:
    memory = CausalEpisodicMemory()
    old = np.array([[0, 0], [0, 1]], dtype=np.uint8)
    new = np.array([[1, 1], [1, 0]], dtype=np.uint8)
    memory.write(key=4, value=2, timestamp=2, refinements=old)
    before_future_write = memory.read(key=4, query_timestamp=5)
    memory.write(key=4, value=9, timestamp=8, refinements=new)
    assert memory.read(key=4, query_timestamp=5) == before_future_write == (2, 1)
    assert memory.read(key=4, query_timestamp=9) == (9, 2)


def test_signal_projection_masks_new_refinements() -> None:
    memory = CausalEpisodicMemory()
    signal = np.array([[0, 1], [1, 1], [0, 0]], dtype=np.uint8)
    memory.write(key=1, value=6, timestamp=1, refinements=signal)
    small = memory.signal(key=1, query_timestamp=2, rounds=1)
    large = memory.signal(key=1, query_timestamp=2, rounds=3)
    assert small is not None and large is not None
    assert np.array_equal(small, large[:1])

