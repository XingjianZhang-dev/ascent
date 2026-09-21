import pytest

from experiments.run_ruler_foundation_headroom import headroom_decision


def test_headroom_gate_passes_only_inside_frozen_interval() -> None:
    passed = headroom_decision([0.0, 1.0, 0.0, 1.0], 0.1, 0.85)
    assert passed["passes"]
    assert passed["observed_score"] == 0.5
    assert passed["observed_errors"] == 2


def test_headroom_gate_distinguishes_hard_from_saturated() -> None:
    hard = headroom_decision([0.0] * 10, 0.1, 0.85)
    saturated = headroom_decision([1.0] * 10, 0.1, 0.85)
    assert hard["failure_reason"] == "too_hard_or_decode_invalid"
    assert saturated["failure_reason"] == "saturated"


def test_headroom_gate_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError):
        headroom_decision([0.5], 0.9, 0.1)
