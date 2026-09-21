import pytest

from experiments.audit_babilong_generative_factorial import (
    clustered_interval,
    remaining_error_elimination,
)


def test_independent_clustered_interval_uses_df2() -> None:
    result = clustered_interval([1 / 12, 1 / 24, 0.0])
    assert result["mean"] == pytest.approx(1 / 24)
    assert result["ci95_low"] == pytest.approx(-0.0618391, abs=1e-6)


def test_remaining_error_elimination_is_derived_from_primitives() -> None:
    result = {
        "foundation": {"mean": 0.25},
        "gain": {"mean": 0.375},
    }
    assert remaining_error_elimination(result) == pytest.approx(0.5)
