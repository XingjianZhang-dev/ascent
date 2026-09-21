import pytest

from experiments.analyze_babilong_generative_fresh_factorial import (
    T95_DF4,
    T95_DF7,
    mean_ci,
)


def test_fresh_interval_uses_five_panel_df4_critical() -> None:
    interval = mean_ci([0.02, 0.04, 0.06, 0.08, 0.10], T95_DF4)
    assert interval["mean"] == pytest.approx(0.06)
    assert interval["ci95_low"] == pytest.approx(0.0207351, abs=1e-6)


def test_combined_interval_uses_eight_panel_df7_critical() -> None:
    interval = mean_ci([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08], T95_DF7)
    assert interval["mean"] == pytest.approx(0.045)
    assert interval["ci95_low"] == pytest.approx(0.0245218, abs=1e-6)
