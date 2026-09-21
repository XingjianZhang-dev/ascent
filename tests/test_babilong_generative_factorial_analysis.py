import pytest

from experiments.analyze_babilong_generative_factorial import mean_ci, paired


def test_registered_interaction_uses_paired_panel_differences() -> None:
    low_model_state = paired([0.50, 0.55, 0.60], [0.40, 0.45, 0.50])
    high_model_state = paired([0.80, 0.85, 0.90], [0.50, 0.55, 0.60])
    assert paired(high_model_state, low_model_state) == pytest.approx([0.2, 0.2, 0.2])


def test_three_panel_interval_uses_df2_critical_value() -> None:
    interval = mean_ci([1 / 12, 1 / 24, 0.0])
    assert interval["mean"] == pytest.approx(1 / 24)
    assert interval["ci95_low"] == pytest.approx(-0.0618391, abs=1e-6)
