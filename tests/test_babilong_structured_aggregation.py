from experiments.analyze_babilong_structured_aggregation import mean_ci


def test_five_panel_interval_uses_student_t_with_four_degrees_of_freedom() -> None:
    result = mean_ci([0.1, 0.2, 0.3, 0.4, 0.5])
    assert result["mean"] == 0.3
    assert result["ci95_low"] < 0.11
    assert result["ci95_high"] > 0.49
