from experiments.analyze_babilong_scale_panel import mean_ci


def test_panel_interval_uses_student_t_with_two_degrees_of_freedom() -> None:
    result = mean_ci([0.2, 0.3, 0.4], critical=4.302652729911275)
    assert result["mean"] == 0.3
    assert result["ci95_low"] < 0.1
    assert result["ci95_high"] > 0.5
