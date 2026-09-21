from experiments.analyze_babilong_certified_factorial import paired_difference


def test_paired_difference_preserves_panel_pairing() -> None:
    assert paired_difference([0.75, 0.5, 1.0], [0.5, 0.5, 0.75]) == [
        0.25,
        0.0,
        0.25,
    ]


def test_null_factorial_interaction_is_not_positive() -> None:
    low_model_effect = paired_difference([0.75] * 5, [0.5] * 5)
    high_model_effect = paired_difference([0.75] * 5, [0.5] * 5)
    assert paired_difference(high_model_effect, low_model_effect) == [0.0] * 5
