import pytest

from experiments.analyze_babilong_around7b_extension import (
    _mapping,
    failure_accounting,
    one_sided_positive_p,
    paired_summary,
)


def test_one_sided_positive_p_handles_deterministic_signs():
    assert one_sided_positive_p([0.2] * 10) == 0.0
    assert one_sided_positive_p([0.0] * 10) == 1.0
    assert one_sided_positive_p([-0.2] * 10) == 1.0


def test_paired_summary_preserves_panel_values():
    result = paired_summary([0.4, 0.7, 0.6], [0.2, 0.3, 0.5])
    assert result["panel_values"] == pytest.approx([0.2, 0.4, 0.1])
    assert result["mean"] == pytest.approx(0.7 / 3)


def test_failure_accounting_retains_invalids_and_regressions():
    rows = [
        {"foundation_answer": None, "ascent_answer": "x", "foundation_score": 0, "ascent_score": 1},
        {"foundation_answer": "x", "ascent_answer": None, "foundation_score": 1, "ascent_score": 0},
        {"foundation_answer": "x", "ascent_answer": "x", "foundation_score": 1, "ascent_score": 1},
    ]
    result = failure_accounting(rows)
    assert result == {
        "samples": 3,
        "foundation_invalid_outputs": 1,
        "ascent_invalid_outputs": 1,
        "wins": 1,
        "regressions": 1,
        "ties": 1,
    }


def test_mapping_fails_on_duplicate_names():
    with pytest.raises(ValueError):
        _mapping(["qwen=/a", "qwen=/b"])
