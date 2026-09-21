import pytest

from ascent.beam_search import normalized_score


def test_normalized_score() -> None:
    assert normalized_score(-4.0, 4, 0.5) == pytest.approx(-2.0)
    assert normalized_score(-4.0, 4, 0.0) == pytest.approx(-4.0)


def test_normalized_score_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="length"):
        normalized_score(-1.0, 0, 1.0)
    with pytest.raises(ValueError, match="alpha"):
        normalized_score(-1.0, 1, -1.0)

