from __future__ import annotations

import pytest

from ascent.digit_grammar import digit_extension


def test_first_piece_allows_only_leading_whitespace_and_ascii_digits() -> None:
    assert digit_extension(" 376", first=True, remaining=7) == "376"
    assert digit_extension("\n42", first=True, remaining=7) == "42"
    assert digit_extension("376", first=True, remaining=7) == "376"
    assert digit_extension("376 ", first=True, remaining=7) is None
    assert digit_extension("3.", first=True, remaining=7) is None


def test_later_piece_is_digits_only_and_respects_remaining_length() -> None:
    assert digit_extension("7469", first=False, remaining=4) == "7469"
    assert digit_extension(" 7469", first=False, remaining=4) is None
    assert digit_extension("7469", first=False, remaining=3) is None


def test_remaining_must_be_positive() -> None:
    with pytest.raises(ValueError):
        digit_extension("1", first=True, remaining=0)
