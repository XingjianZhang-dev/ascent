from __future__ import annotations

import pytest

from ascent.latent_digit_probe import (
    completed_digit_runs,
    digit_class_targets,
    longest_digit_run,
)


def test_digit_class_targets_uses_explicit_token_map() -> None:
    assert digit_class_targets([50, 99, 51], {50: 0, 51: 1}) == [0, 10, 1]


def test_longest_digit_run_prefers_latest_tie_and_caps_length() -> None:
    assert longest_digit_run([10, 1, 2, 10, 3, 4]) == "34"
    assert longest_digit_run([1, 2, 3, 4, 5, 6, 7, 8]) == "1234567"
    assert longest_digit_run([10, 10]) == ""


def test_longest_digit_run_validates_limit() -> None:
    with pytest.raises(ValueError):
        longest_digit_run([1], max_digits=0)


def test_completed_digit_runs_preserve_order_and_drop_partial_tail() -> None:
    classes = [10, 1, 2, 3, 4, 5, 6, 7, 10, 8, 7, 6, 5, 4, 3, 2, 10, 9, 8]
    assert completed_digit_runs(classes, digits_per_run=7) == ["1234567", "8765432"]


def test_completed_digit_runs_honor_max_runs() -> None:
    classes = [1] * 7 + [10] + [2] * 7
    assert completed_digit_runs(classes, digits_per_run=7, max_runs=1) == ["1111111"]
