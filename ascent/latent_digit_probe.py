"""Utilities for a learned digit-evidence decoder over ASCENT latent tokens."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def digit_class_targets(
    token_ids: Iterable[int], digit_token_ids: Mapping[int, int]
) -> list[int]:
    """Map tokenizer IDs to digit classes 0--9 or the non-digit class 10."""
    return [int(digit_token_ids.get(int(token_id), 10)) for token_id in token_ids]


def longest_digit_run(classes: Iterable[int], *, max_digits: int = 7) -> str:
    """Return the latest longest contiguous 0--9 class run."""
    if max_digits < 1:
        raise ValueError("max_digits must be positive")
    best = ""
    current = ""
    for value in classes:
        value = int(value)
        if 0 <= value <= 9:
            current += str(value)
            if len(current) >= len(best):
                best = current
        else:
            current = ""
    return best[:max_digits]


def completed_digit_runs(
    classes: Iterable[int], *, digits_per_run: int = 7, max_runs: int | None = None
) -> list[str]:
    """Return completed digit runs in latent-token order."""
    if digits_per_run < 1:
        raise ValueError("digits_per_run must be positive")
    if max_runs is not None and max_runs < 1:
        raise ValueError("max_runs must be positive when provided")
    runs: list[str] = []
    current = ""

    def finish() -> None:
        nonlocal current
        if len(current) >= digits_per_run and (
            max_runs is None or len(runs) < max_runs
        ):
            runs.append(current[:digits_per_run])
        current = ""

    for value in classes:
        value = int(value)
        if 0 <= value <= 9:
            current += str(value)
        else:
            finish()
    finish()
    return runs
