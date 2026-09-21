"""Task-level output grammar for numeric RULER answers."""

from __future__ import annotations

import re


_FIRST_DIGITS = re.compile(r"[ \t\r\n]*([0-9]+)")
_LATER_DIGITS = re.compile(r"([0-9]+)")


def digit_extension(piece: str, *, first: bool, remaining: int) -> str | None:
    """Return the digits contributed by a tokenizer piece, if grammar-valid.

    The first token may carry tokenizer-introduced leading whitespace. Later
    pieces must contain ASCII digits only. No answer value is supplied here;
    this helper encodes only a public task output format.
    """
    if remaining < 1:
        raise ValueError("remaining must be positive")
    match = (_FIRST_DIGITS if first else _LATER_DIGITS).fullmatch(piece)
    if match is None:
        return None
    digits = match.group(1)
    if len(digits) > remaining:
        return None
    return digits
