"""Evidence extraction from text reconstructed out of ASCENT latent state."""

from __future__ import annotations

import re


_PARTIAL_NIAH_NUMBER = re.compile(
    r"special magic numbers? for (.+?) is:\s*([0-9]+)", re.IGNORECASE
)


def reconstructed_niah_digit_prefix(
    reconstructed_text: str,
    question: str,
    *,
    max_digits: int = 7,
) -> str:
    """Return a query-matched numeric prefix from reconstructed latent text."""
    if max_digits < 1:
        raise ValueError("max_digits must be positive")
    for match in _PARTIAL_NIAH_NUMBER.finditer(reconstructed_text):
        key, digits = match.group(1).strip(), match.group(2)
        if key and key in question:
            return digits[:max_digits]
    return ""
