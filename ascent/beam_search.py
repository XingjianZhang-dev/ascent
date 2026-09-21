"""Small deterministic helpers for matched ASCENT/foundation beam search."""

from __future__ import annotations

import math


def normalized_score(log_probability: float, length: int, alpha: float) -> float:
    """Length-normalize a cumulative log probability for beam ranking."""
    if length <= 0:
        raise ValueError("length must be positive")
    if not math.isfinite(log_probability):
        raise ValueError("log_probability must be finite")
    if alpha < 0.0:
        raise ValueError("alpha must be nonnegative")
    return log_probability / (length**alpha)

