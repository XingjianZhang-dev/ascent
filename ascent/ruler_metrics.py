"""Metrics matching NVIDIA RULER's official synthetic-task evaluator."""

from __future__ import annotations


def string_match_all(prediction: str, references: list[str]) -> float:
    """Return the fraction of references found case-insensitively in a prediction."""
    if not references:
        raise ValueError("references must be nonempty")
    lowered = prediction.lower()
    return sum(reference.lower() in lowered for reference in references) / len(references)


def string_match_part(prediction: str, references: list[str]) -> float:
    """Return one when any reference occurs, matching RULER's QA metric."""
    if not references:
        raise ValueError("references must be nonempty")
    lowered = prediction.lower()
    return float(any(reference.lower() in lowered for reference in references))
