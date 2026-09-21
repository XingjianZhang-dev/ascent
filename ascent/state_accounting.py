"""Exact persistent-state and small-decoder accounting for ASCENT."""

from __future__ import annotations


def latent_state_bytes(tokens: int, width: int, bytes_per_element: int) -> int:
    """Return dense latent-state payload bytes, excluding transient activations."""
    if tokens < 1 or width < 1 or bytes_per_element < 1:
        raise ValueError("tokens, width, and bytes_per_element must be positive")
    return tokens * width * bytes_per_element


def affine_head_parameters(width: int, classes: int) -> int:
    """Return weight-plus-bias parameters for a single affine evidence head."""
    if width < 1 or classes < 2:
        raise ValueError("width must be positive and classes at least two")
    return classes * (width + 1)


def state_element_ratio(tokens: int, width: int, model_parameters: int) -> float:
    """Return persistent latent elements divided by foundation parameters."""
    if model_parameters < 1:
        raise ValueError("model_parameters must be positive")
    return latent_state_bytes(tokens, width, 1) / model_parameters
