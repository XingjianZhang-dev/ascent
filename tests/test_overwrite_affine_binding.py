import pytest

pytest.importorskip("torch")  # optional GPU-stack / audit dependency; skipped in the CPU-only public tree

import numpy as np
import torch

from experiments.run_overwrite_affine_binding import (
    affine_posterior,
    ordered_history,
)


def test_affine_posterior_matches_deterministic_target() -> None:
    first = np.eye(4, dtype=np.float64)[[0, 2, 3]]
    second = np.eye(4, dtype=np.float64)[[1, 3, 0]]
    result = affine_posterior(first, second)
    expected = np.array([1, 7, 6])
    assert np.array_equal(result.argmax(axis=1), expected)
    assert np.allclose(result.sum(axis=1), 1.0)


def test_affine_posterior_preserves_soft_probability_mass() -> None:
    first = np.array([[0.25, 0.75]])
    second = np.array([[0.6, 0.4]])
    result = affine_posterior(first, second)
    expected = np.array([[0.15, 0.10, 0.45, 0.30]])
    assert np.allclose(result, expected)


def test_ordered_history_randomizes_key_order_not_chronology() -> None:
    first_old = torch.tensor([[[1.0]], [[2.0]]])
    first_current = torch.tensor([[[3.0]], [[4.0]]])
    second_old = torch.tensor([[[10.0]], [[20.0]]])
    second_current = torch.tensor([[[30.0]], [[40.0]]])
    result = ordered_history(
        first_old,
        first_current,
        second_old,
        second_current,
        np.array([True, False]),
    )
    expected = torch.tensor(
        [[[1.0], [3.0], [10.0], [30.0]], [[20.0], [40.0], [2.0], [4.0]]]
    )
    assert torch.equal(result, expected)
