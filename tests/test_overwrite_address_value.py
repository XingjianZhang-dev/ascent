import pytest

pytest.importorskip("torch")  # optional GPU-stack / audit dependency; skipped in the CPU-only public tree

import numpy as np
import torch

from experiments.run_overwrite_address_value import (
    masked_mean_hidden,
    ordered_pair,
    posterior_mean,
)


def test_ordered_pair_respects_registered_target_position() -> None:
    target = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
    distractor = torch.tensor([[[10.0]], [[20.0]], [[30.0]]])
    result = ordered_pair(
        target, distractor, np.array([True, False, True], dtype=bool)
    )
    expected = torch.tensor(
        [[[1.0], [10.0]], [[20.0], [2.0]], [[3.0], [30.0]]]
    )
    assert torch.equal(result, expected)


def test_posterior_mean_preserves_probability_weighting() -> None:
    codebook = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 0.0]],
            [[0.0, 1.0], [0.0, 2.0]],
        ]
    )
    probabilities = np.array([[0.25, 0.75], [1.0, 0.0]])
    result = posterior_mean(codebook, probabilities)
    expected = torch.tensor(
        [
            [[0.25, 0.75], [0.5, 1.5]],
            [[1.0, 0.0], [2.0, 0.0]],
        ]
    )
    assert torch.allclose(result, expected)


def test_masked_mean_hidden_ignores_padding_and_supports_variable_lengths() -> None:
    hidden = torch.tensor(
        [
            [[1.0, 3.0], [3.0, 5.0], [99.0, 99.0]],
            [[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    result = masked_mean_hidden(hidden, mask)
    expected = torch.tensor([[2.0, 4.0], [4.0, 6.0]])
    assert torch.allclose(result, expected)
