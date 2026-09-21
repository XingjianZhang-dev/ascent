import numpy as np

from experiments.run_composition_factorial import sum_posterior
from experiments.analyze_composition_coscale import ENDPOINTS


def test_sum_posterior_is_exact_convolution() -> None:
    first = np.array([[0.25, 0.75]])
    second = np.array([[0.6, 0.4]])
    result = sum_posterior(first, second)
    assert np.allclose(result, [[0.15, 0.55, 0.30]])
    assert np.allclose(result.sum(axis=1), 1.0)


def test_point_mass_sum_is_deterministic() -> None:
    first = np.eye(4)[[2]]
    second = np.eye(4)[[3]]
    result = sum_posterior(first, second)
    assert result[0, 5] == 1.0


def test_composition_coscale_endpoint_order_is_fixed() -> None:
    assert [endpoint for endpoint, _ in ENDPOINTS] == [
        "pythia-410m", "pythia-1b", "pythia-2.8b"
    ]
