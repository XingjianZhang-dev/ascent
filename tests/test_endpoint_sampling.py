import json
from pathlib import Path

import numpy as np

from experiments.run_stage_b_endpoint import array_sha256, sample_registered_benchmark


ROOT = Path(__file__).resolve().parents[1]


def test_registered_sampling_is_endpoint_invariant_and_prefix_nested() -> None:
    raw = json.loads((ROOT / "configs" / "stage_b_endpoints.json").read_text())
    first = sample_registered_benchmark(raw)
    second = sample_registered_benchmark(raw)
    keys_a, labels_a, observations_a, _, maximum_rounds = first
    keys_b, labels_b, observations_b, _, _ = second

    assert maximum_rounds == 4
    assert keys_a == keys_b
    assert np.array_equal(labels_a, labels_b)
    assert np.array_equal(observations_a, observations_b)
    assert array_sha256(observations_a[:, :1, :]) == array_sha256(
        observations_b[:, :1, :]
    )
    assert np.array_equal(observations_a[:, :1, :], observations_a[:, :4, :][:, :1, :])
