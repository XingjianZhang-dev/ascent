import math

import numpy as np

from ascent.certified_channel import CertifiedChannelConfig, NoisyRefinementChannel


def test_posterior_is_normalized_and_prefers_clean_label() -> None:
    channel = NoisyRefinementChannel(CertifiedChannelConfig(num_labels=8, crossover_probability=0.1))
    label = 5
    observations = np.repeat(channel.codebook[label][None, None, :], 3, axis=1)
    log_posterior = channel.log_posterior(observations)
    assert np.allclose(np.exp(log_posterior).sum(axis=1), 1.0)
    assert int(np.argmax(log_posterior[0])) == label


def test_exact_gain_is_strict_and_matches_information_increment() -> None:
    channel = NoisyRefinementChannel(CertifiedChannelConfig(num_labels=16, crossover_probability=0.18))
    rows = channel.exact_scale_curve([1, 2, 3, 4], chunk_size=1024)
    assert all(float(row["conditional_information_increment_nats"]) > 0.0 for row in rows)
    for previous, current in zip(rows, rows[1:]):
        gain_difference = float(current["gain_nats"]) - float(previous["gain_nats"])
        assert math.isclose(
            gain_difference,
            float(current["conditional_information_increment_nats"]),
            abs_tol=1e-12,
        )


def test_registered_signal_is_prefix_nested() -> None:
    channel = NoisyRefinementChannel(CertifiedChannelConfig())
    _, observations = channel.sample(np.random.default_rng(7), 1, 4)
    flat = observations[0].reshape(-1)
    for small, large in [(1, 2), (2, 3), (3, 4)]:
        assert np.array_equal(flat[: small * channel.config.code_bits], flat[: large * channel.config.code_bits][: small * channel.config.code_bits])

