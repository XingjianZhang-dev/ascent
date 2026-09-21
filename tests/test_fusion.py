import numpy as np

from ascent.fusion import (
    apply_mixture,
    apply_true_probability_simplex,
    select_clustered_safe_true_probability_simplex,
    select_safe_mixture,
    select_safe_true_probability_simplex,
)


def test_safe_gate_enables_clear_memory_gain() -> None:
    labels = np.arange(400, dtype=np.int64) % 4
    base = np.full((400, 4), 0.25)
    memory = np.full((400, 4), 0.01 / 3)
    memory[np.arange(400), labels] = 0.99
    gate = select_safe_mixture(base, memory, labels)
    assert gate.enabled
    assert gate.mixture == 1.0
    assert gate.calibration_gain_lcb95 > 0.0


def test_safe_gate_falls_back_to_foundation_when_memory_hurts() -> None:
    labels = np.arange(400, dtype=np.int64) % 4
    base = np.full((400, 4), 0.25)
    memory = np.full((400, 4), 0.99 / 3)
    memory[np.arange(400), labels] = 0.01
    gate = select_safe_mixture(base, memory, labels)
    assert not gate.enabled
    assert gate.mixture == 0.0
    assert np.array_equal(apply_mixture(base, memory, gate.mixture), base)


def test_simplex_gate_combines_complementary_paths() -> None:
    base = np.full(1000, 0.1)
    certified = np.where(np.arange(1000) % 2 == 0, 0.95, 0.01)
    latent = np.where(np.arange(1000) % 2 == 1, 0.95, 0.01)
    gate = select_safe_true_probability_simplex(base, certified, latent)
    assert gate.enabled
    assert gate.certified_weight > 0.0
    assert gate.latent_weight > 0.0
    fused = apply_true_probability_simplex(
        base, certified, latent, gate.certified_weight, gate.latent_weight
    )
    assert float(-np.log(fused).mean()) < float(-np.log(base).mean())


def test_simplex_gate_returns_exact_foundation_when_both_paths_hurt() -> None:
    base = np.full(1000, 0.8)
    bad = np.full(1000, 0.01)
    gate = select_safe_true_probability_simplex(base, bad, bad)
    assert not gate.enabled
    assert gate.certified_weight == 0.0
    assert gate.latent_weight == 0.0
    assert gate.candidate_certified_weight == 0.0
    assert gate.candidate_latent_weight == 0.0


def test_clustered_simplex_equal_weights_answers_not_tokens() -> None:
    sample_ids = np.repeat(np.arange(40), np.where(np.arange(40) == 0, 100, 1))
    base = np.full(sample_ids.size, 0.2)
    certified = np.where(sample_ids == 0, 0.99, 0.9)
    latent = np.full(sample_ids.size, 0.01)
    gate = select_clustered_safe_true_probability_simplex(
        base, certified, latent, sample_ids
    )
    assert gate.enabled
    assert gate.certified_weight > 0.0
    assert gate.latent_weight == 0.0
    assert gate.calibration_gain_lcb95 > 0.0


def test_clustered_simplex_requires_multiple_samples() -> None:
    values = np.full(4, 0.5)
    with np.testing.assert_raises(ValueError):
        select_clustered_safe_true_probability_simplex(
            values, values, values, np.zeros(4, dtype=np.int64)
        )


def test_clustered_simplex_retains_unsafe_candidate_for_diagnostics() -> None:
    sample_ids = np.arange(20, dtype=np.int64)
    base = np.full(20, 0.2)
    certified = np.full(20, 0.2)
    certified[0] = 0.99
    latent = np.full(20, 0.2)
    gate = select_clustered_safe_true_probability_simplex(
        base, certified, latent, sample_ids
    )
    assert not gate.enabled
    assert gate.certified_weight == 0.0
    assert gate.latent_weight == 0.0
    assert gate.candidate_certified_weight > 0.0
    assert gate.candidate_latent_weight == 0.0
