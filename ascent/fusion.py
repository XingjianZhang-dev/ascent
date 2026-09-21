"""Cross-fitted conservative fusion for ASCENT candidate distributions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GateSelection:
    mixture: float
    calibration_gain_nats: float
    calibration_gain_lcb95: float
    enabled: bool


@dataclass(frozen=True)
class SimplexGateSelection:
    certified_weight: float
    latent_weight: float
    candidate_certified_weight: float
    candidate_latent_weight: float
    calibration_gain_nats: float
    calibration_gain_lcb95: float
    enabled: bool


def _validate(probabilities: FloatArray) -> FloatArray:
    value = np.asarray(probabilities, dtype=np.float64)
    if value.ndim != 2 or np.any(value <= 0.0):
        raise ValueError("probabilities must be a strictly positive [episodes, labels] array")
    if not np.allclose(value.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("probability rows must sum to one")
    return value


def true_label_nll(probabilities: FloatArray, labels: NDArray[np.int64]) -> FloatArray:
    probs = _validate(probabilities)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (probs.shape[0],):
        raise ValueError("labels must align with probability rows")
    return -np.log(probs[np.arange(labels.size), labels])


def select_safe_mixture(
    foundation: FloatArray,
    memory: FloatArray,
    labels: NDArray[np.int64],
    *,
    grid_size: int = 101,
) -> GateSelection:
    """Select a mixture on calibration data and enable only with positive LCB.

    The admissible grid always includes zero, so the empirical optimizer can
    emulate the foundation. The LCB is computed on paired per-episode NLL gain.
    If it is not strictly positive, the returned gate is exactly zero.
    """

    base = _validate(foundation)
    mem = _validate(memory)
    if base.shape != mem.shape or grid_size < 2:
        raise ValueError("foundation/memory shapes must match and grid_size must be >= 2")
    base_nll = true_label_nll(base, labels)
    candidates = np.linspace(0.0, 1.0, grid_size)
    losses = []
    for mixture in candidates:
        fused = (1.0 - mixture) * base + mixture * mem
        losses.append(float(true_label_nll(fused, labels).mean()))
    selected = float(candidates[int(np.argmin(losses))])
    fused = (1.0 - selected) * base + selected * mem
    gains = base_nll - true_label_nll(fused, labels)
    mean = float(gains.mean())
    se = float(gains.std(ddof=1) / np.sqrt(gains.size)) if gains.size > 1 else float("inf")
    lcb = mean - 1.96 * se
    enabled = bool(selected > 0.0 and lcb > 0.0)
    return GateSelection(
        mixture=selected if enabled else 0.0,
        calibration_gain_nats=mean,
        calibration_gain_lcb95=lcb,
        enabled=enabled,
    )


def apply_mixture(foundation: FloatArray, memory: FloatArray, mixture: float) -> FloatArray:
    base = _validate(foundation)
    mem = _validate(memory)
    if base.shape != mem.shape or not 0.0 <= mixture <= 1.0:
        raise ValueError("invalid shapes or mixture")
    return (1.0 - mixture) * base + mixture * mem


def _validate_true_probabilities(values: FloatArray, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} probabilities must lie in [0, 1]")
    return array


def apply_true_probability_simplex(
    foundation_true: FloatArray,
    certified_true: FloatArray,
    latent_true: FloatArray,
    certified_weight: float,
    latent_weight: float,
) -> FloatArray:
    base = _validate_true_probabilities(foundation_true, name="foundation")
    certified = _validate_true_probabilities(certified_true, name="certified")
    latent = _validate_true_probabilities(latent_true, name="latent")
    if base.shape != certified.shape or base.shape != latent.shape:
        raise ValueError("all true-label probability arrays must align")
    if certified_weight < 0.0 or latent_weight < 0.0 or certified_weight + latent_weight > 1.0:
        raise ValueError("simplex weights must be nonnegative and sum to at most one")
    foundation_weight = 1.0 - certified_weight - latent_weight
    return foundation_weight * base + certified_weight * certified + latent_weight * latent


def select_safe_true_probability_simplex(
    foundation_true: FloatArray,
    certified_true: FloatArray,
    latent_true: FloatArray,
    *,
    grid_size: int = 21,
    epsilon: float = 1e-12,
) -> SimplexGateSelection:
    """Cross-fit a foundation/certified/latent mixture from true probabilities.

    This is sufficient for log-loss calibration and avoids retaining full-vocab
    distributions for natural-text experiments. The grid includes the exact
    foundation corner and the selected path is enabled only when its paired
    calibration-gain lower bound is strictly positive.
    """
    if grid_size < 2 or not 0.0 < epsilon < 1.0:
        raise ValueError("grid_size must be >=2 and epsilon must lie in (0, 1)")
    base = _validate_true_probabilities(foundation_true, name="foundation")
    certified = _validate_true_probabilities(certified_true, name="certified")
    latent = _validate_true_probabilities(latent_true, name="latent")
    if base.shape != certified.shape or base.shape != latent.shape or base.size == 0:
        raise ValueError("all nonempty probability arrays must align")
    grid = np.linspace(0.0, 1.0, grid_size)
    best_loss = float("inf")
    best = (0.0, 0.0)
    for certified_weight in grid:
        for latent_weight in grid:
            if certified_weight + latent_weight > 1.0 + 1e-12:
                continue
            fused = apply_true_probability_simplex(
                base, certified, latent, float(certified_weight), float(latent_weight)
            )
            loss = float(-np.log(np.clip(fused, epsilon, 1.0)).mean())
            if loss < best_loss:
                best_loss = loss
                best = (float(certified_weight), float(latent_weight))
    fused = apply_true_probability_simplex(base, certified, latent, *best)
    gains = -np.log(np.clip(base, epsilon, 1.0)) + np.log(np.clip(fused, epsilon, 1.0))
    mean = float(gains.mean())
    se = float(gains.std(ddof=1) / np.sqrt(gains.size)) if gains.size > 1 else float("inf")
    lcb = mean - 1.96 * se
    enabled = bool(sum(best) > 0.0 and lcb > 0.0)
    return SimplexGateSelection(
        certified_weight=best[0] if enabled else 0.0,
        latent_weight=best[1] if enabled else 0.0,
        candidate_certified_weight=best[0],
        candidate_latent_weight=best[1],
        calibration_gain_nats=mean,
        calibration_gain_lcb95=lcb,
        enabled=enabled,
    )


def select_clustered_safe_true_probability_simplex(
    foundation_true: FloatArray,
    certified_true: FloatArray,
    latent_true: FloatArray,
    sample_ids: NDArray[np.int64],
    *,
    grid_size: int = 21,
    epsilon: float = 1e-12,
) -> SimplexGateSelection:
    """Select a simplex by equal-weight sample NLL and gate on sample clusters.

    Multiple target tokens from one answer are correlated observations. This
    selector first averages token NLL within each sample, then chooses the
    empirical simplex and computes its confidence bound across samples. The
    exact Foundation corner is always admissible.
    """

    if grid_size < 2 or not 0.0 < epsilon < 1.0:
        raise ValueError("grid_size must be >=2 and epsilon must lie in (0, 1)")
    base = _validate_true_probabilities(foundation_true, name="foundation")
    certified = _validate_true_probabilities(certified_true, name="certified")
    latent = _validate_true_probabilities(latent_true, name="latent")
    clusters = np.asarray(sample_ids, dtype=np.int64)
    if (
        base.shape != certified.shape
        or base.shape != latent.shape
        or clusters.shape != base.shape
        or base.size == 0
    ):
        raise ValueError("probabilities and nonempty sample ids must align")
    unique = np.unique(clusters)
    if unique.size < 2:
        raise ValueError("at least two sample clusters are required")

    def clustered_nll(probabilities: FloatArray) -> FloatArray:
        token_nll = -np.log(np.clip(probabilities, epsilon, 1.0))
        return np.asarray(
            [token_nll[clusters == sample].mean() for sample in unique],
            dtype=np.float64,
        )

    base_nll = clustered_nll(base)
    grid = np.linspace(0.0, 1.0, grid_size)
    best_loss = float("inf")
    best = (0.0, 0.0)
    best_nll = base_nll
    for certified_weight in grid:
        for latent_weight in grid:
            if certified_weight + latent_weight > 1.0 + 1e-12:
                continue
            fused = apply_true_probability_simplex(
                base,
                certified,
                latent,
                float(certified_weight),
                float(latent_weight),
            )
            candidate_nll = clustered_nll(fused)
            loss = float(candidate_nll.mean())
            if loss < best_loss:
                best_loss = loss
                best = (float(certified_weight), float(latent_weight))
                best_nll = candidate_nll
    gains = base_nll - best_nll
    mean = float(gains.mean())
    standard_error = float(gains.std(ddof=1) / np.sqrt(gains.size))
    lcb = mean - 1.96 * standard_error
    enabled = bool(sum(best) > 0.0 and lcb > 0.0)
    return SimplexGateSelection(
        certified_weight=best[0] if enabled else 0.0,
        latent_weight=best[1] if enabled else 0.0,
        candidate_certified_weight=best[0],
        candidate_latent_weight=best[1],
        calibration_gain_nats=mean,
        calibration_gain_lcb95=lcb,
        enabled=enabled,
    )
