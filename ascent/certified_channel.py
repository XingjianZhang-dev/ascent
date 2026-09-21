"""Exact registered channel used to test ASCENT's certified proposition.

The answer is fresh episode-level information.  A scale receives a prefix of
independent noisy observations of the answer code, and the certified decoder
returns the exact posterior under the registered binary symmetric channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log, log2
from typing import Iterable

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
UInt8Array = NDArray[np.uint8]


@dataclass(frozen=True)
class CertifiedChannelConfig:
    """Parameters fixed before evaluating the certified channel."""

    num_labels: int = 16
    crossover_probability: float = 0.18

    def __post_init__(self) -> None:
        if self.num_labels < 2 or self.num_labels & (self.num_labels - 1):
            raise ValueError("num_labels must be a power of two and >= 2")
        if not 0.0 < self.crossover_probability < 0.5:
            raise ValueError("crossover_probability must be strictly between 0 and 0.5")

    @property
    def code_bits(self) -> int:
        return int(log2(self.num_labels))


class NoisyRefinementChannel:
    """Append-only BSC observations with an exact Bayes decoder."""

    def __init__(self, config: CertifiedChannelConfig) -> None:
        self.config = config
        shifts = np.arange(config.code_bits - 1, -1, -1, dtype=np.uint64)
        labels = np.arange(config.num_labels, dtype=np.uint64)[:, None]
        self.codebook: UInt8Array = ((labels >> shifts) & 1).astype(np.uint8)
        self._log_match = log(1.0 - config.crossover_probability)
        self._log_flip = log(config.crossover_probability)

    def sample(
        self,
        rng: np.random.Generator,
        num_episodes: int,
        max_rounds: int,
    ) -> tuple[NDArray[np.int64], UInt8Array]:
        """Draw fresh labels and all refinement rounds in one reproducible call."""

        if num_episodes < 1 or max_rounds < 1:
            raise ValueError("num_episodes and max_rounds must be positive")
        labels = rng.integers(0, self.config.num_labels, size=num_episodes, dtype=np.int64)
        clean = self.codebook[labels, None, :]
        flips = (
            rng.random((num_episodes, max_rounds, self.config.code_bits))
            < self.config.crossover_probability
        ).astype(np.uint8)
        return labels, np.bitwise_xor(clean, flips)

    def log_posterior(self, observations: UInt8Array) -> FloatArray:
        """Return log p(label | observations) for every row.

        `observations` has shape ``[episodes, rounds, code_bits]``. The uniform
        prior is exact and independent of any foundation output.
        """

        obs = np.asarray(observations, dtype=np.uint8)
        if obs.ndim != 3 or obs.shape[2] != self.config.code_bits:
            raise ValueError("observations must have shape [episodes, rounds, code_bits]")
        matches = obs[:, :, None, :] == self.codebook[None, None, :, :]
        log_likelihood = np.where(matches, self._log_match, self._log_flip).sum(axis=(1, 3))
        maxima = log_likelihood.max(axis=1, keepdims=True)
        log_norm = maxima + np.log(np.exp(log_likelihood - maxima).sum(axis=1, keepdims=True))
        return log_likelihood - log_norm

    def true_label_nll(self, labels: NDArray[np.int64], observations: UInt8Array) -> FloatArray:
        labels = np.asarray(labels, dtype=np.int64)
        if labels.ndim != 1 or labels.shape[0] != observations.shape[0]:
            raise ValueError("labels must align one-to-one with observations")
        log_post = self.log_posterior(observations)
        return -log_post[np.arange(labels.size), labels]

    def exact_conditional_entropy(self, rounds: int, chunk_size: int = 4096) -> float:
        """Enumerate H(Y | Z_rounds) without Monte Carlo error."""

        if rounds < 1:
            raise ValueError("rounds must be positive")
        observed_bits = rounds * self.config.code_bits
        if observed_bits > 24:
            raise ValueError("exact enumeration is intentionally limited to 24 observed bits")
        pattern_count = 1 << observed_bits
        entropy = 0.0
        prior = 1.0 / self.config.num_labels
        bit_shifts = np.arange(observed_bits - 1, -1, -1, dtype=np.uint64)

        for start in range(0, pattern_count, chunk_size):
            stop = min(start + chunk_size, pattern_count)
            ids = np.arange(start, stop, dtype=np.uint64)[:, None]
            bits = ((ids >> bit_shifts) & 1).astype(np.uint8)
            obs = bits.reshape(stop - start, rounds, self.config.code_bits)
            matches = obs[:, :, None, :] == self.codebook[None, None, :, :]
            log_likelihood = np.where(matches, self._log_match, self._log_flip).sum(axis=(1, 3))
            likelihood = np.exp(log_likelihood)
            evidence = likelihood.mean(axis=1, keepdims=True)
            posterior = (prior * likelihood) / evidence
            joint = prior * likelihood
            entropy -= float(np.sum(joint * np.log(posterior)))
        return entropy

    def exact_scale_curve(
        self, rounds_by_scale: Iterable[int], chunk_size: int = 4096
    ) -> list[dict[str, float | int]]:
        """Compute exact entropy, gain, and conditional-information increments."""

        rounds_list = list(rounds_by_scale)
        if not rounds_list or any(b <= a for a, b in zip(rounds_list, rounds_list[1:])):
            raise ValueError("rounds_by_scale must be non-empty and strictly increasing")
        baseline_nll = log(self.config.num_labels)
        rows: list[dict[str, float | int]] = []
        previous_entropy = baseline_nll
        for scale_index, rounds in enumerate(rounds_list):
            entropy = self.exact_conditional_entropy(rounds, chunk_size=chunk_size)
            increment = previous_entropy - entropy
            rows.append(
                {
                    "scale_index": scale_index,
                    "rounds": rounds,
                    "observed_bits": rounds * self.config.code_bits,
                    "packed_signal_bytes": ceil(rounds * self.config.code_bits / 8),
                    "baseline_nll_nats": baseline_nll,
                    "conditional_entropy_nats": entropy,
                    "gain_nats": baseline_nll - entropy,
                    "conditional_information_increment_nats": increment,
                }
            )
            previous_entropy = entropy
        return rows

    @staticmethod
    def packed_prefix(observations: UInt8Array, rounds: int) -> bytes:
        """Serialize the registered signal in append-only bit order."""

        obs = np.asarray(observations, dtype=np.uint8)
        if obs.ndim != 2 or rounds < 1 or rounds > obs.shape[0]:
            raise ValueError("observations must be [rounds, code_bits] and contain the prefix")
        return np.packbits(obs[:rounds].reshape(-1), bitorder="big").tobytes()

