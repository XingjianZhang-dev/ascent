"""Analytic fail-closed capacity checks for strong exact-KV baselines."""

from __future__ import annotations

from math import log
from typing import Any

from .certified_channel import CertifiedChannelConfig, NoisyRefinementChannel


def exact_kv_comparison(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare clean exact-KV and noisy ASCENT codes on simple uniform recall.

    Both methods use the same address fingerprint and metadata. Queries are
    uniform over writes, retention is slot-limited, a miss emits the uniform
    distribution, and address collisions are omitted equally. This favorable
    simplification cannot reverse the dominance because exact KV uses no more
    bits per slot and carries the full clean value on every hit.
    """

    labels = int(config["num_labels"])
    channel = NoisyRefinementChannel(
        CertifiedChannelConfig(
            num_labels=labels,
            crossover_probability=float(config["crossover_probability"]),
        )
    )
    writes = int(config["num_writes"])
    metadata = int(config["metadata_bits_per_slot"])
    baseline_nll = log(labels)
    rows: list[dict[str, Any]] = []
    for scale in config["scales"]:
        budget = int(scale["state_budget_bits"])
        fingerprint = int(scale["fingerprint_bits"])
        rounds = int(scale["ascent_rounds"])
        raw_slot_bits = fingerprint + channel.config.code_bits + metadata
        ascent_slot_bits = fingerprint + rounds * channel.config.code_bits + metadata
        raw_slots = budget // raw_slot_bits
        ascent_slots = budget // ascent_slot_bits
        raw_hit_rate = min(raw_slots / writes, 1.0)
        ascent_hit_rate = min(ascent_slots / writes, 1.0)
        ascent_hit_gain = float(channel.exact_scale_curve([rounds])[0]["gain_nats"])
        raw_gain = raw_hit_rate * baseline_nll
        ascent_gain = ascent_hit_rate * ascent_hit_gain
        rows.append(
            {
                "scale": scale["name"],
                "state_budget_bits": budget,
                "fingerprint_bits": fingerprint,
                "raw_slot_bits": raw_slot_bits,
                "ascent_slot_bits": ascent_slot_bits,
                "raw_slots": raw_slots,
                "ascent_slots": ascent_slots,
                "raw_hit_rate": raw_hit_rate,
                "ascent_hit_rate": ascent_hit_rate,
                "raw_gain_nats": raw_gain,
                "ascent_gain_nats": ascent_gain,
                "raw_expected_nll_nats": baseline_nll - raw_gain,
                "ascent_expected_nll_nats": baseline_nll - ascent_gain,
                "raw_minus_ascent_gain_nats": raw_gain - ascent_gain,
                "exact_kv_dominates": bool(
                    raw_slot_bits <= ascent_slot_bits
                    and baseline_nll >= ascent_hit_gain
                    and raw_gain >= ascent_gain
                ),
            }
        )
    return rows

