#!/usr/bin/env python3
"""Run the exact and Monte Carlo Stage-A certified-channel experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ascent.certified_channel import CertifiedChannelConfig, NoisyRefinementChannel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(values.size))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def run(config_path: Path, output_path: Path | None = None) -> Path:
    raw: dict[str, Any] = json.loads(config_path.read_text())
    channel = NoisyRefinementChannel(
        CertifiedChannelConfig(
            num_labels=int(raw["num_labels"]),
            crossover_probability=float(raw["crossover_probability"]),
        )
    )
    rounds_by_scale = [int(value) for value in raw["rounds_by_scale"]]
    params = [int(value) for value in raw["model_parameters_by_scale"]]
    widths = [int(value) for value in raw["model_width_by_scale"]]
    if not (len(rounds_by_scale) == len(params) == len(widths)):
        raise ValueError("registered scale arrays must have the same length")

    exact = channel.exact_scale_curve(
        rounds_by_scale,
        chunk_size=int(raw["exact_enumeration_chunk_size"]),
    )
    rng = np.random.default_rng(int(raw["seed"]))
    labels, observations = channel.sample(
        rng,
        int(raw["monte_carlo_episodes"]),
        max(rounds_by_scale),
    )
    baseline = math.log(channel.config.num_labels)
    empirical: list[dict[str, float | int]] = []
    previous_nll = np.full(labels.shape, baseline, dtype=np.float64)
    failures: list[str] = []

    for index, rounds in enumerate(rounds_by_scale):
        nll = channel.true_label_nll(labels, observations[:, :rounds, :])
        gains = baseline - nll
        increments = previous_nll - nll
        nll_mean, nll_low, nll_high = _mean_ci(nll)
        gain_mean, gain_low, gain_high = _mean_ci(gains)
        inc_mean, inc_low, inc_high = _mean_ci(increments)
        row = {
            "scale_index": index,
            "rounds": rounds,
            "model_parameters": params[index],
            "model_width": widths[index],
            "nll_mean_nats": nll_mean,
            "nll_ci95_low": nll_low,
            "nll_ci95_high": nll_high,
            "gain_mean_nats": gain_mean,
            "gain_ci95_low": gain_low,
            "gain_ci95_high": gain_high,
            "increment_mean_nats": inc_mean,
            "increment_ci95_low": inc_low,
            "increment_ci95_high": inc_high,
        }
        empirical.append(row)
        if abs(nll_mean - float(exact[index]["conditional_entropy_nats"])) > float(
            raw["empirical_exact_tolerance_nats"]
        ):
            failures.append(f"scale {index}: Monte Carlo NLL differs from exact NLL")
        if float(exact[index]["conditional_information_increment_nats"]) <= float(
            raw["strict_gain_epsilon_nats"]
        ):
            failures.append(f"scale {index}: exact refinement information is not strict")
        previous_nll = nll

    for index in range(1, len(exact)):
        gain_difference = float(exact[index]["gain_nats"]) - float(exact[index - 1]["gain_nats"])
        identity_error = abs(
            gain_difference - float(exact[index]["conditional_information_increment_nats"])
        )
        if identity_error > float(raw["identity_tolerance_nats"]):
            failures.append(f"scale transition {index - 1}->{index}: information identity failed")

    prefix_checks: list[dict[str, int | bool]] = []
    sample_signal = observations[0]
    for small, large in zip(rounds_by_scale, rounds_by_scale[1:]):
        small_bits = sample_signal[:small].reshape(-1)
        projected_large_bits = sample_signal[:large].reshape(-1)[: small_bits.size]
        passed = bool(np.array_equal(small_bits, projected_large_bits))
        prefix_checks.append({"small_rounds": small, "large_rounds": large, "passed": passed})
        if not passed:
            failures.append(f"prefix nesting failed for rounds {small}->{large}")

    result = {
        "schema_version": 1,
        "status": "passed" if not failures else "failed",
        "experiment": raw["experiment"],
        "started_from_utc": datetime.now(timezone.utc).isoformat(),
        "config": raw,
        "config_sha256": _sha256(config_path),
        "provenance": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "git_commit": _git_value(["rev-parse", "HEAD"]),
            "git_dirty": bool(_git_value(["status", "--porcelain"])),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "exact": exact,
        "empirical": empirical,
        "prefix_checks": prefix_checks,
        "failures": failures,
    }

    if output_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = Path("results/stage_a") / f"certified_channel_{stamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if failures:
        raise RuntimeError("Stage A failed: " + "; ".join(failures))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = run(args.config, args.output)
    print(output)


if __name__ == "__main__":
    main()
