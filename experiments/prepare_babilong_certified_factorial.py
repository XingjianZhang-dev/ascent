#!/usr/bin/env python3
"""Derive the frozen 3x3 certified model-by-state factorial configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ENDPOINTS = ("smollm2-135m", "smollm2-360m-node2", "smollm2-1p7b")
STATE_SLOTS = (1, 2, 8)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive(base: dict[str, Any], base_path: Path) -> dict[str, Any]:
    result = dict(base)
    result.update(
        {
            "experiment": "ascent_babilong_qa78_certified_factorial_smollm2_8k",
            "status": "prospective_certified_factorial_frozen_before_any_factorial_score",
            "factorial_source_config": {
                "path": str(base_path),
                "sha256": sha256_file(base_path),
            },
            "factorial_design": {
                "foundation_endpoints": list(ENDPOINTS),
                "state_fact_slots": list(STATE_SLOTS),
                "cells": 9,
                "panels_per_cell": 5,
                "estimand": (
                    "For each state size, compare ASCENT accuracy and ASCENT-minus-"
                    "Foundation gain across model endpoints; then compare the registered "
                    "co-scaled diagonal against fixed-state rows. Report the full matrix "
                    "even if the foundation-by-state interaction is null or negative."
                ),
                "gate": (
                    "All provenance and Foundation-prediction equality checks pass; the "
                    "co-scaled diagonal has positive adjacent gain increments; and any "
                    "claim that larger models extract more value from the same state "
                    "requires a positive panel-clustered interaction lower bound."
                ),
            },
            "conditions": {
                f"certified_slots_{slots}": {
                    "memory_source": "relevant",
                    "memory_representation": "structured_inventory",
                    "neutral_answer_format": True,
                    "readout_path": "certified_evidence",
                    "state_fact_slots_by_endpoint": {
                        endpoint: slots for endpoint in ENDPOINTS
                    },
                }
                for slots in STATE_SLOTS
            },
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.base.read_text())
    result = derive(base, args.base)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
