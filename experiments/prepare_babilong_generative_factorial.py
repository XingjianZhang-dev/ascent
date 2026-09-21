#!/usr/bin/env python3
"""Derive the frozen 3x3 generative BABILong model-by-state factorial."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ENDPOINTS = (
    "smollm2-135m-instruct",
    "smollm2-360m-instruct",
    "smollm2-1p7b-instruct",
)
STATE_SLOTS = (1, 2, 3)


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
            "experiment": "ascent_babilong_smollm2_8k_generative_factorial",
            "status": "prospective_generative_factorial_frozen_before_any_factorial_decoder_score",
            "factorial_source_config": {
                "path": str(base_path),
                "sha256": sha256_file(base_path),
            },
            "factorial_design": {
                "foundation_endpoints": list(ENDPOINTS),
                "state_fact_slots": list(STATE_SLOTS),
                "cells": 9,
                "panels_per_cell": 3,
                "readout": "foundation_generation",
                "estimand": (
                    "Exact generative ASCENT accuracy in all nine model-by-state cells; "
                    "state main effects, fixed-state model effects, model-by-state "
                    "interactions, and co-scaled improvement beyond memory-only scaling."
                ),
                "gate": (
                    "All provenance and Foundation-invariance checks pass; the co-scaled "
                    "diagonal has positive adjacent gain lower bounds; and both adjacent "
                    "model-by-state interaction lower bounds are positive for the matched "
                    "1-to-2 and 2-to-3 state expansions."
                ),
            },
            "conditions": {
                f"generative_slots_{slots}": {
                    "memory_source": "relevant",
                    "state_fact_slots_by_endpoint": {
                        endpoint: slots for endpoint in ENDPOINTS
                    },
                }
                for slots in STATE_SLOTS
            },
            "interpretation": (
                "Post-confirmation full factorial on the already frozen untouched QA1/QA2/QA3 "
                "panels. It tests whether the positive co-scaled curve reflects actual "
                "model-dependent use of ASCENT state rather than certified state size alone."
            ),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = derive(json.loads(args.base.read_text()), args.base)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
