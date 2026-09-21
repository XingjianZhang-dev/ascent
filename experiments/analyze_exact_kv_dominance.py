#!/usr/bin/env python3
"""Quantify why simple fresh recall cannot justify beating optimized exact KV."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ascent.capacity_diagnostic import exact_kv_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.config.read_text())
    rows = exact_kv_comparison(raw)
    all_dominate = all(row["exact_kv_dominates"] for row in rows)
    payload = {
        "schema_version": 1,
        "status": "proposal_risk_confirmed" if all_dominate else "not_dominated",
        "interpretation": (
            "Optimized clean exact KV dominates the noisy code on this simple-recall abstraction; "
            "do not use this task to claim superiority over matched raw."
        ),
        "config": raw,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
