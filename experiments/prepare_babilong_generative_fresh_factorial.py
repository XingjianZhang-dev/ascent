#!/usr/bin/env python3
"""Freeze the five-panel generative factorial config before decoder scoring."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = load(args.source_config)
    panel_manifest = load(args.panel_manifest)
    design = load(args.design)
    if design["status"] != "frozen_before_any_fresh_panel_decoder_score":
        raise RuntimeError("fresh design is not frozen")
    panels = panel_manifest["panels"]
    expected_starts = design["panel_selection"]["starts"]
    if [panel["start_in_stable_hash_order"] for panel in panels] != expected_starts:
        raise RuntimeError("panel starts differ from the frozen design")
    if any(panel["rows"] != 120 for panel in panels):
        raise RuntimeError("every fresh panel must contain 120 rows")

    config = copy.deepcopy(source)
    config["experiment"] = "ascent_babilong_smollm2_8k_generative_fresh_factorial"
    config["status"] = "prospective_five_panel_factorial_frozen_before_any_fresh_decoder_score"
    config["panels"] = [panel["panel"] for panel in panels]
    config["panel_sha256_by_name"] = {
        panel["panel"]: panel["sha256"] for panel in panels
    }
    config["confirmatory_design_sha256"] = sha256_file(args.design)
    config["factorial_source_config"] = {
        "path": str(args.source_config),
        "sha256": sha256_file(args.source_config),
    }
    config["official_source"] = panel_manifest["official_source"]
    config["fresh_factorial_design"] = design
    config["primary_estimands"] = design["primary_analysis"]
    config["confirmation_gate"] = design["primary_analysis"]["gate"]
    config["supporting_combined_gate"] = design["supporting_analysis"]["gate"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
