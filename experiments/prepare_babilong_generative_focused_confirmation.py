#!/usr/bin/env python3
"""Freeze the focused ten-panel generative confirmation before scoring."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = load(args.source_config)
    manifest = load(args.panel_manifest)
    design = load(args.design)
    if design["status"] != "frozen_before_any_focused_confirmation_decoder_score":
        raise RuntimeError("focused confirmation design is not frozen")
    panels = manifest["panels"]
    if [panel["start_in_stable_hash_order"] for panel in panels] != design[
        "panel_selection"
    ]["starts"]:
        raise RuntimeError("panel starts differ from the frozen design")
    if len(panels) != 10 or any(panel["rows"] != 120 for panel in panels):
        raise RuntimeError("expected ten 120-row panels")

    config = copy.deepcopy(source)
    config["experiment"] = "ascent_babilong_smollm2_8k_generative_focused_confirmation"
    config["status"] = "prospective_focused_ten_panel_confirmation_before_any_decoder_score"
    config["panels"] = [panel["panel"] for panel in panels]
    config["panel_sha256_by_name"] = {
        panel["panel"]: panel["sha256"] for panel in panels
    }
    config["confirmatory_design_sha256"] = sha256_file(args.design)
    config["focused_confirmation_design"] = design
    config["primary_estimands"] = design["primary_analysis"]
    config["confirmation_gate"] = design["primary_analysis"]["gate"]
    config["conditions"] = {
        key: value
        for key, value in source["conditions"].items()
        if key in {"generative_slots_2", "generative_slots_3"}
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
