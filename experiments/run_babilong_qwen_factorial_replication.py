#!/usr/bin/env python3
"""Run assigned endpoint cells from a frozen BABILong factorial."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.run_natural_repeat import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def planned_cells(
    config: dict[str, Any],
    endpoint_names: list[str],
    panels: list[str] | None = None,
) -> list[tuple[str, int, str]]:
    known = {endpoint["name"] for endpoint in config["endpoints"]}
    if not endpoint_names or len(set(endpoint_names)) != len(endpoint_names):
        raise RuntimeError("endpoint assignments must be nonempty and unique")
    unknown = set(endpoint_names) - known
    if unknown:
        raise RuntimeError(f"unknown endpoints: {sorted(unknown)}")
    state_slots = tuple(
        int(value)
        for value in config.get("factorial", {}).get(
            "state_fact_slots", (1, 2, 3)
        )
    )
    if not state_slots or any(value <= 0 for value in state_slots):
        raise RuntimeError("factorial state slots must be nonempty and positive")
    if tuple(sorted(set(state_slots))) != state_slots:
        raise RuntimeError("factorial state slots must be unique and increasing")
    return [
        (endpoint, slots, panel)
        for endpoint in endpoint_names
        for slots in state_slots
        for panel in (config["panels"] if panels is None else panels)
    ]


def valid_completed_cell(
    path: Path,
    config: dict[str, Any],
    config_sha256: str,
    endpoint: str,
    slots: int,
    panel: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return all(
        (
            row.get("status") == config["status"],
            row.get("endpoint", {}).get("name") == endpoint,
            row.get("condition", {}).get("name") == f"generative_slots_{slots}",
            row.get("condition", {}).get("fact_slots") == slots,
            row.get("config", {}).get("sha256") == config_sha256,
            row.get("data", {}).get("sha256")
            == config["panel_sha256_by_name"][panel],
            row.get("parser_accuracy") == 1.0,
            len(row.get("predictions", [])) == int(config["evaluation_samples"]),
            not row.get("environment", {}).get("git_dirty", True),
        )
    )


def write_progress(
    path: Path,
    *,
    assigned: int,
    completed: int,
    skipped: int,
    current: dict[str, Any] | None,
    status: str,
) -> None:
    document = {
        "schema_version": 1,
        "status": status,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "assigned_cells": assigned,
        "completed_cells": completed,
        "skipped_valid_cells": skipped,
        "current_cell": current,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n")


def run(
    config_path: Path,
    data_root: Path,
    model_root: Path,
    output_root: Path,
    node_label: str,
    endpoint_names: list[str],
    phase: str | None = None,
) -> None:
    config = json.loads(config_path.read_text())
    config_sha256 = sha256_file(config_path)
    panels = None
    if phase is not None:
        panels = list(config[f"{phase}_panels"])
    cells = planned_cells(config, endpoint_names, panels)
    node_root = output_root / node_label
    progress_path = output_root / f"progress_{node_label}.json"
    completed = 0
    skipped = 0
    write_progress(
        progress_path,
        assigned=len(cells),
        completed=completed,
        skipped=skipped,
        current=None,
        status="running",
    )
    for endpoint, slots, panel in cells:
        filename = f"generative_slots_{slots}_{panel}_{endpoint}.json"
        output = node_root / filename
        cell = {"endpoint": endpoint, "slots": slots, "panel": panel}
        if valid_completed_cell(
            output, config, config_sha256, endpoint, slots, panel
        ):
            completed += 1
            skipped += 1
            write_progress(
                progress_path,
                assigned=len(cells),
                completed=completed,
                skipped=skipped,
                current=cell,
                status="running",
            )
            continue
        command = [
            sys.executable,
            str(ROOT / "experiments" / "run_babilong_prompt.py"),
            "--config",
            str(config_path),
            "--data",
            str(
                data_root
                / config.get("panel_path_by_name", {}).get(
                    panel, f"{panel}.jsonl"
                )
            ),
            "--endpoint",
            endpoint,
            "--model-dir",
            str(model_root / endpoint),
            "--condition",
            f"generative_slots_{slots}",
            "--output",
            str(output),
        ]
        print(json.dumps({"event": "cell_start", **cell}), flush=True)
        write_progress(
            progress_path,
            assigned=len(cells),
            completed=completed,
            skipped=skipped,
            current=cell,
            status="running",
        )
        subprocess.run(command, cwd=ROOT, check=True)
        if not valid_completed_cell(
            output, config, config_sha256, endpoint, slots, panel
        ):
            raise RuntimeError(f"completed cell failed validation: {output}")
        completed += 1
        print(
            json.dumps(
                {
                    "event": "cell_complete",
                    **cell,
                    "completed_cells": completed,
                    "assigned_cells": len(cells),
                }
            ),
            flush=True,
        )
    write_progress(
        progress_path,
        assigned=len(cells),
        completed=completed,
        skipped=skipped,
        current=None,
        status="complete",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--node-label", required=True)
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--phase", choices=("development", "confirmation"))
    args = parser.parse_args()
    run(
        args.config,
        args.data_root,
        args.model_root,
        args.output_root,
        args.node_label,
        args.endpoint,
        args.phase,
    )


if __name__ == "__main__":
    main()
