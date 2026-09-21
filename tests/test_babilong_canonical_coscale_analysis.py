import json
from pathlib import Path

import pytest

from experiments.analyze_babilong_canonical_coscale_development import (
    _panel_values,
    _scope,
)
from experiments.prepare_babilong_panel import write_panel


def _row(panel: str, task: str, foundation: float, ascent: float) -> dict:
    return {
        "panel": panel,
        "task": task,
        "foundation_score": foundation,
        "ascent_score": ascent,
    }


def test_panel_values_pool_frozen_primary_tasks() -> None:
    rows = [
        _row("p1", "qa1", 0.0, 0.0),
        _row("p1", "qa2", 0.0, 1.0),
        _row("p1", "qa3", 1.0, 1.0),
        _row("p2", "qa2", 0.0, 0.0),
        _row("p2", "qa3", 0.0, 1.0),
    ]
    assert _panel_values(rows, ["p1", "p2"], {"qa2", "qa3"}, "gain") == [
        0.5,
        0.5,
    ]


def test_scope_uses_panel_cluster_inference_and_rwe() -> None:
    rows = [
        _row("p1", "qa2", 0.5, 1.0),
        _row("p2", "qa2", 0.0, 0.5),
    ]
    result = _scope(rows, ["p1", "p2"], {"qa2"})
    assert result["gain"]["mean"] == pytest.approx(0.5)
    assert result["remaining_error_elimination"]["panel_values"] == [1.0, 0.5]


def test_development_config_remains_explicitly_non_promotional() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs/babilong_qwen2p5_canonical_coscale_development.json").read_text()
    )
    assert config["promotion"] is False
    assert config["outcome_visibility_at_freeze"]["qwen2p5-3b-instruct"].startswith(
        "already observed"
    )


def test_16k_confirmation_panel_writer_supports_qa2_qa3_only(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for task in ("qa2", "qa3"):
        rows = [
            {"input": f"{task} input {index}", "question": "where?", "target": "x"}
            for index in range(2)
        ]
        (source / f"{task}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    result = write_panel(
        source,
        tmp_path / "panels",
        panel_name="p",
        start=0,
        rows_per_task=2,
        tasks=("qa2", "qa3"),
    )
    assert result["rows"] == 4
    assert result["tasks"] == ["qa2", "qa3"]
