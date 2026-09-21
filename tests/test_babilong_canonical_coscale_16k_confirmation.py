import json
from pathlib import Path

import pytest

from experiments.analyze_babilong_canonical_coscale_16k_confirmation import (
    _one_sided_positive_p,
    holm_adjust,
)
from experiments.run_natural_repeat import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_16k_panel_hashes_and_no_row_overlap() -> None:
    config = json.loads(
        (ROOT / "configs/babilong_qwen2p5_canonical_coscale_16k_confirmatory.json").read_text()
    )
    assert config["status"].startswith("prospective_16k_confirmation")
    assert config["tasks"] == ["qa2", "qa3"]
    seen: set[str] = set()
    for panel in config["panels"]:
        path = ROOT / "data/babilong_16k_canonical_confirmation" / f"{panel}.jsonl"
        if not path.exists():
            pytest.skip("benchmark row bodies are not redistributed; rebuild the panels with experiments/prepare_babilong_panel.py")
        assert sha256_file(path) == config["panel_sha256_by_name"][panel]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 80
        assert {row["task"] for row in rows} == {"qa2", "qa3"}
        ids = {row["row_id"] for row in rows}
        assert len(ids) == 80
        assert not seen.intersection(ids)
        seen.update(ids)


def test_holm_adjustment_is_step_down_monotone() -> None:
    adjusted = holm_adjust([0.03, 0.01])
    assert adjusted == pytest.approx([0.03, 0.02])


def test_one_sided_positive_test_direction() -> None:
    assert _one_sided_positive_p([0.1] * 10) == 0.0
    assert _one_sided_positive_p([-0.1] * 10) == 1.0
    assert _one_sided_positive_p([0.2, 0.1, 0.0, 0.1]) < 0.05
