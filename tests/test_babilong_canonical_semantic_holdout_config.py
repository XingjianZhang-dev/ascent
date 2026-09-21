import json

import pytest
from pathlib import Path

from experiments.run_natural_repeat import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def test_semantic_holdout_config_is_frozen_and_hash_complete() -> None:
    path = ROOT / "configs/babilong_qwen2p5_canonical_semantic_holdout_8k_confirmatory.json"
    config = json.loads(path.read_text())
    assert config["status"].startswith("prospective_semantic_holdout")
    assert config["independence_scope"]["semantic_overlap_with_full_official_8k_test_pool"] == 0
    assert config["source"]["scope_guardrail"].startswith("This public split was generated for training")
    ids: set[str] = set()
    for panel in config["panels"]:
        data = ROOT / "data/babilong_train_8k_semantic_holdout_confirmation" / f"{panel}.jsonl"
        if not data.exists():
            pytest.skip("benchmark row bodies are not redistributed; rebuild the panels with experiments/prepare_babilong_panel.py")
        assert sha256_file(data) == config["panel_sha256_by_name"][panel]
        rows = [json.loads(line) for line in data.read_text().splitlines()]
        assert len(rows) == 80
        panel_ids = {row["row_id"] for row in rows}
        assert len(panel_ids) == 80
        assert not ids.intersection(panel_ids)
        ids.update(panel_ids)
