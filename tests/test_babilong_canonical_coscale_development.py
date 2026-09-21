import json
from pathlib import Path

from experiments.run_natural_repeat import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_coscale_development_freezes_unseen_small_mid_cells() -> None:
    path = ROOT / "configs" / "babilong_qwen2p5_canonical_coscale_development.json"
    config = json.loads(path.read_text())
    assert config["promotion"] is False
    assert config["status"].endswith("no_promotion")
    assert config["analysis_scope"]["primary_tasks"] == ["qa2", "qa3"]
    assert config["outcome_visibility_at_freeze"] == {
        "qwen2p5-0p5b-instruct": "unseen under canonical representation",
        "qwen2p5-1p5b-instruct": "unseen under canonical representation",
        "qwen2p5-3b-instruct": "already observed in the preceding posthoc diagnostic",
    }
    assert [row["state_fact_slots"] for row in config["endpoints"]] == [1, 2, 3]
    for panel, expected in config["panel_sha256_by_name"].items():
        data = (
            ROOT
            / "data"
            / "babilong_1k_8k_generative_focused_confirmation"
            / f"{panel}.jsonl"
        )
        assert sha256_file(data) == expected
