import json
from pathlib import Path

from experiments.run_natural_repeat import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_event_diagnostic_is_nonpromotional_and_hashed() -> None:
    path = ROOT / "configs" / "babilong_qwen2p5_3b_canonical_event_diagnostic.json"
    config = json.loads(path.read_text())
    assert config["promotion"] is False
    assert config["status"].endswith("no_promotion")
    assert config["conditions"] == {
        "canonical_slots_3": {
            "memory_source": "relevant",
            "memory_representation": "canonical_event_facts",
            "state_fact_slots_by_endpoint": {"qwen2p5-3b-instruct": 3},
        }
    }
    assert "both original raw" in config["mandatory_comparison"]
    for panel, expected in config["panel_sha256_by_name"].items():
        data = (
            ROOT
            / "data"
            / "babilong_1k_8k_generative_focused_confirmation"
            / f"{panel}.jsonl"
        )
        assert sha256_file(data) == expected
