import json
from pathlib import Path

from experiments.run_natural_repeat import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def test_resolved_event_diagnostic_is_explicitly_nonpromotional_and_hashed() -> None:
    path = ROOT / "configs" / "babilong_qwen2p5_3b_resolved_event_diagnostic.json"
    config = json.loads(path.read_text())
    assert config["promotion"] is False
    assert config["status"].endswith("no_promotion")
    assert set(config["conditions"]) == {"resolved_slots_3"}
    assert config["conditions"]["resolved_slots_3"] == {
        "memory_source": "relevant",
        "memory_representation": "resolved_event_facts",
        "state_fact_slots_by_endpoint": {"qwen2p5-3b-instruct": 3},
    }
    for panel, expected in config["panel_sha256_by_name"].items():
        data = (
            ROOT
            / "data"
            / "babilong_1k_8k_generative_focused_confirmation"
            / f"{panel}.jsonl"
        )
        assert sha256_file(data) == expected
