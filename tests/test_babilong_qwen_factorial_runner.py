import json
from pathlib import Path

from experiments.run_babilong_qwen_factorial_replication import planned_cells


def test_node_assignments_cover_all_ninety_factorial_cells_once() -> None:
    config = json.loads(
        Path(
            "configs/babilong_qwen2p5_8k_generative_factorial_replication_frozen.json"
        ).read_text()
    )
    node1 = planned_cells(config, ["qwen2p5-3b-instruct"])
    node2 = planned_cells(
        config, ["qwen2p5-0p5b-instruct", "qwen2p5-1p5b-instruct"]
    )
    assert len(node1) == 30
    assert len(node2) == 60
    assert len(set(node1 + node2)) == 90


def test_runner_uses_registered_nonunit_factorial_slots() -> None:
    config = {
        "endpoints": [{"name": "small"}, {"name": "large"}],
        "panels": ["development"],
        "factorial": {"state_fact_slots": [2, 3, 5]},
    }
    assert planned_cells(config, ["large"]) == [
        ("large", 2, "development"),
        ("large", 3, "development"),
        ("large", 5, "development"),
    ]
