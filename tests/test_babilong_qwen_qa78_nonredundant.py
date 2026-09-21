import hashlib
import json
from pathlib import Path

from ascent.babilong_memory import read_babilong
from experiments.analyze_babilong_qwen_qa78_nonredundant import analyze
from experiments.prepare_babilong_structured_aggregation import projected_answer


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "babilong_qwen2p5_qa78_8k_nonredundant_factorial.json"
DATA = ROOT / "data" / "babilong_qwen_qa78_8k_nonredundant"


def test_nonredundant_config_and_data_are_frozen_score_blind():
    config = json.loads(CONFIG.read_text())
    manifest_path = DATA / "MANIFEST.json"
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == config[
        "data_manifest_sha256"
    ]
    assert config["scale_law"]["slots_per_model_parameter"] == [
        3 / config["endpoints"][0]["model_parameters"],
        7 / config["endpoints"][1]["model_parameters"],
    ]
    assert config["scale_law"]["slots_per_model_parameter"][1] < config[
        "scale_law"
    ]["slots_per_model_parameter"][0]
    source_ids: set[str] = set()
    for name, expected_hash in config["panel_sha256_by_name"].items():
        path = DATA / f"{name}.jsonl"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 24
        assert {task: sum(row["task"] == task for row in rows) for task in config["tasks"]} == {
            "qa7": 12,
            "qa8": 12,
        }
        for row in rows:
            assert row["source_row_id"] not in source_ids
            source_ids.add(row["source_row_id"])
            read = read_babilong(row["input"], row["question"])
            small = projected_answer(read, row["task"], 3)
            large = projected_answer(read, row["task"], 7)
            full = projected_answer(read, row["task"], len(read.facts))
            assert small != large == full
            assert read.answer == row["target"]
    assert len(source_ids) == 144


def _prediction(row: int, foundation: float, ascent: float, condition: str):
    small = condition == "state3"
    suffix = [f"event-{index}" for index in range(3)]
    facts = suffix if small else [f"earlier-{index}" for index in range(4)] + suffix
    return {
        "row_id": f"row-{row}",
        "target": "two",
        "foundation_output": "one",
        "foundation_answer": "one",
        "foundation_score": foundation,
        "ascent_output": "two",
        "ascent_answer": "two",
        "ascent_score": ascent,
        "retained_facts": facts,
        "structured_inventory": "small" if small else "large",
        "structured_count": "one" if small else "two",
    }


def _result(endpoint, condition, config_sha, data_sha, foundation, ascent):
    parameters = 3085938688 if endpoint.startswith("qwen2p5-3b") else 7615616512
    return {
        "endpoint": {"name": endpoint},
        "condition": {
            "name": condition,
            "fact_slots": int(condition.removeprefix("state")),
            "memory_representation": "structured_inventory",
            "readout_path": "foundation_generation",
            "diagnostic_no_promotion": False,
        },
        "model_identity": {"loaded_parameters": parameters},
        "config": {"sha256": config_sha},
        "data": {"sha256": data_sha, "task_counts": {"qa7": 12, "qa8": 12}},
        "environment": {"git_commit": "frozen", "git_dirty": False},
        "parser_accuracy": 1.0,
        "foundation": {"mean": foundation},
        "ascent": {"mean": ascent},
        "gain": {"mean": ascent - foundation},
        "remaining_error_elimination": (ascent - foundation) / (1 - foundation),
        "wins": 12,
        "regressions": 0,
        "predictions": [
            _prediction(index, foundation, ascent, condition) for index in range(24)
        ],
    }


def test_nonredundant_development_analyzer_passes_positive_factorial(tmp_path):
    config = json.loads(CONFIG.read_text())
    config_sha = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    data_sha = config["panel_sha256_by_name"]["headroom_development"]
    output = tmp_path / "development"
    output.mkdir()
    values = {
        ("qwen2p5-3b-instruct", "state3"): (0.10, 0.30),
        ("qwen2p5-3b-instruct", "state7"): (0.10, 0.40),
        ("qwen2p5-7b-instruct", "state3"): (0.20, 0.45),
        ("qwen2p5-7b-instruct", "state7"): (0.20, 0.80),
    }
    for (endpoint, condition), (foundation, ascent) in values.items():
        (output / f"{endpoint}_{condition}.json").write_text(
            json.dumps(
                _result(
                    endpoint,
                    condition,
                    config_sha,
                    data_sha,
                    foundation,
                    ascent,
                )
            )
        )
    result = analyze(CONFIG, tmp_path, "development")
    assert result["provenance"]["all_panel_checks_pass"]
    assert result["panels"][0]["semantic_changes"] == {
        "qwen2p5-3b-instruct": 24,
        "qwen2p5-7b-instruct": 24,
    }
    assert result["gates"]["development_pass"]
