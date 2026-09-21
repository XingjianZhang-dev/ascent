import hashlib
import json
from pathlib import Path

from experiments.analyze_babilong_qwen_qa78_headroom import analyze, mean_ci
from experiments.prepare_babilong_qwen_qa78_headroom import semantic_refinement


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "babilong_qwen2p5_qa78_8k_headroom_factorial.json"


def test_frozen_headroom_config_is_complete_and_parameter_sublinear():
    config = json.loads(CONFIG.read_text())
    assert config["status"].startswith("prospective_")
    assert config["tasks"] == ["qa7", "qa8"]
    assert set(config["conditions"]) == {"state4", "state8"}
    assert [endpoint["name"] for endpoint in config["endpoints"]] == [
        "qwen2p5-3b-instruct",
        "qwen2p5-7b-instruct",
    ]
    ratios = [
        4 / config["endpoints"][0]["model_parameters"],
        8 / config["endpoints"][1]["model_parameters"],
    ]
    assert ratios == config["scale_law"]["slots_per_model_parameter"]
    assert ratios[1] < ratios[0]
    for endpoint in config["endpoints"]:
        assert endpoint["model_type"] == "qwen2"
        assert endpoint["architecture"] == "Qwen2ForCausalLM"
        assert endpoint["native_context_tokens"] >= config["context_tokens"]
        assert endpoint["model_artifacts"]


def _prediction(row: int, foundation_score: float, ascent_score: float, facts: list[str]):
    return {
        "row_id": f"row-{row}",
        "target": "2",
        "foundation_output": "1",
        "foundation_answer": "1",
        "foundation_score": foundation_score,
        "ascent_output": "2",
        "ascent_answer": "2",
        "ascent_score": ascent_score,
        "retained_facts": facts,
        "structured_inventory": "large" if len(facts) == 8 else "small",
        "structured_count": "two" if len(facts) == 8 else "one",
    }


def _result(
    endpoint: str,
    condition: str,
    config_hash: str,
    data_hash: str,
    foundation: float,
    ascent: float,
    commit: str,
):
    fact_slots = int(condition.removeprefix("state"))
    facts = (
        [f"earlier-{index}" for index in range(4)]
        + [f"event-{index}" for index in range(4)]
        if fact_slots == 8
        else [f"event-{index}" for index in range(4)]
    )
    predictions = [
        _prediction(index, foundation, ascent, facts) for index in range(32)
    ]
    return {
        "endpoint": {"name": endpoint},
        "condition": {
            "name": condition,
            "memory_representation": "structured_inventory",
            "readout_path": "foundation_generation",
            "fact_slots": fact_slots,
            "diagnostic_no_promotion": False,
        },
        "config": {"sha256": config_hash},
        "data": {
            "sha256": data_hash,
            "task_counts": {"qa7": 16, "qa8": 16},
        },
        "environment": {"git_commit": commit, "git_dirty": False},
        "parser_accuracy": 1.0,
        "foundation": {"mean": foundation},
        "ascent": {"mean": ascent},
        "gain": {"mean": ascent - foundation},
        "remaining_error_elimination": (ascent - foundation) / (1 - foundation),
        "wins": 20,
        "regressions": 0,
        "predictions": predictions,
    }


def test_development_analyzer_requires_positive_interaction(tmp_path):
    config = json.loads(CONFIG.read_text())
    config_hash = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    data_hash = config["panel_sha256_by_name"]["structured_development"]
    out = tmp_path / "development"
    out.mkdir()
    values = {
        ("qwen2p5-3b-instruct", "state4"): (0.10, 0.30),
        ("qwen2p5-3b-instruct", "state8"): (0.10, 0.40),
        ("qwen2p5-7b-instruct", "state4"): (0.20, 0.45),
        ("qwen2p5-7b-instruct", "state8"): (0.20, 0.80),
    }
    for (endpoint, condition), (foundation, ascent) in values.items():
        path = out / f"{endpoint}_{condition}.json"
        path.write_text(
            json.dumps(
                _result(
                    endpoint,
                    condition,
                    config_hash,
                    data_hash,
                    foundation,
                    ascent,
                    "frozen-commit",
                )
            )
        )
    result = analyze(CONFIG, tmp_path, "development")
    assert result["gates"]["development_pass"]
    contrasts = result["panels"][0]["contrasts"]
    assert contrasts["co_scaled_gain_increment"] > 0
    assert contrasts["model_by_state_interaction"] > 0


def test_mean_ci_has_panel_cluster_df4():
    result = mean_ci([1, 2, 3, 4, 5])
    assert result["clusters"] == 5
    assert result["degrees_of_freedom"] == 4
    assert result["ci95_low"] < 3 < result["ci95_high"]


def test_headroom_selection_requires_semantic_nonredundancy():
    sample = {
        "task": "qa7",
        "question": "How many objects is Mary carrying?",
        "facts": (
            "Mary got the apple there.",
            "Mary took the milk there.",
            "Mary passed the apple to John.",
            "Mary grabbed the apple there.",
            "Mary handed the apple to John.",
            "John handed the apple to Mary.",
            "Mary gave the apple to John.",
        ),
    }
    assert semantic_refinement(sample) == ("none", "one")
