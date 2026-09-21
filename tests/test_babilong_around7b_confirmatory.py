import json
from pathlib import Path

from experiments.freeze_babilong_around7b_extension import EXECUTABLE_STATUS
from experiments.run_natural_repeat import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/babilong_around7b_16k_extension_confirmatory.json"
DESIGN = ROOT / "configs/babilong_around7b_16k_extension_design.json"
SOURCE = ROOT / "configs/babilong_qwen2p5_canonical_coscale_16k_confirmatory.json"


def test_around7b_confirmatory_config_is_fully_authorized_and_pinned() -> None:
    config = json.loads(CONFIG.read_text())
    assert config["status"] == EXECUTABLE_STATUS
    assert config["frozen_utc"] == "2026-08-15T03:52:00+08:00"
    assert config["execution_authorization"]["authorized"] is True
    assert config["prospective_design"]["sha256"] == sha256_file(DESIGN)
    assert config["source_protocol"]["sha256"] == sha256_file(SOURCE)
    assert config["runtime_dependencies"] == {
        "transformers": "4.57.6",
        "tokenizers": "0.22.2",
        "sentencepiece": "0.2.2",
        "safetensors": "0.8.0",
    }


def test_around7b_confirmatory_endpoints_exactly_match_official_manifests() -> None:
    config = json.loads(CONFIG.read_text())
    design = json.loads(DESIGN.read_text())
    endpoints = {row["name"]: row for row in config["endpoints"]}
    roster = {row["name"]: row for row in design["frozen_model_roster"]}
    assert set(endpoints) == set(roster)
    assert {name: row["model_parameters"] for name, row in endpoints.items()} == {
        "qwen2p5-7b-instruct": 7_615_616_512,
        "mistral-7b-instruct-v0p3": 7_248_023_552,
        "falcon3-7b-instruct": 7_455_550_464,
        "granite-3p3-8b-instruct": 8_170_864_640,
        "qwen3-8b-nonthinking": 8_190_735_360,
    }
    for name, endpoint in endpoints.items():
        assert endpoint["repo_id"] == roster[name]["repo_id"]
        assert endpoint["revision"] == roster[name]["revision"]
        assert endpoint["model_artifacts"] == roster[name]["official_weight_artifacts"]
        assert endpoint["state_fact_slots"] == 4
        assert endpoint["batch_size"] == 1


def test_around7b_confirmatory_cells_cover_all_models_and_only_qwen_controls() -> None:
    config = json.loads(CONFIG.read_text())
    names = {row["name"] for row in config["endpoints"]}
    conditions = config["conditions"]
    assert set(conditions["canonical_slots_4"]["state_fact_slots_by_endpoint"]) == names
    assert set(conditions["canonical_slots_4"]["state_fact_slots_by_endpoint"].values()) == {4}
    assert conditions["canonical_fixed_slots_3"]["state_fact_slots_by_endpoint"] == {
        "qwen2p5-7b-instruct": 3
    }
    assert conditions["raw_slots_4"]["state_fact_slots_by_endpoint"] == {
        "qwen2p5-7b-instruct": 4
    }
    assert len(config["panels"]) == 10
    assert config["evaluation_samples"] == 80
