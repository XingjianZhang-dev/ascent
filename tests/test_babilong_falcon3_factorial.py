import hashlib
import json
from pathlib import Path

from experiments.analyze_babilong_falcon3_factorial import analyze


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "babilong_falcon3_8k_raw_235_factorial.json"
DATA = ROOT / "data"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_files(endpoint: dict) -> list[dict]:
    if "model_artifact" in endpoint:
        return [endpoint["model_artifact"]]
    return endpoint["model_artifacts"]


def test_falcon3_factorial_is_frozen_with_decreasing_relative_state() -> None:
    config = json.loads(CONFIG.read_text())
    for directory, expected in config["data_manifest_sha256_by_root"].items():
        assert _sha256(DATA / directory / "MANIFEST.json") == expected
    assert config["development_panels"] == [config["panels"][0]]
    assert config["confirmation_panels"] == config["panels"][1:]
    assert len(config["confirmation_panels"]) == 9
    assert config["factorial"]["state_fact_slots"] == [2, 3, 5]
    ratios = [
        slots / endpoint["model_parameters"]
        for slots, endpoint in zip(
            config["factorial"]["state_fact_slots"],
            config["endpoints"],
            strict=True,
        )
    ]
    assert ratios == config["factorial"]["co_scaled_state_slots_per_parameter"]
    assert ratios[0] > ratios[1] > ratios[2]
    for panel in config["panels"]:
        assert _sha256(DATA / config["panel_path_by_name"][panel]) == config[
            "panel_sha256_by_name"
        ][panel]


def _fake_result(config: dict, endpoint: dict, slots: int, foundation: float, ascent: float):
    facts = [f"event-{index}" for index in range(5)][-slots:]
    predictions = [
        {
            "row_id": f"row-{index}",
            "task": ("qa1", "qa2", "qa3")[index % 3],
            "target": "kitchen",
            "foundation_output": "office",
            "foundation_answer": "office",
            "foundation_score": foundation,
            "foundation_prompt_tokens": 100,
            "retained_facts": facts,
        }
        for index in range(120)
    ]
    return {
        "status": config["status"],
        "endpoint": endpoint,
        "condition": {
            "name": f"generative_slots_{slots}",
            "fact_slots": slots,
            "memory_source": "relevant",
            "memory_representation": "event_facts",
            "readout_path": "foundation_generation",
            "diagnostic_no_promotion": False,
        },
        "config": {"sha256": _sha256(CONFIG)},
        "data": {
            "sha256": config["panel_sha256_by_name"][
                config["development_panels"][0]
            ],
            "path": str(DATA / config["panel_path_by_name"][config["development_panels"][0]]),
        },
        "model_files": _model_files(endpoint),
        "parser_accuracy": 1.0,
        "environment": {"git_dirty": False, "git_commit": "frozen"},
        "foundation": {"mean": foundation},
        "ascent": {"mean": ascent},
        "gain": {"mean": ascent - foundation},
        "predictions": predictions,
    }


def test_falcon3_development_analyzer_passes_positive_registered_factorial(
    tmp_path: Path,
) -> None:
    config = json.loads(CONFIG.read_text())
    values = {
        "falcon3-1b-instruct": {2: (0.10, 0.20), 3: (0.10, 0.30), 5: (0.10, 0.35)},
        "falcon3-3b-instruct": {2: (0.20, 0.25), 3: (0.20, 0.50), 5: (0.20, 0.60)},
        "falcon3-7b-instruct": {2: (0.30, 0.30), 3: (0.30, 0.55), 5: (0.30, 0.90)},
    }
    endpoint_by_name = {row["name"]: row for row in config["endpoints"]}
    panel = config["development_panels"][0]
    for endpoint, cells in values.items():
        node = config["execution"]["node_by_endpoint"][endpoint]
        output = tmp_path / "development" / node
        output.mkdir(parents=True, exist_ok=True)
        for slots, (foundation, ascent) in cells.items():
            path = output / f"generative_slots_{slots}_{panel}_{endpoint}.json"
            path.write_text(
                json.dumps(
                    _fake_result(
                        config,
                        endpoint_by_name[endpoint],
                        slots,
                        foundation,
                        ascent,
                    )
                )
            )
    result = analyze(CONFIG, tmp_path, "development", "frozen")
    assert result["provenance"]["provenance_and_alignment_pass"]
    assert result["gates"]["development_pass"]
