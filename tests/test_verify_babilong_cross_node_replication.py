import json

import pytest

from experiments.run_natural_repeat import sha256_file
from experiments.verify_babilong_cross_node_replication import verify


def _fixtures(tmp_path):
    panel = tmp_path / "panel.jsonl"
    panel.write_text("row\n")
    config = {
        "status": "prospective_around7b_extension_frozen_before_any_new_endpoint_decoder_score",
        "execution_authorization": {"authorized": True},
        "evaluation_samples": 1,
        "panel_sha256_by_name": {"panel": sha256_file(panel)},
        "endpoints": [{"name": "qwen"}],
        "runtime_dependencies": {"transformers": "4.57.6"},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    document = {
        "status": "prospective_around7b_extension_frozen_before_any_new_endpoint_decoder_score",
        "endpoint": {"name": "qwen"},
        "condition": {"name": "canonical"},
        "config": {"sha256": sha256_file(config_path)},
        "data": {"sha256": sha256_file(panel)},
        "model_files": [],
        "model_identity": {},
        "parser_accuracy": 1.0,
        "foundation": {},
        "ascent": {},
        "gain": {},
        "remaining_error_elimination": 0.0,
        "wins": 1,
        "regressions": 0,
        "by_task": {},
        "state": {},
        "predictions": [{"row_id": "one", "foundation_output": "x", "ascent_output": "y"}],
        "environment": {
            "git_commit": "a" * 40,
            "git_dirty": False,
            "transformers": "4.57.6",
        },
    }
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text(json.dumps(document))
    right.write_text(json.dumps(document))
    return config_path, left, right


def test_exact_cross_node_replication_passes(tmp_path):
    config, left, right = _fixtures(tmp_path)
    result = verify(
        config, left, right, endpoint="qwen", condition="canonical", panel="panel"
    )
    assert result["exact_prediction_match"] is True
    assert result["prediction_rows"] == 1


def test_one_changed_output_fails_replication(tmp_path):
    config, left, right = _fixtures(tmp_path)
    document = json.loads(right.read_text())
    document["predictions"][0]["ascent_output"] = "changed"
    right.write_text(json.dumps(document))
    with pytest.raises(RuntimeError, match="predictions"):
        verify(
            config, right, left, endpoint="qwen", condition="canonical", panel="panel"
        )
