from __future__ import annotations

import json

from experiments.analyze_probe_state_accounting import analyze


def test_accounting_report_verifies_decreasing_relative_state(tmp_path) -> None:
    config = {
        "state_dtype": "bfloat16",
        "bytes_per_state_element": 2,
        "evidence_classes": 11,
        "source_experiment_config": "frozen.json",
        "endpoints": [
            {"name": "small", "model_parameters": 1000, "hidden_size": 10, "state_tokens": 2},
            {"name": "large", "model_parameters": 10000, "hidden_size": 20, "state_tokens": 3},
        ],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    report = analyze(path)
    assert report["gates"]["relative_state_ratio_strictly_decreases"]
    assert report["endpoints"][0]["persistent_state_bytes"] == 40
