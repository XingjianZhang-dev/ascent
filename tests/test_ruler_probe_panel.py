from __future__ import annotations

import json

from experiments.analyze_ruler_probe_panel import (
    ENDPOINTS,
    analyze,
    endpoints_from_config,
    state_budget_audit,
)


def test_probe_panel_uses_registered_endpoint_family() -> None:
    config = {
        "endpoints": [
            {"name": "small-instruct", "model_parameters": 10},
            {"name": "middle-instruct", "model_parameters": 20},
            {"name": "large-instruct", "model_parameters": 40},
        ]
    }
    assert endpoints_from_config(config) == (
        ("small", 10),
        ("middle", 20),
        ("large", 40),
    )


def test_probe_panel_requires_positive_seed_clustered_scale_curve(tmp_path) -> None:
    seeds = [1, 2, 3]
    gains = {
        1: [0.60, 0.90, 0.97],
        2: [0.62, 0.91, 0.98],
        3: [0.61, 0.92, 0.99],
    }
    for seed in seeds:
        folder = tmp_path / str(seed)
        folder.mkdir()
        for (endpoint, _), gain in zip(ENDPOINTS, gains[seed]):
            payload = {
                "foundation_score": {"mean": 0.02},
                "ascent_scale_score": {"mean": 0.02 + gain},
                "paired_score_gain": {"mean_nats": gain},
                "probe": {"validation_digit_token_accuracy": 1.0},
                "task_sha256": f"seed-{seed}",
            }
            (folder / f"{endpoint}.json").write_text(json.dumps(payload))
    result = analyze(tmp_path, seeds)
    assert result["status"] == "passed"
    assert result["gates"]["all_adjacent_cluster_lcbs_positive"]


def test_probe_panel_requires_both_audit_inputs(tmp_path) -> None:
    try:
        analyze(tmp_path, [1, 2, 3], config_path=tmp_path / "config.json")
    except FileNotFoundError:
        raise AssertionError("input-pair validation must happen before file reads")
    except ValueError as error:
        assert "provided together" in str(error)


def test_probe_panel_default_boundary_is_short_query(tmp_path) -> None:
    seeds = [1, 2, 3]
    for seed in seeds:
        folder = tmp_path / str(seed)
        folder.mkdir()
        for endpoint, _ in ENDPOINTS:
            payload = {
                "foundation_score": {"mean": 0.0},
                "ascent_scale_score": {"mean": 0.1},
                "paired_score_gain": {"mean_nats": 0.1},
                "probe": {"validation_digit_token_accuracy": 1.0},
                "task_sha256": str(seed),
            }
            (folder / f"{endpoint}.json").write_text(json.dumps(payload))
    assert "short-query" in analyze(tmp_path, seeds)["metric_boundary"]


def test_full_context_state_audit_counts_hidden_width() -> None:
    config = {
        "state_budget_rule": {
            "base_state_tokens": 38,
            "base_hidden_size": 896,
            "alpha": 1.0,
        },
        "endpoints": [
            {
                "name": "small",
                "scale_replay_tokens": 38,
                "hidden_size": 896,
                "model_parameters": 494032768,
            },
            {
                "name": "middle",
                "scale_replay_tokens": 78,
                "hidden_size": 1536,
                "model_parameters": 1543714304,
            },
            {
                "name": "large",
                "scale_replay_tokens": 92,
                "hidden_size": 2048,
                "model_parameters": 3085938688,
            },
        ]
    }
    failed = state_budget_audit(config)
    assert not failed["relative_state_ratio_strictly_decreases"]
    config["endpoints"][1]["scale_replay_tokens"] = 69
    repaired = state_budget_audit(config)
    assert repaired["relative_state_ratio_strictly_decreases"]
    assert not repaired["proposal_width_law_matches"]
    assert repaired["rows"][1]["persistent_state_elements"] == 69 * 1536


def test_full_context_state_audit_requires_proposal_width_law() -> None:
    config = {
        "state_budget_rule": {
            "base_state_tokens": 40,
            "base_hidden_size": 896,
            "alpha": 1.0,
        },
        "endpoints": [
            {
                "name": "small",
                "scale_replay_tokens": 40,
                "hidden_size": 896,
                "model_parameters": 494032768,
            },
            {
                "name": "middle",
                "scale_replay_tokens": 69,
                "hidden_size": 1536,
                "model_parameters": 1543714304,
            },
            {
                "name": "large",
                "scale_replay_tokens": 91,
                "hidden_size": 2048,
                "model_parameters": 3085938688,
            },
        ],
    }
    result = state_budget_audit(config)
    assert result["gate"] == "passed"
    assert result["proposal_width_law_matches"]
