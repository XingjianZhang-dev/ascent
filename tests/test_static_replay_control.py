import numpy as np
from pathlib import Path

from experiments.analyze_static_replay_control import analyze_endpoint, summary
from experiments.analyze_static_replay_panel import paths


def test_summary_reports_positive_lower_bound_for_constant_gain() -> None:
    result = summary(np.full(16, 0.25))
    assert result["mean"] == 0.25
    assert result["ci95_low"] == 0.25


def test_analyze_endpoint_uses_paired_dynamic_static_contrast() -> None:
    result = {
        "same_flop_static_adapter": {
            "same_replay_token_count_as_dynamic": True,
            "same_suffix_layer_index_as_dynamic": True,
            "same_frozen_suffix_call_as_dynamic": True,
            "training_steps": 2,
            "initial_training_loss": 3.0,
            "final_training_loss": 2.5,
            "inference_parameter_bytes_bf16": 64,
        }
    }
    arrays = {
        "base_nll": np.asarray([3.0, 3.0]),
        "rich_fused_nll": np.asarray([1.0, 1.0]),
        "static_adapter_fused_nll": np.asarray([2.0, 2.0]),
    }
    analysis = analyze_endpoint(result, arrays)
    assert analysis["dynamic_rich_gain"]["mean"] == 2.0
    assert analysis["static_adapter_gain"]["mean"] == 1.0
    assert analysis["dynamic_advantage_over_static"]["mean"] == 1.0


def test_panel_paths_route_endpoints_to_separate_nodes() -> None:
    root = Path("results")
    assert paths(root, 20260941, "pythia-410m") == (
        root / "node2" / "seed20260941_pythia-410m.json",
        root / "node2" / "seed20260941_pythia-410m.npz",
    )
    assert paths(root, 20260941, "pythia-2.8b")[0] == (
        root / "node1" / "seed20260941_pythia-2.8b.json"
    )
