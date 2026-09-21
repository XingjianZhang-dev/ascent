from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.render_paper_additional_tables import (
    render_around7b,
    render_mechanism,
    render_systems,
)


ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def test_mechanism_table_uses_passing_immutable_analysis() -> None:
    analysis = load("artifacts/sum_numeric_candidate/confirmation_analysis.json")
    rendered = render_mechanism(analysis)

    assert "1.5B & 2 & .388 & [.311,.465]" in rendered
    assert "3B & 3 & 2.099 & [1.948,2.251]" in rendered
    assert "7B & 5 & 7.406 & [7.151,7.661]" in rendered
    assert "Model$\\times$state: 3B$\\to$7B & -- & .968 & [.671,1.264]" in rendered


def test_mechanism_table_refuses_failed_gate() -> None:
    analysis = load("artifacts/sum_numeric_candidate/confirmation_analysis.json")
    failed = copy.deepcopy(analysis)
    failed["gate_pass"] = False
    with pytest.raises(RuntimeError, match="failed mechanism"):
        render_mechanism(failed)


def test_around7b_table_retains_all_five_models_and_failed_scale_extension() -> None:
    analysis = load("artifacts/around7b_formal/analysis.json")
    assert analysis["registered_gate_pass"] is False
    rendered = render_around7b(analysis)

    for name in (
        "Qwen2.5-7B",
        "Mistral-7B-v0.3",
        "Falcon3-7B",
        "Granite-3.3-8B",
        "Qwen3-8B",
    ):
        assert name in rendered
    assert "Granite-3.3-8B & .0850 & .9413 & .8563 [.8312,.8813]" in rendered


def test_systems_table_preserves_passes_and_failed_wall_gate() -> None:
    systems = load("artifacts/remote_results/babilong_4k_systems/analysis.json")
    flops = load("artifacts/remote_results/flops_e49eae9/analysis.json")
    assert systems["all_gates_pass"] is False
    assert flops["gates"]["frozen_profile_gate_pass"] is False

    rendered = render_systems(systems, flops)
    assert "135M & 589.0 & $2.189\\times10^{-6}$ & .413 & 83.7$\\times$" in rendered
    assert "1.7B & 640.6 & $1.872\\times10^{-7}$ & .279 & 78.2$\\times$" in rendered
