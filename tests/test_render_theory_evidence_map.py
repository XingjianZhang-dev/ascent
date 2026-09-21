import copy
import json
from pathlib import Path

import pytest

from experiments.render_theory_evidence_map import render


ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def inputs() -> tuple[dict, dict, dict, dict]:
    return (
        load(
            "artifacts/remote_results/"
            "babilong_qwen_canonical_16k_confirmation_4fd25e9/analysis.json"
        ),
        load("artifacts/sum_numeric_candidate/confirmation_analysis.json"),
        load("artifacts/around7b_formal/analysis.json"),
        load("artifacts/remote_results/babilong_4k_systems/analysis.json"),
    )


def test_theory_evidence_map_recomputes_headline_values() -> None:
    table = render(*inputs())

    assert "13,824 of 13,824 endpoint--transition row checks" in table
    assert "true by construction" in table and "Design check" in table
    assert r".388$\to$2.099$\to$7.406 nats" in table
    assert "1.711 [1.540,1.882]" in table
    assert ".781 [.539,1.023]" in table
    assert r"10.50$\to$78.88$\to$82.63 points" in table
    assert r"2.189\!\times\!10^{-6}" in table
    assert table.count("Established") == 3
    assert "Identified" in table
    assert "Non-universality" not in table


def test_theory_evidence_map_fails_closed_on_provenance_failure() -> None:
    official, mechanism, around7b, systems = inputs()
    invalid = copy.deepcopy(mechanism)
    invalid["provenance"]["exact_nested_observation_prefixes"] = False

    with pytest.raises(RuntimeError, match="provenance"):
        render(official, invalid, around7b, systems)
