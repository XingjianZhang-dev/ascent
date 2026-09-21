import copy
import json
from pathlib import Path

import pytest

from experiments.render_cross_task_table import load_single_analysis, render


ROOT = Path(__file__).resolve().parents[1]


def inputs() -> tuple[dict, dict, dict, dict]:
    qwen_multiquery = load_single_analysis(
        ROOT
        / "artifacts/remote_results/"
        "qwen_niah_fullctx_16k_multiquery_width_linear_confirmation_372xxx.tar.gz"
    )
    smol_multiquery = load_single_analysis(
        ROOT
        / "artifacts/remote_results/"
        "smollm2_4k_multiquery_width075_confirm_376101_376202_376303.tar.gz"
    )
    smol_cwe = json.loads(
        (
            ROOT
            / "artifacts/remote_results/cwe_confirm_c749f24/analysis.json"
        ).read_text()
    )
    qwen_cwe = json.loads(
        (
            ROOT
            / "artifacts/remote_results/qwen_cwe_confirm_cd8352f/analysis.json"
        ).read_text()
    )
    return qwen_multiquery, smol_multiquery, smol_cwe, qwen_cwe


def test_cross_task_table_recomputes_all_four_studies() -> None:
    table = render(*inputs())

    assert r".0438$\to$.1208$\to$.3188" in table
    assert r".1042$\to$.2167$\to$.5229" in table
    assert r".2628$\to$.4378$\to$.9561" in table
    assert r".2417$\to$.6144$\to$.7100" in table
    assert ".0956 [-.0043,.1954]" in table
    assert table.count("Two positive increments") == 3
    assert table.count("Positive scaling slope") == 1


def test_cross_task_table_fails_closed_if_confirmed_gate_is_removed() -> None:
    qwen_multiquery, smol_multiquery, smol_cwe, qwen_cwe = inputs()
    invalid = copy.deepcopy(smol_multiquery)
    invalid["gates"]["all_adjacent_cluster_lcbs_positive"] = False

    with pytest.raises(RuntimeError, match="confirmation gate"):
        render(qwen_multiquery, invalid, smol_cwe, qwen_cwe)
