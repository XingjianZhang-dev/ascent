import json
from pathlib import Path

import pytest

from experiments.run_babilong_qrag_retrieval import (
    RetrievalOnlyFeedback,
    frozen_panel_rows,
    install_unused_vllm_import_shim,
    qvalue_keep_count,
    sentence_chunks,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "babilong_qrag_direct_peer_frozen.json"


def test_sentence_chunks_match_frozen_boundary_semantics() -> None:
    pattern = r"(?<=[.!?])\s+|(?<=\n\n)"
    assert sentence_chunks("One. Two!\n\nThree?", pattern) == [
        "One.",
        "Two!",
        "Three?",
    ]
    with pytest.raises(RuntimeError, match="empty document"):
        sentence_chunks(" \n ", pattern)


def test_qvalue_filter_stops_at_first_value_at_or_below_threshold() -> None:
    assert qvalue_keep_count([0.9, 0.7, 0.5, 0.8], 0.5) == 2
    assert qvalue_keep_count([0.5, 0.9], 0.5) == 0
    assert qvalue_keep_count([0.9, 0.8], 0.5) == 2


def test_retrieval_only_vllm_shim_fails_if_accidentally_used(monkeypatch) -> None:
    modules = __import__("sys").modules
    monkeypatch.delitem(modules, "vllm", raising=False)
    try:
        assert install_unused_vllm_import_shim()
        import vllm

        with pytest.raises(RuntimeError, match="shim was instantiated"):
            vllm.LLM()
    finally:
        modules.pop("vllm", None)


def test_retrieval_only_feedback_never_changes_episode_termination() -> None:
    feedback = RetrievalOnlyFeedback()
    feedback.reset({"pred_idx": []}, {"sf_idx": []})
    assert feedback.get_feedback({}, {}, truncated=False) == {
        "reward": 0.0,
        "terminated": False,
    }
    assert feedback.get_feedback({}, {}, truncated=True) == {
        "reward": 0.0,
        "terminated": False,
    }
    assert isinstance(feedback.copy(), RetrievalOnlyFeedback)


def test_frozen_panel_adapter_selects_all_and_only_registered_task_rows() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    data = (
        ROOT
        / "data"
        / "babilong_1k_8k_generative_focused_confirmation"
        / "generative_focused_confirmation_panel_1.jsonl"
    )
    for task in ("qa2", "qa3"):
        rows, source = frozen_panel_rows(manifest, data, task)
        assert len(rows) == 40
        assert {row["task"] for row in rows} == {task}
        assert len({row["row_id"] for row in rows}) == 40
        assert data.stem in source["panel_sha256_by_name"]
