import numpy as np

from experiments.analyze_ruler_qa_evidence import paired_interval, per_example_slopes
from experiments.run_ruler_qa_dual_path_nll import (
    compact_evidence_content,
    resolve_relative_injection_depth,
)
from experiments.run_ruler_qa_evidence import (
    evidence_content,
    remaining_error_elimination,
    safe_nll_gate,
)


def test_safe_nll_gate_enables_only_positive_lower_bound() -> None:
    base = np.array([2.0, 2.1, 1.9, 2.2])
    clearly_better = base - 0.5
    tied = base.copy()
    assert safe_nll_gate(base, clearly_better)["enabled"]
    assert not safe_nll_gate(base, tied)["enabled"]


def test_evidence_content_discloses_causal_retrieval_contract() -> None:
    content = evidence_content("Document 2:\nEvidence")
    assert "without using the answer" in content
    assert content.endswith("Return only the answer to the original question.")


def test_remaining_error_elimination_respects_accuracy_ceiling() -> None:
    assert remaining_error_elimination(0.5, 1.0) == 1.0
    assert remaining_error_elimination(1.0, 1.0) is None


def test_per_example_scale_slope_is_positive_for_increasing_gains() -> None:
    gains = np.array([[0.1, 0.2], [0.3, 0.5], [0.8, 1.0]])
    slopes = per_example_slopes(gains, np.array([0.5e9, 1.5e9, 3.0e9]))
    assert np.all(slopes > 0.0)
    assert paired_interval(slopes)["ci95_low"] > 0.0


def test_compact_dual_path_prompt_discloses_no_label_retrieval() -> None:
    content = compact_evidence_content(
        "Document 3: Beta Harbor\nBeta Harbor is located in Norway.",
        "Where is Beta Harbor?",
    )
    assert "without using the answer or support labels" in content
    assert "Question: Where is Beta Harbor?" in content
    assert content.endswith("Return only the answer and no explanation.")


def test_replay_depth_override_is_explicitly_diagnostic() -> None:
    assert resolve_relative_injection_depth(0.875, None) == (0.875, False)
    assert resolve_relative_injection_depth(0.875, 0.625) == (0.625, True)


def test_replay_depth_rejects_boundaries() -> None:
    for invalid in (0.0, 1.0, -0.1, 1.1):
        with np.testing.assert_raises(ValueError):
            resolve_relative_injection_depth(0.875, invalid)
