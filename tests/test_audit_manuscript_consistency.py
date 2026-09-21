from __future__ import annotations
import pytest

pytest.importorskip("bibtexparser")  # optional GPU-stack / audit dependency; skipped in the CPU-only public tree


from pathlib import Path

from experiments.audit_manuscript_consistency import audit


ROOT = Path(__file__).resolve().parents[1]


def test_manuscript_numbers_and_references_are_consistent() -> None:
    result = audit(ROOT)
    assert result["pass"], result["failures"]
    assert result["independently_recalculated_statistical_summaries"] >= 350
    assert result["promoted_narrative_claims_checked"] >= 30
    assert result["references"]["cited_keys"] >= 80
