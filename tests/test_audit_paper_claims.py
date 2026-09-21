from __future__ import annotations
import pytest

pytest.importorskip("bibtexparser")  # optional GPU-stack / audit dependency; skipped in the CPU-only public tree


from pathlib import Path

from experiments.audit_paper_claims import audit


ROOT = Path(__file__).resolve().parents[1]


def test_manuscript_claims_match_immutable_analyses() -> None:
    result = audit(ROOT)
    assert result["pass"], result["failed_checks"]
    assert 150 <= result["abstract_words"] <= 250
