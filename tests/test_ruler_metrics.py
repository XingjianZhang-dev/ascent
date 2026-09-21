import pytest
from pathlib import Path

from ascent.ruler_metrics import string_match_all, string_match_part
from experiments.run_ruler_niah import verify_model_artifact


def test_string_match_all_matches_ruler_fractional_credit() -> None:
    assert string_match_all("A, b", ["A", "B", "C"]) == pytest.approx(2 / 3)
    assert string_match_all("nothing", ["A"]) == 0.0


def test_string_match_all_rejects_empty_references() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        string_match_all("", [])


def test_string_match_part_matches_ruler_qa_any_reference_rule() -> None:
    assert string_match_part("the answer is Alpha", ["beta", "alpha"]) == 1.0
    assert string_match_part("nothing", ["beta", "alpha"]) == 0.0
    with pytest.raises(ValueError):
        string_match_part("", [])


def test_verify_model_artifact_accepts_all_frozen_shards(tmp_path: Path) -> None:
    import hashlib

    rows = []
    for name, content in (("model-00001-of-00002.safetensors", b"one"), ("model-00002-of-00002.safetensors", b"two")):
        path = tmp_path / name
        path.write_bytes(content)
        rows.append({"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    assert len(verify_model_artifact(tmp_path, rows)) == 2
