from pathlib import Path

import pytest

from experiments.run_ruler_niah import verify_model_artifact


def test_model_artifact_manifest_and_gate(tmp_path: Path) -> None:
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"frozen-weight-bytes")
    expected = {
        "path": "model.safetensors",
        "bytes": 19,
        "sha256": "374ad2246cccc8b95db50077607071b7a280a48503ac59c66ec7e0785b383dc9",
    }
    assert verify_model_artifact(tmp_path, expected) == [expected]


def test_model_artifact_gate_rejects_wrong_hash(tmp_path: Path) -> None:
    (tmp_path / "pytorch_model.bin").write_bytes(b"actual")
    expected = {"path": "pytorch_model.bin", "bytes": 6, "sha256": "wrong"}
    with pytest.raises(RuntimeError, match="model artifact mismatch"):
        verify_model_artifact(tmp_path, expected)


def test_model_artifact_gate_requires_weights(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no model weights"):
        verify_model_artifact(tmp_path, None)
