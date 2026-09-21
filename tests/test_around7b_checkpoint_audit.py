import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from experiments.audit_around7b_checkpoint import (
    frozen_roster_entry,
    frozen_weight_paths,
    repository_environment,
    tensor_manifest,
    validate_safetensors_index_metadata,
    validate_non_thinking_template,
    verify_official_critical_manifest,
    verify_official_weight_manifest,
)


def test_frozen_weight_manifest_covers_each_indexed_tensor_once(tmp_path) -> None:
    first = tmp_path / "model-00001-of-00002.safetensors"
    second = tmp_path / "model-00002-of-00002.safetensors"
    save_file({"a": np.zeros((2, 3), dtype=np.float32)}, first)
    save_file({"b": np.zeros((4,), dtype=np.int64)}, second)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 56},
                "weight_map": {"a": first.name, "b": second.name},
            }
        )
    )
    paths, indexed_tensors, indexed_payload_bytes = frozen_weight_paths(tmp_path)
    manifest = tensor_manifest(paths)
    assert indexed_tensors == 2
    assert indexed_payload_bytes == 56
    assert manifest["tensors"] == 2
    assert manifest["parameters"] == 10
    assert manifest["tensor_payload_bytes"] == 56
    assert all(len(row["sha256"]) == 64 for row in manifest["files"])


def test_duplicate_tensor_across_shards_is_rejected(tmp_path) -> None:
    paths = [tmp_path / "a.safetensors", tmp_path / "b.safetensors"]
    for path in paths:
        save_file({"duplicate": np.zeros((1,), dtype=np.float32)}, path)
    with pytest.raises(RuntimeError, match="more than one shard"):
        tensor_manifest(paths)


def test_roster_lookup_requires_exactly_one_entry() -> None:
    design = {"frozen_model_roster": [{"name": "model-a"}, {"name": "model-b"}]}
    assert frozen_roster_entry(design, "model-b") == {"name": "model-b"}
    with pytest.raises(RuntimeError, match="entry count is 0"):
        frozen_roster_entry(design, "missing")


def test_repository_environment_records_the_current_commit() -> None:
    environment = repository_environment()
    assert len(environment["git_commit"]) == 40
    assert isinstance(environment["git_dirty"], bool)


def test_official_manifest_comparison_is_exact() -> None:
    files = [{"path": "model.safetensors", "bytes": 123, "sha256": "a" * 64}]
    weights = {"files": [{**files[0], "tensors": 1, "parameters": 2}]}
    verify_official_weight_manifest(weights, {"official_weight_artifacts": files})
    altered = [{**files[0], "bytes": 124}]
    with pytest.raises(RuntimeError, match="official pinned LFS"):
        verify_official_weight_manifest(weights, {"official_weight_artifacts": altered})


def test_qwen3_official_empty_think_marker_is_not_active_reasoning() -> None:
    assert validate_non_thinking_template(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    with pytest.raises(RuntimeError, match="empty closed think block"):
        validate_non_thinking_template("<think>reasoning</think>\n")
    with pytest.raises(RuntimeError, match="after </think>"):
        validate_non_thinking_template("<think>\n</think>answer prefill")


def test_official_critical_manifest_supports_git_blob_and_lfs_digests(tmp_path) -> None:
    plain = tmp_path / "config.json"
    lfs = tmp_path / "tokenizer.json"
    plain.write_bytes(b"plain")
    lfs.write_bytes(b"large-content")
    from experiments.audit_around7b_checkpoint import git_blob_sha1
    from experiments.run_natural_repeat import sha256_file

    roster = {
        "official_critical_artifacts": [
            {"path": plain.name, "bytes": 5, "git_blob_sha1": git_blob_sha1(plain)},
            {"path": lfs.name, "bytes": 13, "sha256": sha256_file(lfs)},
        ]
    }
    verified = verify_official_critical_manifest([plain, lfs], roster)
    assert {row["path"] for row in verified} == {plain.name, lfs.name}
    roster["official_critical_artifacts"][0]["bytes"] = 6
    with pytest.raises(RuntimeError, match="byte mismatch"):
        verify_official_critical_manifest([plain, lfs], roster)


def test_official_index_payload_anomaly_is_retained_not_hidden() -> None:
    result = validate_safetensors_index_metadata(
        {"tensors": 255, "tensor_payload_bytes": 14_911_100_928},
        255,
        14_911_113_216,
    )
    assert result["tensor_count_match"] is True
    assert result["payload_bytes_match"] is False
    assert result["payload_delta_bytes"] == -12_288
    with pytest.raises(RuntimeError, match="tensor count"):
        validate_safetensors_index_metadata(
            {"tensors": 254, "tensor_payload_bytes": 1}, 255, 1
        )
