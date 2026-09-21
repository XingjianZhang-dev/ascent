import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "babilong_qrag_direct_peer_frozen.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_qrag_direct_peer_manifest_is_complete_and_consistent() -> None:
    manifest = json.loads(MANIFEST.read_text())
    source = manifest["source_panel_config"]
    source_path = ROOT / source["path"]
    source_config = json.loads(source_path.read_text())

    assert sha256_file(source_path) == source["sha256"]
    assert source["executed_commit"] == (
        "c9af9422c9df033ce81ea07edb0cb4f30733e0a0"
    )
    assert manifest["status"].startswith("prospective_before_any_qrag_score")
    assert manifest["qrag_code"]["commit"] == (
        "42358d78ac491843763b90677f07237471c97086"
    )
    assert manifest["qrag_base_embedder"] == {
        "repo_id": "facebook/contriever",
        "revision": "2bd46a25019aeea091fd42d1f0fd4801675cf699",
        "reason": (
            "The released checkpoint config names revision 'main'. This "
            "immutable commit was resolved and frozen before any Q-RAG score "
            "was produced on the ASCENT panels."
        ),
    }
    assert set(manifest["qrag_checkpoints"]) == {"qa2", "qa3"}
    assert all(
        len(spec["model_sha256"]) == 64
        and spec["model_bytes"] == 3936641879
        for spec in manifest["qrag_checkpoints"].values()
    )

    evaluation = manifest["evaluation"]
    assert evaluation["panels"] == source_config["panels"]
    assert evaluation["reader_endpoints"] == [
        endpoint["name"] for endpoint in source_config["endpoints"]
    ]
    assert evaluation["retrieval_rows"] == (
        len(evaluation["tasks"])
        * len(evaluation["panels"])
        * evaluation["rows_per_task_per_panel"]
    )
    assert evaluation["reader_generations"] == (
        evaluation["retrieval_rows"] * len(evaluation["reader_endpoints"])
    )
    assert evaluation["retrieval"] == {
        "max_steps": 6,
        "random": False,
        "qvalue_filter": True,
        "stopping_threshold": 0.5,
        "sort_selected_chunks_by_document_index": True,
    }

    immutability = manifest["immutability"]
    assert all(immutability.values())
    assert manifest["analysis"]["degrees_of_freedom"] == 9
    assert manifest["analysis"]["peer_noninferiority_margin"] == 0.05
