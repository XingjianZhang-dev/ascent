import json
from pathlib import Path

import pytest

from experiments.run_babilong_qrag_reader import (
    load_retrieval_document,
    qrag_prompt,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "babilong_qrag_direct_peer_frozen.json"


def test_qrag_prompt_preserves_frozen_passage_order_and_answer_contract() -> None:
    prompt = qrag_prompt("qa3", "Where is Mary?", ["First.", "Second."])
    assert prompt.index("First.") < prompt.index("Second.")
    assert "exactly one lowercase location word" in prompt
    assert prompt.endswith("Question: Where is Mary?\nAnswer:")
    empty = qrag_prompt("qa2", "Where is John?", [])
    assert "No passage survived" in empty


def test_retrieval_document_validation_rejects_manifest_drift(tmp_path: Path) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    document = {
        "status": manifest["status"],
        "manifest": {"sha256": "wrong"},
        "panel": manifest["evaluation"]["panels"][0],
        "task": "qa2",
        "qrag": {
            "code_commit": manifest["qrag_code"]["commit"],
            "checkpoint_model_sha256": manifest["qrag_checkpoints"]["qa2"][
                "model_sha256"
            ],
        },
        "data": {"rows": 40},
        "rows": [{"row_id": str(index)} for index in range(40)],
    }
    path = tmp_path / "retrieval.json"
    path.write_text(json.dumps(document))
    with pytest.raises(RuntimeError, match="manifest hash mismatch"):
        load_retrieval_document(
            path,
            manifest,
            "expected",
            manifest["evaluation"]["panels"][0],
            "qa2",
        )
