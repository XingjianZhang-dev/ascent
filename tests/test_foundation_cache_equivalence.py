from __future__ import annotations

import json

from experiments.check_foundation_cache_equivalence import compare


def _payload(cache: bool, prediction: str) -> dict[str, object]:
    return {
        "endpoint": {"repo_id": "official/model"},
        "task_sha256": "task",
        "runtime": {"foundation_decode_cache": cache},
        "predictions": [
            {
                "index": 20,
                "outputs": ["1234567"],
                "foundation_prediction": prediction,
                "ascent_prediction": prediction + " evidence",
                "foundation_score": 0.0,
                "ascent_score": 1.0,
                "ascent_evidence_values": ["1234567"],
            }
        ],
    }


def test_cache_equivalence_requires_byte_identical_predictions(tmp_path) -> None:
    reference = tmp_path / "reference.json"
    cached = tmp_path / "cached.json"
    reference.write_text(json.dumps(_payload(False, "answer")))
    cached.write_text(json.dumps(_payload(True, "answer")))
    assert compare(reference, cached)["status"] == "passed"
    cached.write_text(json.dumps(_payload(True, "different")))
    result = compare(reference, cached)
    assert result["status"] == "failed"
    assert not result["gates"]["every_compared_prediction_identical"]
