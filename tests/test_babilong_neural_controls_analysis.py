from pathlib import Path

from experiments.analyze_babilong_neural_controls import path_for, score_signature


def test_path_for_routes_middle_endpoint_to_node2() -> None:
    root = Path("results")
    assert path_for(
        root, 2, "smollm2-360m-instruct", "scale_bge_m3"
    ) == root / "node2" / "panel2_smollm2-360m-instruct_scale_bge_m3.json"
    assert path_for(
        root, 3, "smollm2-1p7b-instruct", "scale_relevant"
    ) == root / "node1" / "panel3_smollm2-1p7b-instruct_scale_relevant.json"


def test_score_signature_preserves_row_alignment() -> None:
    result = {
        "predictions": [
            {
                "row_id": "a",
                "foundation_location": "garden",
                "foundation_score": 1.0,
            },
            {
                "row_id": "b",
                "foundation_location": None,
                "foundation_score": 0.0,
            },
        ]
    }
    assert score_signature(result, "foundation") == [
        ("a", "garden", 1.0),
        ("b", None, 0.0),
    ]
