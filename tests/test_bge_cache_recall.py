from experiments.analyze_bge_cache_recall import audit_rows


def test_audit_rows_reports_slot_dependent_exact_recall() -> None:
    data = [
        {
            "row_id": "row-1",
            "input": "Mary went to the kitchen. Mary moved to the garden.",
            "question": "Where is Mary?",
            "target": "garden",
        }
    ]
    cache = [
        {
            "row_id": "row-1",
            "ranked_passages": [
                {"passage": "Mary went to the kitchen."},
                {"passage": "Mary moved to the garden."},
            ],
        }
    ]
    result = audit_rows(data, cache, slot_counts=(1, 2))
    assert result["1"]["exact_support_recall"] == 0.0
    assert result["2"]["exact_support_recall"] == 1.0


def test_audit_rows_rejects_mismatched_ids() -> None:
    data = [
        {
            "row_id": "expected",
            "input": "Mary went to the kitchen.",
            "question": "Where is Mary?",
            "target": "kitchen",
        }
    ]
    cache = [{"row_id": "other", "ranked_passages": []}]
    try:
        audit_rows(data, cache)
    except RuntimeError as error:
        assert "row IDs" in str(error)
    else:
        raise AssertionError("mismatched row IDs should fail")
