from experiments.analyze_babilong_certified_controls import CONDITIONS, ENDPOINTS


def test_certified_control_matrix_is_complete() -> None:
    assert len(CONDITIONS) == 4
    assert len(ENDPOINTS) == 3
    assert "fixed_one_slot" in CONDITIONS
    assert "matched_raw_event_fifo" in CONDITIONS
