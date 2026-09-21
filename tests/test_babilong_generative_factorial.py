import json
from pathlib import Path

from experiments.prepare_babilong_generative_factorial import (
    ENDPOINTS,
    STATE_SLOTS,
    derive,
)


def test_generative_factorial_has_nine_frozen_cells(tmp_path: Path) -> None:
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps({"conditions": {"old": {}}}))
    result = derive(json.loads(base_path.read_text()), base_path)

    assert result["factorial_design"]["cells"] == 9
    assert result["factorial_design"]["readout"] == "foundation_generation"
    for slots in STATE_SLOTS:
        condition = result["conditions"][f"generative_slots_{slots}"]
        assert set(condition["state_fact_slots_by_endpoint"]) == set(ENDPOINTS)
        assert set(condition["state_fact_slots_by_endpoint"].values()) == {slots}
