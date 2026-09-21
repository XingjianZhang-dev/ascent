import json
from pathlib import Path

from experiments.prepare_babilong_certified_factorial import (
    ENDPOINTS,
    STATE_SLOTS,
    derive,
)


def test_factorial_config_has_all_nine_cells(tmp_path: Path) -> None:
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps({"conditions": {"old": {}}}))
    result = derive(json.loads(base_path.read_text()), base_path)

    assert tuple(result["factorial_design"]["foundation_endpoints"]) == ENDPOINTS
    assert tuple(result["factorial_design"]["state_fact_slots"]) == STATE_SLOTS
    assert result["factorial_design"]["cells"] == 9
    assert set(result["conditions"]) == {
        "certified_slots_1",
        "certified_slots_2",
        "certified_slots_8",
    }
    for slots in STATE_SLOTS:
        condition = result["conditions"][f"certified_slots_{slots}"]
        assert condition["readout_path"] == "certified_evidence"
        assert set(condition["state_fact_slots_by_endpoint"].values()) == {slots}
