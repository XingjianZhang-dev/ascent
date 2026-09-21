import json
from pathlib import Path

import pytest

from ascent.babilong_memory import canonical_babilong_event_sources, read_babilong
from ascent.target_blindness import TargetAccessError, TargetBlindRow, fact_provenance_record

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/babilong_16k_canonical_confirmation/canonical_16k_confirmation_panel_1.jsonl"


def first_row() -> dict:
    if not PANEL.exists():
        pytest.skip("benchmark row bodies are not redistributed; rebuild the panels with experiments/prepare_babilong_panel.py")
    return json.loads(PANEL.read_text().splitlines()[0])


def test_blinded_row_blocks_target_and_iteration_until_reveal() -> None:
    row = TargetBlindRow(first_row())
    assert row["question"]
    with pytest.raises(TargetAccessError):
        row["target"]
    with pytest.raises(TargetAccessError):
        dict(row)
    assert row.blocked_attempts == 2
    row.reveal("test")
    assert row["target"] == first_row()["target"]
    record = row.record()
    assert record["keys_read_before_reveal"] == ["question"]
    assert record["target_read_before_reveal"] is False
    assert record["revealed_phase"] == "test"
    with pytest.raises(RuntimeError):
        row.reveal("again")


def test_provenance_record_holds_for_canonical_state_on_a_real_row() -> None:
    raw = first_row()
    row = TargetBlindRow(raw)
    read = read_babilong(row["input"], row["question"], history_slots=256, enabled_tasks=("qa2", "qa3"))
    retained = read.facts[-3:]
    canonical = canonical_babilong_event_sources(row["input"])
    rendered = [canonical[fact.character_position] for fact in retained]
    record = fact_provenance_record(row["input"], retained, rendered)
    assert record["state_derived_from_input_only"] is True
    assert record["retained_facts"] == len(retained)
    assert row.blocked_attempts == 0 and row.revealed_phase is None


def test_provenance_record_detects_a_foreign_string() -> None:
    raw = first_row()
    read = read_babilong(raw["input"], raw["question"], history_slots=256, enabled_tasks=("qa2", "qa3"))
    retained = read.facts[-2:]
    rendered = [retained[0].source, "the answer is hallway."]
    record = fact_provenance_record(raw["input"], retained, rendered)
    assert record["all_rendered_strings_derived_from_own_fact"] is False
    assert record["state_derived_from_input_only"] is False
