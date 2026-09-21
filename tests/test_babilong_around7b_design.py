import json
from pathlib import Path

from experiments.run_natural_repeat import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def test_around7b_design_is_score_blind_and_not_executable() -> None:
    design = json.loads(
        (ROOT / "configs/babilong_around7b_16k_extension_design.json").read_text()
    )
    assert design["status"].endswith("not_executable_pending_weight_audit")
    assert design["execution_authorization"]["authorized"] is False
    assert "No decoder score" in design["score_visibility_at_freeze"]["new_endpoints"]
    assert len(design["execution_authorization"]["required_before_first_score"]) == 6


def test_around7b_design_reuses_the_exact_frozen_official_protocol() -> None:
    design = json.loads(
        (ROOT / "configs/babilong_around7b_16k_extension_design.json").read_text()
    )
    source = design["source_panel_protocol"]
    assert sha256_file(ROOT / source["config_path"]) == source["config_sha256"]
    assert sha256_file(ROOT / source["manifest_path"]) == source["manifest_sha256"]
    assert source["panels"] == 10
    assert source["samples_per_panel"] == 80
    assert source["tasks"] == ["qa2", "qa3"]
    assert source["context_tokens"] == 16384


def test_around7b_roster_and_conditions_are_frozen_without_cherry_picking() -> None:
    design = json.loads(
        (ROOT / "configs/babilong_around7b_16k_extension_design.json").read_text()
    )
    roster = design["frozen_model_roster"]
    assert len(roster) == 5
    assert len({row["name"] for row in roster}) == len(roster)
    assert len({row["repo_id"] for row in roster}) == len(roster)
    assert all(len(row["revision"]) == 40 for row in roster)
    official_shards = [
        shard for row in roster for shard in row["official_weight_artifacts"]
    ]
    assert len(official_shards) == 20
    assert all(shard["path"].endswith(".safetensors") for shard in official_shards)
    assert all(shard["bytes"] > 0 and len(shard["sha256"]) == 64 for shard in official_shards)
    critical = [shard for row in roster for shard in row["official_critical_artifacts"]]
    assert len(critical) == 35
    assert all(
        ("sha256" in row) != ("git_blob_sha1" in row) and row["bytes"] > 0
        for row in critical
    )
    assert roster[-1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert design["registered_conditions"]["all_models"]["canonical_slots_4"] == {
        "memory_source": "relevant",
        "memory_representation": "canonical_event_facts",
        "state_fact_slots": 4,
    }
    assert "No model, task, panel, invalid output, or regression may be removed" in design[
        "primary_analysis"
    ]["gate"]
