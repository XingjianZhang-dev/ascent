from types import SimpleNamespace

import pytest

from experiments.run_babilong_prompt import (
    certified_state_answer,
    configure_decoder_tokenizer,
    corrupt_locations,
    extract_task_answer,
    frozen_model_artifact_spec,
    optional_certified_state_answer,
    prompt_input_token_budget,
    remaining_error_elimination,
)
from experiments.diagnose_babilong_decode_context import (
    build_ascent_prompt,
    build_operator_prompt,
)


def test_location_corruption_is_deterministic_and_changes_every_location() -> None:
    source = "Mary moved to the garden. John went to the office."
    assert corrupt_locations(source) == (
        "Mary moved to the hallway. John went to the bathroom."
    )


def test_qa7_answer_extraction_uses_count_words() -> None:
    assert extract_task_answer("The answer is two.", "qa7") == "two"
    assert extract_task_answer("2", "qa7") == "two"
    assert extract_task_answer("10", "qa7") is None


def test_qa8_answer_extraction_preserves_object_order() -> None:
    assert extract_task_answer("football, then milk", "qa8") == "football,milk"
    assert extract_task_answer("She carries nothing.", "qa8") == "nothing"


def test_operator_prompt_is_target_blind_and_places_empty_rule_last() -> None:
    row = {
        "task": "qa8",
        "input": "Daniel took the football there.",
        "question": "What is Daniel carrying? ",
    }
    prompt = build_operator_prompt(row, 1)
    assert "Daniel took the football there." in prompt
    assert "Question: What is Daniel carrying?\nAnswer:" in prompt
    assert "reply nothing" in prompt
    assert "target" not in prompt.lower()


def test_frozen_prompt_can_strip_only_question_boundary_whitespace() -> None:
    row = {
        "task": "qa7",
        "input": "Mary got the milk there.",
        "question": "How many objects is Mary carrying? ",
    }
    original = build_ascent_prompt(row, 1)
    stripped = build_ascent_prompt(row, 1, strip_question=True)
    assert original != stripped
    assert original.replace("carrying? \n", "carrying?\n") == stripped


def test_remaining_error_elimination_respects_accuracy_ceiling() -> None:
    assert remaining_error_elimination([0, 0, 1, 1], [1, 1, 1, 1]) == 1.0
    assert remaining_error_elimination([1, 1], [1, 1]) is None


def test_prompt_budget_reserves_generation_inside_total_context() -> None:
    assert prompt_input_token_budget(8192, 12) == 8180


def test_frozen_model_artifact_spec_accepts_single_and_sharded_weights() -> None:
    single = {"model_artifact": {"path": "model.safetensors"}}
    sharded = {
        "model_artifacts": [
            {"path": "model-00001-of-00002.safetensors"},
            {"path": "model-00002-of-00002.safetensors"},
        ]
    }
    assert frozen_model_artifact_spec(single) == single["model_artifact"]
    assert frozen_model_artifact_spec(sharded) == sharded["model_artifacts"]


def test_frozen_model_artifact_spec_rejects_missing_or_ambiguous_manifest() -> None:
    with pytest.raises(RuntimeError, match="exactly one"):
        frozen_model_artifact_spec({})
    with pytest.raises(RuntimeError, match="exactly one"):
        frozen_model_artifact_spec(
            {
                "model_artifact": {"path": "model.safetensors"},
                "model_artifacts": [{"path": "model.safetensors"}],
            }
        )


def test_decoder_tokenizer_preserves_terminal_question_under_truncation() -> None:
    tokenizer = SimpleNamespace(
        pad_token_id=None,
        pad_token=None,
        eos_token="<eos>",
        padding_side="right",
        truncation_side="right",
    )
    configure_decoder_tokenizer(tokenizer)
    assert tokenizer.pad_token == "<eos>"
    assert tokenizer.padding_side == "left"
    assert tokenizer.truncation_side == "left"


def test_certified_path_returns_only_task_relevant_state() -> None:
    assert certified_state_answer("qa7", "two", "apple,football") == "two"
    assert certified_state_answer("qa8", "two", "apple,football") == "apple,football"


def test_generative_qa1_path_does_not_construct_qa78_certified_answer() -> None:
    assert optional_certified_state_answer(
        "foundation_generation", "qa1", None, None
    ) is None
