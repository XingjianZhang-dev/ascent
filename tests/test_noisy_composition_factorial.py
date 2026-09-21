import json
import hashlib
from pathlib import Path

import numpy as np

from experiments.prepare_noisy_composition_factorial import (
    exact_nll,
    exact_sum_nll,
    make_panel,
    posterior,
    target_probability,
)
from experiments.run_noisy_composition_prompt import (
    ascent_prompt,
    extract_answer,
    foundation_prompt,
)
from experiments.run_noisy_composition_candidate import (
    ascent_prompt as candidate_ascent_prompt,
    candidate_probabilities,
    foundation_prompt as candidate_foundation_prompt,
)
from experiments.analyze_noisy_composition_factorial import (
    T95_BONFERRONI_DF8_M6,
    T95_DF8,
    interval,
)
from experiments.analyze_noisy_composition_candidate import (
    T95_BONFERRONI_DF8 as CANDIDATE_BONFERRONI_CRITICALS,
    T95_DF8 as CANDIDATE_T95_DF8,
    interval as candidate_interval,
)


def test_categorical_posterior_normalizes_and_refines() -> None:
    weak = posterior([3, 3], labels=8, eta=0.28)
    rich = posterior([3, 3, 3, 3, 3], labels=8, eta=0.28)
    assert np.isclose(weak.sum(), 1.0)
    assert np.isclose(rich.sum(), 1.0)
    assert rich[3] > weak[3] > 1 / 8


def test_affine_target_posterior_is_valid() -> None:
    first = np.full(8, 1 / 8)
    second = np.full(8, 1 / 8)
    probabilities = [target_probability(first, second, target, 8) for target in range(8)]
    assert np.allclose(probabilities, np.full(8, 1 / 8))
    assert np.isclose(sum(probabilities), 1.0)


def test_generated_panel_has_strict_exact_information_curve() -> None:
    rows = make_panel(
        seed=2026081501,
        panel_index=1,
        rows=120,
        labels=8,
        eta=0.28,
        max_rounds=5,
    )
    curve = [exact_nll(rows, rounds, 8, 0.28) for rounds in (2, 3, 5)]
    assert curve[0] > curve[1] > curve[2]
    assert all(row["target"] == (2 * row["first_value"] + row["second_value"]) % 8 for row in rows)


def test_prompt_prefixes_are_nested_and_target_absent() -> None:
    row = make_panel(
        seed=7,
        panel_index=1,
        rows=1,
        labels=8,
        eta=0.28,
        max_rounds=5,
    )[0]
    weak = ascent_prompt(row, 2, 0.28)
    rich = ascent_prompt(row, 5, 0.28)
    assert ", ".join(map(str, row["first_observations"][:2])) in weak
    assert ", ".join(map(str, row["first_observations"])) in rich
    assert "No memory observations" in foundation_prompt(row)
    assert "first_value" not in weak and "second_value" not in weak


def test_answer_extraction_uses_last_standalone_candidate() -> None:
    assert extract_answer("4") == 4
    assert extract_answer("The result is 6.") == 6
    assert extract_answer("Compute 2 times A, so the answer is 5") == 5
    assert extract_answer("unknown") is None
    assert extract_answer("14") is None
    assert extract_answer("I considered 2.\nFINAL: 5\nThen ignored 1.") == 5
    assert extract_answer("14", maximum=14) == 14


def test_candidate_prompt_is_target_isolated_and_has_no_worked_answer() -> None:
    row = make_panel(
        seed=11,
        panel_index=1,
        rows=1,
        labels=8,
        eta=0.28,
        max_rounds=5,
    )[0]
    altered = dict(row)
    altered["target"] = (row["target"] + 1) % 8
    assert candidate_foundation_prompt(row) == candidate_foundation_prompt(altered)
    assert candidate_ascent_prompt(row, 5, 0.28) == candidate_ascent_prompt(
        altered, 5, 0.28
    )
    assert "Example" not in candidate_ascent_prompt(row, 5, 0.28)


def test_qwen_candidate_design_is_frozen_and_relative_state_decreases() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs" / "posttraining_noisy_composition_qwen_candidate.json").read_text()
    )
    assert config["development_panels"] == [config["panels"][0]]
    assert len(config["confirmation_panels"]) == 9
    assert config["require_model_by_state_interactions"] is True
    ratios = config["factorial"]["co_scaled_rounds_per_parameter"]
    assert ratios[0] > ratios[1] > ratios[2]
    for panel, relative in config["panel_path_by_name"].items():
        actual = hashlib.sha256((root / "data" / relative).read_bytes()).hexdigest()
        assert actual == config["panel_sha256_by_name"][panel]


def test_qwen_cot_design_is_fresh_length_safe_and_target_isolated() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs" / "posttraining_noisy_composition_qwen_cot.json").read_text()
    )
    assert config["prompt_style"] == "qwen_cot"
    assert config["max_new_tokens"] == 96
    assert config["minimum_final_marker_coverage"] == 0.95
    assert len(config["confirmation_panels"]) == 9
    row = make_panel(
        seed=19,
        panel_index=1,
        rows=1,
        labels=8,
        eta=0.28,
        max_rounds=5,
    )[0]
    changed = dict(row)
    changed["target"] = (row["target"] + 1) % 8
    assert foundation_prompt(row, "qwen_cot") == foundation_prompt(changed, "qwen_cot")
    assert ascent_prompt(row, 5, 0.28, "qwen_cot") == ascent_prompt(
        changed, 5, 0.28, "qwen_cot"
    )
    assert "Example" not in ascent_prompt(row, 5, 0.28, "qwen_cot")


def test_posterior_replay_is_nested_target_isolated_dual_path() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs" / "posttraining_noisy_composition_posterior_replay.json").read_text()
    )
    assert config["prompt_style"] == "posterior_replay"
    assert config["minimum_answer_coverage"] == 0.99
    ratios = config["factorial"]["co_scaled_rounds_per_parameter"]
    assert ratios[0] > ratios[1] > ratios[2]
    row = make_panel(
        seed=23,
        panel_index=1,
        rows=1,
        labels=8,
        eta=0.28,
        max_rounds=5,
    )[0]
    changed = dict(row)
    changed["target"] = (row["target"] + 1) % 8
    weak = ascent_prompt(row, 2, 0.28, "posterior_replay")
    rich = ascent_prompt(row, 5, 0.28, "posterior_replay")
    assert weak == ascent_prompt(changed, 2, 0.28, "posterior_replay")
    assert rich == ascent_prompt(changed, 5, 0.28, "posterior_replay")
    assert ", ".join(map(str, row["first_observations"][:2])) in weak
    assert ", ".join(map(str, row["first_observations"])) in rich
    assert "Certified A posterior:" in rich and "Certified B MAP:" in rich


def test_sum_replay_has_strict_information_and_frozen_standard_composition() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs" / "posttraining_noisy_composition_sum_replay.json").read_text()
    )
    assert config["construction"]["target_rule"] == "first_value + second_value"
    assert config["prompt_style"] == "sum_posterior_replay"
    rows = make_panel(
        seed=2026082001,
        panel_index=1,
        rows=256,
        labels=8,
        eta=0.28,
        max_rounds=5,
    )
    for row in rows:
        row["target"] = row["first_value"] + row["second_value"]
    curve = [exact_sum_nll(rows, rounds, 8, 0.28) for rounds in (2, 3, 5)]
    assert curve[0] > curve[1] > curve[2]
    row = rows[0]
    changed = dict(row)
    changed["target"] = (row["target"] + 1) % 15
    assert ascent_prompt(row, 5, 0.28, "sum_posterior_replay") == ascent_prompt(
        changed, 5, 0.28, "sum_posterior_replay"
    )


def test_compact_map_sum_exposes_decoder_values_not_raw_noise() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs" / "posttraining_noisy_composition_map_sum.json").read_text()
    )
    assert config["prompt_style"] == "map_sum_replay"
    assert config["max_new_tokens"] == 8
    row = make_panel(
        seed=29,
        panel_index=1,
        rows=1,
        labels=8,
        eta=0.28,
        max_rounds=5,
    )[0]
    row["target"] = row["first_value"] + row["second_value"]
    changed = dict(row)
    changed["target"] = (row["target"] + 1) % 15
    prompt = ascent_prompt(row, 5, 0.28, "map_sum_replay")
    assert prompt == ascent_prompt(changed, 5, 0.28, "map_sum_replay")
    assert "Decoded A:" in prompt and "Decoded B:" in prompt
    assert "posterior" not in prompt.lower()
    assert "readings" not in prompt.lower()


def test_bonferroni_confirmation_interval_is_predeclared_and_stricter() -> None:
    values = [0.10, 0.12, 0.08, 0.14, 0.09, 0.11, 0.13, 0.07, 0.15]
    ordinary = interval(values, "confirmation")
    familywise = interval(
        values,
        "confirmation",
        critical_value=T95_BONFERRONI_DF8_M6,
    )
    assert T95_BONFERRONI_DF8_M6 > T95_DF8
    assert familywise["mean"] == ordinary["mean"]
    assert familywise["standard_error"] == ordinary["standard_error"]
    assert familywise["ci95_low"] < ordinary["ci95_low"]
    assert familywise["ci95_high"] > ordinary["ci95_high"]


def test_sum_candidate_prompt_is_target_isolated_and_uses_fixed_labels() -> None:
    labels = list("ABCDEFGHIJKLMNO")
    row = make_panel(
        seed=31,
        panel_index=1,
        rows=1,
        labels=8,
        eta=0.28,
        max_rounds=5,
    )[0]
    row["target"] = row["first_value"] + row["second_value"]
    changed = dict(row)
    changed["target"] = (row["target"] + 1) % 15
    style = "sum_posterior_candidate_labels"
    assert candidate_foundation_prompt(row, style, labels) == candidate_foundation_prompt(
        changed, style, labels
    )
    prompt = candidate_ascent_prompt(row, 5, 0.28, style, labels)
    assert prompt == candidate_ascent_prompt(changed, 5, 0.28, style, labels)
    assert "A=0" in prompt and "O=14" in prompt
    assert "Certified A posterior:" in prompt and "Certified B MAP:" in prompt


def test_eight_estimand_candidate_familywise_interval_is_stricter() -> None:
    values = [0.20, 0.22, 0.18, 0.24, 0.19, 0.21, 0.23, 0.17, 0.25]
    ordinary = candidate_interval(values, "confirmation")
    familywise = candidate_interval(
        values,
        "confirmation",
        critical_value=CANDIDATE_BONFERRONI_CRITICALS[8],
    )
    assert CANDIDATE_BONFERRONI_CRITICALS[8] > CANDIDATE_T95_DF8
    assert familywise["ci95_low"] < ordinary["ci95_low"]
    assert familywise["ci95_high"] > ordinary["ci95_high"]


def test_numeric_sum_candidate_prompt_is_target_isolated() -> None:
    candidates = [f"{value:02d}" for value in range(15)]
    row = make_panel(
        seed=37,
        panel_index=1,
        rows=1,
        labels=8,
        eta=0.28,
        max_rounds=5,
    )[0]
    row["target"] = row["first_value"] + row["second_value"]
    changed = dict(row)
    changed["target"] = (row["target"] + 1) % 15
    style = "sum_posterior_candidate_numeric"
    assert candidate_foundation_prompt(row, style, candidates) == candidate_foundation_prompt(
        changed, style, candidates
    )
    prompt = candidate_ascent_prompt(row, 5, 0.28, style, candidates)
    assert prompt == candidate_ascent_prompt(changed, 5, 0.28, style, candidates)
    assert "exactly two digits" in prompt
    assert "Certified A posterior:" in prompt and "Certified B MAP:" in prompt


def test_two_token_candidate_probability_matches_joint_sequence_score() -> None:
    import torch

    class Batch(dict):
        def __getattr__(self, name):
            return self[name]

        def to(self, _device):
            return self

    class Tokenizer:
        def __call__(self, texts, **_kwargs):
            count = len(texts)
            return Batch(
                input_ids=torch.full((count, 1), 9, dtype=torch.long),
                attention_mask=torch.ones((count, 1), dtype=torch.long),
            )

    class Output:
        def __init__(self, logits):
            self.logits = logits

    class Model:
        def __init__(self):
            self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        def parameters(self):
            yield self.anchor

        def __call__(self, input_ids, **_kwargs):
            logits = torch.zeros((*input_ids.shape, 10), dtype=torch.float32)
            if input_ids.shape[1] == 1:
                logits[:, -1, 0] = 0.0
                logits[:, -1, 1] = np.log(2.0)
            else:
                for row, first in enumerate(input_ids[:, -1].tolist()):
                    if first == 0:
                        logits[row, -1, 2] = 0.0
                        logits[row, -1, 3] = np.log(3.0)
                    elif first == 1:
                        logits[row, -1, 2] = np.log(4.0)
            return Output(logits)

    sequences = [[0, 2], [0, 3], [1, 2]]
    actual = candidate_probabilities(Model(), Tokenizer(), ["prompt"], sequences, 1)[0]
    base = torch.zeros(10)
    base[1] = np.log(2.0)
    conditional_zero = torch.zeros(10)
    conditional_zero[3] = np.log(3.0)
    conditional_one = torch.zeros(10)
    conditional_one[2] = np.log(4.0)
    scores = torch.tensor(
        [
            base.log_softmax(0)[0] + conditional_zero.log_softmax(0)[2],
            base.log_softmax(0)[0] + conditional_zero.log_softmax(0)[3],
            base.log_softmax(0)[1] + conditional_one.log_softmax(0)[2],
        ]
    )
    expected = scores.softmax(0).numpy()
    assert np.allclose(actual, expected, atol=1e-7)
    assert np.isclose(actual.sum(), 1.0)
