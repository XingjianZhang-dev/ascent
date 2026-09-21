from __future__ import annotations

import json

from experiments.audit_ruler_full_context_data import audit_task


class FakeTokenizer:
    def __call__(self, text, **_kwargs):
        return {"input_ids": list(range(len(text.split())))}

    def apply_chat_template(self, messages, **_kwargs):
        return list(range(len(messages[0]["content"].split()) + 2))


def test_full_context_data_audit_requires_exact_ordered_parser(tmp_path) -> None:
    text = (
        "One of the special magic numbers for alpha is: 1234567.\n"
        "One of the special magic numbers for beta is: 7654321.\n"
        "What are the special magic numbers for alpha and beta?"
    )
    row = {
        "index": 23129,
        "input": text,
        "outputs": ["1234567", "7654321", "1111111", "2222222"],
        "answer_prefix": "Answer:",
    }
    path = tmp_path / "validation.jsonl"
    path.write_text(json.dumps(row) + "\n")
    failed = audit_task(path, FakeTokenizer())
    assert failed["status"] == "failed"
    row["input"] = (
        text.replace(
            "What are",
            "One of the special magic numbers for gamma is: 1111111.\n"
            "One of the special magic numbers for delta is: 2222222.\nWhat are",
        )
    )
    row["input"] = row["input"].replace("alpha and beta?", "alpha, beta, gamma and delta?")
    path.write_text(json.dumps(row) + "\n")
    passed = audit_task(path, FakeTokenizer())
    assert passed["status"] == "passed"
    assert passed["parser_exact_rows"] == 1


def test_multivalue_audit_matches_official_order_independent_metric(tmp_path) -> None:
    row = {
        "index": 1,
        "input": (
            "One of the special magic numbers for alpha is: 1234567.\n"
            "One of the special magic numbers for alpha is: 7654321.\n"
            "One of the special magic numbers for alpha is: 1111111.\n"
            "One of the special magic numbers for alpha is: 2222222.\n"
            "What are all the special magic numbers for alpha?"
        ),
        "outputs": ["2222222", "1111111", "7654321", "1234567"],
        "answer_prefix": "Answer:",
    }
    path = tmp_path / "validation.jsonl"
    path.write_text(json.dumps(row) + "\n")
    passed = audit_task(path, FakeTokenizer(), "niah_all_values")
    assert passed["status"] == "passed"
    assert passed["parser_match_semantics"] == "exact_multiset_official_string_match_all"

    row["outputs"][-1] = "2222222"
    path.write_text(json.dumps(row) + "\n")
    failed = audit_task(path, FakeTokenizer(), "niah_all_values")
    assert failed["status"] == "failed"


def test_full_context_data_audit_allows_index_collision_not_row_duplicate(tmp_path) -> None:
    def make_row(value: str, key: str):
        return {
            "index": 197,
            "input": (
                f"One of the special magic numbers for {key} is: {value}.\n"
                f"What is the special magic number for {key}?"
            ),
            "outputs": [value, "1", "2", "3"],
            "answer_prefix": "Answer:",
        }

    path = tmp_path / "validation.jsonl"
    first = make_row("1234567", "alpha")
    second = make_row("7654321", "beta")
    path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    # This fixture has four registered outputs but only one parsed query value,
    # so isolate the collision fields from the expected parser failure.
    result = audit_task(path, FakeTokenizer())
    assert result["duplicate_index_occurrences"] == {"197": 2}
    assert result["gates"]["unique_full_rows"]
    assert result["gates"]["unique_inputs"]
    path.write_text(json.dumps(first) + "\n" + json.dumps(first) + "\n")
    duplicated = audit_task(path, FakeTokenizer())
    assert not duplicated["gates"]["unique_full_rows"]
    assert not duplicated["gates"]["unique_inputs"]
