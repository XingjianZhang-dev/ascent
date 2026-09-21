from __future__ import annotations

import pytest

from ascent.latent_evidence import reconstructed_niah_digit_prefix


def test_extracts_partial_query_matched_numeric_evidence() -> None:
    text = " of the special magic numbers for tasty-bust is: 98477"
    question = "What is the special magic number for tasty-bust?"
    assert reconstructed_niah_digit_prefix(text, question) == "98477"


def test_rejects_evidence_for_a_different_key() -> None:
    text = "special magic number for other-key is: 1234567"
    question = "What is the special magic number for tasty-bust?"
    assert reconstructed_niah_digit_prefix(text, question) == ""


def test_caps_evidence_and_validates_limit() -> None:
    text = "special magic numbers for key is: 123456789"
    assert reconstructed_niah_digit_prefix(text, "number for key", max_digits=7) == "1234567"
    with pytest.raises(ValueError):
        reconstructed_niah_digit_prefix(text, "number for key", max_digits=0)
