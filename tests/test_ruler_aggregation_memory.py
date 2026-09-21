import pytest

from ascent.ruler_aggregation_memory import (
    read_ruler_common_words,
    read_ruler_frequency,
)


def prompt(context: str, question: str = "Which coded words?") -> str:
    return (
        "Read the following coded text and track the frequency of each coded "
        "word. Find the three most frequently appeared coded words. "
        f"{context}\nQuestion: {question}"
    )


def test_frequency_reader_orders_by_count_then_first_occurrence() -> None:
    read = read_ruler_frequency(
        prompt("aaaaaa bbbbbb cccccc bbbbbb aaaaaa bbbbbb cccccc")
    )
    assert read.top_words[:3] == ("bbbbbb", "aaaaaa", "cccccc")
    assert [entry.count for entry in read.entries] == [2, 3, 2]


def test_frequency_reader_excludes_instruction_and_question() -> None:
    read = read_ruler_frequency(
        prompt("aaaaaa bbbbbb aaaaaa", question="Please ignore bbbbbb bbbbbb")
    )
    assert read.top_words[:2] == ("aaaaaa", "bbbbbb")


def test_frequency_reader_enforces_frozen_bound() -> None:
    with pytest.raises(RuntimeError, match="slot bound"):
        read_ruler_frequency(prompt("aaaaaa bbbbbb"), memory_slots=1)


def test_frequency_reader_requires_official_markers() -> None:
    with pytest.raises(ValueError, match="context marker"):
        read_ruler_frequency("aaaaaa bbbbbb")


def test_common_words_reader_handles_numbered_multiword_items() -> None:
    text = (
        "Below is a numbered list of words. In these words, some appear more "
        "often than others. Memorize the ones that appear most often.\n"
        "1. creme brulee 2. chalk 3. creme brulee 4. oyster 5. chalk "
        "6. creme brulee\nQuestion: What are the common words?"
    )
    read = read_ruler_common_words(text)
    assert read.top_words[:3] == ("creme brulee", "chalk", "oyster")
    assert [entry.count for entry in read.entries] == [3, 2, 1]


def test_common_words_reader_excludes_question_and_enforces_bound() -> None:
    text = (
        "Below is a numbered list of words. Memorize the ones that appear most "
        "often.\n1. chalk 2. oyster 3. chalk\nQuestion: oyster oyster"
    )
    assert read_ruler_common_words(text).top_words[:2] == ("chalk", "oyster")
    with pytest.raises(RuntimeError, match="slot bound"):
        read_ruler_common_words(text, memory_slots=1)
