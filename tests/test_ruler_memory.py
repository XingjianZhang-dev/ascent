from ascent.ruler_memory import read_ruler_niah, read_ruler_niah_all_values


def test_ruler_read_is_causal_and_follows_query_order() -> None:
    text = (
        "One of the special magic numbers for amber-key is: 1234567. filler\n"
        "One of the special magic numbers for blue-key is: 7654321. filler\n"
        "What are all the special magic numbers for blue-key and amber-key?"
    )
    read = read_ruler_niah(text)
    assert read.values == ("7654321", "1234567")
    assert all(fact.character_position < text.rfind(read.question) for fact in read.facts)
    assert read.persistent_payload_bytes > 0


def test_ruler_read_obeys_bounded_lru_capacity() -> None:
    text = (
        "One of the special magic numbers for evicted is: 1111111.\n"
        "One of the special magic numbers for retained is: 2222222.\n"
        "What is the special magic number for evicted?"
    )
    read = read_ruler_niah(text, memory_slots=1)
    assert read.values == ()


def test_ruler_read_handles_uuid_format() -> None:
    value = "0d5d4ddf-39be-4a4f-8da9-50a920c382a7"
    text = (
        f"One of the special magic uuids for loutish-pine is: {value}. filler\n"
        "What is the special magic uuid for loutish-pine mentioned in the provided text?"
    )
    assert read_ruler_niah(text).values == (value,)


def test_ruler_multivalue_read_preserves_repeated_key_order() -> None:
    text = (
        "One of the special magic numbers for amber-key is: 1234567. filler\n"
        "One of the special magic numbers for amber-key is: 7654321. filler\n"
        "What are all the special magic numbers for amber-key?"
    )
    latest = read_ruler_niah(text)
    multivalue = read_ruler_niah_all_values(text)
    assert latest.values == ("7654321",)
    assert multivalue.values == ("1234567", "7654321")
    assert all(
        fact.character_position < text.rfind(multivalue.question)
        for fact in multivalue.facts
    )


def test_ruler_multivalue_read_has_bounded_event_capacity() -> None:
    text = (
        "One of the special magic numbers for amber-key is: 1111111.\n"
        "One of the special magic numbers for amber-key is: 2222222.\n"
        "What are all the special magic numbers for amber-key?"
    )
    assert read_ruler_niah_all_values(text, memory_slots=1).values == ("2222222",)
