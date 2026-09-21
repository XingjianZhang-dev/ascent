from ascent.generic_retrieval import lexical_tokens, retrieve_lexical_chain


def test_generic_vocabulary_excludes_question_scaffolding() -> None:
    assert lexical_tokens("Where was the token before the place?") == ("token", "place")


def test_recency_breaks_exact_lexical_ties() -> None:
    text = "Alex moved north. Noise is here. Alex moved south."
    result = retrieve_lexical_chain(text, "Where is Alex?", count=1)
    assert result.passages == ("Alex moved south.",)


def test_selected_passage_expands_generic_chain() -> None:
    text = "Rin entered amber. Rin obtained widget. Rin entered cobalt."
    result = retrieve_lexical_chain(text, "Where is widget?", count=2)
    assert result.passages == ("Rin obtained widget.", "Rin entered cobalt.")
    assert result.corpus_utf8_bytes == len(text.encode("utf-8"))


def test_retrieval_is_target_blind_and_chronological() -> None:
    text = "Mira reached alpha. Mira acquired parcel. Mira reached beta."
    first = retrieve_lexical_chain(text, "Where is parcel?", count=3)
    second = retrieve_lexical_chain(text, "Where is parcel?", count=3)
    assert first == second
    assert list(first.character_positions) == sorted(first.character_positions)
