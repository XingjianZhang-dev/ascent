import pytest

from ascent.ruler_qa_memory import (
    bm25_scores,
    document_sketch,
    encode_ruler_qa_state,
    parse_ruler_qa,
    read_ruler_qa_graph,
    serialize_document_sketches,
    serialize_documents,
    serialize_ruler_qa_state,
)


PROMPT = """Answer the question based on the given documents.

The following are given documents.

Document 1:
Alpha City
Alpha City is the birthplace of the scientist. Its sister city is Beta Harbor.

Document 2:
Noise Article
This unrelated article discusses painting and music.

Document 3:
Beta Harbor
Beta Harbor is located in Norway.

Answer the question based on the given documents. Only give me the answer and do not output any other words.

Question: In which country is the scientist's sister city?"""


def test_parse_ruler_qa_extracts_documents_and_question() -> None:
    question, documents = parse_ruler_qa(PROMPT)
    assert question == "In which country is the scientist's sister city?"
    assert [document.number for document in documents] == [1, 2, 3]
    assert documents[0].title == "Alpha City"


def test_graph_expansion_places_linked_second_hop_before_noise() -> None:
    read = read_ruler_qa_graph(PROMPT, seed_documents=1)
    assert read.graph_ranked[0].title == "Alpha City"
    assert read.graph_ranked[1].title == "Beta Harbor"
    assert read.irrelevant_ranked[0].title == "Noise Article"
    assert "Document 3:\nBeta Harbor" in serialize_documents(read.graph_ranked[:2])


def test_bm25_and_reader_validate_inputs() -> None:
    _, documents = parse_ruler_qa(PROMPT)
    assert len(bm25_scores("scientist birthplace", documents)) == 3
    with pytest.raises(ValueError):
        read_ruler_qa_graph(PROMPT, seed_documents=0)
    with pytest.raises(ValueError):
        parse_ruler_qa("no documents")


def test_document_sketches_are_query_conditioned_and_nested() -> None:
    read = read_ruler_qa_graph(PROMPT, seed_documents=1)
    sketch = document_sketch(read.graph_ranked[0], read.question)
    assert "birthplace of the scientist" in sketch
    small = serialize_document_sketches(read.graph_ranked[:2], read.question)
    large = serialize_document_sketches(read.graph_ranked[:3], read.question)
    assert large.startswith(small)


def test_registered_state_serialization_supports_complete_documents() -> None:
    read = read_ruler_qa_graph(PROMPT, seed_documents=1)
    full = serialize_ruler_qa_state(
        read.graph_ranked[:2], read.question, mode="full_documents"
    )
    assert "Its sister city is Beta Harbor" in full
    with pytest.raises(ValueError):
        serialize_ruler_qa_state(read.graph_ranked, read.question, mode="unknown")


def test_independent_document_encoding_constructs_exact_token_prefix() -> None:
    class BoundaryTokenizer:
        def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
            assert not add_special_tokens
            return {"input_ids": [len(text), sum(text.encode()) % 997]}

    read = read_ruler_qa_graph(PROMPT, seed_documents=1)
    tokenizer = BoundaryTokenizer()
    small = encode_ruler_qa_state(
        tokenizer, read.graph_ranked[:2], read.question, mode="full_documents"
    )
    large = encode_ruler_qa_state(
        tokenizer, read.graph_ranked[:3], read.question, mode="full_documents"
    )
    assert large[: len(small)] == small
