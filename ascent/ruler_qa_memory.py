"""Query-conditioned document addressing for official RULER QA prompts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


_TOKEN = re.compile(r"[a-z0-9]+")
_DOCUMENT = re.compile(
    r"(?:^|\n\n)Document\s+(\d+):\n(.*?)(?=\n\nDocument\s+\d+:\n|\Z)",
    re.DOTALL,
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do",
    "does", "for", "from", "had", "has", "have", "how", "in", "is",
    "it", "of", "on", "or", "that", "the", "their", "this", "to",
    "was", "were", "what", "when", "where", "which", "who", "why",
    "with",
}


@dataclass(frozen=True)
class QADocument:
    number: int
    title: str
    text: str

    @property
    def source(self) -> str:
        return f"Document {self.number}:\n{self.text}"


@dataclass(frozen=True)
class RulerQARead:
    question: str
    documents: tuple[QADocument, ...]
    query_only_ranked: tuple[QADocument, ...]
    graph_ranked: tuple[QADocument, ...]
    irrelevant_ranked: tuple[QADocument, ...]


def lexical_tokens(text: str) -> list[str]:
    return [token for token in _TOKEN.findall(text.lower()) if token not in _STOP]


def parse_ruler_qa(prompt: str) -> tuple[str, tuple[QADocument, ...]]:
    """Extract the final question and numbered documents without using answers."""

    marker = "The following are given documents.\n\n"
    if marker not in prompt:
        raise ValueError("RULER QA document marker is missing")
    remainder = prompt.split(marker, 1)[1]
    question_marker = "\n\nQuestion: "
    if question_marker not in remainder:
        raise ValueError("RULER QA question marker is missing")
    context, question = remainder.rsplit(question_marker, 1)
    preamble = "\n\nAnswer the question based on the given documents."
    if preamble in context:
        context = context.split(preamble, 1)[0]
    documents: list[QADocument] = []
    for match in _DOCUMENT.finditer(context.strip()):
        text = match.group(2).strip()
        title = text.splitlines()[0].strip()
        documents.append(QADocument(int(match.group(1)), title, text))
    if not documents:
        raise ValueError("RULER QA prompt contains no numbered documents")
    numbers = [document.number for document in documents]
    if len(numbers) != len(set(numbers)):
        raise ValueError("RULER QA document numbers must be unique")
    return question.strip(), tuple(documents)


def bm25_scores(
    question: str,
    documents: tuple[QADocument, ...],
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> list[float]:
    """Deterministic BM25 over document title and body."""

    if not documents:
        raise ValueError("BM25 requires at least one document")
    query = lexical_tokens(question)
    tokenized = [lexical_tokens(document.text) for document in documents]
    average_length = sum(map(len, tokenized)) / len(tokenized)
    frequencies: list[dict[str, int]] = []
    document_frequency: dict[str, int] = {}
    for tokens in tokenized:
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        frequencies.append(counts)
        for token in counts:
            document_frequency[token] = document_frequency.get(token, 0) + 1
    scores: list[float] = []
    for counts, tokens in zip(frequencies, tokenized, strict=True):
        score = 0.0
        length_normalizer = 1.0 - b + b * len(tokens) / max(average_length, 1.0)
        for token in query:
            frequency = counts.get(token, 0)
            if not frequency:
                continue
            df = document_frequency[token]
            inverse_frequency = math.log(1.0 + (len(documents) - df + 0.5) / (df + 0.5))
            score += inverse_frequency * frequency * (k1 + 1.0) / (
                frequency + k1 * length_normalizer
            )
        scores.append(score)
    return scores


def read_ruler_qa_graph(prompt: str, *, seed_documents: int = 2) -> RulerQARead:
    """Rank query seeds, then add documents explicitly linked by their titles."""

    question, documents = parse_ruler_qa(prompt)
    if not 0 < seed_documents <= len(documents):
        raise ValueError("seed_documents must fit the document collection")
    scores = bm25_scores(question, documents)
    base_indices = sorted(range(len(documents)), key=lambda i: (-scores[i], i))
    ranked: list[int] = list(base_indices[:seed_documents])
    ranked_set = set(ranked)
    for source_index in tuple(ranked):
        source = documents[source_index].text.lower()
        linked = [
            index
            for index, document in enumerate(documents)
            if index not in ranked_set
            and len(document.title.strip()) >= 4
            and document.title.lower() in source
        ]
        linked.sort(key=lambda i: (-scores[i], i))
        ranked.extend(linked)
        ranked_set.update(linked)
    graph_frontier = set(ranked_set)
    ranked.extend(index for index in base_indices if index not in ranked_set)
    irrelevant_indices = [
        index for index in reversed(base_indices) if index not in graph_frontier
    ] + [index for index in reversed(base_indices) if index in graph_frontier]
    irrelevant = tuple(documents[index] for index in irrelevant_indices)
    return RulerQARead(
        question=question,
        documents=documents,
        query_only_ranked=tuple(documents[index] for index in base_indices),
        graph_ranked=tuple(documents[index] for index in ranked),
        irrelevant_ranked=irrelevant,
    )


def serialize_documents(documents: tuple[QADocument, ...]) -> str:
    return "\n\n".join(document.source for document in documents)


def document_sketch(document: QADocument, question: str) -> str:
    """Return a deterministic query-conditioned title plus one body sentence."""

    lines = document.text.splitlines()
    body = " ".join(lines[1:]).strip() if len(lines) > 1 else document.text
    sentences = [sentence.strip() for sentence in _SENTENCE.split(body) if sentence.strip()]
    if not sentences:
        sentences = [body or document.title]
    query_terms = lexical_tokens(question) + lexical_tokens(document.title)
    counts: dict[str, int] = {}
    for token in query_terms:
        counts[token] = counts.get(token, 0) + 1

    def score(sentence: str) -> float:
        tokens = lexical_tokens(sentence)
        frequencies: dict[str, int] = {}
        for token in tokens:
            frequencies[token] = frequencies.get(token, 0) + 1
        overlap = sum(
            counts[token] * min(frequency, 2)
            for token, frequency in frequencies.items()
            if token in counts
        )
        return float(overlap) / math.sqrt(max(len(tokens), 1))

    best_index = max(range(len(sentences)), key=lambda index: (score(sentences[index]), -index))
    return f"Document {document.number}: {document.title}\n{sentences[best_index]}"


def serialize_document_sketches(
    documents: tuple[QADocument, ...], question: str
) -> str:
    """Serialize independently computed sketches so every prefix stays nested."""

    return "\n\n".join(document_sketch(document, question) for document in documents)


def serialize_ruler_qa_state(
    documents: tuple[QADocument, ...], question: str, *, mode: str
) -> str:
    """Serialize a registered state without exposing target annotations."""

    if mode == "document_sketches":
        return serialize_document_sketches(documents, question)
    if mode == "full_documents":
        return serialize_documents(documents)
    raise ValueError(f"unsupported RULER QA state serialization: {mode}")


def encode_ruler_qa_state(
    tokenizer: Any,
    documents: tuple[QADocument, ...],
    question: str,
    *,
    mode: str,
) -> list[int]:
    """Encode each append-only document independently with a fixed delimiter.

    Tokenizing a whole text prefix can change its final BPE token when more
    bytes are appended. Independent per-document encoding plus an explicit
    delimiter makes token-id nesting constructive rather than accidental.
    """

    delimiter = tokenizer("\n\n", add_special_tokens=False)["input_ids"]
    if not delimiter:
        raise ValueError("the registered document delimiter must tokenize")
    encoded: list[int] = []
    for document in documents:
        segment = serialize_ruler_qa_state((document,), question, mode=mode)
        segment_ids = tokenizer(segment, add_special_tokens=False)["input_ids"]
        if not segment_ids:
            raise ValueError("a registered state document cannot be empty")
        encoded.extend(int(value) for value in segment_ids)
        encoded.extend(int(value) for value in delimiter)
    return encoded
