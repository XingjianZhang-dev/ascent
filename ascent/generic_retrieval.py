"""Task-agnostic deterministic lexical retrieval controls.

This module intentionally contains no BABILong entity, location, object, or
event vocabulary.  It is a generic full-corpus retrieval baseline, not the
bounded causal ASCENT write state.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


_TOKEN = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "before",
        "did",
        "do",
        "does",
        "how",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
    }
)


@dataclass(frozen=True)
class LexicalRetrieval:
    passages: tuple[str, ...]
    character_positions: tuple[int, ...]
    corpus_utf8_bytes: int


def lexical_tokens(text: str) -> tuple[str, ...]:
    """Return generic alphanumeric content tokens."""
    return tuple(
        token
        for token in (match.group(0).lower() for match in _TOKEN.finditer(text))
        if len(token) > 1 and token not in _STOP_WORDS
    )


def split_passages(text: str) -> list[tuple[int, str]]:
    """Split text at generic punctuation and retain character positions."""
    sentences: list[tuple[int, str]] = []
    offset = 0
    for raw in _SENTENCE_BOUNDARY.split(text):
        sentence = raw.strip()
        if sentence:
            position = text.find(sentence, offset)
            if position < 0:
                raise RuntimeError("sentence offset reconstruction failed")
            sentences.append((position, sentence))
            offset = position + len(sentence)
    return sentences


def retrieve_lexical_chain(text: str, question: str, *, count: int) -> LexicalRetrieval:
    """Retrieve a deterministic BM25-like lexical chain with generic expansion.

    Original question terms receive weight two.  After each selection, content
    terms in that passage are added with weight one, allowing generic multi-hop
    entity expansion.  Exact-score ties prefer the most recent passage.  The
    returned passages are restored to chronological order for the decoder.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    sentences = split_passages(text)
    if not sentences:
        return LexicalRetrieval((), (), len(text.encode("utf-8")))
    documents = [Counter(lexical_tokens(sentence)) for _, sentence in sentences]
    document_frequency = Counter(token for document in documents for token in document)
    question_terms = Counter(lexical_tokens(question))
    expansion_terms: Counter[str] = Counter()
    chosen: list[int] = []
    document_count = len(documents)
    for _ in range(min(count, document_count)):
        weights: Counter[str] = Counter(
            {token: 2 * frequency for token, frequency in question_terms.items()}
        )
        weights.update(expansion_terms)
        best: tuple[float, int] | None = None
        for index, document in enumerate(documents):
            if index in chosen:
                continue
            score = 0.0
            for token, weight in weights.items():
                frequency = document.get(token, 0)
                if not frequency:
                    continue
                df = document_frequency[token]
                inverse_frequency = math.log(
                    1.0 + (document_count - df + 0.5) / (df + 0.5)
                )
                score += weight * inverse_frequency * frequency / (frequency + 1.2)
            candidate = (score, index)
            if best is None or candidate > best:
                best = candidate
        if best is None or best[0] <= 0.0:
            break
        chosen.append(best[1])
        expansion_terms.update(documents[best[1]])
    chronological = sorted(chosen)
    return LexicalRetrieval(
        passages=tuple(sentences[index][1] for index in chronological),
        character_positions=tuple(sentences[index][0] for index in chronological),
        corpus_utf8_bytes=len(text.encode("utf-8")),
    )
