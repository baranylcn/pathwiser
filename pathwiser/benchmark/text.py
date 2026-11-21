"""Text normalisation and similarity helpers."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

_WS = re.compile(r"\s+")


_NON_ALNUM = re.compile(r"[^a-z0-9 ]")


_TOKEN = re.compile(r"[a-z0-9]+")


GREETING_ONLY = re.compile(
    r"^\W*(hi|hello|hey|hiya|howdy|greetings|good\s+(morning|afternoon|evening)|"
    r"yes|no|ok|okay|thanks|thank you|sure)\W*$",
    re.IGNORECASE,
)


STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "to",
        "of",
        "in",
        "on",
        "for",
        "and",
        "or",
        "is",
        "are",
        "be",
        "was",
        "were",
        "am",
        "my",
        "me",
        "i",
        "you",
        "your",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "with",
        "at",
        "by",
        "from",
        "as",
        "about",
        "user",
        "customer",
        "users",
        "wants",
        "want",
        "asks",
        "ask",
        "has",
        "have",
        "had",
        "they",
        "their",
        "them",
        "he",
        "she",
        "we",
        "our",
        "us",
        "can",
        "could",
        "would",
        "should",
        "will",
        "do",
        "does",
        "did",
        "not",
        "no",
        "how",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "whats",
        "how",
        "many",
        "much",
        "there",
        "any",
        "some",
        "if",
        "please",
        "like",
        "need",
        "so",
        "than",
        "then",
        "too",
        "very",
        "just",
        "also",
        "more",
        "most",
    ]
)


def clean_text(text: str) -> str:
    """Collapse whitespace; keep case and punctuation (the content is otherwise untouched)."""
    return _WS.sub(" ", text).strip()


def norm_text(text: str) -> str:
    """Aggressive normalisation used only for de-duplication and equality checks."""
    return _WS.sub(" ", _NON_ALNUM.sub("", text.lower())).strip()


def n_words(text: str) -> int:
    return len(text.split())


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokens(a)), set(tokens(b))
    union = len(sa | sb)
    return len(sa & sb) / union if union else 0.0


class TfIdf:
    """Minimal TF-IDF with L2-normalised sparse vectors (dicts). Pure stdlib, deterministic."""

    def __init__(self, docs: Sequence[str]):
        n = len(docs)
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(tokens(d)))
        self.idf = {w: math.log((1 + n) / (1 + c)) + 1.0 for w, c in df.items()}

    def vector(self, text: str) -> dict[str, float]:
        tf = Counter(t for t in tokens(text) if t in self.idf)
        vec = {w: c * self.idf[w] for w, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        return {w: v / norm for w, v in vec.items()} if norm else {}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(w, 0.0) for w, v in a.items())
