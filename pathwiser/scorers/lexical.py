"""Training-free lexical scorers: random reference, word overlap and TF-IDF cosine."""

from __future__ import annotations

import hashlib
import re

import numpy as np

from pathwiser.scorers.base import context_text, split_scores

_TOKEN = re.compile(r"[a-z0-9']+")


class RandomScorer:
    name = "random"

    def __init__(self) -> None:
        self.params: dict = {}

    def fit(self, train: list[dict], val: list[dict] | None = None) -> None:
        pass

    def score(self, records: list[dict]) -> list[np.ndarray]:
        out = []
        for r in records:
            seed = int(hashlib.sha256(r["example_id"].encode("utf-8")).hexdigest()[:8], 16)
            out.append(np.random.default_rng(seed).random(len(r["input"]["conditions"])))
        return out


def _stem(tok: str) -> str:
    tok = tok.strip("'").removesuffix("'s")
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def content_words(text: str) -> frozenset[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return frozenset(
        w for w in (_stem(t) for t in _TOKEN.findall(text.lower())) if len(w) > 1 and w not in ENGLISH_STOP_WORDS
    )


class LexicalOverlap:
    """|M ∩ C| / sqrt(|M|·|C|) over content words (stop words removed, naive plural stemming)."""

    def __init__(self, use_context: bool = False) -> None:
        self.use_context = use_context
        self.name = "lexical_ctx" if use_context else "lexical"
        self.params = {"use_context": use_context}

    def fit(self, train: list[dict], val: list[dict] | None = None) -> None:
        pass

    def score(self, records: list[dict]) -> list[np.ndarray]:
        cache: dict[str, frozenset[str]] = {}

        def words(text: str) -> frozenset[str]:
            if text not in cache:
                cache[text] = content_words(text)
            return cache[text]

        out = []
        for r in records:
            text = r["input"]["message"] + (" " + context_text(r) if self.use_context else "")
            m = words(text)
            s = np.zeros(len(r["input"]["conditions"]))
            for j, cond in enumerate(r["input"]["conditions"]):
                c = words(cond)
                if m and c:
                    s[j] = len(m & c) / np.sqrt(len(m) * len(c))
            out.append(s)
        return out


class TfidfCosine:
    """Cosine of TF-IDF vectors. Context (if used) is added to the message vector with weight ``context_weight``."""

    needs_train = True

    def __init__(self, use_context: bool = False, context_weight: float = 0.5) -> None:
        self.use_context = use_context
        self.context_weight = context_weight
        self.name = "tfidf_ctx" if use_context else "tfidf"
        self.params = {
            "use_context": use_context,
            "context_weight": context_weight if use_context else None,
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "stop_words": "english",
            "fitted_on": "train",
        }
        self.vec = None

    def fit(self, train: list[dict], val: list[dict] | None = None) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        corpus = [r["input"]["message"] for r in train]
        corpus += sorted({c for r in train for c in r["input"]["conditions"]})
        if self.use_context:
            corpus += [t for t in (context_text(r) for r in train) if t]
        self.vec = TfidfVectorizer(
            lowercase=True, strip_accents="unicode", stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        ).fit(corpus)

    def score(self, records: list[dict]) -> list[np.ndarray]:
        from sklearn.preprocessing import normalize

        if self.vec is None:
            raise RuntimeError("fit() must be called before score()")
        q = self.vec.transform([r["input"]["message"] for r in records])
        if self.use_context:
            q = normalize(q + self.context_weight * self.vec.transform([context_text(r) for r in records]))
        uniq = sorted({c for r in records for c in r["input"]["conditions"]})
        row = {c: i for i, c in enumerate(uniq)}
        cmat = self.vec.transform(uniq)
        ex_idx = np.repeat(np.arange(len(records)), [len(r["input"]["conditions"]) for r in records])
        cond_idx = np.array([row[c] for r in records for c in r["input"]["conditions"]])
        sims = np.asarray(q[ex_idx].multiply(cmat[cond_idx]).sum(axis=1)).ravel()
        return split_scores(sims, records)
