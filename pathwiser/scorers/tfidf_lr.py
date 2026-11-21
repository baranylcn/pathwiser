"""TF-IDF and lexical pair features into a logistic regression."""

from __future__ import annotations

import time

import numpy as np

from pathwiser.scorers.base import context_text
from pathwiser.scorers.lexical import LexicalOverlap
from pathwiser.scorers.training import (
    candidate_sizes,
    hardware,
    pair_labels_and_weights,
    relative_features,
    split_flat,
)


def _rowdot(a, b) -> np.ndarray:
    return np.asarray(a.multiply(b).sum(axis=1)).ravel()


class TfidfLogReg:
    needs_train = True
    needs_val = False
    name = "tfidf_lr"
    BASE = ("tfidf_cos_message", "tfidf_cos_context", "lexical_message", "lexical_message_context")

    def __init__(self, C: float = 1.0, seed: int = 42) -> None:
        self.C, self.seed = C, seed
        self.params = {
            "C": C,
            "seed": seed,
            "features": "tfidf+lexical similarities, listwise-relative, length",
            "fitted_on": "train (all)",
        }
        self.info: dict = {}
        self.vec = self.scaler = self.model = None

    def _features(self, records: list[dict]) -> np.ndarray:
        sizes = candidate_sizes(records)
        q_msg = self.vec.transform([r["input"]["message"] for r in records])
        q_ctx = self.vec.transform([context_text(r) for r in records])
        uniq = sorted({c for r in records for c in r["input"]["conditions"]})
        row = {c: i for i, c in enumerate(uniq)}
        cmat = self.vec.transform(uniq)
        ex_idx = np.repeat(np.arange(len(records)), sizes)
        cond_idx = np.array([row[c] for r in records for c in r["input"]["conditions"]])
        cond_rows = cmat[cond_idx]
        base = np.column_stack(
            [
                _rowdot(q_msg[ex_idx], cond_rows),
                _rowdot(q_ctx[ex_idx], cond_rows),
                np.concatenate(LexicalOverlap(False).score(records)),
                np.concatenate(LexicalOverlap(True).score(records)),
            ]
        )
        cond_words = np.array([len(c.split()) for r in records for c in r["input"]["conditions"]])
        msg_words = np.repeat([len(r["input"]["message"].split()) for r in records], sizes)
        has_ctx = np.repeat([float(bool(r["input"]["context"])) for r in records], sizes)
        return np.hstack(
            [
                base,
                relative_features(base, sizes),
                np.column_stack([np.log1p(cond_words), np.log1p(msg_words), has_ctx, np.log(np.repeat(sizes, sizes))]),
            ]
        )

    def fit(self, train: list[dict], val: list[dict] | None = None) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        t0 = time.perf_counter()
        corpus = [r["input"]["message"] for r in train] + sorted({c for r in train for c in r["input"]["conditions"]})
        corpus += [t for t in (context_text(r) for r in train) if t]
        self.vec = TfidfVectorizer(
            lowercase=True, strip_accents="unicode", stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        ).fit(corpus)
        x = self._features(train)
        y, w = pair_labels_and_weights(train)
        self.scaler = StandardScaler().fit(x)
        self.model = LogisticRegression(C=self.C, max_iter=2000, random_state=self.seed)
        self.model.fit(self.scaler.transform(x), y, sample_weight=w)
        self.info = {
            "train_examples": len(train),
            "train_pairs": int(len(y)),
            "train_seconds": round(time.perf_counter() - t0, 1),
            "n_features": int(x.shape[1]),
            "hardware": hardware(),
        }

    def score(self, records: list[dict]) -> list[np.ndarray]:
        if self.model is None:
            raise RuntimeError("fit() must be called before score()")
        flat = self.model.decision_function(self.scaler.transform(self._features(records)))
        return split_flat(flat, candidate_sizes(records))
