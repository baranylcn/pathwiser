"""Score fusion: one score per candidate from the scores of several models."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from pathwiser.decision.scaling import Standardizer


class Fusion:
    """Pointwise logistic regression over candidate-level features of several models."""

    def __init__(self, models: tuple[str, ...], C: float = 1.0) -> None:
        self.models, self.C = tuple(models), C

    def _features(self, by_model: dict[str, list[np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
        cols, sizes = [], np.array([len(s) for s in by_model[self.models[0]]])
        for m in self.models:
            if [len(s) for s in by_model[m]] != list(sizes):
                raise ValueError(f"{m}: candidate counts differ from {self.models[0]}")
            s = by_model[m]
            flat = np.concatenate(s)
            best = np.concatenate([np.full(len(v), v.max()) for v in s])
            mean = np.concatenate([np.full(len(v), v.mean()) for v in s])
            cols += [flat, flat - best, flat - mean]
        return np.column_stack(cols), sizes

    def fit(self, by_model: dict[str, list[np.ndarray]], labels: list[int]) -> Fusion:
        X, sizes = self._features(by_model)
        y = np.zeros(len(X))
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        for st, lab in zip(starts, labels):
            if lab >= 0:
                y[st + lab] = 1.0
        w = np.repeat(1.0 / sizes, sizes)  # every example weighs the same
        self.scaler = Standardizer().fit(X)
        self.lr = LogisticRegression(C=self.C, max_iter=2000).fit(
            self.scaler.transform(X), y, sample_weight=w * (len(X) / w.sum())
        )
        return self

    def transform(self, by_model: dict[str, list[np.ndarray]]) -> list[np.ndarray]:
        X, sizes = self._features(by_model)
        z = self.lr.decision_function(self.scaler.transform(X))
        return list(np.split(z, np.cumsum(sizes)[:-1]))
