"""Learned abstention: probability that routing to the best candidate is correct, from listwise score features."""

from __future__ import annotations

import math

import numpy as np
from sklearn.linear_model import LogisticRegression

from pathwiser.decision.scaling import EPS, Standardizer
from pathwiser.decision.threshold import tune_threshold

RELATIVE = ("margin", "gap_mean", "z_top", "z_margin", "std", "log_k")


FULL = ("top", "second") + RELATIVE


FEATURE_SETS = {"relative": RELATIVE, "full": FULL}


def abstention_features(scores: list[np.ndarray]) -> np.ndarray:
    """Listwise features of each score vector (rows follow ``scores``; every vector has at least two candidates)."""
    rows = []
    for s in scores:
        s = np.asarray(s, dtype=float)
        if len(s) < 2:
            raise ValueError("abstention features need at least two candidates")
        order = np.sort(s)[::-1]
        top, second = order[0], order[1]
        mean, std = s.mean(), s.std()
        others = (s.sum() - top) / (len(s) - 1)
        rows.append(
            [
                top,
                second,
                top - second,
                top - others,
                (top - mean) / (std + EPS),
                (top - second) / (std + EPS),
                std,
                math.log(len(s)),
            ]
        )
    cols = ("top", "second", "margin", "gap_mean", "z_top", "z_margin", "std", "log_k")
    arr = np.array(rows)
    return arr[:, [cols.index(c) for c in FULL]]


def select_features(X: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    return X[:, [FULL.index(n) for n in names]]


class AbstentionModel:
    """P(routing to the best candidate is correct | listwise score features), plus the route/no_match cut."""

    def __init__(self, feature_set: str = "relative", C: float = 1.0) -> None:
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {sorted(FEATURE_SETS)}")
        self.feature_set, self.C = feature_set, C
        self.names = FEATURE_SETS[feature_set]

    def fit(self, scores: list[np.ndarray], route_correct: np.ndarray, is_no_match: np.ndarray) -> AbstentionModel:
        X = select_features(abstention_features(scores), self.names)
        self.scaler = Standardizer().fit(X)
        self.lr = LogisticRegression(C=self.C, max_iter=2000).fit(
            self.scaler.transform(X), np.asarray(route_correct, dtype=int)
        )
        p = self.proba(scores)
        self.threshold = tune_threshold(p, route_correct, is_no_match)["tau"]
        return self

    def proba(self, scores: list[np.ndarray]) -> np.ndarray:
        X = select_features(abstention_features(scores), self.names)
        return self.lr.predict_proba(self.scaler.transform(X))[:, 1]
