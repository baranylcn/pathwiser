"""Feature standardisation shared by the learned decision components."""

from __future__ import annotations

import numpy as np

EPS = 1e-6


class Standardizer:
    def fit(self, X: np.ndarray) -> Standardizer:
        self.mean, self.scale = X.mean(axis=0), X.std(axis=0)
        self.scale = np.where(self.scale < EPS, 1.0, self.scale)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.scale
