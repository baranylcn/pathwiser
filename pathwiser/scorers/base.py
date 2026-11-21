"""The scorer interface."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Scorer(Protocol):
    name: str

    def fit(self, train: list[dict], val: list[dict] | None = None) -> None: ...

    def score(self, records: list[dict]) -> list[np.ndarray]: ...


def context_text(record: dict) -> str:
    return " ".join(t["content"] for t in record["input"]["context"])


def split_scores(flat: np.ndarray, records: list[dict]) -> list[np.ndarray]:
    out, i = [], 0
    for r in records:
        k = len(r["input"]["conditions"])
        out.append(flat[i : i + k])
        i += k
    return out
