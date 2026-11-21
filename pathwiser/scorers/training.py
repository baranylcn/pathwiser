"""Shared helpers of the trained baselines: example sampling, listwise (relative) features, hardware record."""

from __future__ import annotations

import os
import platform

import numpy as np


def sample_examples(records: list[dict], n: int | None, seed: int) -> list[dict]:
    """Deterministic subset of ``n`` examples, drawn without replacement with probability proportional to ``weight``.

    The training weights already balance the datasets, so sampling by weight keeps that balance in a smaller
    training set (used when a baseline cannot afford the full split). ``n=None`` or ``n >= len`` returns everything.
    """
    if n is None or n >= len(records):
        return list(records)
    w = np.array([r["weight"] for r in records], dtype=float)
    idx = np.random.default_rng(seed).choice(len(records), size=n, replace=False, p=w / w.sum())
    return [records[i] for i in sorted(idx)]


def relative_features(base: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Listwise context for pointwise models.

    ``base`` is (pairs, d), rows grouped by example (``sizes`` = candidates per example). For every column returns
    two more columns: the value minus the best OTHER candidate's value, and the within-example z-score.
    """
    diff = np.empty_like(base)
    z = np.empty_like(base)
    start = 0
    for k in sizes:
        blk = base[start : start + k]
        top = np.partition(blk, k - 2, axis=0)[k - 2 :]  # two largest per column (k >= 2)
        best, second = top[1], top[0]
        diff[start : start + k] = blk - np.where(blk >= best, second, best)
        z[start : start + k] = (blk - blk.mean(axis=0)) / (blk.std(axis=0) + 1e-6)
        start += k
    return np.hstack([diff, z])


def candidate_sizes(records: list[dict]) -> np.ndarray:
    return np.array([len(r["input"]["conditions"]) for r in records])


def pair_labels_and_weights(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """One row per (example, candidate): is_gold (0 for every candidate of a no_match example) and the example weight."""
    y, w = [], []
    for r in records:
        k, label = len(r["input"]["conditions"]), r["target"]["label"]
        y.extend(int(j == label) for j in range(k))
        w.extend([r["weight"]] * k)
    return np.array(y), np.array(w, dtype=float)


def split_flat(flat: np.ndarray, sizes: np.ndarray) -> list[np.ndarray]:
    return list(np.split(np.asarray(flat), np.cumsum(sizes)[:-1]))


def hardware() -> dict:
    info = {"cpu_count": os.cpu_count(), "processor": platform.processor(), "platform": platform.platform()}
    try:
        import torch

        info["torch_threads"] = torch.get_num_threads()
        info["cuda"] = bool(torch.cuda.is_available())
    except ImportError:
        pass
    return info
