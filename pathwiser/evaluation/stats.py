"""Uncertainty and calibration helpers.

* ``cluster_bootstrap``: confidence intervals that respect correlation inside a cluster (all examples of one scenario / dialogue
  share their candidates and wording). Resampling single examples would give intervals that are too narrow. Differences between
  systems are paired: every system is evaluated on the same resampled clusters.
* ``ece`` / ``brier``: calibration of a predicted probability against a binary outcome.
"""

from __future__ import annotations

import numpy as np

from pathwiser.evaluation.metrics import Prediction

METRICS = ("top1", "decision_accuracy", "false_routing_rate", "abstain_rate", "no_match_recall", "no_match_precision")


def terms(preds: list[Prediction], abstain: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-example numerator / denominator of each metric (a ratio of sums, so it can be bootstrapped by cluster)."""
    abstain = np.asarray(abstain, dtype=bool)
    label = np.array([p.label for p in preds])
    arg = np.array([p.arg for p in preds])
    rank = np.array([p.rank if p.rank is not None else 0 for p in preds])
    route, nm = label >= 0, label < 0
    ok = route & ~abstain & (arg == label)
    wrong = (route & ~abstain & (arg != label)) | (nm & ~abstain)
    one = np.ones(len(preds))
    return {
        "top1": ((route & (rank == 1)).astype(float), route.astype(float)),
        "decision_accuracy": ((ok | (nm & abstain)).astype(float), one),
        "false_routing_rate": (wrong.astype(float), one),
        "abstain_rate": (abstain.astype(float), one),
        "no_match_recall": ((nm & abstain).astype(float), nm.astype(float)),
        "no_match_precision": ((nm & abstain).astype(float), abstain.astype(float)),
    }


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def cluster_bootstrap(
    systems: dict[str, tuple[list[Prediction], np.ndarray]],
    clusters: list[str],
    reference: str | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> dict:
    """Point estimates and percentile intervals per system and metric; paired differences against ``reference`` if given.

    ``systems``: name -> (predictions, abstain mask), all aligned with ``clusters`` (one cluster id per example).
    """
    ids = sorted(set(clusters))
    index = {c: i for i, c in enumerate(ids)}
    cid = np.array([index[c] for c in clusters])
    C = len(ids)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(C, np.full(C, 1.0 / C), size=n_boot).astype(
        float
    )  # (n_boot, C): how often each cluster is drawn
    lo_q, hi_q = (1 - level) / 2 * 100, (1 + level) / 2 * 100

    def pct(x: np.ndarray, q: float) -> float | None:
        return None if np.isnan(x).all() else float(np.nanpercentile(x, q))

    def r4(x: float | None) -> float | None:
        return None if x is None or np.isnan(x) else round(float(x), 4)

    per: dict[str, dict] = {}
    for name, (preds, abstain) in systems.items():
        if len(preds) != len(cid):
            raise ValueError(f"{name}: {len(preds)} predictions for {len(cid)} clusters entries")
        per[name] = {}
        for metric, (num, den) in terms(preds, abstain).items():
            cn, cd = np.zeros(C), np.zeros(C)
            np.add.at(cn, cid, num)
            np.add.at(cd, cid, den)
            boot = _ratio(counts @ cn, counts @ cd)
            per[name][metric] = {
                "value": float(_ratio(num.sum(), den.sum())),
                "lo": pct(boot, lo_q),
                "hi": pct(boot, hi_q),
                "_boot": boot,
            }
    out: dict = {"n_clusters": C, "n_boot": n_boot, "level": level, "systems": {}, "vs_reference": {}}
    for name, metrics in per.items():
        out["systems"][name] = {m: {k: r4(v) for k, v in d.items() if k != "_boot"} for m, d in metrics.items()}
    if reference is not None:
        for name, metrics in per.items():
            if name == reference:
                continue
            out["vs_reference"][name] = {}
            for m, d in metrics.items():
                diff = d["_boot"] - per[reference][m]["_boot"]
                out["vs_reference"][name][m] = {
                    "diff": r4(d["value"] - per[reference][m]["value"]),
                    "lo": r4(pct(diff, lo_q)),
                    "hi": r4(pct(diff, hi_q)),
                }
    return out


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins on [0, 1]."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)
