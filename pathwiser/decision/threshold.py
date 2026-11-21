"""Threshold decision rule: route to the best candidate, or abstain (``no_match``) when its score is below a threshold."""

from __future__ import annotations

import math

import numpy as np

NO_MATCH = -1


def decide(scores: np.ndarray, tau: float) -> int:
    """Index of the best candidate (first one on ties), or NO_MATCH if the best score is below ``tau``."""
    best = int(np.argmax(scores))
    return best if scores[best] >= tau else NO_MATCH


def tune_threshold(tops: np.ndarray, route_correct: np.ndarray, is_no_match: np.ndarray) -> dict:
    """Threshold that maximises decision accuracy.

    ``tops``: best score per example; ``route_correct``: the best candidate is the gold one; ``is_no_match``: no_match examples.
    An example is decided correctly when it is routed to the gold condition, or when it is a no_match example and the rule abstains.
    Every distinct split point is evaluated exactly; ties prefer fewer abstentions.
    """
    tops = np.asarray(tops, dtype=float)
    n = len(tops)
    if n == 0:
        raise ValueError("cannot tune a threshold without examples")
    order = np.argsort(tops, kind="stable")
    t = tops[order]
    nm = np.asarray(is_no_match, dtype=bool)[order]
    rc = np.asarray(route_correct, dtype=bool)[order]
    cum_nm = np.concatenate([[0], np.cumsum(nm)])  # abstaining on the j lowest scores
    cum_rc = np.concatenate([[0], np.cumsum(rc)])
    correct = cum_nm + (cum_rc[-1] - cum_rc)  # correct decisions for j = 0..n
    valid = np.ones(n + 1, dtype=bool)
    valid[1:n] = t[:-1] != t[1:]  # only split between distinct score values
    j = int(np.argmax(np.where(valid, correct, -1)))
    if j == 0:
        tau = float(t[0])  # abstain on nothing
    elif j == n:
        tau = math.nextafter(float(t[-1]), math.inf)  # abstain on everything
    else:
        tau = float((t[j - 1] + t[j]) / 2)
    return {"tau": tau, "decision_accuracy": float(correct[j] / n), "abstain_rate": float(j / n)}
