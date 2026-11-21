"""Tests for the threshold decision rule."""

from __future__ import annotations

import unittest

import numpy as np

from pathwiser.decision.threshold import decide, tune_threshold


def rec(eid, conds, label, message="m", dataset="d", paired_with=None, slices=(), context=()):
    return {
        "example_id": eid,
        "input": {
            "message": message,
            "conditions": list(conds),
            "context": [{"role": "user", "content": c} for c in context],
        },
        "target": {"label": label},
        "meta": {
            "dataset": dataset,
            "is_paired": paired_with is not None,
            "paired_with": paired_with,
            "slices": list(slices),
            "gold_seen_exact": None,
        },
    }


class TestDecision(unittest.TestCase):
    def test_decide_routes_or_abstains(self):
        s = np.array([0.1, 0.7, 0.3])
        self.assertEqual(decide(s, 0.5), 1)
        self.assertEqual(decide(s, 0.8), -1)
        self.assertEqual(decide(np.array([0.5, 0.5]), 0.0), 0)  # ties -> first

    def test_tune_threshold_matches_brute_force(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            n = 60
            tops = np.round(rng.random(n), 1)  # many ties on purpose
            nm = rng.random(n) < 0.3
            rc = (~nm) & (rng.random(n) < 0.6)
            res = tune_threshold(tops, rc, nm)
            best = max(
                ((nm & (tops < t)) | (rc & (tops >= t))).mean() for t in np.concatenate([tops, [tops.max() + 1]])
            )
            self.assertAlmostEqual(res["decision_accuracy"], best)
            tau = res["tau"]
            self.assertAlmostEqual(((nm & (tops < tau)) | (rc & (tops >= tau))).mean(), best)

    def test_tune_threshold_without_signal_abstains_on_everything_when_that_is_best(self):
        res = tune_threshold(np.array([0.5] * 10), np.zeros(10, bool), np.ones(10, bool))
        self.assertEqual(res["abstain_rate"], 1.0)
        self.assertGreater(res["tau"], 0.5)


if __name__ == "__main__":
    unittest.main()
