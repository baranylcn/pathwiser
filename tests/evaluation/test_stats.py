"""Tests for bootstrap intervals and calibration helpers."""

from __future__ import annotations

import unittest

import numpy as np

from pathwiser.evaluation.metrics import predictions
from pathwiser.evaluation.stats import brier, cluster_bootstrap, ece
from tests.helpers import make_world


class TestStats(unittest.TestCase):
    def test_ece_and_brier(self):
        p = np.array([0.9, 0.9, 0.1, 0.1])
        self.assertAlmostEqual(ece(p, np.array([1, 1, 0, 0])), 0.1)
        self.assertAlmostEqual(ece(np.array([0.5, 0.5]), np.array([1, 0])), 0.0)
        self.assertAlmostEqual(brier(np.array([1.0, 0.0]), np.array([1, 0])), 0.0)
        self.assertAlmostEqual(brier(np.array([0.5, 0.5]), np.array([1, 0])), 0.25)

    def test_cluster_bootstrap_intervals_and_paired_difference(self):
        records, scores = make_world(n_per=120)
        clusters = [r["example_id"].split("_")[0] + str(int(r["example_id"].split("_")[1]) // 6) for r in records]
        good = predictions(records, scores["a"])
        poor = predictions(records, scores["b"])
        no_abstain = np.zeros(len(records), dtype=bool)
        out = cluster_bootstrap(
            {"a": (good, no_abstain), "b": (poor, no_abstain)}, clusters, reference="b", n_boot=300, seed=1
        )
        a = out["systems"]["a"]["top1"]
        self.assertTrue(a["lo"] <= a["value"] <= a["hi"])
        self.assertGreater(out["vs_reference"]["a"]["top1"]["diff"], 0.1)
        self.assertGreater(out["vs_reference"]["a"]["top1"]["lo"], 0.0)  # clearly better, interval excludes zero
        again = cluster_bootstrap(
            {"a": (good, no_abstain), "b": (poor, no_abstain)}, clusters, reference="b", n_boot=300, seed=1
        )
        self.assertEqual(out, again)  # seeded -> reproducible
        with self.assertRaises(ValueError):
            cluster_bootstrap({"a": (good, no_abstain)}, clusters[:-1])


if __name__ == "__main__":
    unittest.main()
