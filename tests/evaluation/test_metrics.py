"""Tests for the evaluation metrics."""

from __future__ import annotations

import unittest

import numpy as np

from pathwiser.evaluation.metrics import evaluate_split, predictions, summarize, tune


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


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.records = [
            rec("a", ["x", "y", "z"], 0, slices=["cond:seen"]),
            rec("b", ["x", "y", "z"], 2, slices=["cond:unseen"]),
            rec("c", ["x", "y"], -1),  # no_match, low score -> correct abstain
            rec("d", ["x", "y"], -1),  # no_match, high score -> false route
        ]
        self.scores = [np.array([0.9, 0.2, 0.1]), np.array([0.8, 0.3, 0.5]), np.array([0.1, 0.2]), np.array([0.7, 0.6])]

    def test_ranking_and_decision_numbers(self):
        m = summarize(predictions(self.records, self.scores), 0.4)
        self.assertEqual((m["n"], m["n_route"], m["n_no_match"]), (4, 2, 2))
        self.assertEqual(m["top1"], 0.5)  # a right, b wrong (gold ranked 2nd)
        self.assertAlmostEqual(m["mrr"], 0.75)
        self.assertEqual(m["top3"], 1.0)
        self.assertEqual(m["decision_accuracy"], 0.5)  # a ok, c ok; b wrong route, d false route
        self.assertEqual(m["no_match_recall"], 0.5)
        self.assertEqual(m["no_match_precision"], 1.0)
        self.assertEqual(m["false_routing_rate"], 0.5)  # b (wrong) + d (routed, nothing applies)
        self.assertEqual(m["abstain_rate"], 0.25)

    def test_ties_broken_by_position_and_nan_rejected(self):
        p = predictions([rec("t", ["x", "y"], 1)], [np.array([0.5, 0.5])])[0]
        self.assertEqual((p.arg, p.rank), (0, 2))
        with self.assertRaises(ValueError):
            predictions([rec("t", ["x", "y"], 1)], [np.array([0.5, np.nan])])
        with self.assertRaises(ValueError):
            predictions([rec("t", ["x", "y"], 1)], [np.array([0.5])])

    def test_main_and_paired_are_reported_separately_and_slices_grouped(self):
        recs = self.records + [rec("a2", ["x", "y", "z"], 0, paired_with="a", slices=["cond:seen"])]
        scores = self.scores + [np.array([0.2, 0.9, 0.1])]
        rep = evaluate_split(recs, scores, 0.4)
        self.assertEqual(rep["overall"]["n"], 4)  # twin not counted
        self.assertEqual(rep["paired_twins"]["n"], 1)
        self.assertEqual(rep["paraphrase_consistency"]["same_selected_candidate"], 0.0)
        self.assertEqual(rep["slices"]["cond:seen"]["n"], 1)
        self.assertIsNone(rep["slices"]["cond:seen"]["no_match_recall"])  # undefined: no no_match examples there

    def test_tune_uses_only_what_it_is_given(self):
        preds = predictions(self.records, self.scores)
        res = tune(preds)
        self.assertGreaterEqual(res["decision_accuracy"], summarize(preds, 0.4)["decision_accuracy"])


if __name__ == "__main__":
    unittest.main()
