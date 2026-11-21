"""Tests for fusion, abstention, decision systems and leave-one-dataset-out selection."""

from __future__ import annotations

import unittest

import numpy as np

from pathwiser.decision.abstention import abstention_features
from pathwiser.decision.fusion import Fusion
from pathwiser.decision.selection import leave_one_dataset_out, select_system
from pathwiser.decision.system import Pipeline, System, align
from pathwiser.evaluation.metrics import predictions, summarize, summarize_decisions
from tests.helpers import make_world


class TestFeatures(unittest.TestCase):
    def test_features_of_a_known_vector(self):
        X = abstention_features([np.array([1.0, 3.0, 2.0, 0.0])])
        row = dict(zip(("top", "second", "margin", "gap_mean", "z_top", "z_margin", "std", "log_k"), X[0]))
        self.assertAlmostEqual(row["top"], 3.0)
        self.assertAlmostEqual(row["margin"], 1.0)
        self.assertAlmostEqual(row["gap_mean"], 3.0 - 1.0)  # mean of the other three
        self.assertAlmostEqual(row["log_k"], np.log(4))
        with self.assertRaises(ValueError):
            abstention_features([np.array([1.0])])

    def test_relative_features_ignore_a_constant_shift(self):
        s = np.array([0.2, 1.4, 0.7])
        a, b = abstention_features([s]), abstention_features([s + 10.0])
        names = ("top", "second", "margin", "gap_mean", "z_top", "z_margin", "std", "log_k")
        for n in ("margin", "gap_mean", "z_top", "z_margin", "std", "log_k"):
            self.assertAlmostEqual(a[0, names.index(n)], b[0, names.index(n)])


class TestSystem(unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(ValueError):
            System(())
        with self.assertRaises(ValueError):
            System(("a", "b"))  # several models need fusion
        with self.assertRaises(ValueError):
            System(("a",), abstention="magic")
        self.assertLess(System(("a",)).complexity, System(("a",), "relative").complexity)
        self.assertLess(System(("a",), "full").complexity, System(("a", "b"), "tau", True).complexity)
        self.assertEqual(System(("a", "b"), "relative", True).label, "a+b|fused|relative")


class TestFusionAndPipeline(unittest.TestCase):
    def setUp(self):
        self.records, self.scores = make_world()

    def test_fusion_ranks_at_least_as_well_as_the_best_single_model(self):
        labels = [r["target"]["label"] for r in self.records]
        fused = Fusion(("a", "b")).fit(self.scores, labels).transform(self.scores)

        def top1(sc):
            return np.mean([int(np.argmax(s)) == y for s, y in zip(sc, labels) if y >= 0])

        self.assertGreaterEqual(top1(fused), top1(self.scores["a"]) - 0.01)
        self.assertGreater(top1(fused), top1(self.scores["b"]))

    def test_fusion_is_deterministic_and_checks_shapes(self):
        labels = [r["target"]["label"] for r in self.records]
        x = Fusion(("a", "b")).fit(self.scores, labels).transform(self.scores)
        y = Fusion(("a", "b")).fit(self.scores, labels).transform(self.scores)
        np.testing.assert_allclose(np.concatenate(x), np.concatenate(y))
        bad = {"a": self.scores["a"], "b": [s[:-1] for s in self.scores["b"]]}
        with self.assertRaises(ValueError):
            Fusion(("a", "b")).fit(bad, labels)

    def test_pipeline_refuses_paired_examples(self):
        recs = [dict(r, meta=dict(r["meta"], is_paired=True)) if i == 0 else r for i, r in enumerate(self.records)]
        with self.assertRaises(ValueError):
            Pipeline(System(("a",))).fit(recs, self.scores)

    def test_threshold_pipeline_matches_the_plain_threshold_rule(self):
        pipe = Pipeline(System(("a",))).fit(self.records, self.scores)
        dec = pipe.decide(self.records, self.scores)
        self.assertIsNone(dec.probability)
        from pathwiser.evaluation.metrics import tune

        tau = tune(predictions(self.records, self.scores["a"]))["tau"]
        self.assertEqual(pipe.tau, tau)
        self.assertEqual(summarize_decisions(dec.preds, dec.abstain), summarize(dec.preds, tau))

    def test_calibrated_pipeline_returns_probabilities(self):
        pipe = Pipeline(System(("a",), "relative")).fit(self.records, self.scores)
        dec = pipe.decide(self.records, self.scores)
        self.assertEqual(len(dec.probability), len(self.records))
        self.assertTrue(((dec.probability >= 0) & (dec.probability <= 1)).all())
        self.assertEqual(list(dec.abstain), list(dec.probability < pipe.abstainer.threshold))


class TestLeaveOneDatasetOut(unittest.TestCase):
    def test_relative_features_survive_a_domain_shift_that_breaks_a_global_threshold(self):
        records, scores = make_world()
        tau = leave_one_dataset_out(System(("a",)), records, scores)
        rel = leave_one_dataset_out(System(("a",), "relative"), records, scores)
        self.assertGreater(rel["macro_decision_accuracy"], tau["macro_decision_accuracy"] + 0.10)
        self.assertEqual(sorted(rel["per_dataset"]), ["ds0", "ds1", "ds2"])
        self.assertGreater(rel["se"], 0)

    def test_held_out_dataset_is_never_used_for_fitting(self):
        records, scores = make_world()
        seen = []

        class Spy(Pipeline):
            def fit(self, recs, by_model):
                seen.append({r["meta"]["dataset"] for r in recs})
                return super().fit(recs, by_model)

        import pathwiser.decision.selection as policy

        original = policy.Pipeline
        policy.Pipeline = Spy
        try:
            leave_one_dataset_out(System(("a",)), records, scores)
        finally:
            policy.Pipeline = original
        self.assertEqual(sorted(len(s) for s in seen), [2, 2, 2])
        self.assertEqual(sorted(tuple(sorted(s)) for s in seen), [("ds0", "ds1"), ("ds0", "ds2"), ("ds1", "ds2")])

    def test_select_system_prefers_the_simpler_one_within_one_se(self):
        res = [
            {"system": "complex", "complexity": 5, "macro_decision_accuracy": 0.80, "se": 0.02},
            {"system": "simple", "complexity": 0, "macro_decision_accuracy": 0.79, "se": 0.02},
            {"system": "bad", "complexity": 0, "macro_decision_accuracy": 0.60, "se": 0.02},
        ]
        out = select_system(res)
        self.assertEqual((out["chosen"], out["best"]), ("simple", "complex"))
        res[1]["macro_decision_accuracy"] = 0.70
        self.assertEqual(select_system(res)["chosen"], "complex")

    def test_generalisation_guard_removes_a_near_tie_that_fails_on_unseen_services(self):
        res = [
            {
                "system": "simple_but_bad_on_unseen",
                "complexity": 2,
                "macro_decision_accuracy": 0.811,
                "se": 0.017,
                "slice_top1": {"svc:unseen": 0.855},
            },
            {
                "system": "fused",
                "complexity": 3,
                "macro_decision_accuracy": 0.820,
                "se": 0.015,
                "slice_top1": {"svc:unseen": 0.892},
            },
            {
                "system": "best_fused",
                "complexity": 4,
                "macro_decision_accuracy": 0.827,
                "se": 0.017,
                "slice_top1": {"svc:unseen": 0.893},
            },
            {
                "system": "reference",
                "complexity": 0,
                "macro_decision_accuracy": 0.800,
                "se": 0.017,
                "slice_top1": {"svc:unseen": 0.902},
            },
        ]
        out = select_system(res)
        self.assertEqual(out["chosen"], "fused")
        self.assertEqual(out["choice_without_guard"], "simple_but_bad_on_unseen")
        self.assertEqual(out["excluded_by_guard"], ["simple_but_bad_on_unseen"])
        self.assertAlmostEqual(out["guard"]["best_value"], 0.902)

    def test_guard_is_inactive_without_slice_information_or_when_it_would_remove_everything(self):
        plain = [
            {"system": "complex", "complexity": 5, "macro_decision_accuracy": 0.80, "se": 0.02},
            {"system": "simple", "complexity": 0, "macro_decision_accuracy": 0.79, "se": 0.02},
        ]
        out = select_system(plain)
        self.assertEqual((out["chosen"], out["excluded_by_guard"]), ("simple", []))
        allbad = [
            {
                "system": "a",
                "complexity": 1,
                "macro_decision_accuracy": 0.80,
                "se": 0.02,
                "slice_top1": {"svc:unseen": 0.80},
            },
            {
                "system": "b",
                "complexity": 0,
                "macro_decision_accuracy": 0.79,
                "se": 0.02,
                "slice_top1": {"svc:unseen": 0.70},
            },
            {
                "system": "outside",
                "complexity": 0,
                "macro_decision_accuracy": 0.50,
                "se": 0.02,
                "slice_top1": {"svc:unseen": 0.95},
            },
        ]
        out = select_system(allbad)  # guard (0.95 - 0.02) would drop both near systems: not applied
        self.assertEqual(out["chosen"], "b")
        self.assertEqual(out["excluded_by_guard"], [])

    def test_lodo_reports_slice_top1_when_slices_exist(self):
        records, scores = make_world(n_per=60)
        for i, r in enumerate(records):
            r["meta"]["slices"] = ["svc:unseen"] if i % 3 == 0 else ["svc:seen"]
        r = leave_one_dataset_out(System(("a",)), records, scores)
        self.assertIn("svc:unseen", r["slice_top1"])
        self.assertTrue(0 <= r["slice_top1"]["svc:unseen"] <= 1)
        self.assertNotIn("cond:unseen", r["slice_top1"])

    def test_align_keeps_models_in_sync(self):
        records, scores = make_world(n_per=10)
        keep = np.array([i % 2 == 0 for i in range(len(records))])
        r, s = align(records, scores, keep)
        self.assertEqual(len(r), keep.sum())
        self.assertTrue(all(len(v) == len(r) for v in s.values()))


if __name__ == "__main__":
    unittest.main()
