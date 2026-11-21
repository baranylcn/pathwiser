"""Tests for the specification sampler."""

from __future__ import annotations

import copy
import random
import unittest
from collections import Counter

from pathwiser.synth import spec as SP
from pathwiser.synth import styles as S
from pathwiser.synth import taxonomy as T
from tests.helpers import fake_scenario


class TestApportion(unittest.TestCase):
    def test_counts_follow_the_weights_exactly(self):
        w = {"a": 0.5, "b": 0.3, "c": 0.2}
        for n in (7, 10, 33, 100):
            out = SP.apportion(w, n, random.Random(1))
            self.assertEqual(len(out), n)
            for k, share in w.items():
                self.assertLessEqual(abs(out.count(k) - share * n), 1)

    def test_order_is_shuffled_but_reproducible(self):
        a = SP.apportion({"x": 0.5, "y": 0.5}, 20, random.Random(3))
        self.assertEqual(a, SP.apportion({"x": 0.5, "y": 0.5}, 20, random.Random(3)))
        self.assertNotEqual(a, sorted(a))


class TestSampler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = SP.sample_specs("eval_v2", 80, 7, S.EVAL_PERSONA_WEIGHTS, "t")

    def test_deterministic(self):
        self.assertEqual(self.specs, SP.sample_specs("eval_v2", 80, 7, S.EVAL_PERSONA_WEIGHTS, "t"))
        self.assertNotEqual(self.specs, SP.sample_specs("eval_v2", 80, 8, S.EVAL_PERSONA_WEIGHTS, "t"))

    def test_domains_come_from_the_split_and_are_used_evenly(self):
        allowed = {d["id"] for _, d in T.domains_by_split(T.load(), "eval_v2")}
        counts = Counter(s["domain"] for s in self.specs)
        self.assertTrue(set(counts) <= allowed)
        self.assertEqual(len(counts), 36)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_requested_mix_appears(self):
        info = SP.summary(self.specs)
        self.assertEqual(info["scenarios"], 80)
        self.assertGreaterEqual(info["examples"], 350)
        self.assertAlmostEqual(info["no_match_share"], 0.22, delta=0.04)
        for p in S.HELDOUT_PERSONAS:
            self.assertGreaterEqual(info["personas"][p], 30)
        self.assertEqual(info["archetypes"]["intent_router"], 40)
        self.assertGreaterEqual(min(info["k"]), 2)
        self.assertLessEqual(max(info["k"]), 10)

    def test_training_weights_never_produce_heldout_personas(self):
        tr = SP.sample_specs("train", 60, 3, S.TRAIN_PERSONA_WEIGHTS, "tr")
        used = {e["persona"] for s in tr for e in s["examples"]}
        self.assertFalse(used & set(S.HELDOUT_PERSONAS))
        used_noise = {e["noise"] for s in tr for e in s["examples"]}
        self.assertTrue(used_noise <= set(S.NOISE_SEEN))  # held-out noise is evaluation-only too
        with self.assertRaises(ValueError):
            SP.sample_specs("train", 10, 3, S.EVAL_PERSONA_WEIGHTS, "tr")  # held-out persona weights are refused
        val = SP.sample_specs("synth_val", 20, 3, S.TRAIN_PERSONA_WEIGHTS, "sv")
        self.assertFalse({e["noise"] for s in val for e in s["examples"]} & set(S.NOISE_HELDOUT))
        test = SP.sample_specs("synth_test", 30, 3, S.EVAL_PERSONA_WEIGHTS, "st")
        self.assertTrue({e["persona"] for s in test for e in s["examples"]} & set(S.HELDOUT_PERSONAS))

    def test_structural_rules(self):
        for s in self.specs:
            arch = S.ARCHETYPES[s["archetype"]]
            self.assertIn(s["k"], arch["k"])
            for e in s["examples"]:
                if arch["context_required"]:
                    self.assertNotEqual(e["context"], "none")
                if e["kind"] == "no_match" and e["nm"] == "sibling_removed":
                    self.assertGreaterEqual(s["k"], 3)
                    self.assertNotEqual(e["intent"], e["sibling"])
                if e["kind"] == "route" and e["persona"] == "hinting":
                    self.assertTrue({"indirect", "context_dependent"} & set(e["tags"]))
                if e["kind"] == "no_match":
                    self.assertEqual(e["tags"], [])
            self.assertGreaterEqual(sum(e["kind"] == "route" for e in s["examples"]), 3)

    def test_validation_rejects_wrong_domain_and_bad_plans(self):
        with self.assertRaises(ValueError):
            SP.validate_specs(self.specs, "train")  # eval domains are not train domains
        leak = SP.sample_specs("train", 20, 4, S.TRAIN_PERSONA_WEIGHTS, "lk")
        leak[0]["examples"][0].update(persona="neutral", noise="text_speak")
        with self.assertRaises(ValueError):
            SP.validate_specs(leak, "train")  # held-out noise in a training split
        bad = copy.deepcopy(self.specs)
        bad[0]["examples"][0]["context"] = "nonsense"
        with self.assertRaises(ValueError):
            SP.validate_specs(bad, "eval_v2")
        bad = copy.deepcopy(self.specs)
        bad[1]["k"] = 99
        with self.assertRaises(ValueError):
            SP.validate_specs(bad, "eval_v2")
        with self.assertRaises(ValueError):
            SP.sample_specs("nope", 3, 1, S.EVAL_PERSONA_WEIGHTS, "x")


class TestMatchScenario(unittest.TestCase):
    def setUp(self):
        self.spec = SP.sample_specs("eval_v2", 12, 5, S.EVAL_PERSONA_WEIGHTS, "m")[3]
        self.good = fake_scenario(self.spec)

    def test_conforming_scenario_has_no_problems(self):
        self.assertEqual(SP.match_scenario(self.good, self.spec), [])

    def test_each_kind_of_deviation_is_reported(self):
        def broken(mutate):
            sc = copy.deepcopy(self.good)
            mutate(sc)
            return SP.match_scenario(sc, self.spec)

        self.assertTrue(broken(lambda sc: sc.update(id="other")))
        self.assertTrue(broken(lambda sc: sc["conditions"].pop(next(reversed(sc["conditions"])))))
        self.assertTrue(broken(lambda sc: sc["examples"].pop()))
        route = next(i for i, p in enumerate(self.spec["examples"]) if p["kind"] == "route")
        other = next(c for c in SP.LETTERS[: self.spec["k"]] if c != self.spec["examples"][route]["target"])
        self.assertTrue(broken(lambda sc: sc["examples"][route].update(gold=other)))
        self.assertTrue(
            broken(
                lambda sc: sc["examples"][route].update(
                    ctx=[["u", "x"], ["a", "y"], ["u", "z"]]
                    if self.spec["examples"][route]["context"] == "none"
                    else []
                )
            )
        )
        nm = next((i for i, p in enumerate(self.spec["examples"]) if p["kind"] == "no_match"), None)
        self.assertIsNotNone(nm)
        self.assertTrue(
            broken(
                lambda sc: sc["examples"][nm].update(
                    nm="oos" if self.spec["examples"][nm]["nm"] != "oos" else "sibling_removed"
                )
            )
        )

    def test_sibling_removed_must_remove_exactly_the_planned_condition(self):
        spec = next(
            s
            for s in SP.sample_specs("eval_v2", 40, 9, S.EVAL_PERSONA_WEIGHTS, "q")
            if any(e["kind"] == "no_match" and e["nm"] == "sibling_removed" for e in s["examples"])
        )
        sc = fake_scenario(spec)
        self.assertEqual(SP.match_scenario(sc, spec), [])
        i = next(i for i, e in enumerate(spec["examples"]) if e["kind"] == "no_match" and e["nm"] == "sibling_removed")
        sc["examples"][i]["cands"] = list(SP.LETTERS[: spec["k"]])  # nothing removed
        self.assertTrue(SP.match_scenario(sc, spec))


if __name__ == "__main__":
    unittest.main()
