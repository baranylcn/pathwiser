"""Tests for the planned noise and the adjudicated drops."""

from __future__ import annotations

import unittest

from pathwiser.synth import prepare as P
from pathwiser.synth import styles as S
from tests.helpers import make_batch


class TestPrepare(unittest.TestCase):
    def setUp(self):
        self.specs, self.scenarios = make_batch()
        self.prepared = P.prepare(
            self.scenarios, self.specs, id_prefix="t2", dataset="toy_set", tier="T", model_split="test"
        )

    def test_noise_follows_the_plan_and_keeps_the_clean_text(self):
        spec_by_id = {s["id"]: s for s in self.specs}
        changed = 0
        for sc in self.prepared:
            for plan, ex in zip(spec_by_id[sc["id"]]["examples"], sc["examples"]):
                meta = ex["meta_extra"]
                self.assertEqual(meta["noise_planned"], plan["noise"])
                if plan["noise"] == "none":
                    self.assertEqual((ex["msg"], meta["noise_applied"]), (meta["clean_message"], "none"))
                if meta["noise_applied"] != "none":
                    changed += 1
                    self.assertNotEqual(ex["msg"], meta["clean_message"])
                self.assertIn(f"noise:{meta['noise_applied']}", ex["slices_extra"])
                self.assertIn(f"persona:{plan['persona']}", ex["slices_extra"])
        self.assertGreater(changed, 10)

    def test_only_user_turns_are_damaged(self):
        for sc0, sc in zip(self.scenarios, self.prepared):
            for e0, e in zip(sc0["examples"], sc["examples"]):
                for (r0, t0), (r, t) in zip(e0["ctx"], e["ctx"]):
                    self.assertEqual(r0, r)
                    if r == "a":
                        self.assertEqual(t0, t)

    def test_prepare_is_deterministic_and_does_not_mutate_the_input(self):
        again = P.prepare(self.scenarios, self.specs, id_prefix="t2", dataset="toy_set", tier="T", model_split="test")
        self.assertEqual(self.prepared, again)
        self.assertTrue(all("meta_extra" not in ex for sc in self.scenarios for ex in sc["examples"]))

    def test_heldout_style_flag(self):
        spec_by_id = {s["id"]: s for s in self.specs}
        for sc in self.prepared:
            for plan, ex in zip(spec_by_id[sc["id"]]["examples"], sc["examples"]):
                m = ex["meta_extra"]
                expect = plan["persona"] in S.HELDOUT_PERSONAS or m["noise_applied"] in S.NOISE_HELDOUT
                self.assertEqual(m["style_split"], "heldout" if expect else "seen")


class TestDrops(unittest.TestCase):
    def test_dropping_keeps_the_other_examples_ids_and_text(self):
        specs, scenarios = make_batch(6, 9)
        sid = specs[0]["id"]
        full = P.prepare(scenarios, specs, id_prefix="t2", dataset="d", tier="T", model_split="test")
        sc2, sp2 = P.apply_drops(scenarios, specs, [{"scenario": sid, "i": 1, "reason": "contested"}])
        self.assertEqual(len(sp2[0]["examples"]), len(specs[0]["examples"]) - 1)
        self.assertEqual(len(sc2[0]["examples"]), len(scenarios[0]["examples"]) - 1)
        self.assertEqual(len(scenarios[0]["examples"]), len(specs[0]["examples"]))  # inputs untouched
        part = P.prepare(sc2, sp2, id_prefix="t2", dataset="d", tier="T", model_split="test")
        from pathwiser.synth.scenarios import build_records

        ids_full = [r["example_id"] for r in build_records(full)]
        ids_part = [r["example_id"] for r in build_records(part)]
        self.assertNotIn(f"t2_{sid}_01", ids_part)
        self.assertEqual(set(ids_full) - set(ids_part), {f"t2_{sid}_01"})  # nothing else changed id
        texts_full = {r["example_id"]: r["input"]["message"] for r in build_records(full)}
        for r in build_records(part):
            self.assertEqual(r["input"]["message"], texts_full[r["example_id"]])  # same noise, same text

    def test_dropping_a_missing_example_is_an_error(self):
        specs, scenarios = make_batch(4, 9)
        with self.assertRaises(ValueError):
            P.apply_drops(scenarios, specs, [{"scenario": specs[0]["id"], "i": 99, "reason": "x"}])

    def test_a_scenario_with_a_missing_example_is_not_masked_by_dropping(self):
        specs, scenarios = make_batch(4, 9)
        scenarios[1]["examples"].pop()
        sc, sp = P.apply_drops(scenarios, specs, [{"scenario": specs[0]["id"], "i": 0, "reason": "x"}])
        self.assertEqual(
            len(sp[1]["examples"]), len(specs[1]["examples"])
        )  # spec of the broken scenario is NOT truncated
        from pathwiser.synth import check as C

        self.assertTrue(C.validate_generated(sc, sp))  # so the mismatch is still reported
        with self.assertRaises(ValueError):
            P.apply_drops(scenarios, specs, [{"scenario": specs[1]["id"], "i": 0, "reason": "x"}])


if __name__ == "__main__":
    unittest.main()
