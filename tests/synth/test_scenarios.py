"""Tests for scenario loading, validation and record building."""

from __future__ import annotations

import copy
import json
import unittest

from pathwiser.paths import DATA_DIR
from pathwiser.storage.hashing import record_text_sha
from pathwiser.synth import check
from pathwiser.synth.prepare import apply_drops
from pathwiser.synth.scenarios import ScenarioError, build_records, load_scenarios, validate_scenarios
from pathwiser.synth.verify import attach_verification


def _scenario() -> dict:
    return {
        "id": "s1",
        "domain": "demo",
        "node": {"name": "router", "description": "Decide what the user wants."},
        "conditions": {
            "a": "User wants to book a table",
            "b": "User wants to cancel a booking",
            "c": "User asks about opening hours",
        },
        "examples": [
            {"gold": "a", "ctx": [], "msg": "table for two tonight?"},
            {"gold": "b", "ctx": [["a", "Hi!"]], "msg": "I can't make it anymore"},
            {
                "gold": None,
                "nm": "sibling_removed",
                "intent": "c",
                "cands": ["a", "b"],
                "ctx": [["u", "hello"], ["a", "Hi"]],
                "msg": "when do you close on friday",
            },
            {"gold": None, "nm": "oos", "ctx": [], "msg": "what's the capital of peru"},
        ],
    }


class TestValidate(unittest.TestCase):
    def test_valid_scenario_passes(self):
        validate_scenarios([_scenario()])

    def test_rejects_gold_outside_candidates(self):
        s = _scenario()
        s["examples"][0]["cands"] = ["b", "c"]
        with self.assertRaises(ScenarioError):
            validate_scenarios([s])

    def test_rejects_sibling_removed_with_intent_still_in_candidates(self):
        s = _scenario()
        s["examples"][2]["cands"] = ["a", "b", "c"]
        with self.assertRaises(ScenarioError):
            validate_scenarios([s])

    def test_rejects_no_match_without_kind_and_duplicate_messages(self):
        s = _scenario()
        del s["examples"][3]["nm"]
        with self.assertRaises(ScenarioError):
            validate_scenarios([s])
        s = _scenario()
        s["examples"][1]["msg"] = s["examples"][0]["msg"]
        with self.assertRaises(ScenarioError):
            validate_scenarios([s])

    def test_rejects_duplicate_conditions(self):
        s = _scenario()
        s["conditions"]["c"] = s["conditions"]["a"]
        with self.assertRaises(ScenarioError):
            validate_scenarios([s])


class TestBuild(unittest.TestCase):
    def test_labels_point_at_the_gold_text_and_build_is_deterministic(self):
        s = _scenario()
        recs = build_records([s])
        again = build_records([copy.deepcopy(s)])
        self.assertEqual(recs, again)
        for r, ex in zip(recs, s["examples"]):
            if ex["gold"] is None:
                self.assertEqual(
                    r["target"], {"label": -1, "label_null": len(r["input"]["conditions"]), "has_match": 0}
                )
            else:
                self.assertEqual(r["input"]["conditions"][r["target"]["label"]], s["conditions"][ex["gold"]])

    def test_slices_and_meta(self):
        recs = {r["example_id"]: r for r in build_records([_scenario()])}
        self.assertIn("ctx:none", recs["rl_s1_00"]["meta"]["slices"])
        self.assertIn("ctx:first_turn", recs["rl_s1_01"]["meta"]["slices"])
        self.assertIn("ctx:later_turn", recs["rl_s1_02"]["meta"]["slices"])
        self.assertIn("nm_kind:sibling_removed", recs["rl_s1_02"]["meta"]["slices"])
        self.assertIn("risk:sibling_partial_validity", recs["rl_s1_02"]["meta"]["slices"])
        self.assertEqual(recs["rl_s1_03"]["meta"]["label_risk"], "none")
        self.assertFalse(recs["rl_s1_00"]["meta"]["is_paired"])


class TestVerification(unittest.TestCase):
    def _entry(self, rec: dict, picks: list[str], alt: str | None = None) -> dict:
        return {
            "verified_sha": record_text_sha(rec),
            "passes": {p: {"pick": pick, "alt": alt, "note": ""} for p, pick in zip("AB", picks)},
        }

    def test_statuses_and_agreement(self):
        recs = build_records([_scenario()])
        gold_text = recs[0]["input"]["conditions"][recs[0]["target"]["label"]]
        other = next(c for c in recs[0]["input"]["conditions"] if c != gold_text)
        entries = {
            recs[0]["example_id"]: self._entry(
                recs[0], [gold_text, gold_text], alt=other
            ),  # verified, agrees, alt flagged
            recs[1]["example_id"]: self._entry(recs[1], ["NONE", "NONE"]),  # verified, disagrees (gold is a route)
            recs[2]["example_id"]: {
                "verified_sha": "stale",
                "passes": {"A": {"pick": "NONE", "alt": None, "note": ""}},
            },
        }
        summary = attach_verification(recs, {"examples": entries, "note": "n"})
        v = [r["meta"]["verification"] for r in recs]
        self.assertEqual(
            v[0], {"status": "blind_verified", "passes": 2, "all_agree_with_intended": True, "alt_flagged": True}
        )
        self.assertFalse(v[1]["all_agree_with_intended"])
        self.assertEqual(v[2]["status"], "changed_after_verification")
        self.assertEqual(v[3]["status"], "unverified")
        self.assertEqual(
            summary["status_counts"], {"blind_verified": 2, "changed_after_verification": 1, "unverified": 1}
        )
        self.assertEqual(summary["all_passes_agree_with_intended"], 1)

    def test_no_verification_file_means_everything_unverified(self):
        recs = build_records([_scenario()])
        summary = attach_verification(recs, None)
        self.assertEqual(summary["blind_verified_examples"], 0)
        self.assertTrue(all(r["meta"]["verification"]["status"] == "unverified" for r in recs))


class TestAuthoredSets(unittest.TestCase):
    """The committed scenario files must stay valid; this guards edits made by hand."""

    def test_realistic_v1_scenarios_are_valid(self):
        scenarios, _ = load_scenarios(DATA_DIR / "curated" / "realistic_v1")
        validate_scenarios(scenarios)
        self.assertGreater(len(build_records(scenarios)), 300)

    def test_realistic_v2_scenarios_follow_their_specs_after_the_adjudicated_drops(self):
        src = DATA_DIR / "curated" / "realistic_v2"
        scenarios, _ = load_scenarios(src)
        specs = json.loads((src / "specs.json").read_text(encoding="utf-8"))
        dropped = json.loads((src / "dropped.json").read_text(encoding="utf-8"))
        scenarios, specs = apply_drops(scenarios, specs, dropped)
        self.assertEqual(check.validate_generated(scenarios, specs), [])
        self.assertEqual(sum(len(s["examples"]) for s in scenarios), 434)


if __name__ == "__main__":
    unittest.main()
