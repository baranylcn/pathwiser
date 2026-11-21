"""End-to-end tests of the evaluation-set build."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pathwiser.synth import build as B
from tests.helpers import FakeEncoder, make_batch


class TestExportSet(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.specs, self.scenarios = make_batch(8, 5)
        src = root / "src"
        src.mkdir()
        (src / "specs.json").write_text(json.dumps(self.specs), encoding="utf-8")
        (src / "generation.json").write_text(json.dumps({"sampler_version": "1"}), encoding="utf-8")
        (src / "scenarios_00.json").write_text(json.dumps(self.scenarios), encoding="utf-8")
        ref = root / "ref"
        ref.mkdir()
        (ref / "train.jsonl").write_text(
            json.dumps(
                {"input": {"message": "an unrelated reference message", "conditions": ["unrelated condition text"]}}
            )
            + "\n",
            encoding="utf-8",
        )
        B.SETS["toy"] = {
            "src": src,
            "out": root / "out",
            "version": "toy-v2",
            "id_prefix": "t2",
            "dataset": "toy_set",
            "tier": "T",
            "model_split": "test",
            "specs": True,
            "criteria": True,
            "references": [],
            "purpose": "unit test",
            "generator": "none",
        }
        self.addCleanup(B.SETS.pop, "toy", None)
        self.ref, self.out = ref, root / "out"

    def test_end_to_end(self):
        m = B.build_set("toy", log=lambda s: None, encoder=FakeEncoder(), reference_dirs=[self.ref])
        records = [json.loads(line) for line in (self.out / "test.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), m["n_examples"])
        self.assertEqual(m["n_scenarios"], 8)
        r = records[0]
        self.assertEqual(r["meta"]["dataset"], "toy_set")
        self.assertTrue(r["example_id"].startswith("t2_t_"))
        for key in (
            "persona",
            "style_split",
            "noise_planned",
            "noise_applied",
            "clean_message",
            "tags",
            "archetype",
            "condition_style",
            "group",
        ):
            self.assertIn(key, r["meta"])
        self.assertTrue(any(s.startswith("persona:") for s in r["meta"]["slices"]))
        self.assertFalse(m["quality"]["human_audited"])
        self.assertFalse(m["acceptance"]["at_least_75_scenarios"])  # 8 scenarios only
        self.assertFalse(m["acceptance"]["all"])
        self.assertTrue(m["acceptance"]["no_exact_leakage"])
        self.assertEqual(m["quality"]["blind_verified"]["status_counts"], {"unverified": m["n_examples"]})
        self.assertFalse(m["acceptance"]["all_examples_blind_verified"])

    def test_refuses_scenarios_that_do_not_follow_their_specs(self):
        broken = json.loads((self.out.parent / "src" / "scenarios_00.json").read_text(encoding="utf-8"))
        broken[0]["examples"].pop()
        (self.out.parent / "src" / "scenarios_00.json").write_text(json.dumps(broken), encoding="utf-8")
        with self.assertRaises(ValueError):
            B.build_set("toy", log=lambda s: None, encoder=FakeEncoder(), reference_dirs=[self.ref])
        self.assertFalse((self.out / "test.jsonl").exists())

    def test_exact_leak_fails_the_acceptance_check(self):
        leaked = json.loads((self.ref / "train.jsonl").read_text(encoding="utf-8"))
        leaked["input"]["message"] = self.scenarios[0]["examples"][0]["msg"]
        (self.ref / "train.jsonl").write_text(json.dumps(leaked) + "\n", encoding="utf-8")
        m = B.build_set("toy", log=lambda s: None, encoder=FakeEncoder(), reference_dirs=[self.ref])
        self.assertFalse(m["acceptance"]["no_exact_leakage"])


if __name__ == "__main__":
    unittest.main()
