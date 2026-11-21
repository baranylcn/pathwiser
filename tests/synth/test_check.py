"""Tests for the checks of generated scenarios."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from pathwiser.synth import check as C
from pathwiser.synth import spec as SP
from pathwiser.synth import styles as S
from tests.helpers import good_scenario


class TestCheck(unittest.TestCase):
    def setUp(self):
        self.specs = SP.sample_specs("eval_v2", 10, 11, S.EVAL_PERSONA_WEIGHTS, "c")
        self.scenarios = [good_scenario(s) for s in self.specs]

    def test_conforming_batch_passes(self):
        self.assertEqual(C.validate_generated(self.scenarios, self.specs), [])

    def test_missing_and_extra_scenarios(self):
        self.assertTrue(any("no scenario written" in p for p in C.validate_generated(self.scenarios[1:], self.specs)))
        extra = copy.deepcopy(self.scenarios) + [dict(copy.deepcopy(self.scenarios[0]), id="zzz")]
        self.assertTrue(any("without a spec" in p for p in C.validate_generated(extra, self.specs)))

    def test_style_rules(self):
        by_style = {s["condition_style"]: i for i, s in enumerate(self.specs)}
        for style, i in by_style.items():
            sc = copy.deepcopy(self.scenarios[i])
            letter = next(iter(sc["conditions"]))
            sc["conditions"][letter] = (
                "x"
                if style != "imperative_terse"
                else "A very long sentence that cannot be an imperative phrase at all"
            )
            problems = C.style_problems(sc, self.specs[i])
            self.assertTrue(any(f"style {style!r}" in p for p in problems), (style, problems))

    def test_node_message_and_note_rules(self):
        sc = copy.deepcopy(self.scenarios[0])
        sc["node"]["name"] = "Bad Name"
        sc["examples"][0]["msg"] = " ".join(["word"] * 100)
        sc["examples"][1]["why"] = ""
        problems = C.style_problems(sc, self.specs[0])
        self.assertTrue(any("snake_case" in p for p in problems))
        self.assertTrue(any("words" in p and "message" in p for p in problems))
        self.assertTrue(any("'why'" in p for p in problems))

    def test_copying_the_condition_is_flagged(self):
        spec = next(s for s in self.specs if s["examples"][0]["kind"] == "route")
        i = self.specs.index(spec)
        sc = copy.deepcopy(self.scenarios[i])
        gold = spec["examples"][0]["target"]
        sc["examples"][0]["msg"] = sc["conditions"][gold]
        self.assertTrue(any("copies" in p for p in C.style_problems(sc, spec)))

    def test_duplicate_messages_are_caught_by_the_structural_check(self):
        sc = copy.deepcopy(self.scenarios)
        sc[0]["examples"][1]["msg"] = sc[0]["examples"][0]["msg"]
        self.assertTrue(any("duplicate" in p for p in C.validate_generated(sc, self.specs)))

    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            sp, sc = Path(tmp) / "specs.json", Path(tmp) / "sc.json"
            sp.write_text(json.dumps(self.specs), encoding="utf-8")
            sc.write_text(json.dumps(self.scenarios), encoding="utf-8")
            self.assertEqual(C.main(["--specs", str(sp), "--scenarios", str(sc)]), 0)
            sc.write_text(json.dumps(self.scenarios[:-1]), encoding="utf-8")
            self.assertEqual(C.main(["--specs", str(sp), "--scenarios", str(sc)]), 1)


if __name__ == "__main__":
    unittest.main()
