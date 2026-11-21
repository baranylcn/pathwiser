"""End-to-end tests of the report layer on a synthetic world with stored scores."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pathwiser.reports.encoders import adoption_verdict
from pathwiser.reports.external import external_report, render_external
from pathwiser.reports.final import final_report
from pathwiser.reports.selection import candidate_systems, dev_selection, system_from_label
from pathwiser.reports.store import Store
from pathwiser.storage.scores import save_scores, score_path
from tests.helpers import full_records, make_world

VERSION = "toy-model"


class TestReportLayer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.data, self.scores, self.ext, self.out = root / "data", root / "scores", root / "ext", root / "reports"
        for d in (self.data, self.ext):
            d.mkdir()
        (self.data / "manifest.json").write_text(
            json.dumps({"version": VERSION, "source_version": "toy"}), encoding="utf-8"
        )
        self.recs = {}
        for split, seed, d in (("val", 1, self.data), ("test", 2, self.data), ("realistic", 3, self.ext)):
            world, by_model = make_world(seed=seed, n_per=150)
            recs = full_records(world, split[0])
            fname = "test.jsonl" if split == "realistic" else f"{split}.jsonl"
            (d / fname).write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
            self.recs[split] = recs
            for name, key in (("st_lgbm", "a"), ("st_message", "b")):  # model names the candidate list knows
                save_scores(score_path(self.scores, VERSION, name, split), recs, by_model[key])
        self.store = Store(self.data, self.scores, {"realistic": self.ext})

    def test_candidates_depend_on_available_models(self):
        labels = [s.label for s in candidate_systems(["st_lgbm", "st_message"])]
        self.assertIn("st_lgbm|tau", labels)
        self.assertIn("st_lgbm+st_message|fused|full", labels)
        self.assertFalse(any("cross_encoder" in s for s in labels))
        self.assertTrue(
            any("cross_encoder" in s.label for s in candidate_systems(["st_lgbm", "st_message", "cross_encoder"]))
        )
        with self.assertRaises(KeyError):
            system_from_label("nope", ["st_lgbm"])

    def test_trained_variants_get_singles_and_a_fusion_with_the_base_model(self):
        labels = [s.label for s in candidate_systems(["st_lgbm", "st_lgbm__bge_small"])]
        self.assertIn("st_lgbm__bge_small|tau", labels)
        self.assertIn("st_lgbm+st_lgbm__bge_small|fused|full", labels)
        self.assertFalse(any("bge" in s for s in [x.label for x in candidate_systems(["st_lgbm"])]))

    def test_dev_selection_uses_validation_only_and_writes_the_selection(self):
        # make test / external scores unreadable: selection must not touch them
        for split in ("test", "realistic"):
            for name in ("st_lgbm", "st_message"):
                score_path(self.scores, VERSION, name, split).unlink()
        sel = dev_selection(self.store, ["st_lgbm", "st_message"], self.out, log=lambda s: None)
        self.assertIn(sel["chosen"], [s.label for s in candidate_systems(["st_lgbm", "st_message"])])
        self.assertEqual(json.loads((self.out / "selection.json").read_text(encoding="utf-8"))["chosen"], sel["chosen"])
        lodo = json.loads((self.out / "lodo.json").read_text(encoding="utf-8"))
        self.assertEqual(len(lodo["results"]), sel["n_candidates"])
        self.assertEqual(sorted(lodo["results"][0]["per_dataset"]), ["ds0", "ds1", "ds2"])

    def test_final_report_reads_the_selection_and_never_picks(self):
        with self.assertRaises(FileNotFoundError):
            final_report(self.store, "realistic", 50, self.out, log=lambda s: None)  # no selection yet -> refuses
        sel = dev_selection(self.store, ["st_lgbm", "st_message"], self.out, log=lambda s: None)
        res = final_report(self.store, "realistic", 50, self.out, log=lambda s: None)
        self.assertEqual(res["chosen"], sel["chosen"])
        self.assertIn(sel["chosen"], res["systems"])
        self.assertIn("st_lgbm|tau", res["systems"])  # the plain-threshold reference is always reported
        ext = res["external"]["realistic"]
        self.assertEqual(ext["reference"], "st_lgbm|tau")
        self.assertIn(sel["chosen"], ext["bootstrap"]["systems"])
        text = (self.out / "final.json").read_text(encoding="utf-8")
        self.assertNotIn("NaN", text)
        self.assertIn(sel["chosen"], (self.out / "summary.md").read_text(encoding="utf-8"))

    def test_external_report_uses_validation_tau_and_scenario_bootstrap(self):
        out = external_report(self.store, "realistic", ["st_lgbm", "st_message"], "st_message", 50, self.out)
        self.assertEqual(out["n_examples"], 450)
        self.assertGreater(out["n_clusters"], 10)
        m = out["models"]["st_lgbm"]
        self.assertIn("validation", m["tau_source"])
        lo, hi, v = (out["bootstrap"]["systems"]["st_lgbm"]["top1"][k] for k in ("lo", "hi", "value"))
        self.assertTrue(lo <= v <= hi)
        self.assertIn("st_lgbm", out["bootstrap"]["vs_reference"])
        self.assertIn("Transfer", render_external(out))
        self.assertTrue((self.out / "realistic_baselines.md").exists())


class TestAdoptionRule(unittest.TestCase):
    REF = {"top1": 0.893, "lodo_macro": 0.800, "lodo_se": 0.017, "svc_unseen_top1": 0.85}

    def cand(self, **kw):
        return dict(self.REF, **kw)

    def test_adopt_when_validation_holds_and_external_gains(self):
        v = adoption_verdict(
            self.cand(), self.REF, {"diff": 0.03, "lo": -0.01, "hi": 0.07}, {"diff": 0.02, "lo": -0.03, "hi": 0.07}
        )
        self.assertTrue(v["adopt"])
        self.assertEqual(v["confidence"], "tentative")  # interval includes 0
        v = adoption_verdict(
            self.cand(), self.REF, {"diff": 0.06, "lo": 0.01, "hi": 0.1}, {"diff": 0.0, "lo": -0.03, "hi": 0.03}
        )
        self.assertEqual((v["adopt"], v["confidence"]), (True, "confirmed"))

    def test_reject_when_validation_is_worse(self):
        ext = {"diff": 0.08, "lo": 0.03, "hi": 0.12}
        self.assertFalse(
            adoption_verdict(self.cand(top1=0.885), self.REF, ext, ext)["adopt"]
        )  # Top-1 loses more than 0.005
        self.assertFalse(
            adoption_verdict(self.cand(lodo_macro=0.78), self.REF, ext, ext)["adopt"]
        )  # more than one s.e. below
        self.assertTrue(adoption_verdict(self.cand(lodo_macro=0.79), self.REF, ext, ext)["adopt"])  # within one s.e.
        self.assertIsNone(adoption_verdict(self.cand(top1=0.885), self.REF, ext, ext)["confidence"])

    def test_reject_when_external_does_not_gain(self):
        ok = {"diff": 0.01, "lo": -0.02, "hi": 0.04}
        self.assertFalse(
            adoption_verdict(self.cand(), self.REF, {"diff": 0.0, "lo": -0.03, "hi": 0.03}, ok)["adopt"]
        )  # Top-1 gain must be > 0
        self.assertFalse(
            adoption_verdict(self.cand(), self.REF, ok, {"diff": -0.01, "lo": -0.05, "hi": 0.03})["adopt"]
        )  # decision accuracy must not drop

    def test_unseen_service_warning_is_independent_of_the_verdict(self):
        ext = {"diff": 0.05, "lo": 0.01, "hi": 0.09}
        v = adoption_verdict(self.cand(svc_unseen_top1=0.82), self.REF, ext, ext)
        self.assertTrue(v["adopt"])
        self.assertTrue(v["unseen_service_warning"])
        self.assertFalse(
            adoption_verdict(self.cand(svc_unseen_top1=0.84), self.REF, ext, ext)["unseen_service_warning"]
        )


if __name__ == "__main__":
    unittest.main()
