"""End-to-end tests of the benchmark pipeline and the baseline protocol on a synthetic raw tree."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from pathwiser.benchmark.config import load_config
from pathwiser.benchmark.pipeline import build
from pathwiser.benchmark.records import export
from pathwiser.benchmark.schema import PipelineError
from pathwiser.paths import REPO_ROOT as ROOT
from pathwiser.reports.baselines import run_baseline, write_summary
from pathwiser.storage.dataset import load_split


def quiet(s):

    return None


@unittest.skipUnless((ROOT / "data" / "raw" / "sgd").exists(), "data/raw not present")
class TestPipelineIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        cls.bench = base / "bench"
        build(load_config(), ROOT, out_dir=cls.bench, verify=False, limit=40, log=quiet)
        cls.model = base / "model"
        cls.manifest = export(cls.bench, cls.model, log=quiet)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_export_is_byte_reproducible(self):
        again = Path(self.tmp.name) / "model2"
        m2 = export(self.bench, again, log=quiet)
        for name in ("train.jsonl", "val.jsonl", "test.jsonl", "SCHEMA.md"):
            self.assertEqual(self.manifest["files"][name]["sha256"], m2["files"][name]["sha256"], name)

    def test_records_match_the_benchmark_and_leak_nothing(self):
        src = {}
        for s in ("train", "val", "test"):
            for line in (self.bench / f"{s}.jsonl").read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                src[r["example_id"]] = r
        for split in ("train", "val", "test"):
            recs = load_split(split, self.model)
            self.assertEqual(len(recs), self.manifest["files"][f"{split}.jsonl"]["rows"])
            for m in recs:
                r = src[m["example_id"]]
                k = len(m["input"]["conditions"])
                self.assertEqual(set(m), {"example_id", "split", "input", "target", "weight", "meta"})
                self.assertEqual(k, len(r["candidates"]))
                self.assertEqual(m["split"], split)
                gold = r["gold_cid"]
                if gold is None:
                    self.assertEqual(
                        (m["target"]["label"], m["target"]["label_null"], m["target"]["has_match"]), (-1, k, 0)
                    )
                    self.assertIsNone(m["meta"]["gold_seen_exact"])
                else:
                    i = m["target"]["label"]
                    self.assertEqual(r["candidates"][i]["cid"], gold)
                    self.assertEqual(m["input"]["conditions"][i], r["candidates"][i]["description"].strip())
                    self.assertEqual(m["target"]["label_null"], i)
                text = m["input"]["query"] + "\n".join(m["input"]["conditions"])
                for meta in r["candidate_meta"]:
                    self.assertNotIn(meta["condition_uid"], text)
                self.assertTrue(m["input"]["query"].endswith(m["input"]["message"]))
                if split != "train":
                    self.assertEqual(m["weight"], 1.0)
                else:
                    self.assertFalse(m["meta"]["is_paired"])

    def test_train_weights_average_to_one(self):
        w = [m["weight"] for m in load_split("train", self.model)]
        self.assertAlmostEqual(sum(w) / len(w), 1.0, places=3)

    def test_tampered_source_is_rejected(self):
        bad = Path(self.tmp.name) / "tampered"
        shutil.copytree(self.bench, bad)
        with open(bad / "train.jsonl", "a", encoding="utf-8") as f:
            f.write("\n")
        with self.assertRaises(PipelineError):
            export(bad, Path(self.tmp.name) / "never", log=quiet)

    def test_baselines_run_and_report(self):
        out = Path(self.tmp.name) / "reports"
        for name in ("random", "lexical", "tfidf"):
            res = run_baseline(name, self.model, out, final=True, st_model="", log=quiet)
            self.assertEqual(set(res["splits"]), {"val", "test"})
            o = res["splits"]["test"]["overall"]
            self.assertTrue(0 <= o["top1"] <= 1 and 0 <= o["decision_accuracy"] <= 1)
            self.assertEqual(res["protocol"]["threshold_tuned_on"], "val main examples")
        summary = write_summary(out).read_text(encoding="utf-8")
        self.assertIn("## Test (τ from validation)", summary)
        self.assertIn("`tfidf`", summary)

    def test_val_only_run_never_touches_test(self):
        out = Path(self.tmp.name) / "reports_val"
        res = run_baseline("lexical", self.model, out, final=False, st_model="", log=quiet)
        self.assertEqual(set(res["splits"]), {"val"})
        self.assertFalse(res["protocol"]["test_evaluated"])


if __name__ == "__main__":
    unittest.main()
