"""Tests for the score dump command."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pathwiser.scorers.fit import run, store_name
from pathwiser.storage.dataset import load_split
from pathwiser.storage.scores import available, load_info, load_scores, score_path
from tests.helpers import synth


def _write_split(dirpath: Path, name: str, records: list[dict]) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


class TestDump(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.data, self.ext, self.store = root / "data", root / "ext", root / "scores"
        for split, seed in (("train", 1), ("val", 2), ("test", 3)):
            _write_split(self.data, split, synth(30, seed))
        (self.data / "manifest.json").write_text(
            json.dumps({"version": "toy-model", "source_version": "toy"}), encoding="utf-8"
        )
        _write_split(self.ext, "test", synth(12, 4))
        self.extras = {"realistic": self.ext}
        self.quiet = lambda s: None

    def _run(self, splits, **kw):
        return run("lexical", splits, self.data, self.store, self.extras, log=self.quiet, **kw)

    def test_scores_are_stored_and_loadable(self):
        paths = self._run(["val", "test", "realistic"], final=True)
        self.assertEqual(sorted(paths), ["realistic", "test", "val"])
        self.assertEqual(available(self.store, "toy-model", "lexical"), ["realistic", "test", "val"])
        for split in ("val", "test"):
            self.assertEqual(len(load_scores(paths[split], load_split(split, self.data))), 30)
        self.assertEqual(len(load_scores(paths["realistic"], load_split("test", self.ext))), 12)
        info = load_info(paths["val"])
        self.assertEqual((info["scorer"], info["split"], info["n_examples"]), ("lexical", "val", 30))

    def test_test_split_requires_final(self):
        with self.assertRaises(ValueError):
            self._run(["val", "test"])
        self.assertEqual(available(self.store, "toy-model", "lexical"), [])  # nothing is scored before the check passes

    def test_unknown_split_and_existing_files(self):
        with self.assertRaises(ValueError):
            self._run(["nope"])
        paths = self._run(["val"])
        stamp = paths["val"].stat().st_mtime_ns
        self._run(["val"])  # kept
        self.assertEqual(paths["val"].stat().st_mtime_ns, stamp)
        self._run(["val"], force=True)  # recomputed
        self.assertNotEqual(paths["val"].stat().st_mtime_ns, stamp)

    def test_tag_names_a_variant(self):
        self.assertEqual(store_name("cross_encoder", "hn8k"), "cross_encoder__hn8k")
        self.assertEqual(store_name("cross_encoder", None), "cross_encoder")
        run("lexical", ["val"], self.data, self.store, {}, tag="v2", log=self.quiet)
        self.assertTrue(score_path(self.store, "toy-model", "lexical__v2", "val").exists())

    def test_load_is_only_for_loadable_baselines(self):
        with self.assertRaises(ValueError):
            self._run(["val"], load=Path(self.tmp.name))


if __name__ == "__main__":
    unittest.main()
