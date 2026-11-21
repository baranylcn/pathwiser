"""Tests for the persistent score store."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pathwiser.storage.scores import (
    ScoreMismatch,
    available,
    load_info,
    load_scores,
    save_scores,
    score_path,
    text_fingerprint,
)


def _records(n: int = 5) -> list[dict]:
    out = []
    for i in range(n):
        k = 2 + i % 3
        out.append(
            {
                "example_id": f"ex{i}",
                "input": {
                    "message": f"msg {i}",
                    "context": [{"role": "user", "content": "hi"}] if i % 2 else [],
                    "conditions": [f"cond {i}-{j}" for j in range(k)],
                },
            }
        )
    return out


class TestScoreStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.recs = _records()
        rng = np.random.default_rng(0)
        self.scores = [rng.normal(size=len(r["input"]["conditions"])) for r in self.recs]
        self.path = score_path(self.tmp.name, "v1-model", "st_lgbm", "val")

    def test_round_trip(self):
        save_scores(self.path, self.recs, self.scores, {"scorer": "x", "seconds": 1.5})
        back = load_scores(self.path, self.recs)
        for a, b in zip(self.scores, back):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)  # stored as float32
        self.assertEqual(load_info(self.path), {"scorer": "x", "seconds": 1.5})
        self.assertEqual(available(self.tmp.name, "v1-model", "st_lgbm"), ["val"])
        self.assertEqual(available(self.tmp.name, "v1-model", "nothing"), [])

    def test_refuses_scores_for_different_records(self):
        save_scores(self.path, self.recs, self.scores)
        with self.assertRaises(ScoreMismatch):
            load_scores(self.path, self.recs[:-1])  # fewer examples
        swapped = copy.deepcopy(self.recs)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        with self.assertRaises(ScoreMismatch):
            load_scores(self.path, swapped)  # order / ids
        edited = copy.deepcopy(self.recs)
        edited[2]["input"]["message"] = "a different message"
        with self.assertRaises(ScoreMismatch):
            load_scores(self.path, edited)  # same ids and sizes, edited text
        edited = copy.deepcopy(self.recs)
        edited[3]["input"]["conditions"][0] = "another condition"
        with self.assertRaises(ScoreMismatch):
            load_scores(self.path, edited)

    def test_refuses_misaligned_or_non_finite_scores(self):
        with self.assertRaises(ScoreMismatch):
            save_scores(self.path, self.recs, self.scores[:-1])
        bad = [s.copy() for s in self.scores]
        bad[0][0] = np.nan
        with self.assertRaises(ScoreMismatch):
            save_scores(self.path, self.recs, bad)
        self.assertFalse(Path(self.path).exists())

    def test_fingerprint_is_sensitive_to_context(self):
        a = text_fingerprint(self.recs)
        b = copy.deepcopy(self.recs)
        b[1]["input"]["context"][0]["content"] = "hello"
        self.assertNotEqual(a, text_fingerprint(b))


if __name__ == "__main__":
    unittest.main()
