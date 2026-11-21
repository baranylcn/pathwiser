"""Tests for the leakage checks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pathwiser.synth import leakage as L
from tests.helpers import FakeEncoder


class TestLeakage(unittest.TestCase):
    def test_exact_and_duplicates(self):
        ref = ["Cancel my order", "Where is my parcel"]
        self.assertEqual(
            L.exact_leaks({"a": "  cancel   my ORDER ", "b": "something new"}, ref),
            [{"id": "a", "text": "  cancel   my ORDER "}],
        )
        dups = L.duplicates_within({"a": "Hello there", "b": "hello   THERE", "c": "other"})
        self.assertEqual([d["id"] for d in dups], ["b"])

    def test_tfidf_near_duplicates(self):
        ref = ["i would like to cancel my subscription starting next month please", "the weather is nice today"]
        hits = L.tfidf_near(
            {
                "x": "I would like to cancel my subscription starting next month",
                "y": "turbine blade vibration alarm in tower three",
            },
            ref,
        )
        self.assertEqual([h["id"] for h in hits], ["x"])
        self.assertGreaterEqual(hits[0]["similarity"], L.MSG_THRESHOLD)
        self.assertEqual(L.tfidf_near({}, ref), [])

    def test_embedding_near_duplicates_with_a_fake_encoder(self):
        ref = ["Customer wants to cancel an order", "Report a lost item"]
        hits = L.embedding_near(
            {"a": "Customer wants to cancel an order", "b": "Quote for a new elevator installation"},
            ref,
            encoder=FakeEncoder(),
        )
        self.assertEqual([h["id"] for h in hits], ["a"])

    def test_read_reference_collects_unique_texts(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)

            def rec(m, cs):
                return json.dumps({"input": {"message": m, "conditions": cs}}) + "\n"

            (d / "train.jsonl").write_text(
                rec("Hi there", ["a b", "c d"]) + rec("hi  THERE", ["a b"]), encoding="utf-8"
            )
            msgs, conds = L.read_reference(d)
            self.assertEqual((len(msgs), sorted(conds)), (1, ["a b", "c d"]))


if __name__ == "__main__":
    unittest.main()
