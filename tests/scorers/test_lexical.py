"""Tests for the training-free scorers."""

from __future__ import annotations

import unittest

import numpy as np

from pathwiser.scorers.lexical import LexicalOverlap, RandomScorer, TfidfCosine, content_words


def rec(eid, conds, label, message="m", dataset="d", paired_with=None, slices=(), context=()):
    return {
        "example_id": eid,
        "input": {
            "message": message,
            "conditions": list(conds),
            "context": [{"role": "user", "content": c} for c in context],
        },
        "target": {"label": label},
        "meta": {
            "dataset": dataset,
            "is_paired": paired_with is not None,
            "paired_with": paired_with,
            "slices": list(slices),
            "gold_seen_exact": None,
        },
    }


class TestScorers(unittest.TestCase):
    def test_content_words_stems_plurals_and_drops_stop_words(self):
        self.assertEqual(
            content_words("The user wants to cancel the orders"), frozenset({"user", "want", "cancel", "order"})
        )

    def test_lexical_prefers_overlapping_condition(self):
        r = rec(
            "x",
            ["The user wants to track a package", "The user wants to cancel an order"],
            1,
            message="please cancel my order",
        )
        s = LexicalOverlap().score([r])[0]
        self.assertGreater(s[1], s[0])

    def test_lexical_context_variant_uses_context(self):
        r = rec(
            "x",
            ["Search for flights", "Book a hotel room"],
            0,
            message="yes please",
            context=["I need a flight to Paris"],
        )
        self.assertEqual(LexicalOverlap(False).score([r])[0].max(), 0.0)
        self.assertGreater(LexicalOverlap(True).score([r])[0][0], 0.0)

    def test_tfidf_fits_on_train_only_and_is_deterministic(self):
        train = [
            rec("t1", ["Cancel an order", "Track a parcel"], 0, message="cancel my order"),
            rec("t2", ["Cancel an order", "Track a parcel"], 1, message="where is my parcel"),
        ]
        test = [rec("v", ["Cancel an order", "Track a parcel"], 1, message="track the parcel please")]
        a, b = TfidfCosine(), TfidfCosine()
        a.fit(train)
        b.fit(train)
        sa, sb = a.score(test)[0], b.score(test)[0]
        self.assertTrue(np.array_equal(sa, sb))
        self.assertGreater(sa[1], sa[0])
        with self.assertRaises(RuntimeError):
            TfidfCosine().score(test)

    def test_random_scorer_is_deterministic_per_example(self):
        r = rec("same", ["a", "b", "c"], 0)
        self.assertTrue(np.array_equal(RandomScorer().score([r])[0], RandomScorer().score([r])[0]))
        self.assertFalse(
            np.array_equal(RandomScorer().score([r])[0], RandomScorer().score([rec("other", ["a", "b", "c"], 0)])[0])
        )


if __name__ == "__main__":
    unittest.main()
