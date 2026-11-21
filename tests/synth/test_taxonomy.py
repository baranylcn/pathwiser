"""Tests for the business-context taxonomy and its splits."""

from __future__ import annotations

import copy
import unittest

from pathwiser.synth import taxonomy as T


class TestTaxonomy(unittest.TestCase):
    def setUp(self):
        self.tax = T.load()

    def test_committed_taxonomy_is_valid_and_complete(self):
        T.validate(self.tax)
        self.assertEqual(sum(len(g["domains"]) for g in self.tax["groups"]), 198)
        self.assertEqual(len(self.tax["groups"]), 33)

    def test_committed_splits_follow_the_rules(self):
        counts = {s: sum(1 for g in self.tax["groups"] if g["split"] == s) for s in T.SPLITS}
        self.assertEqual({k: counts[k] for k in T.SPLIT_COUNTS}, T.SPLIT_COUNTS)
        for g in self.tax["groups"]:
            if g["near"]:
                self.assertEqual(g["split"], "train")
        self.assertEqual(
            T.assign_splits(self.tax), {g["id"]: g["split"] for g in self.tax["groups"]}
        )  # seeded -> reproducible

    def test_eval_domains_are_not_in_train_groups(self):
        ev = {d["id"] for _, d in T.domains_by_split(self.tax, "eval_v2")}
        tr = {d["id"] for _, d in T.domains_by_split(self.tax, "train")}
        self.assertEqual(len(ev), 36)
        self.assertFalse(ev & tr)

    def test_validation_catches_blocked_names_near_groups_and_duplicates(self):
        bad = copy.deepcopy(self.tax)
        bad["groups"][0]["domains"][0]["id"] = "restaurants"
        with self.assertRaises(ValueError):
            T.validate(bad)
        bad = copy.deepcopy(self.tax)
        near = next(g for g in bad["groups"] if g["near"])
        near["split"] = "eval_v2"
        with self.assertRaises(ValueError):
            T.validate(bad)
        bad = copy.deepcopy(self.tax)
        bad["groups"][1]["domains"][0]["id"] = bad["groups"][0]["domains"][0]["id"]
        with self.assertRaises(ValueError):
            T.validate(bad)

    def test_no_domain_reuses_a_realistic_v1_context(self):
        blocked = T.realistic_v1_blocklist()
        self.assertGreater(len(blocked), 60)
        ids = {d["id"] for g in self.tax["groups"] for d in g["domains"]}
        self.assertFalse(ids & blocked)

    def test_domains_by_split_requires_stored_splits(self):
        tax = copy.deepcopy(self.tax)
        del tax["groups"][0]["split"]
        with self.assertRaises(ValueError):
            T.domains_by_split(tax, "train")


if __name__ == "__main__":
    unittest.main()
