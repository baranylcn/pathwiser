"""Tests for the benchmark pipeline building blocks."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pathwiser.benchmark.candidates import Pool, choose_k, plan_no_match, plan_route
from pathwiser.benchmark.config import load_config, validate
from pathwiser.benchmark.pipeline import build
from pathwiser.benchmark.schema import Condition, DropLog, Moment, PipelineError, SourceData
from pathwiser.benchmark.seeding import rng_for, stable_int, stable_unit
from pathwiser.benchmark.sources.sgd import UnionFind
from pathwiser.benchmark.splits import assign_partitions, assign_splits, dedupe_single_turn
from pathwiser.benchmark.text import TfIdf, cosine, norm_text
from pathwiser.paths import REPO_ROOT as ROOT


def cond(uid: str, family: str, desc: str, group: str | None = None, partition: str = "train") -> Condition:
    return Condition(
        uid=uid,
        dataset="t",
        family=family,
        group=group or uid,
        descriptions={"orig": desc},
        description_origin="test",
        reviewed=True,
        partition=partition,
    )


def make_pool() -> Pool:
    specs = [
        ("a1", "cards", "Block a lost credit card"),
        ("a2", "cards", "Order a replacement credit card"),
        ("a3", "cards", "Activate a new credit card"),
        ("b1", "transfer", "Send money to another account"),
        ("b2", "transfer", "Cancel a pending money transfer"),
        ("c1", "travel", "Book a flight to a city"),
        ("c2", "travel", "Reserve a hotel room for the night"),
        ("d1", "music", "Play a song by artist"),
        # same meaning as a1, other service: must never co-occur with a1
        ("a1x", "cards", "Block a lost or stolen credit card", "a1"),
    ]
    return Pool([cond(s[0], s[1], s[2], s[3] if len(s) > 3 else None) for s in specs])


class TestCommon(unittest.TestCase):
    def test_stable_int_matches_contract_and_is_stable(self):
        expected = int.from_bytes(hashlib.sha256(b"42\x1fx").digest()[:8], "big")
        self.assertEqual(stable_int(42, "x"), expected)
        self.assertNotEqual(stable_int(42, "x"), stable_int(43, "x"))
        self.assertNotEqual(stable_int("a", "bc"), stable_int("ab", "c"))

    def test_stable_unit_range(self):
        vals = [stable_unit(1, i) for i in range(2000)]
        self.assertTrue(all(0.0 <= v < 1.0 for v in vals))
        self.assertAlmostEqual(sum(vals) / len(vals), 0.5, delta=0.05)

    def test_rng_independent_streams(self):
        self.assertEqual(rng_for(1, "k").random(), rng_for(1, "k").random())
        self.assertNotEqual(rng_for(1, "k").random(), rng_for(1, "j").random())

    def test_union_find_root_is_smallest(self):
        uf = UnionFind(["c", "b", "a", "d"])
        uf.union("c", "b")
        uf.union("b", "d")
        self.assertEqual({uf.find(x) for x in "bcd"}, {"b"})
        self.assertEqual(uf.find("a"), "a")

    def test_tfidf_ranks_related_text_first(self):
        docs = ["block a lost credit card", "play a song by an artist", "book a flight to paris"]
        t = TfIdf(docs)
        q = t.vector("my credit card is lost")
        scores = [cosine(q, t.vector(d)) for d in docs]
        self.assertEqual(scores.index(max(scores)), 0)

    def test_norm_text(self):
        self.assertEqual(norm_text("  Hello,   WORLD!! "), "hello world")


class TestCandidates(unittest.TestCase):
    CCFG = {
        "size_weights": {"2": 1, "3": 1, "4": 1, "5": 1, "8": 1},
        "negative_strategy_weights": {"family": 0.4, "nearest": 0.4, "random": 0.2},
        "nearest_window_multiplier": 3,
    }

    def test_route_invariants(self):
        pool = make_pool()
        for i in range(400):
            gold = pool.conditions[i % len(pool)]
            plan = plan_route(pool, gold, rng_for(7, i), self.CCFG)
            self.assertIsNotNone(plan)
            uids = [c.uid for c, _, _ in plan.members]
            groups = [c.group for c, _, _ in plan.members]
            descs = [norm_text(c.descriptions["orig"]) for c, _, _ in plan.members]
            self.assertEqual(sum(1 for _, r, _ in plan.members if r == "gold"), 1)
            self.assertIn(gold.uid, uids)
            self.assertEqual(len(set(groups)), len(groups), "equivalent conditions co-occur")
            self.assertEqual(len(set(descs)), len(descs))
            self.assertGreaterEqual(len(uids), 2)
            self.assertLessEqual(len(uids), 8)

    def test_equivalent_condition_never_a_negative(self):
        pool = make_pool()
        gold = pool.by_uid["a1"]
        for i in range(200):
            plan = plan_route(pool, gold, rng_for(3, i), self.CCFG)
            self.assertNotIn("a1x", [c.uid for c, _, _ in plan.members])

    def test_determinism_and_gold_position_uniformity(self):
        pool = make_pool()
        gold = pool.by_uid["b1"]
        a = plan_route(pool, gold, rng_for(5, "m"), self.CCFG)
        b = plan_route(pool, gold, rng_for(5, "m"), self.CCFG)
        self.assertEqual([(c.uid, r, s) for c, r, s in a.members], [(c.uid, r, s) for c, r, s in b.members])
        pos = {}
        for i in range(3000):
            p = plan_route(pool, gold, rng_for(11, i), {**self.CCFG, "size_weights": {"4": 1}})
            idx = [r for _, r, _ in p.members].index("gold")
            pos[idx] = pos.get(idx, 0) + 1
        for idx in range(4):
            self.assertAlmostEqual(pos[idx] / 3000, 0.25, delta=0.04)

    def test_no_match_kinds(self):
        pool = make_pool()
        gold = pool.by_uid["a2"]
        for i in range(300):
            p = plan_no_match(pool, gold, "msg", "other_family", rng_for(2, i), self.CCFG)
            self.assertIsNotNone(p)
            self.assertTrue(all(c.family != gold.family for c, _, _ in p.members))
            self.assertTrue(all(r == "negative" for _, r, _ in p.members))
            self.assertEqual(p.label_risk, "none")
            q = plan_no_match(pool, gold, "msg", "sibling_removed", rng_for(2, i), self.CCFG)
            self.assertTrue(all(c.group != gold.group for c, _, _ in q.members))
            if any(c.family == gold.family for c, _, _ in q.members):
                self.assertEqual(q.label_risk, "sibling_partial_validity")
            o = plan_no_match(pool, None, "block my lost credit card", "oos", rng_for(2, i), self.CCFG)
            self.assertIsNone(o.removed_gold_uid)
            self.assertGreaterEqual(len(o.members), 2)

    def test_infeasible_returns_none(self):
        tiny = Pool([cond("x", "f", "Do thing one"), cond("y", "f", "Do thing two")])
        self.assertIsNone(plan_no_match(tiny, tiny.by_uid["x"], "m", "other_family", rng_for(1), self.CCFG))
        self.assertIsNone(choose_k(rng_for(1), {"5": 1}, 3))


class TestSplits(unittest.TestCase):
    def _moment(self, mid, group, base, gold, msg="hello there"):
        return Moment(mid, "t", group, base, [], msg, gold, None, True, 0, "none")

    def test_partitions_and_unknown_family_error(self):
        conds = {"a": cond("a", "f1", "x y z"), "b": cond("b", "f2", "x y z"), "c": cond("c", "f3", "x y z")}
        assign_partitions("t", conds, {"val": ["f2"], "test": ["f3"]})
        self.assertEqual([conds[k].partition for k in "abc"], ["train", "val", "test"])
        with self.assertRaises(PipelineError):
            assign_partitions("t", conds, {"val": ["nope"]})

    def test_dialogue_routing_keeps_dialogue_whole(self):
        conds = {"a": cond("a", "f1", "x y z"), "b": cond("b", "f2", "x y z", partition="test")}
        ms = [
            self._moment("m1", "d1", "train", "a"),
            self._moment("m2", "d1", "train", "b"),
            self._moment("m3", "d2", "val", "a"),
        ]
        sd = SourceData("t", "T", "L", conds, ms, "dialogue_routed")
        assign_splits(sd)
        self.assertEqual([m.split for m in ms], ["test", "test", "val"])

    def test_partition_routing_and_oos(self):
        conds = {"a": cond("a", "f1", "x y z"), "b": cond("b", "f2", "x y z", partition="val")}
        ms = [
            self._moment("m1", "g1", "train", "a"),
            self._moment("m2", "g2", "train", "b"),
            self._moment("m3", "g3", "test", None),
        ]
        sd = SourceData("t", "T", "L", conds, ms, "partition_routed", True)
        assign_splits(sd)
        self.assertEqual([m.split for m in ms], ["train", "val", "test"])

    def test_dedupe_conflicts_duplicates_and_cross_dataset_leakage(self):
        conds = {"a": cond("a", "f", "x y z"), "b": cond("b", "f", "x y z")}

        def mk(mid, text, gold, split):
            return self._moment(mid, mid, split, gold, text)

        ms1 = [
            mk("1", "Book a flight!", "a", "train"),
            mk("2", "book a flight", "a", "train"),  # dup within split
            mk("3", "same text", "a", "train"),
            mk("4", "same text", "b", "val"),  # label conflict
            mk("5", "only test", "a", "test"),
            mk("6", "Leaky one", "a", "test"),
        ]
        ms2 = [mk("7", "leaky one", "a", "train")]  # other dataset, train
        for m in ms1 + ms2:
            m.split = m.base_split
        s1 = SourceData("d1", "T", "L", dict(conds), ms1, "official", True)
        s2 = SourceData("d2", "T", "L", dict(conds), ms2, "official", True)
        drops = DropLog()
        dedupe_single_turn([s1, s2], drops)
        kept1 = sorted(m.moment_id for m in s1.moments)
        self.assertEqual(kept1, ["1", "5"])  # 2 dup, 3+4 conflict, 6 leaks from d2 train
        self.assertEqual([m.moment_id for m in s2.moments], ["7"])

    def test_dedupe_within_dataset_when_other_dataset_owns_first_occurrence(self):
        # Regression: d1 sorts before d2, so d1 owns the first occurrence; both d2 copies used to survive.
        conds = {"a": cond("a", "f", "x y z")}

        def mk(mid, text, split):
            return self._moment(mid, mid, split, "a", text)

        s1 = SourceData("d1", "T", "L", dict(conds), [mk("1", "what is the weather", "test")], "official", True)
        s2 = SourceData(
            "d2",
            "T",
            "L",
            dict(conds),
            [mk("2", "what is the weather", "test"), mk("3", "what is the weather!", "test")],
            "official",
            True,
        )
        for sd in (s1, s2):
            for m in sd.moments:
                m.split = m.base_split
        dedupe_single_turn([s1, s2], DropLog())
        self.assertEqual([m.moment_id for m in s1.moments], ["1"])  # same split, other dataset: kept by design
        self.assertEqual([m.moment_id for m in s2.moments], ["2"])  # same dataset + split: exactly one copy


class TestConfig(unittest.TestCase):
    def test_default_config_valid(self):
        cfg = load_config()
        self.assertEqual(len(cfg["_hash"]), 64)

    def test_rejects_leaky_sgdx_variants(self):
        cfg = load_config()
        bad = copy.deepcopy(cfg)
        bad["sgdx"]["train_variant_weights"]["v5"] = 0.1  # v5 is the held-out test variant
        with self.assertRaises(PipelineError):
            validate(bad)

    def test_rejects_bad_sizes_and_fractions(self):
        cfg = load_config()
        for mutate in (
            lambda c: c["candidates"]["size_weights"].update({"1": 1}),
            lambda c: c["no_match"].update({"fraction": 1.0}),
            lambda c: c["datasets"]["hwu64"]["split_fractions"].update({"train": 0.9}),
        ):
            bad = copy.deepcopy(cfg)
            mutate(bad)
            with self.assertRaises(PipelineError):
                validate(bad)


@unittest.skipUnless((ROOT / "data" / "raw" / "sgd").exists(), "data/raw not present")
class TestEndToEndSmoke(unittest.TestCase):
    def test_smoke_build_validates_and_is_reproducible(self):
        cfg = load_config()
        hashes = []
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(2):
                out = Path(tmp) / f"run{i}"
                res = build(cfg, ROOT, out_dir=out, verify=False, limit=40, log=lambda s: None)
                self.assertEqual(res["validation"]["n_checks"], len(res["validation"]["checks"]))
                man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
                hashes.append({k: v["sha256"] for k, v in man["files"].items()})
                self.assertTrue((out / "train.jsonl").stat().st_size > 0)
                if i == 0:
                    self._check_csv_matches_jsonl(out)
        self.assertEqual(hashes[0], hashes[1], "build is not reproducible")

    def _check_csv_matches_jsonl(self, out: Path):
        import csv

        examples = [
            json.loads(line)
            for s in ("train", "val", "test")
            for line in (out / f"{s}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        with open(out / "examples_flat.csv", encoding="utf-8-sig", newline="") as f:
            flat = list(csv.DictReader(f))
        with open(out / "candidates_long.csv", encoding="utf-8-sig", newline="") as f:
            long_rows = list(csv.DictReader(f))
        self.assertEqual(len(flat), len(examples))
        self.assertEqual(len(long_rows), sum(len(e["candidates"]) for e in examples))
        by_id = {e["example_id"]: e for e in examples}
        for row in flat[:500]:
            e = by_id[row["example_id"]]
            self.assertEqual(row["current_message"], e["current_message"])
            self.assertEqual(row["label_type"], e["label_type"])
            if e["label_type"] == "route":
                gold = next(c for c in e["candidates"] if c["cid"] == e["gold_cid"])
                self.assertEqual(row["gold_description"], gold["description"])
        self.assertEqual(
            sum(int(r["is_gold"]) for r in long_rows), sum(1 for e in examples if e["label_type"] == "route")
        )


@unittest.skipUnless((ROOT / "data" / "raw" / "sgd").exists(), "data/raw not present")
class TestValidatorCatchesViolations(unittest.TestCase):
    """Inject corruption into a real (small) build and require validate() to reject it."""

    @classmethod
    def setUpClass(cls):
        from pathwiser.benchmark.pipeline import prepare

        cls.cfg = load_config()
        cls.prep = prepare(cls.cfg, ROOT, verify=False, limit=120, log=lambda s: None)

    def fresh(self):
        return copy.deepcopy(self.prep.examples)

    def assert_rejected(self, examples, needle):
        from pathwiser.benchmark.validate import validate as run_validate

        with self.assertRaises(PipelineError) as cm:
            run_validate(self.cfg, self.prep.sources, examples)
        self.assertIn(needle, str(cm.exception))

    def test_clean_build_passes(self):
        from pathwiser.benchmark.validate import validate as run_validate

        report = run_validate(self.cfg, self.prep.sources, self.prep.examples)
        self.assertEqual(report["n_failed"], 0)

    def test_unseen_condition_in_train_is_rejected(self):
        ex = self.fresh()
        src = next(
            e
            for e in ex["test"]
            if e["label_type"] == "route" and "cond:unseen" in e["slices"] and e["source"]["dataset"] != "sgd"
        )
        moved = copy.deepcopy(src)
        moved["split"] = "train"
        ex["train"].append(moved)
        self.assert_rejected(ex, "partition")

    def test_wrong_gold_cid_is_rejected(self):
        ex = self.fresh()
        e = next(x for x in ex["val"] if x["label_type"] == "route")
        e["gold_cid"] = "c999"
        self.assert_rejected(ex, "gold_cid_consistent")

    def test_group_crossing_splits_is_rejected(self):
        ex = self.fresh()
        e = next(x for x in ex["train"] if x["source"]["dataset"] == "abcd")
        clone = copy.deepcopy(e)
        clone["example_id"], clone["split"] = "ex_forged", "test"
        ex["test"].append(clone)
        self.assert_rejected(ex, "split_groups_disjoint_train_test")

    def test_single_turn_text_leak_is_rejected(self):
        ex = self.fresh()
        tr = next(x for x in ex["train"] if x["source"]["dataset"] == "banking77")
        te = next(x for x in ex["test"] if x["source"]["dataset"] == "banking77")
        te["current_message"] = tr["current_message"]
        self.assert_rejected(ex, "single_turn_text_disjoint_banking77_train_test")

    def test_duplicate_text_within_dataset_split_is_rejected(self):
        ex = self.fresh()
        tr = next(x for x in ex["train"] if x["source"]["dataset"] == "banking77")
        clone = copy.deepcopy(tr)
        clone["example_id"] = "ex_forged_dup"
        ex["train"].append(clone)
        self.assert_rejected(ex, "single_turn_text_unique_within_dataset_split")

    def test_duplicate_equivalent_candidate_is_rejected(self):
        ex = self.fresh()
        e = next(x for x in ex["train"] if x["label_type"] == "route")
        e["candidate_meta"][0]["equivalence_group_id"] = e["candidate_meta"][1]["equivalence_group_id"]
        self.assert_rejected(ex, "equivalence_groups_unique_in_example")

    def test_context_over_limit_is_rejected(self):
        ex = self.fresh()
        e = next(x for x in ex["train"] if x["source"]["dataset"] == "sgd")
        e["context"] = [{"role": "user", "content": "x"}] * (self.cfg["context_max_turns"] + 1)
        self.assert_rejected(ex, "context_within_limit")


@unittest.skipUnless((ROOT / "data" / "raw" / "sgd").exists(), "data/raw not present")
class TestCrossProcessReproducibility(unittest.TestCase):
    def test_identical_output_under_different_hash_seeds(self):
        import os
        import subprocess
        import sys

        manifests = []
        with tempfile.TemporaryDirectory() as tmp:
            for hs in ("0", "12345"):
                out = Path(tmp) / f"hs{hs}"
                env = {**os.environ, "PYTHONHASHSEED": hs}
                r = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pathwiser.benchmark",
                        "--smoke",
                        "40",
                        "--skip-verify-raw",
                        "--out",
                        str(out),
                    ],
                    cwd=ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(r.returncode, 0, r.stderr[-800:])
                man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
                manifests.append({k: v["sha256"] for k, v in man["files"].items()})
        self.assertEqual(manifests[0], manifests[1])


if __name__ == "__main__":
    unittest.main()
