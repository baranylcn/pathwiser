"""Tests for the trained scorers, the cross-encoder training loop and the encoder prefixes."""

from __future__ import annotations

import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pathwiser.evaluation.metrics import predictions, summarize
from pathwiser.scorers.cross_encoder import CrossEncoderScorer, build_group, format_query
from pathwiser.scorers.embedding_pair import EmbeddingPairModel
from pathwiser.scorers.registry import ALL_BASELINES, TRAINED_BASELINES, build
from pathwiser.scorers.tfidf_lr import TfidfLogReg
from pathwiser.scorers.training import pair_labels_and_weights, relative_features, sample_examples, split_flat
from tests.helpers import FakeEncoder, synth


class TestHelpers(unittest.TestCase):
    def test_sample_examples_is_deterministic_and_follows_weights(self):
        recs = synth(400, 1)
        for r in recs[:50]:
            r["weight"] = 20.0
        a, b = sample_examples(recs, 100, 7), sample_examples(recs, 100, 7)
        self.assertEqual([r["example_id"] for r in a], [r["example_id"] for r in b])
        self.assertEqual(len({r["example_id"] for r in a}), 100)
        heavy = sum(r["weight"] == 20.0 for r in a)
        self.assertGreater(heavy, 30)  # 12% of the pool carry 70% of the mass
        self.assertEqual(len(sample_examples(recs, None, 7)), 400)
        self.assertEqual(len(sample_examples(recs, 10_000, 7)), 400)

    def test_relative_features_match_brute_force(self):
        rng = np.random.default_rng(0)
        sizes = np.array([2, 3, 5])
        base = np.round(rng.random((10, 2)), 1)
        rel = relative_features(base, sizes)
        diff, z = rel[:, :2], rel[:, 2:]
        start = 0
        for k in sizes:
            blk = base[start : start + k]
            for i in range(k):
                others = np.delete(blk, i, axis=0)
                np.testing.assert_allclose(diff[start + i], blk[i] - others.max(axis=0))
            np.testing.assert_allclose(z[start : start + k].mean(axis=0), 0, atol=1e-6)
            start += k

    def test_pair_labels_and_split(self):
        recs = synth(30, 2)
        y, w = pair_labels_and_weights(recs)
        self.assertEqual(len(y), sum(len(r["input"]["conditions"]) for r in recs))
        self.assertEqual(int(y.sum()), sum(r["target"]["label"] >= 0 for r in recs))
        parts = split_flat(y, np.array([len(r["input"]["conditions"]) for r in recs]))
        self.assertEqual(len(parts), len(recs))


class TestPairModels(unittest.TestCase):
    def setUp(self):
        self.train, self.val = synth(300, 3), synth(120, 4)

    def top1(self, scorer):
        scores = scorer.score(self.val)
        return summarize(predictions(self.val, scores))["top1"], scores

    def test_tfidf_lr_learns_and_is_deterministic(self):
        a, b = TfidfLogReg(), TfidfLogReg()
        a.fit(self.train)
        b.fit(self.train)
        t1, s1 = self.top1(a)
        _, s2 = self.top1(b)
        self.assertGreater(t1, 0.9)
        for x, y in zip(s1, s2):
            np.testing.assert_allclose(x, y)
        with self.assertRaises(RuntimeError):
            TfidfLogReg().score(self.val)

    def test_embedding_models_with_fake_encoder_and_cache(self):
        for learner in ("lr", "lgbm"):
            if learner == "lgbm" and importlib.util.find_spec("lightgbm") is None:
                continue
            with tempfile.TemporaryDirectory() as tmp:
                m = EmbeddingPairModel(learner, train_examples=200, cache_dir=tmp)
                m._st = FakeEncoder()
                m.fit(self.train)
                t1, s1 = self.top1(m)
                self.assertGreater(t1, 0.9, learner)
                self.assertTrue(list(Path(tmp).glob("emb_*.npy")), "embeddings were not cached")
                m2 = EmbeddingPairModel(learner, train_examples=200, cache_dir=tmp)
                m2._st = FakeEncoder()
                m2.fit(self.train)
                _, s2 = self.top1(m2)
                for x, y in zip(s1, s2):
                    np.testing.assert_allclose(x, y, atol=1e-6)
                self.assertEqual(m.info["train_examples"], 200)

    def test_registry_knows_every_trained_baseline(self):
        for name in TRAINED_BASELINES:
            self.assertIn(name, ALL_BASELINES)
        self.assertIsInstance(build("tfidf_lr"), TfidfLogReg)
        self.assertEqual(build("st_lgbm", params={"train_examples": 10}).params["train_examples"], 10)
        with self.assertRaises(ValueError):
            build("lexical", params={"x": 1})


class TestCrossEncoder(unittest.TestCase):
    def test_format_query_keeps_message_first_and_newest_context_first(self):
        r = {
            "input": {
                "message": "cancel it",
                "context": [
                    {"role": "user", "content": "old"},
                    {"role": "assistant", "content": "mid"},
                    {"role": "user", "content": "new"},
                ],
            }
        }
        self.assertEqual(format_query(r, "[SEP]", 2), "cancel it [SEP] user: new [SEP] assistant: mid")
        self.assertEqual(format_query(r, "[SEP]", 0), "cancel it")
        self.assertEqual(format_query({"input": {"message": "hi", "context": []}}, "[SEP]", 4), "hi")

    def test_build_group_targets(self):
        rng = random.Random(0)
        route = synth(1, 5, k=5, no_match=0.0)[0]
        for _ in range(20):
            chosen, target = build_group(route, 3, rng)
            self.assertEqual(len(chosen), 3)
            self.assertEqual(chosen[target], route["target"]["label"])
            self.assertEqual(len(set(chosen)), 3)
        nm = synth(1, 6, k=5, no_match=1.0)[0]
        chosen, target = build_group(nm, 3, rng)
        self.assertEqual((len(chosen), target), (3, 3))
        small = synth(1, 7, k=2, no_match=0.0)[0]
        chosen, _ = build_group(small, 4, rng)
        self.assertEqual(len(chosen), 2)

    def test_training_loop_and_scoring_with_a_tiny_random_model(self):
        try:
            from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification

            tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
        except Exception as e:  # pragma: no cover
            self.skipTest(f"tokenizer unavailable offline: {e}")

        class Tiny(CrossEncoderScorer):
            def _load_model(self):
                import torch

                torch.manual_seed(0)
                cfg = BertConfig(
                    vocab_size=tok.vocab_size,
                    hidden_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    intermediate_size=64,
                    num_labels=1,
                )
                return tok, BertForSequenceClassification(cfg)

        train, val = synth(48, 8), synth(10, 9)
        a = Tiny(
            train_examples=48, batch_examples=8, max_len=48, lr=5e-3, save_dir=None, threads=2, eval_batch_pairs=16
        )
        a.fit(train)
        self.assertEqual(a.info["steps"], 6)
        self.assertEqual(a.info["train_examples"], 48)
        scores = a.score(val)
        self.assertEqual([len(s) for s in scores], [len(r["input"]["conditions"]) for r in val])
        self.assertTrue(all(np.isfinite(s).all() for s in scores))
        b = Tiny(
            train_examples=48, batch_examples=8, max_len=48, lr=5e-3, save_dir=None, threads=2, eval_batch_pairs=16
        )
        b.fit(train)
        for x, y in zip(scores, b.score(val)):
            np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-4)  # same seed -> same model -> same scores
        with self.assertRaises(RuntimeError):
            Tiny(save_dir=None).score(val)


def _tiny_class():
    """A CrossEncoderScorer whose backbone is a 1-layer random BERT (offline); skips when the tokenizer is not cached."""
    try:
        from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification

        tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
    except Exception as e:  # pragma: no cover
        raise unittest.SkipTest(f"tokenizer unavailable offline: {e}") from e

    class Tiny(CrossEncoderScorer):
        def _load_model(self):
            import torch

            torch.manual_seed(0)
            cfg = BertConfig(
                vocab_size=tok.vocab_size,
                hidden_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=64,
                num_labels=1,
            )
            return tok, BertForSequenceClassification(cfg)

        def _load_pretrained(self, directory):
            return AutoTokenizer.from_pretrained(directory), BertForSequenceClassification.from_pretrained(directory)

        def _load_encoder(self):
            return FakeEncoder()

    return Tiny


TINY = {"train_examples": 48, "batch_examples": 8, "max_len": 48, "lr": 5e-3, "threads": 2, "eval_batch_pairs": 16}


class TestCrossEncoderTraining(unittest.TestCase):
    def test_build_group_hard_order_always_includes_the_hardest_wrong_candidates(self):
        rec = synth(1, 11, k=5, no_match=0.0)[0]
        gold = rec["target"]["label"]
        order = [j for j in range(5) if j != gold][::-1]
        order.insert(2, gold)  # gold somewhere in the ranking: it must be skipped, not counted
        rng = random.Random(0)
        for _ in range(30):
            chosen, target = build_group(rec, 4, rng, order, n_hard=2)
            hardest = [j for j in order if j != gold][:2]
            self.assertTrue(set(hardest) <= set(chosen))
            self.assertEqual(chosen[target], gold)
            self.assertEqual(len(set(chosen)), 4)
        nm = synth(1, 12, k=5, no_match=1.0)[0]
        chosen, target = build_group(nm, 3, rng, list(range(5)), n_hard=2)
        self.assertTrue({0, 1} <= set(chosen))
        self.assertEqual(target, 3)

    def test_build_group_without_hard_order_is_unchanged(self):
        """Random negatives use the same random stream as before hard negatives existed, so earlier results stay reproducible."""
        rec = synth(1, 13, k=5, no_match=0.0)[0]
        a, b = random.Random(5), random.Random(5)
        label = rec["target"]["label"]
        others = [j for j in range(5) if j != label]
        legacy = b.sample(others, 3) + [label]
        b.shuffle(legacy)
        chosen, _ = build_group(rec, 4, a)
        self.assertEqual(chosen, legacy)

    def test_hard_negative_training_and_orders(self):
        Tiny = _tiny_class()
        train = synth(48, 21, k=5)
        a = Tiny(negatives="hard", n_hard=2, save_dir=None, **TINY)
        orders = a._hard_orders([r for r in train if r["target"]["label"] >= 0])
        for o, r in zip(orders, [r for r in train if r["target"]["label"] >= 0]):
            self.assertEqual(sorted(o), list(range(5)))
            self.assertEqual(o[0], r["target"]["label"])  # the message overlaps its own topic most
        a.fit(train)
        self.assertEqual(a.info["negatives"], "hard")
        self.assertEqual(a.params["n_hard"], 2)
        with self.assertRaises(ValueError):
            Tiny(negatives="easy")

    def test_resume_after_interruption_matches_uninterrupted_training(self):
        Tiny = _tiny_class()
        train, val = synth(48, 31), synth(10, 32)

        class Interrupted(Tiny):
            def _after_step(self, step):
                if step == 3:
                    raise KeyboardInterrupt

        ref = Tiny(save_dir=None, **TINY)
        ref.fit(train)
        with tempfile.TemporaryDirectory() as tmp:
            first = Interrupted(save_dir=None, checkpoint_dir=tmp, checkpoint_every=2, **TINY)
            with self.assertRaises(KeyboardInterrupt):
                first.fit(train)
            self.assertTrue((Path(tmp) / "ckpt.pt").exists())
            second = Tiny(save_dir=None, checkpoint_dir=tmp, checkpoint_every=2, **TINY)
            second.fit(train)
            self.assertTrue(second.info["resumed"])
            self.assertEqual(second.info["steps"], ref.info["steps"])
            self.assertFalse((Path(tmp) / "ckpt.pt").exists())  # finished runs leave no checkpoint behind
            for x, y in zip(ref.score(val), second.score(val)):
                np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-4)

    def test_checkpoint_from_other_settings_is_ignored(self):
        Tiny = _tiny_class()
        train = synth(48, 33)

        class Stop(Tiny):
            def _after_step(self, step):
                if step == 3:
                    raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(KeyboardInterrupt):
                Stop(save_dir=None, checkpoint_dir=tmp, checkpoint_every=2, **TINY).fit(train)
            other = Tiny(save_dir=None, checkpoint_dir=tmp, checkpoint_every=2, **dict(TINY, lr=1e-3))
            other.fit(train)
            self.assertFalse(other.info["resumed"])

    def test_load_reproduces_saved_model_and_scoring_settings(self):
        Tiny = _tiny_class()
        train, val = synth(48, 41), synth(10, 42)
        with tempfile.TemporaryDirectory() as tmp:
            a = Tiny(save_dir=tmp, **TINY)
            a.fit(train)
            b = Tiny(
                max_len=999, max_context_turns=0, save_dir=None
            )  # constructor values must be overridden by the saved ones
            b.load(tmp)
            self.assertEqual((b.max_len, b.max_context_turns), (48, 4))
            for x, y in zip(a.score(val), b.score(val)):
                np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-4)
            self.assertEqual(b.info["train_examples"], 48)

    def test_validation_monitor_curve(self):
        Tiny = _tiny_class()
        train, val = synth(48, 51), synth(20, 52)
        a = Tiny(save_dir=None, monitor_examples=15, **TINY)
        self.assertTrue(a.needs_val)
        a.fit(train, val)
        curve = a.info["validation_curve"]
        self.assertEqual(curve[-1]["step"], a.info["steps"])
        self.assertEqual(len(curve), 4)
        self.assertTrue(all(0 <= c["top1"] <= 1 and 0 < c["mrr"] <= 1 for c in curve))
        self.assertFalse(Tiny(save_dir=None, **TINY).needs_val)


class TestEncoderPrefixes(unittest.TestCase):
    def test_sentence_encoder_applies_query_and_doc_prefixes(self):
        from pathwiser.scorers.encoder import SentenceEncoder

        seen: list[str] = []

        class Spy(FakeEncoder):
            def encode(self, texts, **kw):
                seen.extend(texts)
                return super().encode(texts, **kw)

        rec = synth(2, 5, k=3, no_match=0.0)
        plain = SentenceEncoder("m", "message")
        plain._model = FakeEncoder()
        enc = SentenceEncoder("m", "message", query_prefix="query: ", doc_prefix="passage: ")
        enc._model = Spy()
        out = enc.score(rec)
        self.assertTrue(seen and all(t.startswith(("query: ", "passage: ")) for t in seen))
        self.assertEqual(enc.params["query_prefix"], "query: ")
        self.assertNotIn("query_prefix", plain.params)  # default params are unchanged
        self.assertEqual([len(o) for o in out], [3, 3])

    def test_build_passes_prefixes_to_st_baselines_only(self):
        b = build("st_message", params={"query_prefix": "query: "})
        self.assertEqual(b.query_prefix, "query: ")
        self.assertEqual(build("st_lgbm", params={"doc_prefix": "passage: "}).doc_prefix, "passage: ")
        with self.assertRaises(ValueError):
            build("lexical", params={"query_prefix": "x"})


if __name__ == "__main__":
    unittest.main()
