"""Tests for the model-ready record export."""

from __future__ import annotations

import unittest

from pathwiser.benchmark.records import ModelDataConfig, Preprocessor, dataset_weights


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


class TestModelDataPreprocessing(unittest.TestCase):
    def test_normalisation_keeps_case_and_collapses_whitespace(self):
        p = Preprocessor(ModelDataConfig())
        self.assertEqual(p.norm("  Hello ​  World\n\n!  "), "Hello World !")

    def test_context_drops_oldest_turns_first_and_never_touches_message(self):
        p = Preprocessor(ModelDataConfig(max_context_words=8, max_context_turns=6))
        turns = [
            {"role": "user", "content": "one two three four five"},
            {"role": "assistant", "content": "six seven"},
            {"role": "user", "content": "eight nine"},
        ]
        kept = p.context(turns)
        self.assertEqual([t["content"] for t in kept], ["six seven", "eight nine"])
        self.assertEqual(p.stats["context_turns_dropped_examples"], 1)

    def test_oversized_newest_turn_keeps_its_tail(self):
        p = Preprocessor(ModelDataConfig(max_context_words=3))
        kept = p.context([{"role": "user", "content": "a b c d e f"}])
        self.assertEqual(kept[0]["content"], "d e f")

    def test_query_has_no_condition_ids_and_node_ablation_works(self):
        node = {"name": "order_routing", "description": "Pick an operation."}
        q = Preprocessor(ModelDataConfig()).query(node, [{"role": "user", "content": "hi"}], "cancel it")
        self.assertIn("Workflow node: order routing. Pick an operation.", q)
        self.assertTrue(q.endswith("Current user message: cancel it"))
        q2 = Preprocessor(ModelDataConfig(include_node=False)).query(node, [], "cancel it")
        self.assertEqual(q2, "Current user message: cancel it")

    def test_dataset_weights_average_to_one(self):
        from collections import Counter

        counts = Counter({("big", False): 900, ("big", True): 100, ("small", False): 90, ("small", True): 10})
        cfg = ModelDataConfig(dataset_weight_power=0.5, risky_label_weight=0.5)
        w = dataset_weights(counts, cfg)
        mean = sum(w[ds] * n * (0.5 if risky else 1.0) for (ds, risky), n in counts.items()) / sum(counts.values())
        self.assertAlmostEqual(mean, 1.0)
        self.assertGreater(w["small"], w["big"])
        flat = dataset_weights(counts, ModelDataConfig(dataset_weight_power=0.0, risky_label_weight=1.0))
        self.assertAlmostEqual(flat["big"], flat["small"])

    def test_config_rejects_nonsense(self):
        with self.assertRaises(ValueError):
            ModelDataConfig(max_k=0)
        with self.assertRaises(ValueError):
            ModelDataConfig(risky_label_weight=0)


if __name__ == "__main__":
    unittest.main()
