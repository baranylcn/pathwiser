"""Tests for the baseline summary rendering."""

from __future__ import annotations

import unittest


class TestSummaryRendering(unittest.TestCase):
    def test_cost_table_is_not_overwritten_by_training_table(self):
        from pathwiser.reports.baselines import render_summary

        split = {"overall": {}, "datasets": {"d": {}}, "slices": {}, "paraphrase_consistency": None}
        base = {
            "data": {"version": "v", "train_sha256": "a" * 64, "val_sha256": "b" * 64, "test_sha256": "c" * 64},
            "splits": {"val": split},
            "latency_ms_per_example": {"val": 1.0},
            "threshold": {"global": 0.5, "val_decision_accuracy_at_global": 0.7},
            "fit_seconds": 3.0,
        }
        results = [
            dict(base, baseline="plain", params={}),
            dict(
                base,
                baseline="learned",
                kind="trained",
                params={"embedding_model": "m"},
                training={"train_examples": 10, "train_pairs": 40, "steps": 2, "train_seconds": 60, "final_loss": 0.5},
            ),
        ]
        text = render_summary(results)
        cost = text.split("## Thresholds, fitting cost, size")[1]
        self.assertIn("| `plain` | 0.5000 |", cost)
        self.assertIn("| `learned` | 0.5000 |", cost)
        self.assertNotIn("Train pairs", cost)


if __name__ == "__main__":
    unittest.main()
