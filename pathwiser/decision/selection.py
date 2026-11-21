"""Selection of a decision system by leave-one-dataset-out cross-validation on the validation split."""

from __future__ import annotations

import math

import numpy as np

from pathwiser.decision.system import Pipeline, System, align
from pathwiser.evaluation.metrics import Prediction, slice_groups, summarize, summarize_decisions

GUARD_SLICES = ("svc:unseen", "seen_exact:equivalent_only", "cond:unseen")  # the benchmark's own generalisation slices


GUARD_TOLERANCE = 0.02


def leave_one_dataset_out(
    system: System,
    records: list[dict],
    by_model: dict[str, list[np.ndarray]],
    fusion_C: float = 1.0,
    abstention_C: float = 1.0,
) -> dict:
    """Fit on all datasets but one, decide on the held-out one; repeat for every dataset (validation main examples only)."""
    datasets = np.array([r["meta"]["dataset"] for r in records])
    per_dataset: dict[str, dict] = {}
    pooled_preds: list[Prediction] = []
    pooled_abstain: list[bool] = []
    for held in sorted(set(datasets)):
        tr_recs, tr_scores = align(records, by_model, datasets != held)
        te_recs, te_scores = align(records, by_model, datasets == held)
        dec = Pipeline(system, fusion_C, abstention_C).fit(tr_recs, tr_scores).decide(te_recs, te_scores)
        per_dataset[held] = summarize_decisions(dec.preds, dec.abstain)
        pooled_preds += dec.preds
        pooled_abstain += list(dec.abstain)
    acc = np.array([m["decision_accuracy"] for m in per_dataset.values()])
    groups = slice_groups(pooled_preds)
    slice_top1 = {tag: summarize(groups[tag])["top1"] for tag in GUARD_SLICES if tag in groups}
    return {
        "system": system.label,
        "complexity": system.complexity,
        "per_dataset": per_dataset,
        "slice_top1": slice_top1,
        "macro_decision_accuracy": float(acc.mean()),
        "se": float(acc.std(ddof=1) / math.sqrt(len(acc))) if len(acc) > 1 else 0.0,
        "macro_top1": float(np.mean([m["top1"] for m in per_dataset.values() if m["top1"] is not None])),
        "pooled": summarize_decisions(pooled_preds, pooled_abstain),
    }


def select_system(results: list[dict], guard: str = "svc:unseen", tolerance: float = GUARD_TOLERANCE) -> dict:
    """Choose one system from leave-one-dataset-out results (validation only).

    Systems within one standard error of the best macro decision accuracy stay. Of those, systems whose out-of-fold Top-1 on ``guard``
    is more than ``tolerance`` below the best value over all candidates are dropped, unless that would drop all of them. The least
    complex remaining system wins; ties go to the higher accuracy.
    """
    best = max(results, key=lambda r: r["macro_decision_accuracy"])
    near = [r for r in results if r["macro_decision_accuracy"] >= best["macro_decision_accuracy"] - best["se"]]
    values = [r["slice_top1"][guard] for r in results if guard in r.get("slice_top1", {})]
    top_guard = max(values) if values else None
    kept = [
        r
        for r in near
        if top_guard is None or guard not in r.get("slice_top1", {}) or r["slice_top1"][guard] >= top_guard - tolerance
    ]
    applied = bool(kept) and len(kept) < len(near)
    pool = kept or near
    chosen = min(pool, key=lambda r: (r["complexity"], -r["macro_decision_accuracy"]))
    unguarded = min(near, key=lambda r: (r["complexity"], -r["macro_decision_accuracy"]))
    return {
        "chosen": chosen["system"],
        "best": best["system"],
        "within_one_se": [r["system"] for r in near],
        "excluded_by_guard": [r["system"] for r in near if r not in pool] if applied else [],
        "choice_without_guard": unguarded["system"],
        "guard": {"slice": guard, "tolerance": tolerance, "best_value": top_guard},
        "rule": "highest macro decision accuracy under leave-one-dataset-out; among systems within one standard error of the best, drop those "
        f"whose out-of-fold Top-1 on {guard} is more than {tolerance} below the best candidate's; of the rest, the least complex",
    }
