"""Diagnostic: is a weak decision accuracy on an external set caused by the threshold or by the ranking?"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pathwiser.evaluation.metrics import predictions, summarize_decisions, tune
from pathwiser.reports.store import Store, write_json


def threshold_diagnostic(store: Store, model: str, extra: str, out_dir: Path) -> dict:
    """Is a poor decision accuracy on an external set caused by the threshold or by the ranking?

    DIAGNOSTIC ONLY: the oracle threshold is tuned on the external set itself, which is exactly what must never be done for
    a real result. It bounds how much any threshold-based decision layer could gain there. Nothing is selected from it.
    """
    val_recs, val_sc = store.main("val", [model])
    ext = store.records(extra)
    ext_pred = predictions(ext, store.scores(model, extra))
    val_pred = predictions(val_recs, val_sc[model])
    tau = tune(val_pred)["tau"]
    oracle = tune(ext_pred)
    never = summarize_decisions(ext_pred, np.zeros(len(ext_pred), dtype=bool))

    def medians(preds):
        grp = {
            "correct_route": [p.top for p in preds if p.label >= 0 and p.arg == p.label],
            "wrong_route": [p.top for p in preds if p.label >= 0 and p.arg != p.label],
            "no_match": [p.top for p in preds if p.label < 0],
        }
        return {k: float(np.median(v)) if v else None for k, v in grp.items()}

    result = {
        "model": model,
        "set": extra,
        "validation_tau": {"tau": tau, **summarize_decisions(ext_pred, np.array([p.top < tau for p in ext_pred]))},
        "oracle_tau_on_external_set": {
            "tau": oracle["tau"],
            **summarize_decisions(ext_pred, np.array([p.top < oracle["tau"] for p in ext_pred])),
        },
        "never_abstain": never,
        "median_best_score_by_outcome": {"validation": medians(val_pred), extra: medians(ext_pred)},
        "note": "oracle tau is tuned on the external set: an upper bound for threshold policies, never a result",
    }
    write_json(out_dir / f"threshold_diagnostic_{model}_{extra}.json", result)
    return result
