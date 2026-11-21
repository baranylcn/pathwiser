"""Metrics for dynamic candidate selection. Model-agnostic: works on per-example score vectors.

Two layers:
* ranking (needs no threshold): Top-1 / Top-3 / MRR over examples that have a gold condition,
* decision (needs a threshold): decision accuracy, no_match precision/recall, false routing rate, ...

Conventions
* Ties are broken by candidate position. Positions are shuffled in the data, so this is unbiased.
* ``false_routing_rate`` = (routed to a wrong condition + routed although no condition applies) / all examples.
* Metrics that are undefined for a group (e.g. ranking on a group with no route examples) are ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pathwiser.decision.threshold import tune_threshold

SLICE_PREFIXES = (
    "dataset",
    "tier",
    "label",
    "cond",
    "svc",
    "ctx",
    "msg",
    "k",
    "nm_kind",
    "risk",
    "desc",
    "seen_exact",
    "persona",
    "style_split",
    "noise",
    "tag",
    "archetype",
    "cstyle",
    "group",
)


@dataclass(frozen=True)
class Prediction:
    example_id: str
    dataset: str
    label: int  # gold index, or -1 for no_match
    k: int
    top: float  # best score
    arg: int  # index of the best candidate (first on ties)
    rank: int | None  # 1-based rank of the gold candidate (None for no_match)
    tags: tuple[str, ...]  # slice tags such as "cond:unseen"
    paired: bool
    paired_with: str | None


def predictions(records: list[dict], scores: list[np.ndarray]) -> list[Prediction]:
    if len(records) != len(scores):
        raise ValueError("records and scores must be aligned")
    out = []
    for r, s in zip(records, scores):
        s = np.asarray(s, dtype=float)
        k = len(r["input"]["conditions"])
        if s.shape != (k,):
            raise ValueError(f"{r['example_id']}: expected {k} scores, got shape {s.shape}")
        if not np.isfinite(s).all():
            raise ValueError(f"{r['example_id']}: non-finite score")
        label = r["target"]["label"]
        order = np.argsort(-s, kind="stable")
        rank = int(np.nonzero(order == label)[0][0]) + 1 if label >= 0 else None
        meta = r["meta"]
        tags = list(meta["slices"])
        if "cond:seen" in tags:
            tags.append("seen_exact:exact" if meta.get("gold_seen_exact") else "seen_exact:equivalent_only")
        out.append(
            Prediction(
                r["example_id"],
                meta["dataset"],
                label,
                k,
                float(s[order[0]]),
                int(order[0]),
                rank,
                tuple(tags),
                meta["is_paired"],
                meta["paired_with"],
            )
        )
    return out


def _r(x: float | None) -> float | None:
    return None if x is None else round(float(x), 4)


def _tau_for(p: Prediction, tau: float | dict | None) -> float | None:
    return tau.get(p.dataset) if isinstance(tau, dict) else tau


def summarize(preds: list[Prediction], tau: float | dict | None = None) -> dict:
    """Ranking metrics, plus decision metrics when ``tau`` is given (a float, or a dataset -> float mapping)."""
    route = [p for p in preds if p.label >= 0]
    out: dict = {"n": len(preds), "n_route": len(route), "n_no_match": len(preds) - len(route)}
    if route:
        out["top1"] = _r(np.mean([p.rank == 1 for p in route]))
        out["top3"] = _r(np.mean([p.rank <= 3 for p in route]))
        out["mrr"] = _r(np.mean([1 / p.rank for p in route]))
        out["top1_chance"] = _r(np.mean([1 / p.k for p in route]))
    else:
        out.update(top1=None, top3=None, mrr=None, top1_chance=None)
    if tau is None:
        return out
    return _decision_block(preds, [p.top < _tau_for(p, tau) for p in preds], out)


def summarize_decisions(preds: list[Prediction], abstain) -> dict:
    """Like ``summarize`` but the route / no_match decision of every example is given (any decision policy), not a threshold."""
    return _decision_block(preds, list(abstain), summarize(preds))


def _decision_block(preds: list[Prediction], abstain_flags, out: dict) -> dict:
    if len(abstain_flags) != len(preds):
        raise ValueError("one abstain decision per prediction is required")
    route = [p for p in preds if p.label >= 0]
    ok = wrong = false_abstain = nm_abstain = nm_routed = 0
    for p, abstain in zip(preds, abstain_flags):
        if p.label >= 0:
            if abstain:
                false_abstain += 1
            elif p.arg == p.label:
                ok += 1
            else:
                wrong += 1
        elif abstain:
            nm_abstain += 1
        else:
            nm_routed += 1
    n, n_route, n_nm = len(preds), len(route), out["n_no_match"]
    pred_nm = nm_abstain + false_abstain
    prec = nm_abstain / pred_nm if pred_nm else None
    rec = nm_abstain / n_nm if n_nm else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec is not None and rec is not None else None)
    out.update(
        decision_accuracy=_r((ok + nm_abstain) / n),
        route_accuracy=_r(ok / n_route) if n_route else None,  # routed to the gold condition
        wrong_route_rate=_r(wrong / n_route) if n_route else None,  # routed to another condition
        false_abstain_rate=_r(false_abstain / n_route) if n_route else None,
        no_match_precision=_r(prec),
        no_match_recall=_r(rec),
        no_match_f1=_r(f1),
        false_routing_rate=_r((wrong + nm_routed) / n),
        abstain_rate=_r(pred_nm / n),
    )
    return out


def tune(preds: list[Prediction]) -> dict:
    """Decision threshold fitted on ``preds`` (use validation main examples only)."""
    return tune_threshold(
        np.array([p.top for p in preds]),
        np.array([p.label >= 0 and p.arg == p.label for p in preds]),
        np.array([p.label < 0 for p in preds]),
    )


def paraphrase_consistency(preds: list[Prediction], tau: float | dict | None) -> dict | None:
    """SGD-X twins: same moment, held-out paraphrase of the condition descriptions."""
    by_id = {p.example_id: p for p in preds}
    pairs = [(by_id[p.paired_with], p) for p in preds if p.paired and p.paired_with in by_id]
    if not pairs:
        return None
    same_pick = [m.arg == t.arg for m, t in pairs]
    out = {"n_pairs": len(pairs), "same_selected_candidate": _r(np.mean(same_pick))}
    rp = [(m, t) for m, t in pairs if m.label >= 0]
    if rp:
        out["top1_main"] = _r(np.mean([m.rank == 1 for m, _ in rp]))
        out["top1_paraphrase"] = _r(np.mean([t.rank == 1 for _, t in rp]))
    if tau is not None:
        out["same_abstain_decision"] = _r(
            np.mean([(m.top < _tau_for(m, tau)) == (t.top < _tau_for(t, tau)) for m, t in pairs])
        )
    return out


def slice_groups(preds: list[Prediction]) -> dict[str, list[Prediction]]:
    groups: dict[str, list[Prediction]] = {}
    for p in preds:
        for tag in p.tags:
            if tag.split(":", 1)[0] in SLICE_PREFIXES:
                groups.setdefault(tag, []).append(p)
    return dict(sorted(groups.items()))


def evaluate_split(
    records: list[dict], scores: list[np.ndarray], tau: float | None, tau_by_dataset: dict[str, float] | None = None
) -> dict:
    """Full report of one split. Main examples and SGD-X paraphrase twins are reported separately."""
    preds = predictions(records, scores)
    main = [p for p in preds if not p.paired]
    twins = [p for p in preds if p.paired]
    by_ds: dict[str, list[Prediction]] = {}
    for p in main:
        by_ds.setdefault(p.dataset, []).append(p)
    report = {
        "overall": summarize(main, tau),
        "datasets": {ds: summarize(v, tau) for ds, v in sorted(by_ds.items())},
        "slices": {name: summarize(v, tau) for name, v in slice_groups(main).items()},
        "paired_twins": summarize(twins, tau) if twins else None,
        "paraphrase_consistency": paraphrase_consistency(preds, tau),
    }
    if tau_by_dataset:
        report["overall_dataset_thresholds"] = summarize(main, tau_by_dataset)
        report["datasets_own_threshold"] = {ds: summarize(v, tau_by_dataset) for ds, v in sorted(by_ds.items())}
    return report
