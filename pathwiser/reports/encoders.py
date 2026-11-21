"""Zero-shot encoder scan and the comparison of a candidate scorer against a reference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from pathwiser.decision.selection import leave_one_dataset_out
from pathwiser.decision.system import System
from pathwiser.evaluation.metrics import predictions, slice_groups, summarize, tune
from pathwiser.evaluation.stats import cluster_bootstrap
from pathwiser.reports.store import Store, ci, fmt, fmt_diff, write_json


def encoder_scan(store: Store, names: list[str], reference: str, extra: str, n_boot: int, out_dir: Path) -> dict:
    """Zero-shot comparison of pretrained encoders. Validation numbers are the selection evidence (overall, per-dataset macro and
    the unseen-service slice, which mimics a domain shift); the external set is a confirmation with paired scenario-bootstrap
    intervals. Nothing is fitted; test is not read."""
    val_recs = store.records("val")
    keep = np.array([not r["meta"]["is_paired"] for r in val_recs])
    ext = store.records(extra)
    clusters = [r["meta"]["dialogue_id"] for r in ext]
    rows, systems = {}, {}
    for n in names:
        if not (store.has(n, "val") and store.has(n, extra)):
            print(f"  skipping {n}: needs stored val and {extra} scores", file=sys.stderr)
            continue
        sc = store.scores(n, "val")
        vp = [p for p, k in zip(predictions(val_recs, sc), keep) if k]
        groups = slice_groups(vp)
        by_ds = {}
        for p in vp:
            by_ds.setdefault(p.dataset, []).append(p)
        macro = float(np.mean([summarize(v)["top1"] for v in by_ds.values() if summarize(v)["top1"] is not None]))
        tau = tune(vp)["tau"]
        ep = predictions(ext, store.scores(n, extra))
        abstain = np.array([p.top < tau for p in ep])
        systems[n] = (ep, abstain)
        rows[n] = {
            "val": {
                "top1": summarize(vp)["top1"],
                "mrr": summarize(vp)["mrr"],
                "macro_top1_over_datasets": macro,
                "top1_svc_unseen": summarize(groups["svc:unseen"])["top1"] if "svc:unseen" in groups else None,
                "top1_cond_unseen": summarize(groups["cond:unseen"])["top1"] if "cond:unseen" in groups else None,
            },
            "tau_val": tau,
        }
    boot = cluster_bootstrap(systems, clusters, reference=reference if reference in systems else None, n_boot=n_boot)
    for n in rows:
        rows[n][extra] = boot["systems"][n]
        rows[n][f"{extra}_vs_{reference}"] = boot["vs_reference"].get(n)
    result = {
        "set": extra,
        "reference": reference,
        "n_clusters": boot["n_clusters"],
        "n_boot": boot["n_boot"],
        "models": rows,
        "protocol": "zero-shot cosine of message and condition embeddings; no fitting; test not read",
    }
    write_json(out_dir / "zero_shot_scan.json", result)
    L = [
        "# Zero-shot encoder scan",
        "",
        f"{result['protocol']}. Validation = selection evidence; `{extra}` = confirmation "
        f"(95% scenario-bootstrap intervals, paired difference to `{reference}`).",
        "",
        f"| Encoder | val Top-1 | val macro Top-1 (datasets) | val Top-1 unseen service | {extra} Top-1 | Δ vs `{reference}` ({extra}) |",
        "|---|---:|---:|---:|---|---|",
    ]
    for n, v in rows.items():
        d = v[f"{extra}_vs_{reference}"]
        t = d["top1"] if d else None
        delta = "(reference)" if t is None else f"{fmt(t['diff'])} [{fmt(t['lo'])}, {fmt(t['hi'])}]"
        L.append(
            f"| `{n}` | {fmt(v['val']['top1'])} | {fmt(v['val']['macro_top1_over_datasets'])} | {fmt(v['val']['top1_svc_unseen'])} | "
            f"{ci(v[extra]['top1'])} | {delta} |"
        )
    (out_dir / "zero_shot_scan.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    return result


ADOPTION_RULE = {
    "statement": "A candidate (same model family, different frozen encoder) replaces the reference only if it does not lose on the benchmark "
    "validation split and gains on the external set. Fixed before the candidate was trained or scored.",
    "validation_not_worse": "leave-one-dataset-out macro decision accuracy >= reference - 1 standard error of the reference, "
    "AND validation Top-1 >= reference - 0.005",
    "external_gain": "paired external-set difference to the reference: Top-1 > 0 AND decision accuracy >= 0 (point estimates)",
    "confirmed_if": "the 95% scenario-bootstrap interval of the Top-1 difference excludes 0; otherwise the adoption is only tentative",
    "flag": "unseen-service validation Top-1 more than 0.02 below the reference is reported as a warning",
    "test": "not read for this comparison; read once, for the adopted model only",
}


def adoption_verdict(cand: dict, ref: dict, ext_top1: dict, ext_decision: dict) -> dict:
    """``cand`` / ``ref``: {top1, lodo_macro, lodo_se, svc_unseen_top1}; ``ext_*``: paired difference {diff, lo, hi}."""
    val_ok = (cand["lodo_macro"] >= ref["lodo_macro"] - ref["lodo_se"]) and (cand["top1"] >= ref["top1"] - 0.005)
    ext_ok = (
        ext_top1["diff"] is not None
        and ext_top1["diff"] > 0
        and ext_decision["diff"] is not None
        and ext_decision["diff"] >= 0
    )
    confirmed = bool(ext_ok and ext_top1["lo"] is not None and ext_top1["lo"] > 0)
    warn = (
        cand["svc_unseen_top1"] is not None
        and ref["svc_unseen_top1"] is not None
        and cand["svc_unseen_top1"] < ref["svc_unseen_top1"] - 0.02
    )
    return {
        "validation_not_worse": bool(val_ok),
        "external_gain": bool(ext_ok),
        "adopt": bool(val_ok and ext_ok),
        "confidence": ("confirmed" if confirmed else "tentative") if (val_ok and ext_ok) else None,
        "unseen_service_warning": bool(warn),
    }


def compare_candidate(store: Store, candidate: str, reference: str, extra: str, n_boot: int, out_dir: Path) -> dict:
    """Validation (selection evidence) and external set (confirmation) for a candidate scorer against a reference scorer."""
    val_recs, val_sc = store.main("val", [candidate, reference])
    stats = {}
    for m in (candidate, reference):
        preds = predictions(val_recs, val_sc[m])
        groups = slice_groups(preds)
        lodo = leave_one_dataset_out(System((m,)), val_recs, {m: val_sc[m]})
        stats[m] = {
            "top1": summarize(preds)["top1"],
            "mrr": summarize(preds)["mrr"],
            "lodo_macro": lodo["macro_decision_accuracy"],
            "lodo_se": lodo["se"],
            "lodo_per_dataset": {d: v["decision_accuracy"] for d, v in lodo["per_dataset"].items()},
            "svc_unseen_top1": summarize(groups["svc:unseen"])["top1"] if "svc:unseen" in groups else None,
            "tau_val": tune(preds)["tau"],
        }
    ext = store.records(extra)
    systems = {}
    for m in (candidate, reference):
        ep = predictions(ext, store.scores(m, extra))
        systems[m] = (ep, np.array([p.top < stats[m]["tau_val"] for p in ep]))
    boot = cluster_bootstrap(systems, [r["meta"]["dialogue_id"] for r in ext], reference=reference, n_boot=n_boot)
    diff = boot["vs_reference"][candidate]
    verdict = adoption_verdict(stats[candidate], stats[reference], diff["top1"], diff["decision_accuracy"])
    result = {
        "candidate": candidate,
        "reference": reference,
        "rule": ADOPTION_RULE,
        "validation": stats,
        extra: boot["systems"],
        f"{extra}_difference_to_reference": diff,
        "verdict": verdict,
    }
    write_json(out_dir / f"compare_{candidate}_vs_{reference}.json", result)
    c, r = stats[candidate], stats[reference]
    L = [f"# `{candidate}` vs `{reference}`", "", "Rule fixed before the candidate was trained:", ""]
    L += [f"- {k}: {v}" for k, v in ADOPTION_RULE.items() if k != "statement"]
    L += [
        "",
        ADOPTION_RULE["statement"],
        "",
        "## Validation (benchmark)",
        "",
        "| | Top-1 | MRR | unseen-service Top-1 | LODO macro decision acc. (s.e.) |",
        "|---|---:|---:|---:|---:|",
    ]
    for m, v in ((candidate, c), (reference, r)):
        L.append(
            f"| `{m}` | {fmt(v['top1'])} | {fmt(v['mrr'])} | {fmt(v['svc_unseen_top1'])} | {fmt(v['lodo_macro'])} ({fmt(v['lodo_se'])}) |"
        )
    L += [
        "",
        f"## External set `{extra}` (95% scenario-bootstrap intervals; threshold from validation)",
        "",
        "| | Top-1 | Decision acc. | False routing |",
        "|---|---|---|---|",
    ]
    for m in (candidate, reference):
        b = boot["systems"][m]
        L.append(f"| `{m}` | {ci(b['top1'])} | {ci(b['decision_accuracy'])} | {ci(b['false_routing_rate'])} |")
    L += [
        "",
        f"Paired difference (candidate minus reference): Top-1 {fmt_diff(diff['top1'])}; decision accuracy {fmt_diff(diff['decision_accuracy'])}; "
        f"false routing {fmt_diff(diff['false_routing_rate'])}.",
        "",
        "## Verdict",
        "",
        f"- validation not worse: **{verdict['validation_not_worse']}**",
        f"- external gain: **{verdict['external_gain']}**",
        f"- adopt: **{verdict['adopt']}**" + (f" ({verdict['confidence']})" if verdict["confidence"] else ""),
        f"- unseen-service warning: {verdict['unseen_service_warning']}",
    ]
    (out_dir / f"compare_{candidate}_vs_{reference}.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    return result
