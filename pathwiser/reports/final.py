"""Evaluation of the pre-selected decision system on test and on an external set."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pathwiser.decision.system import Pipeline, System
from pathwiser.evaluation.metrics import Prediction, summarize_decisions
from pathwiser.evaluation.stats import brier, cluster_bootstrap, ece
from pathwiser.reports.selection import system_from_label
from pathwiser.reports.store import GENERALISATION, Store, ci, fmt, fmt_diff, slice_metrics, write_json


def _calibration(probability: np.ndarray | None, preds: list[Prediction]) -> dict | None:
    """Does P(routing to the best candidate is correct) match how often it is? (ECE and Brier score.)"""
    if probability is None:
        return None
    y = np.array([p.label >= 0 and p.arg == p.label for p in preds], dtype=float)
    return {"ece": ece(probability, y), "brier": brier(probability, y)}


def _paraphrase(dec) -> dict | None:
    """SGD-X twins: same pick, and same decision (both abstain, or both route to the same candidate)."""
    by_id = {p.example_id: i for i, p in enumerate(dec.preds)}
    pairs = [(by_id[p.paired_with], i) for i, p in enumerate(dec.preds) if p.paired and p.paired_with in by_id]
    if not pairs:
        return None

    def same_decision(a: int, b: int) -> bool:
        if dec.abstain[a] or dec.abstain[b]:
            return bool(dec.abstain[a] and dec.abstain[b])
        return dec.preds[a].arg == dec.preds[b].arg

    return {
        "n_pairs": len(pairs),
        "same_selected_candidate": float(np.mean([dec.preds[a].arg == dec.preds[b].arg for a, b in pairs])),
        "same_decision": float(np.mean([same_decision(a, b) for a, b in pairs])),
    }


def final_report(
    store: Store, extra: str | None, n_boot: int, out_dir: Path, log=print, selection_dir: Path | None = None
) -> dict:
    """``selection_dir``: where ``selection.json`` is read from (default ``out_dir``), so another external set can be reported elsewhere."""
    selection = json.loads(((selection_dir or out_dir) / "selection.json").read_text(encoding="utf-8"))
    models = selection["models_with_validation_scores"]
    chosen = system_from_label(selection["chosen"], models)
    refs = [System((m,), "tau") for m in ("st_message", "st_lgbm", "cross_encoder") if m in models]
    systems = [chosen] + [r for r in refs if r.label != chosen.label]
    unguarded = selection.get(
        "choice_without_guard"
    )  # what the rule would have chosen without the guard: reported, not chosen
    if unguarded and unguarded != chosen.label and all(unguarded != s.label for s in systems):
        systems.append(system_from_label(unguarded, models))
    needed = sorted({m for s in systems for m in s.models})
    val_recs, val_scores = store.main("val", needed)
    fitted = {s.label: Pipeline(s).fit(val_recs, {m: val_scores[m] for m in s.models}) for s in systems}
    result = {"selection": selection, "chosen": chosen.label, "systems": {}, "external": {}}
    test_main: dict = {}
    for split in ["test"] + ([extra] if extra else []):
        recs = store.records(split)
        by_model = {m: store.scores(m, split) for m in needed}
        decisions = {s.label: fitted[s.label].decide(recs, {m: by_model[m] for m in s.models}) for s in systems}
        main_idx = np.array([not r["meta"]["is_paired"] for r in recs])
        block = {}
        for label, dec in decisions.items():
            main_preds = [p for p, keep in zip(dec.preds, main_idx) if keep]
            main_abs = dec.abstain[main_idx]
            block[label] = {
                "overall": summarize_decisions(main_preds, main_abs),
                "slices": slice_metrics(main_preds, main_abs),
                "calibration": _calibration(None if dec.probability is None else dec.probability[main_idx], main_preds),
                "paraphrase": _paraphrase(dec),
            }
            if split == "test":
                result["systems"].setdefault(label, {})["test"] = block[label]
                test_main[label] = (main_preds, main_abs)
                log(
                    f"  test  {label:50s} top1={block[label]['overall']['top1']} dec={block[label]['overall']['decision_accuracy']} "
                    f"false_routing={block[label]['overall']['false_routing_rate']} nm_f1={block[label]['overall']['no_match_f1']}"
                )
        if split == "test":
            # fewer resamples than the small external set: ~30k dialogue clusters make each resample large
            main_clusters = [r["meta"]["dialogue_id"] for r, keep in zip(recs, main_idx) if keep]
            reference = (
                System(("st_lgbm",), "tau").label if System(("st_lgbm",), "tau").label in test_main else chosen.label
            )
            tb = cluster_bootstrap(test_main, main_clusters, reference=reference, n_boot=min(n_boot, 400))
            result["test_bootstrap"] = {
                "reference": reference,
                **{k: tb[k] for k in ("n_clusters", "n_boot", "level", "systems", "vs_reference")},
            }
        if split != "test":
            clusters = [r["meta"]["dialogue_id"] for r in recs]
            reference = (
                System(("st_lgbm",), "tau").label if System(("st_lgbm",), "tau").label in decisions else refs[0].label
            )
            boot = cluster_bootstrap(
                {l: (d.preds, d.abstain) for l, d in decisions.items()}, clusters, reference=reference, n_boot=n_boot
            )
            result["external"].setdefault(split, {})["reference"] = reference
            result["external"][split].update(
                {
                    "systems": block,
                    "bootstrap": {k: boot[k] for k in ("n_clusters", "n_boot", "level", "systems", "vs_reference")},
                }
            )
            for label in decisions:
                o = block[label]["overall"]
                log(
                    f"  {split}  {label:50s} top1={o['top1']} dec={o['decision_accuracy']} false_routing={o['false_routing_rate']} nm_f1={o['no_match_f1']}"
                )
    write_json(out_dir / "final.json", result)
    (out_dir / "summary.md").write_text(render_final(result), encoding="utf-8", newline="\n")
    return result


def render_final(r: dict) -> str:
    def cell(label):  # labels contain '|'; escape it inside table cells
        return f"`{label}`".replace("|", "\\|")

    sel = r["selection"]
    L = [
        "# Decision layer: results",
        "",
        f"Chosen on validation only (leave-one-dataset-out): **`{r['chosen']}`**. Rule: {sel['rule']}. "
        f"{sel['n_candidates']} candidates were compared; {sel['caveat']}.",
        "",
        "## Test (benchmark)",
        "",
        "| System | Top-1 | Decision acc. | no_match F1 | False routing | Abstain rate | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, v in r["systems"].items():
        o = v["test"]["overall"]
        cal = v["test"]["calibration"]
        L.append(
            f"| {cell(label)}{' (chosen)' if label == r['chosen'] else ''} | {fmt(o['top1'])} | {fmt(o['decision_accuracy'])} | "
            f"{fmt(o['no_match_f1'])} | {fmt(o['false_routing_rate'])} | {fmt(o['abstain_rate'])} | {fmt(cal['ece']) if cal else '–'} |"
        )
    gen = [g for g in GENERALISATION if any(g in v["test"]["slices"] for v in r["systems"].values())]
    if gen:
        L += [
            "",
            "Generalisation slices on test (Top-1 / decision accuracy; `cond:unseen` = no equivalent of the gold condition in train, "
            "`svc:unseen` = service not in train):",
            "",
            "| System | " + " | ".join(gen) + " |",
            "|---|" + "---:|" * len(gen),
        ]
        for label, v in r["systems"].items():
            sl = v["test"]["slices"]
            L.append(
                f"| {cell(label)} | "
                + " | ".join(
                    f"{fmt(sl[g]['top1'])} / {fmt(sl[g]['decision_accuracy'])}" if g in sl else "–" for g in gen
                )
                + " |"
            )
    tb = r.get("test_bootstrap")
    if tb and tb["vs_reference"]:
        L += [
            "",
            f"Paired differences on test to `{tb['reference']}` (95% bootstrap over {tb['n_clusters']} dialogues/examples, {tb['n_boot']} resamples):",
            "",
            "| System | Δ Top-1 | Δ Decision acc. | Δ False routing |",
            "|---|---|---|---|",
        ]
        for label, d in tb["vs_reference"].items():
            L.append(
                f"| {cell(label)} | {fmt_diff(d['top1'])} | {fmt_diff(d['decision_accuracy'])} | {fmt_diff(d['false_routing_rate'])} |"
            )
    for name, ext in r["external"].items():
        L += [
            "",
            f"## External set `{name}` (95% scenario-bootstrap intervals)",
            "",
            "| System | Top-1 | Decision acc. | False routing | Abstain rate | no_match recall |",
            "|---|---|---|---|---|---|",
        ]
        for label, s in ext["bootstrap"]["systems"].items():
            L.append(
                f"| {cell(label)}{' (chosen)' if label == r['chosen'] else ''} | {ci(s['top1'])} | {ci(s['decision_accuracy'])} | "
                f"{ci(s['false_routing_rate'])} | {ci(s['abstain_rate'])} | {ci(s['no_match_recall'])} |"
            )
        if ext["bootstrap"]["vs_reference"]:
            L += [
                "",
                f"Paired difference to the reference system `{ext['reference']}`:",
                "",
                "| System | Δ Top-1 | Δ Decision acc. | Δ False routing |",
                "|---|---|---|---|",
            ]
            for label, d in ext["bootstrap"]["vs_reference"].items():
                L.append(
                    f"| {cell(label)} | {fmt_diff(d['top1'])} | {fmt_diff(d['decision_accuracy'])} | {fmt_diff(d['false_routing_rate'])} |"
                )
    return "\n".join(L) + "\n"
