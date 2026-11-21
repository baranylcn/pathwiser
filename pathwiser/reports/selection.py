"""Choice of a decision system on the validation split."""

from __future__ import annotations

from pathlib import Path

from pathwiser.decision.selection import leave_one_dataset_out, select_system
from pathwiser.decision.system import System
from pathwiser.reports.store import Store, write_json


def candidate_systems(models: list[str]) -> list[System]:
    singles = [m for m in ("st_message", "st_lr", "st_lgbm", "cross_encoder") if m in models]
    # trained variants of st_lgbm that use another frozen encoder are stored as ``st_lgbm__<tag>``
    variants = sorted(m for m in models if m.startswith("st_lgbm__"))
    out = [System((m,), a) for m in singles + variants for a in ("tau", "relative", "full")]
    fusions = []
    cheap = tuple(m for m in ("st_lgbm", "st_lr", "st_message", "tfidf_lr", "lexical") if m in models)
    if len(cheap) > 1:
        fusions.append(cheap)
    if "st_lgbm" in models and "st_message" in models:
        fusions.append(("st_lgbm", "st_message"))
    if "st_lgbm" in models:
        fusions += [("st_lgbm", v) for v in variants]
    if "cross_encoder" in models and "st_lgbm" in models:
        fusions.append(("st_lgbm", "cross_encoder"))
    if "cross_encoder" in models:
        fusions.append(cheap + ("cross_encoder",))
    seen = set()
    for f in fusions:
        if f in seen:
            continue
        seen.add(f)
        out += [System(f, a, fuse=True) for a in ("tau", "relative", "full")]
    return out


def dev_selection(store: Store, models: list[str], out_dir: Path, log=print) -> dict:
    models = [m for m in models if store.has(m, "val")]
    recs, by_model = store.main("val", models)
    systems = candidate_systems(models)
    results = []
    for s in systems:
        r = leave_one_dataset_out(s, recs, by_model)
        results.append(r)
        log(
            f"  {s.label:60s} macro decision acc={r['macro_decision_accuracy']:.4f} (se {r['se']:.4f}) "
            f"pooled top1={r['pooled']['top1']} false routing={r['pooled']['false_routing_rate']}"
        )
    selection = select_system(results)
    selection["models_with_validation_scores"] = models
    selection["n_candidates"] = len(results)
    selection["caveat"] = (
        "the maximum over many candidates is optimistic; the selection itself is computed from validation only, but the "
        "generalisation guard and the candidate list were chosen after the external set had been looked at "
        "(docs/decision-layer.md); the test split was not used"
    )
    write_json(out_dir / "lodo.json", {"selection": selection, "results": results})
    write_json(out_dir / "selection.json", selection)
    return selection


def system_from_label(label: str, models: list[str]) -> System:
    for s in candidate_systems(models):
        if s.label == label:
            return s
    raise KeyError(f"system {label!r} is not a candidate for models {models}")
