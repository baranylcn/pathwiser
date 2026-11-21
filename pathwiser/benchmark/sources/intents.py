"""Single-turn intent datasets: Banking77, CLINC150 (+ out-of-scope), HWU64.

All three yield moments with an empty context (``context_origin='none'``): no context is fabricated.
Condition descriptions come from data/curated/*.json (authored for this project, flagged unreviewed).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from pathwiser.benchmark.fileio import read_json
from pathwiser.benchmark.schema import Condition, DropLog, Moment, PipelineError, SourceData
from pathwiser.benchmark.seeding import stable_unit
from pathwiser.benchmark.sources.curated import load_curated
from pathwiser.benchmark.text import clean_text


def _condition(ds: str, label: str, family: str, desc: str, meta: dict) -> Condition:
    uid = f"{ds}:{label}"
    return Condition(
        uid=uid,
        dataset=ds,
        family=family,
        group=uid,
        descriptions={"orig": desc.strip()},
        description_origin=meta["description_origin"],
        reviewed=bool(meta["reviewed"]),
        extra={"label": label},
    )


def _moment(
    ds: str, mid: str, group: str, split: str, text: str, gold: str | None, extra: dict | None = None
) -> Moment:
    return Moment(
        moment_id=mid,
        dataset=ds,
        group_id=group,
        base_split=split,
        context=[],
        message=text,
        gold_uid=gold,
        turn_idx=None,
        is_first_user_turn=True,
        n_context_total=0,
        context_origin="none",
        extra=extra or {},
    )


def load_banking77(cfg: dict, raw: Path, curated: Path, drops: DropLog, limit: int | None) -> SourceData:
    dcfg = cfg["datasets"]["banking77"]
    root = raw / "banking77" / "banking_data"
    labels = set(read_json(root / "categories.json"))
    meta, cur = load_curated(curated / "banking77_conditions.json", labels, "banking77")
    conds = {}
    for label in sorted(labels):
        fam = cur[label].get("family")
        if not fam:
            raise PipelineError(f"banking77: '{label}' has no family in the curated file")
        conds[f"banking77:{label}"] = _condition("banking77", label, fam, cur[label]["description"], meta)
    moments: list[Moment] = []
    for official in ("train", "test"):
        with open(root / f"{official}.csv", encoding="utf-8", newline="") as f:
            for idx, row in enumerate(csv.DictReader(f)):
                if limit is not None and idx >= limit:
                    break
                text, label = clean_text(row["text"]), row["category"]
                if f"banking77:{label}" not in conds:
                    raise PipelineError(f"banking77: unknown label '{label}'")
                if official == "train":
                    split = "val" if stable_unit(cfg["seed"], "b77-val", idx) < dcfg["val_fraction"] else "train"
                else:
                    split = "test"
                mid = f"banking77:{official}:{idx}"
                moments.append(_moment("banking77", mid, mid, split, text, f"banking77:{label}"))
    return SourceData("banking77", "T2", cfg["licenses"]["banking77"], conds, moments, "partition_routed", True)


def load_clinc150(cfg: dict, raw: Path, curated: Path, drops: DropLog, limit: int | None) -> SourceData:
    root = raw / "clinc150" / "data"
    domains = read_json(root / "domains.json")
    label_domain = {lab: dom for dom, labs in domains.items() for lab in labs}
    meta, cur = load_curated(curated / "clinc150_conditions.json", set(label_domain), "clinc150")
    conds = {
        f"clinc150:{lab}": _condition("clinc150", lab, label_domain[lab], cur[lab]["description"], meta)
        for lab in sorted(label_domain)
    }
    data = read_json(root / "data_full.json")
    moments: list[Moment] = []
    plan = (
        ("train", "train", False),
        ("val", "val", False),
        ("test", "test", False),
        ("oos_train", "train", True),
        ("oos_val", "val", True),
        ("oos_test", "test", True),
    )
    for key, split, oos in plan:
        for idx, (text, label) in enumerate(data[key]):
            if limit is not None and idx >= limit:
                break
            if oos != (label == "oos"):
                raise PipelineError(f"clinc150: label '{label}' inconsistent with split key '{key}'")
            if not oos and f"clinc150:{label}" not in conds:
                raise PipelineError(f"clinc150: unknown label '{label}'")
            mid = f"clinc150:{key}:{idx}"
            moments.append(
                _moment(
                    "clinc150",
                    mid,
                    mid,
                    split,
                    clean_text(text),
                    None if oos else f"clinc150:{label}",
                    {"oos": True} if oos else None,
                )
            )
    return SourceData("clinc150", "T2", cfg["licenses"]["clinc150"], conds, moments, "partition_routed", True)


_UNK = re.compile(r"<unk>", re.IGNORECASE)
_HWU_KEEP = re.compile(r"[^a-z0-9' ]+")


def _hwu_text(row: dict) -> str | None:
    """answer_normalised when present, else the raw answer (used by general_* rows).
    Both are lower-cased and stripped of punctuation so that punctuation cannot reveal the label."""
    for key in ("answer_normalised", "answer"):
        t = row.get(key) or ""
        if t and t != "null":
            t = _HWU_KEEP.sub(" ", _UNK.sub(" ", t.lower()))
            t = clean_text(t)
            return t or None
    return None


def load_hwu64(cfg: dict, raw: Path, curated: Path, drops: DropLog, limit: int | None) -> SourceData:
    dcfg = cfg["datasets"]["hwu64"]
    path = raw / "hwu64" / "AnnotatedData" / "NLU-Data-Home-Domain-Annotated-All.csv"
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    all_labels = {f"{r['scenario']}_{r['intent']}" for r in rows}
    excluded = set(dcfg["excluded_labels"])
    if excluded - all_labels:
        raise PipelineError(f"hwu64: excluded_labels not in data: {sorted(excluded - all_labels)}")
    labels = all_labels - excluded
    meta, cur = load_curated(curated / "hwu64_conditions.json", labels, "hwu64")
    conds = {
        f"hwu64:{lab}": _condition("hwu64", lab, lab.split("_")[0], cur[lab]["description"], meta)
        for lab in sorted(labels)
    }
    fr = dcfg["split_fractions"]
    moments: list[Moment] = []
    for idx, r in enumerate(rows):
        if limit is not None and len(moments) >= limit:
            break
        label = f"{r['scenario']}_{r['intent']}"
        if label in excluded:
            drops.add("hwu64", "excluded_label")
            continue
        if (r.get("status") or "").startswith("IRR_"):
            drops.add("hwu64", "irrelevant_status")  # the dataset's own post-processing drops these
            continue
        text = _hwu_text(r)
        if text is None:
            drops.add("hwu64", "no_text")
            continue
        u = stable_unit(cfg["seed"], "hwu-split", r["userid"])  # split by worker: no style leakage
        split = "train" if u < fr["train"] else ("val" if u < fr["train"] + fr["val"] else "test")
        moments.append(_moment("hwu64", f"hwu64:r{idx}", f"hwu64:user:{r['userid']}", split, text, f"hwu64:{label}"))
    return SourceData("hwu64", "T2", cfg["licenses"]["hwu64"], conds, moments, "partition_routed", True)
