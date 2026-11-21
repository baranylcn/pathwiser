"""Schema-Guided Dialogue (+ SGD-X) loader.

Conditions: every intent in the train/dev/test schemas, with the original description plus the five
crowd-sourced SGD-X paraphrases (aligned by position: SGD-X renames services and intents).
Moments: a user turn where a service's ``active_intent`` becomes a new non-NONE value.
Splits: the official ones (dev -> val). ``partition`` follows schema presence (train / dev-only / test-only).
"""

from __future__ import annotations

import glob
from collections.abc import Iterable
from pathlib import Path

from pathwiser.benchmark.fileio import read_json
from pathwiser.benchmark.schema import Condition, DropLog, Moment, PipelineError, SourceData
from pathwiser.benchmark.text import clean_text, jaccard, norm_text


class UnionFind:
    """Union-find whose root is always the lexicographically smallest member."""

    def __init__(self, items: Iterable[str]):
        self.parent = {i: i for i in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            lo, hi = sorted((ra, rb))
            self.parent[hi] = lo


DIRS = (("train", "train"), ("dev", "val"), ("test", "test"))
VARIANTS = ("v1", "v2", "v3", "v4", "v5")
JACCARD_EQUIV = 0.6


def domain_of(service: str) -> str:
    return service.rsplit("_", 1)[0]


def load_schemas(root: Path):
    """Return (conditions, service_descriptions, service_partition)."""
    orig = {d: read_json(root / d / "schema.json") for d, _ in DIRS}
    var = {v: {d: read_json(root / "sgd_x" / "data" / v / d / "schema.json") for d, _ in DIRS} for v in VARIANTS}
    conds: dict[str, Condition] = {}
    svc_desc: dict[str, dict[str, str]] = {}
    svc_part: dict[str, str] = {}
    part_of_dir = {"train": "train", "dev": "val", "test": "test"}
    for d, _ in DIRS:
        for si, svc in enumerate(orig[d]):
            name = svc["service_name"]
            vsvc = {}
            for v in VARIANTS:
                if len(var[v][d]) != len(orig[d]):
                    raise PipelineError(f"sgd: SGD-X {v}/{d} has a different number of services")
                vsvc[v] = var[v][d][si]
                if len(vsvc[v]["intents"]) != len(svc["intents"]):
                    raise PipelineError(f"sgd: SGD-X {v}/{d}/{name} intent count differs; cannot align")
            if name not in svc_desc:
                svc_desc[name] = {
                    "orig": clean_text(svc["description"]),
                    **{v: clean_text(vsvc[v]["description"]) for v in VARIANTS},
                }
                svc_part[name] = part_of_dir[d]
            for ii, it in enumerate(svc["intents"]):
                uid = f"sgd:{name}:{it['name']}"
                text = clean_text(it["description"])
                if uid in conds:
                    if conds[uid].descriptions["orig"] != text:
                        raise PipelineError(f"sgd: {uid} has different descriptions across splits")
                    continue
                descs = {"orig": text}
                for v in VARIANTS:
                    descs[v] = clean_text(vsvc[v]["intents"][ii]["description"])
                conds[uid] = Condition(
                    uid=uid,
                    dataset="sgd",
                    family=domain_of(name),
                    group=uid,
                    descriptions=descs,
                    description_origin="source_schema+sgdx_crowdsourced",
                    reviewed=True,
                    partition=svc_part[name],
                    extra={"service": name, "intent": it["name"], "is_transactional": bool(it.get("is_transactional"))},
                )
    _assign_equivalence_groups(conds)
    return conds, svc_desc, svc_part


def _assign_equivalence_groups(conds: dict[str, Condition]) -> None:
    """Merge cross-service conditions that mean the same thing (same domain, same intent name,
    identical description, or token-Jaccard >= 0.6) so they never co-occur in one candidate set."""
    uf = UnionFind(conds)
    uids = sorted(conds)
    for i, a in enumerate(uids):
        ca = conds[a]
        for b in uids[i + 1 :]:
            cb = conds[b]
            if ca.family != cb.family or ca.extra["service"] == cb.extra["service"]:
                continue
            da, db = ca.descriptions["orig"], cb.descriptions["orig"]
            if (
                ca.extra["intent"] == cb.extra["intent"]
                or norm_text(da) == norm_text(db)
                or jaccard(da, db) >= JACCARD_EQUIV
            ):
                uf.union(a, b)
    for uid, c in conds.items():
        c.group = uf.find(uid)


def load(cfg: dict, raw: Path, curated: Path, drops: DropLog, limit: int | None) -> SourceData:
    root = raw / "sgd"
    conds, svc_desc, _ = load_schemas(root)
    max_turns = cfg["context_max_turns"]
    moments: list[Moment] = []
    for dirname, split in DIRS:
        produced = 0
        for path in sorted(glob.glob(str(root / dirname / "dialogues_*.json"))):
            for dlg in read_json(Path(path)):
                if limit is not None and produced >= limit:
                    break
                produced += _extract(dlg, dirname, split, conds, max_turns, moments, drops)
            if limit is not None and produced >= limit:
                break
    sd = SourceData("sgd", "T1", cfg["licenses"]["sgd"], conds, moments, "official", single_turn=False)
    sd.service_descriptions = svc_desc  # type: ignore[attr-defined]
    return sd


def _extract(
    dlg: dict, dirname: str, split: str, conds: dict[str, Condition], max_turns: int, out: list[Moment], drops: DropLog
) -> int:
    # SGD dialogue ids are NOT unique across train/dev/test (e.g. '1_00000' exists in several splits),
    # so every id is namespaced with the official split directory.
    did = f"{dirname}/{dlg['dialogue_id']}"
    prev: dict[str, str] = {}
    hist: list[dict] = []
    n = 0
    for ti, turn in enumerate(dlg["turns"]):
        if turn["speaker"] == "USER":
            new = []
            for fr in turn["frames"]:
                ai = (fr.get("state") or {}).get("active_intent")
                if ai is None:
                    continue
                if ai != "NONE" and prev.get(fr["service"]) != ai:
                    new.append((fr["service"], ai))
                prev[fr["service"]] = ai
            new = sorted(set(new))
            if len(new) > 1:
                drops.add("sgd", "multi_gold_turn")
            elif new:
                service, intent = new[0]
                uid = f"sgd:{service}:{intent}"
                text = clean_text(turn["utterance"])
                if uid not in conds:
                    drops.add("sgd", "gold_without_schema_condition")
                elif not text:
                    drops.add("sgd", "empty_message")
                else:
                    out.append(
                        Moment(
                            moment_id=f"sgd:{did}:{ti}:{service}",
                            dataset="sgd",
                            group_id=f"sgd:{did}",
                            base_split=split,
                            context=hist[-max_turns:] if max_turns else [],
                            message=text,
                            gold_uid=uid,
                            turn_idx=ti,
                            is_first_user_turn=(ti == 0),
                            n_context_total=len(hist),
                            context_origin="natural" if hist else "none",
                            extra={"service": service},
                        )
                    )
                    n += 1
        role = "user" if turn["speaker"] == "USER" else "assistant"
        hist.append({"role": role, "content": clean_text(turn["utterance"])})
    return n
