"""Authored scenarios: loading, validation and conversion to model-ready records.

A scenario is a workflow node, its candidate conditions (keyed by a letter) and a list of examples. Format of an example:

    {"gold": "a" | null, "ctx": [["u" | "a", text], ...], "msg": text,
     "cands": [letters]            optional subset of the conditions (default: all)
     "nm": "sibling_removed" | "in_domain_uncovered" | "oos"     required when gold is null
     "intent": letter}             sibling_removed only: the wanted condition, which is not among ``cands``
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from pathwiser.benchmark.records import ModelDataConfig, Preprocessor
from pathwiser.storage.hashing import sha256_file

NO_MATCH_KINDS = ("sibling_removed", "in_domain_uncovered", "oos")


ROLES = {"u": "user", "a": "assistant"}


class ScenarioError(ValueError):
    pass


def k_bucket(k: int) -> str:
    return "k:2-3" if k <= 3 else ("k:4-6" if k <= 6 else "k:7+")


def load_scenarios(src: Path) -> tuple[list[dict], dict[str, str]]:
    files = sorted(src.glob("scenarios_*.json"))
    if not files:
        raise ScenarioError(f"no scenarios_*.json in {src}")
    scenarios: list[dict] = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ScenarioError(f"{f.name}: expected a JSON list of scenarios")
        scenarios.extend(data)
    return scenarios, {f.name: sha256_file(f) for f in files}


def validate_scenarios(scenarios: list[dict], max_k: int = 10) -> None:
    """Hard checks on the authoring data; raises ScenarioError listing every problem."""
    problems: list[str] = []
    seen_ids: set[str] = set()
    messages: Counter = Counter()
    for sc in scenarios:
        sid = sc.get("id", "?")
        if sid in seen_ids:
            problems.append(f"{sid}: duplicate scenario id")
        seen_ids.add(sid)
        conds = sc.get("conditions", {})
        if len(set(conds.values())) != len(conds):
            problems.append(f"{sid}: duplicate condition descriptions")
        if not all(isinstance(v, str) and v.strip() for v in conds.values()):
            problems.append(f"{sid}: empty condition description")
        if not (sc.get("node", {}).get("name") and sc.get("node", {}).get("description")):
            problems.append(f"{sid}: node needs name and description")
        if not sc.get("examples"):
            problems.append(f"{sid}: no examples")
        for i, ex in enumerate(sc.get("examples", [])):
            tag = f"{sid}[{i}]"
            cands = ex.get("cands", list(conds))
            if not all(c in conds for c in cands) or len(set(cands)) != len(cands):
                problems.append(f"{tag}: cands must be distinct condition ids")
                continue
            if not 2 <= len(cands) <= max_k:
                problems.append(f"{tag}: {len(cands)} candidates outside [2, {max_k}]")
            if not str(ex.get("msg", "")).strip():
                problems.append(f"{tag}: empty message")
            messages[ex.get("msg", "").strip().lower()] += 1
            for t in ex.get("ctx", []):
                if len(t) != 2 or t[0] not in ROLES or not str(t[1]).strip():
                    problems.append(f"{tag}: bad context turn {t!r}")
            if ex.get("gold") is not None:
                if ex["gold"] not in cands:
                    problems.append(f"{tag}: gold {ex['gold']!r} not among cands")
                if "nm" in ex:
                    problems.append(f"{tag}: route example must not set 'nm'")
            else:
                if ex.get("nm") not in NO_MATCH_KINDS:
                    problems.append(f"{tag}: no_match needs nm in {NO_MATCH_KINDS}")
                if ex.get("nm") == "sibling_removed" and (ex.get("intent") not in conds or ex.get("intent") in cands):
                    problems.append(f"{tag}: sibling_removed needs 'intent' that exists but is not in cands")
                if ex.get("nm") != "sibling_removed" and "intent" in ex:
                    problems.append(f"{tag}: 'intent' only for sibling_removed")
    problems += [f"duplicate message across examples: {m!r}" for m, n in messages.items() if n > 1]
    if problems:
        raise ScenarioError("invalid scenarios:\n  " + "\n  ".join(problems))


def build_records(scenarios: list[dict], cfg: ModelDataConfig | None = None) -> list[dict]:
    pp = Preprocessor(cfg or ModelDataConfig())
    out: list[dict] = []
    for sc in scenarios:
        conds = sc["conditions"]
        # optional per-scenario settings; the defaults describe a plain, spec-less scenario set
        dataset, tier, prefix = sc.get("dataset", "realistic"), sc.get("tier", "R"), sc.get("id_prefix", "rl")
        for i, ex in enumerate(sc["examples"]):
            eid = f"{prefix}_{sc['id']}_{ex.get('example_index', i):02d}"  # explicit index keeps ids stable when examples are dropped
            cids = list(ex.get("cands", list(conds)))
            random.Random(int(hashlib.sha256(eid.encode()).hexdigest()[:16], 16)).shuffle(cids)
            k = len(cids)
            label = cids.index(ex["gold"]) if ex.get("gold") is not None else -1
            ctx = pp.context([{"role": ROLES[r], "content": t} for r, t in ex.get("ctx", [])])
            msg = pp.norm(ex["msg"])
            node = {"name": sc["node"]["name"], "description": pp.norm(sc["node"]["description"])}
            # same meaning as the benchmark: first_turn = the message is the first user turn of the conversation
            ctx_kind = (
                "ctx:none"
                if not ctx
                else ("ctx:later_turn" if any(t["role"] == "user" for t in ctx) else "ctx:first_turn")
            )
            risky = ex.get("nm") == "sibling_removed"
            slices = {
                f"dataset:{dataset}",
                f"tier:{tier}",
                f"domain:{sc['domain']}",
                k_bucket(k),
                ctx_kind,
                "desc:orig",
                "label:route" if label >= 0 else "label:no_match",
                *ex.get("slices_extra", []),
            }
            if label < 0:
                slices.add(f"nm_kind:{ex['nm']}")
            if risky:
                slices.add("risk:sibling_partial_validity")
            if len(msg.split()) <= 4:
                slices.add("msg:short")
            out.append(
                {
                    "example_id": eid,
                    "split": sc.get("model_split", "test"),
                    "input": {
                        "node": node,
                        "context": ctx,
                        "message": msg,
                        "query": pp.query(node, ctx, msg),
                        "conditions": [pp.norm(conds[c]) for c in cids],
                    },
                    "target": {"label": label, "label_null": label if label >= 0 else k, "has_match": int(label >= 0)},
                    "weight": 1.0,
                    "meta": {
                        "dataset": dataset,
                        "tier": tier,
                        "label_type": "route" if label >= 0 else "no_match",
                        "nm_kind": ex.get("nm"),
                        "label_risk": "sibling_partial_validity" if risky else "none",
                        "n_candidates": k,
                        "is_paired": False,
                        "paired_with": None,
                        "description_variant": "orig",
                        "slices": sorted(slices),
                        "cids": cids,
                        "condition_uids": [f"{dataset}:{sc['id']}:{c}" for c in cids],
                        "families": [sc["domain"]] * k,
                        "dialogue_id": f"{dataset}:{sc['id']}",
                        "gold_seen_exact": None,
                        "intended_condition": ex.get("intent"),
                        **ex.get("meta_extra", {}),
                    },
                }
            )
    return out
