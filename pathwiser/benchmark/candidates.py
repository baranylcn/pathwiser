"""Candidate-set construction.

For a moment we build either
  * a *route* example: the gold condition plus hard negatives, or
  * a *no_match* example: candidates none of which is correct, in one of three kinds
      other_family     gold removed together with its whole family (clean: nothing partially valid)
      sibling_removed  only the gold's equivalence group removed; same-family siblings may remain
                       (harder, but a sibling can be partially valid -> flagged ``label_risk``)
      oos              out-of-scope utterance; candidates are the conditions nearest to the message.

Negatives are drawn with three strategies (family / nearest by TF-IDF / random), never include a condition
equivalent to the gold and never contain two equivalent conditions or two identical descriptions.
All randomness comes from the RNG passed in, so results are reproducible and order independent.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from pathwiser.benchmark.schema import Condition
from pathwiser.benchmark.seeding import weighted_choice
from pathwiser.benchmark.text import TfIdf, cosine, norm_text


class Pool:
    """The conditions available for one (dataset, split), with cached TF-IDF neighbours."""

    def __init__(self, conditions: Sequence[Condition]):
        self.conditions = sorted(conditions, key=lambda c: c.uid)
        self.by_uid = {c.uid: c for c in self.conditions}
        self.tfidf = TfIdf([c.descriptions["orig"] for c in self.conditions])
        self.vecs = {c.uid: self.tfidf.vector(c.descriptions["orig"]) for c in self.conditions}
        self._neighbors: dict[str, list[Condition]] = {}

    def __len__(self) -> int:
        return len(self.conditions)

    def neighbors(self, uid: str) -> list[Condition]:
        if uid not in self._neighbors:
            v = self.vecs[uid]
            ranked = sorted((-cosine(v, self.vecs[o.uid]), o.uid) for o in self.conditions if o.uid != uid)
            self._neighbors[uid] = [self.by_uid[u] for _, u in ranked]
        return self._neighbors[uid]

    def rank_text(self, text: str) -> list[Condition]:
        v = self.tfidf.vector(text)
        ranked = sorted((-cosine(v, self.vecs[o.uid]), o.uid) for o in self.conditions)
        return [self.by_uid[u] for _, u in ranked]


@dataclass
class Plan:
    members: list[tuple[Condition, str, str]]  # (condition, role gold|negative, strategy) in final order
    kind: str  # "route" | "other_family" | "sibling_removed" | "oos"
    label_risk: str  # "none" | "sibling_partial_validity"
    removed_gold_uid: str | None = None


def n_groups(conds: Sequence[Condition]) -> int:
    return len({c.group for c in conds})


def choose_k(rng: random.Random, size_weights: dict[str, float], max_k: int) -> int | None:
    options = [(int(k), w) for k, w in sorted(size_weights.items(), key=lambda kv: int(kv[0])) if int(k) <= max_k]
    if not options:
        return None
    return int(weighted_choice(rng, options))


def select_negatives(
    pool: Pool,
    rng: random.Random,
    n: int,
    eligible: Sequence[Condition],
    ranked_nearest: Sequence[Condition],
    anchor_family: str | None,
    used_groups: set[str],
    used_desc: set[str],
    ccfg: dict,
) -> list[tuple[Condition, str]]:
    eligible_uids = {c.uid for c in eligible}
    family_list = [c for c in eligible if anchor_family is not None and c.family == anchor_family]
    rng.shuffle(family_list)
    ranked = [c for c in ranked_nearest if c.uid in eligible_uids]
    window = max(n * ccfg["nearest_window_multiplier"], n + 2)
    head = ranked[:window]
    rng.shuffle(head)
    nearest_list = head + ranked[window:]
    random_list = list(eligible)
    rng.shuffle(random_list)
    lists = {"family": family_list, "nearest": nearest_list, "random": random_list}
    ptr = {"family": 0, "nearest": 0, "random": 0}
    weights = sorted(ccfg["negative_strategy_weights"].items())

    def take(name: str) -> Condition | None:
        lst, i = lists[name], ptr[name]
        while i < len(lst):
            c = lst[i]
            i += 1
            if c.group not in used_groups and norm_text(c.descriptions["orig"]) not in used_desc:
                ptr[name] = i
                return c
        ptr[name] = i
        return None

    chosen: list[tuple[Condition, str]] = []
    for _ in range(n):
        strat = str(weighted_choice(rng, weights))
        for name in [strat] + [s for s in ("nearest", "family", "random") if s != strat]:
            c = take(name)
            if c is not None:
                chosen.append((c, name))
                used_groups.add(c.group)
                used_desc.add(norm_text(c.descriptions["orig"]))
                break
        else:
            break
    return chosen


def plan_route(pool: Pool, gold: Condition, rng: random.Random, ccfg: dict) -> Plan | None:
    eligible = [c for c in pool.conditions if c.group != gold.group]
    k = choose_k(rng, ccfg["size_weights"], 1 + n_groups(eligible))
    if k is None:
        return None
    picked = select_negatives(
        pool,
        rng,
        k - 1,
        eligible,
        pool.neighbors(gold.uid),
        gold.family,
        {gold.group},
        {norm_text(gold.descriptions["orig"])},
        ccfg,
    )
    if not picked:
        return None
    members = [(gold, "gold", "gold")] + [(c, "negative", s) for c, s in picked]
    rng.shuffle(members)
    return Plan(members, "route", "none")


def plan_no_match(
    pool: Pool, gold: Condition | None, message: str, kind: str, rng: random.Random, ccfg: dict
) -> Plan | None:
    if kind == "oos":
        eligible = list(pool.conditions)
        ranked = pool.rank_text(message)
        removed = None
    else:
        if gold is None:
            return None
        if kind == "other_family":
            eligible = [c for c in pool.conditions if c.family != gold.family and c.group != gold.group]
        elif kind == "sibling_removed":
            eligible = [c for c in pool.conditions if c.group != gold.group]
        else:
            raise ValueError(kind)
        ranked = pool.neighbors(gold.uid)
        removed = gold.uid
    if n_groups(eligible) < 2:
        return None
    eligible_uids = {c.uid for c in eligible}
    if kind == "sibling_removed":
        anchor_family = gold.family  # type: ignore[union-attr]
    else:
        top = next((c for c in ranked if c.uid in eligible_uids), None)
        anchor_family = top.family if top else None
    k = choose_k(rng, ccfg["size_weights"], n_groups(eligible))
    if k is None:
        return None
    picked = select_negatives(pool, rng, k, eligible, ranked, anchor_family, set(), set(), ccfg)
    if len(picked) < 2:
        return None
    members = [(c, "negative", s) for c, s in picked]
    rng.shuffle(members)
    risk = "none"
    if gold is not None and any(c.family == gold.family for c, _, _ in members):
        risk = "sibling_partial_validity"
    return Plan(members, kind, risk, removed)
