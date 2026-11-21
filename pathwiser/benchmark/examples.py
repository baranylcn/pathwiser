"""Turn moments into final examples (candidate sets, node, slices, provenance)."""

from __future__ import annotations

import hashlib

from pathwiser.benchmark.candidates import Plan, Pool, plan_no_match, plan_route
from pathwiser.benchmark.schema import DropLog, Moment, SourceData
from pathwiser.benchmark.seeding import rng_for, stable_unit, weighted_choice
from pathwiser.benchmark.sources.sgd import domain_of
from pathwiser.benchmark.text import n_words


def _k_bucket(k: int) -> str:
    return "k:2-3" if k <= 3 else ("k:4-6" if k <= 6 else "k:7+")


class ExampleFactory:
    def __init__(self, cfg: dict, sources: dict[str, SourceData], pools: dict[tuple[str, str], Pool], drops: DropLog):
        self.cfg = cfg
        self.sources = sources
        self.pools = pools
        self.drops = drops
        self.seed = cfg["seed"]
        self.ccfg = cfg["candidates"]
        self.train_groups = {
            ds: {c.group for c in sd.conditions.values() if c.partition == "train"} for ds, sd in sources.items()
        }
        sgd = sources.get("sgd")
        self.svc_desc = getattr(sgd, "service_descriptions", {}) if sgd else {}
        self.train_variants = sorted(cfg["sgdx"]["train_variant_weights"].items())

    def _plan(self, m: Moment) -> Plan | None:
        sd = self.sources[m.dataset]
        pool = self.pools[(m.dataset, m.split)]
        gold = sd.conditions[m.gold_uid] if m.gold_uid else None
        rng = rng_for(self.seed, "plan", m.moment_id)
        nm_cfg = self.cfg["no_match"]
        kinds = sorted(nm_cfg["kind_weights"].items())
        if gold is None:
            return plan_no_match(pool, None, m.message, "oos", rng, self.ccfg)
        if stable_unit(self.seed, "want-no-match", m.moment_id) < nm_cfg["fraction"]:
            first = str(weighted_choice(rng_for(self.seed, "nm-kind", m.moment_id), kinds))
            for kind in [first] + [k for k, _ in kinds if k != first]:
                plan = plan_no_match(
                    pool, gold, m.message, kind, rng_for(self.seed, "plan-nm", m.moment_id, kind), self.ccfg
                )
                if plan is not None:
                    return plan
            self.drops.add(m.dataset, "no_match_infeasible_fell_back_to_route")
        return plan_route(pool, gold, rng, self.ccfg)

    def _node(self, ds: str, plan: Plan, variant: str) -> dict:
        nodes = self.cfg["nodes"]
        if ds == "sgd":
            services = sorted({c.extra["service"] for c, _, _ in plan.members})
            if len(services) == 1:
                desc = self.svc_desc[services[0]]
                text = desc.get(variant) or desc["orig"]
                return {
                    "name": f"{domain_of(services[0]).lower()}_routing",
                    "description": nodes["sgd_single_service"]["description"].format(service_description=text),
                    "origin": "sgd_service_schema",
                }
            return {**nodes["generic"], "origin": "generic"}
        return {**nodes["static"][ds], "origin": "dataset_static"}

    def _slices(self, m: Moment, plan: Plan, k: int, variant: str, paired: bool, gold) -> list[str]:
        sd = self.sources[m.dataset]
        s = {f"tier:{sd.tier}", f"dataset:{m.dataset}", _k_bucket(k)}
        s.add("label:route" if plan.kind == "route" else "label:no_match")
        if plan.kind != "route":
            s.add(f"nm_kind:{plan.kind}")
        if plan.label_risk != "none":
            s.add(f"risk:{plan.label_risk}")
        if plan.kind == "route":
            s.add("cond:seen" if gold.group in self.train_groups[m.dataset] else "cond:unseen")
            if m.dataset == "sgd":
                s.add("svc:seen" if gold.partition == "train" else "svc:unseen")
        else:
            s.add("cond:none")
        s.add(
            "ctx:none"
            if m.context_origin == "none"
            else ("ctx:first_turn" if m.is_first_user_turn else "ctx:later_turn")
        )
        if n_words(m.message) <= 4:
            s.add("msg:short")
        s.add("desc:heldout_paraphrase" if paired else ("desc:orig" if variant == "orig" else "desc:train_variant"))
        return sorted(s)

    def _render(self, m: Moment, plan: Plan, variant: str, paired_with: str | None) -> dict:
        sd = self.sources[m.dataset]
        gold = sd.conditions[m.gold_uid] if (m.gold_uid and plan.kind == "route") else None
        k = len(plan.members)
        cands, meta, gold_cid, gold_pos = [], [], None, None
        for i, (c, role, strat) in enumerate(plan.members):
            cid = f"c{i}"
            cands.append({"cid": cid, "description": c.describe(variant)})
            meta.append(
                {
                    "cid": cid,
                    "condition_uid": c.uid,
                    "family": c.family,
                    "equivalence_group_id": c.group,
                    "role": role,
                    "negative_strategy": None if role == "gold" else strat,
                    "description_variant": variant if variant in c.descriptions else "orig",
                    "description_origin": c.description_origin,
                    "reviewed": c.reviewed,
                }
            )
            if role == "gold":
                gold_cid, gold_pos = cid, i
        eid = (
            "ex_"
            + hashlib.sha256(f"{m.moment_id}|{variant}|{'paired' if paired_with else 'main'}".encode()).hexdigest()[:16]
        )
        return {
            "example_id": eid,
            "dataset_version": self.cfg["version"],
            "split": m.split,
            "tier": sd.tier,
            "source": {
                "dataset": m.dataset,
                "moment_id": m.moment_id,
                "dialogue_id": m.group_id,
                "turn_idx": m.turn_idx,
                "license": sd.license,
            },
            "split_group_id": m.group_id,
            "context": m.context,
            "current_message": m.message,
            "node": self._node(m.dataset, plan, variant),
            "candidates": cands,
            "gold_cid": gold_cid,
            "label_type": "route" if plan.kind == "route" else "no_match",
            "slices": self._slices(m, plan, k, variant, paired_with is not None, gold),
            "construction": {
                "kind": plan.kind,
                "label_risk": plan.label_risk,
                "n_candidates": k,
                "gold_position": gold_pos,
                "description_variant": variant,
                "removed_gold_uid": plan.removed_gold_uid,
                "paired_with": paired_with,
            },
            "candidate_meta": meta,
            "features": {
                "msg_words": n_words(m.message),
                "n_context_turns": len(m.context),
                "n_context_total": m.n_context_total,
                "is_first_user_turn": m.is_first_user_turn,
                "context_origin": m.context_origin,
            },
        }

    def make(self, m: Moment) -> list[dict]:
        plan = self._plan(m)
        if plan is None:
            self.drops.add(m.dataset, "no_feasible_candidate_set")
            return []
        sg = self.cfg["sgdx"]
        if m.dataset == "sgd":
            if m.split == "train":
                variant = str(weighted_choice(rng_for(self.seed, "sgdx", m.moment_id), self.train_variants))
            else:
                variant = "orig"
        else:
            variant = "orig"
        main = self._render(m, plan, variant, None)
        out = [main]
        if m.dataset == "sgd" and m.split in ("val", "test"):
            held = sg["val_variant"] if m.split == "val" else sg["test_variant"]
            out.append(self._render(m, plan, held, main["example_id"]))
        return out
