"""Preparation of authored scenarios for the build: adjudicated drops and the planned noise."""

from __future__ import annotations

import copy

from pathwiser.synth import styles as S
from pathwiser.synth.noise import apply_noise


def apply_drops(scenarios: list[dict], specs: list[dict], dropped: list[dict]) -> tuple[list[dict], list[dict]]:
    """Remove adjudicated examples (``dropped.json``: scenario id, example index ``i``, reason) from scenarios and specs.

    The remaining examples keep their original index (``example_index``), so their ids, noise and verification records do not change.
    """
    todo = {(d["scenario"], d["i"]) for d in dropped}
    spec_by_id = {s["id"]: s for s in specs}
    found = set()
    new_specs, new_scenarios = [], []
    for sc in scenarios:
        spec = spec_by_id[sc["id"]]
        if len(spec["examples"]) != len(sc["examples"]):
            # never let zip() hide a scenario that does not follow its spec: leave it untouched so the conformance check reports it
            if any(s == sc["id"] for s, _ in todo):
                raise ValueError(f"{sc['id']}: cannot drop examples from a scenario that does not match its spec")
            new_specs.append(spec)
            new_scenarios.append(sc)
            continue
        keep = [(plan, ex) for plan, ex in zip(spec["examples"], sc["examples"]) if (sc["id"], plan["i"]) not in todo]
        found |= {(sc["id"], plan["i"]) for plan in spec["examples"]} & todo
        new_specs.append(dict(spec, examples=[p for p, _ in keep]))
        new_scenarios.append(dict(sc, examples=[e for _, e in keep]))
    missing = todo - found
    if missing:
        raise ValueError(f"dropped examples that do not exist: {sorted(missing)}")
    return new_scenarios, new_specs


def prepare(
    scenarios: list[dict], specs: list[dict], *, id_prefix: str, dataset: str, tier: str, model_split: str
) -> list[dict]:
    """Deep copies of the scenarios with noise applied and the per-example metadata / slice tags attached."""
    spec_by_id = {s["id"]: s for s in specs}
    out = []
    for sc0 in scenarios:
        spec = spec_by_id[sc0["id"]]
        sc = copy.deepcopy(sc0)
        sc.update(domain=spec["domain"], dataset=dataset, tier=tier, id_prefix=id_prefix, model_split=model_split)
        for plan, ex in zip(spec["examples"], sc["examples"]):
            key = f"{id_prefix}_{spec['id']}_{plan['i']:02d}"
            clean_msg, clean_ctx = ex["msg"], [list(t) for t in ex.get("ctx", [])]
            msg, msg_hit = apply_noise(clean_msg, plan["noise"], key)
            ctx, ctx_hit = [], False
            for j, (role, text) in enumerate(clean_ctx):
                if role == "u":
                    text, hit = apply_noise(text, plan["noise"], f"{key}:c{j}")
                    ctx_hit = ctx_hit or hit
                ctx.append([role, text])
            applied = plan["noise"] if (msg_hit or ctx_hit) else "none"
            heldout = plan["persona"] in S.HELDOUT_PERSONAS or applied in S.NOISE_HELDOUT
            ex["msg"], ex["ctx"] = msg, ctx
            ex["example_index"] = plan["i"]
            ex["slices_extra"] = sorted(
                {
                    f"persona:{plan['persona']}",
                    f"style_split:{'heldout' if heldout else 'seen'}",
                    f"noise:{applied}",
                    f"archetype:{spec['archetype']}",
                    f"cstyle:{spec['condition_style']}",
                    f"group:{spec['group']}",
                    *[f"tag:{t}" for t in plan["tags"]],
                }
            )
            ex["meta_extra"] = {
                "persona": plan["persona"],
                "style_split": "heldout" if heldout else "seen",
                "noise_planned": plan["noise"],
                "noise_applied": applied,
                "clean_message": clean_msg,
                "clean_context": clean_ctx,
                "tags": plan["tags"],
                "archetype": spec["archetype"],
                "condition_style": spec["condition_style"],
                "group": spec["group"],
                "domain": spec["domain"],
                "language": spec["language"],
                "rationale": ex.get("why"),
                "spec_id": spec["id"],
            }
        out.append(sc)
    return out
