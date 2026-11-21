"""Scenario specifications: what the generating model is asked to write. Made by code with a seed, never by the model.

A specification fixes domain, node archetype, number of candidate conditions ``k``, the style of the condition descriptions and, for every
example, the target condition (a letter), the ``no_match`` kind, the context kind, difficulty tags, persona and noise. Counts of archetypes
and personas are apportioned exactly (largest remainder) rather than drawn independently, so a requested mix really appears in the data.

``validate_specs`` / ``summary`` check and describe a list of specs; ``match_scenario`` checks a generated scenario against its spec.
"""

from __future__ import annotations

import random
import string
from collections import Counter

from pathwiser.synth import styles as S
from pathwiser.synth import taxonomy as T

SAMPLER_VERSION = "1"
HELDOUT_STYLE_SPLITS = ("eval_v2", "synth_test")  # held-out personas and noise are only allowed in these splits
LETTERS = string.ascii_lowercase[:10]


def apportion(weights: dict[str, float], n: int, rng: random.Random) -> list[str]:
    """``n`` labels whose counts follow ``weights`` as exactly as integers allow (largest remainder), in random order."""
    total = sum(weights.values())
    quota = {k: w / total * n for k, w in weights.items()}
    base = {k: int(q) for k, q in quota.items()}
    order = sorted(quota, key=lambda k: (quota[k] - base[k], rng.random()), reverse=True)
    for k in order[: n - sum(base.values())]:
        base[k] += 1
    out = [k for k, c in sorted(base.items()) for _ in range(c)]
    rng.shuffle(out)
    return out


def _choice(weights: dict, rng: random.Random):
    keys = sorted(weights, key=str)
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def _example_plan(k: int, archetype: dict, n_examples: int, n_nm: int, rng: random.Random) -> list[dict]:
    letters = list(LETTERS[:k])
    n_route = n_examples - n_nm
    perm = rng.sample(letters, k)
    targets = perm[:n_route] if n_route <= k else perm + [rng.choice(letters) for _ in range(n_route - k)]
    plans: list[dict] = []
    for t in targets:
        plans.append({"kind": "route", "target": t})
    for _ in range(n_nm):
        allowed = {kind: w for kind, w in S.NM_WEIGHTS.items() if kind != "sibling_removed" or k >= 3}
        kind = _choice(allowed, rng)
        plan = {"kind": "no_match", "nm": kind}
        if kind == "sibling_removed":
            intent = rng.choice(letters)
            plan.update(intent=intent, sibling=rng.choice([c for c in letters if c != intent]))
        plans.append(plan)
    rng.shuffle(plans)
    for i, p in enumerate(plans):
        p["i"] = i
        needs_ctx = archetype["context_required"]
        p["context"] = (
            _choice({"first_turn": 0.4, "later_turn": 0.6}, rng) if needs_ctx else _choice(S.CONTEXT_WEIGHTS, rng)
        )
    return plans


def sample_specs(
    split: str,
    n_scenarios: int,
    seed: int,
    persona_weights: dict[str, float],
    id_prefix: str,
    taxonomy: dict | None = None,
    start_index: int = 0,
) -> list[dict]:
    """Specifications for ``n_scenarios`` scenarios in the domains of ``split``. Deterministic in all arguments."""
    tax = taxonomy or T.load()
    domains = T.domains_by_split(tax, split)
    if not domains:
        raise ValueError(f"no domains for split {split!r}")
    heldout_ok = split in HELDOUT_STYLE_SPLITS
    if not heldout_ok and any(w > 0 and p in S.HELDOUT_PERSONAS for p, w in persona_weights.items()):
        raise ValueError(f"held-out personas are only allowed in {HELDOUT_STYLE_SPLITS}, not in {split!r}")
    rng = random.Random(seed)
    # domains are used as evenly as possible
    pool: list[tuple[str, dict]] = []
    while len(pool) < n_scenarios:
        block = domains[:]
        rng.shuffle(block)
        pool += block
    pool = pool[:n_scenarios]
    arch_ids = apportion({a: v["weight"] for a, v in S.ARCHETYPES.items()}, n_scenarios, rng)
    style_ids = apportion({s: w for s, (w, _) in S.CONDITION_STYLES.items()}, n_scenarios, rng)
    specs = []
    for idx, ((group, dom), arch_id, style_id) in enumerate(zip(pool, arch_ids, style_ids)):
        arch = S.ARCHETYPES[arch_id]
        k = _choice(arch["k"], rng)
        n_examples = rng.choice([5, 5, 6])
        n_nm = 1 if rng.random() < 0.85 else 2
        n_nm = min(n_nm, n_examples - 3)
        specs.append(
            {
                "id": f"{id_prefix}_{start_index + idx:03d}",
                "split": split,
                "group": group,
                "domain": dom["id"],
                "domain_description": dom["description"],
                "archetype": arch_id,
                "k": k,
                "condition_style": style_id,
                "language": "en",
                "examples": _example_plan(k, arch, n_examples, n_nm, rng),
            }
        )
    # personas: exact apportionment over ALL examples, then noise per example given the persona
    flat = [(s, e) for s in specs for e in s["examples"]]
    for (s, e), persona in zip(flat, apportion(persona_weights, len(flat), rng)):
        e["persona"] = persona
        e["noise"] = _choice(
            {n: w for n, w in S.NOISE_BY_PERSONA[persona].items() if heldout_ok or n in S.NOISE_SEEN}, rng
        )
        tags: list[str] = []
        if e["kind"] == "route":
            if e["context"] != "none" and rng.random() < 0.35:
                tags.append("context_dependent")
            elif persona == "hinting":
                tags.append("indirect")
            elif persona == "multi_topic":
                tags.append("distractor")
            else:
                neg_ok = s["archetype"] in ("intent_router", "confirmation", "post_answer_followup", "triage")
                roll = rng.random()
                if roll < 0.30:
                    tags.append("indirect")
                elif roll < 0.38 and neg_ok:
                    tags.append("negation")
                elif roll < 0.48:
                    tags.append("distractor")
        e["tags"] = tags
    validate_specs(specs, split)
    return specs


def validate_specs(specs: list[dict], split: str, taxonomy: dict | None = None) -> None:
    tax = taxonomy or T.load()
    allowed = {d["id"] for _, d in T.domains_by_split(tax, split)}
    problems: list[str] = []
    ids = [s["id"] for s in specs]
    if len(set(ids)) != len(ids):
        problems.append("duplicate scenario ids")
    for s in specs:
        if s["domain"] not in allowed:
            problems.append(f"{s['id']}: domain {s['domain']} is not in split {split}")
        arch = S.ARCHETYPES[s["archetype"]]
        if s["k"] not in arch["k"]:
            problems.append(f"{s['id']}: k={s['k']} not allowed for {s['archetype']}")
        letters = set(LETTERS[: s["k"]])
        n_route = 0
        for e in s["examples"]:
            if e["context"] not in S.CONTEXT_KINDS or (arch["context_required"] and e["context"] == "none"):
                problems.append(f"{s['id']}[{e['i']}]: bad context {e['context']}")
            if e["persona"] not in S.PERSONAS or e["noise"] not in S.NOISE_BY_PERSONA[e["persona"]]:
                problems.append(f"{s['id']}[{e['i']}]: persona/noise combination")
            elif split not in HELDOUT_STYLE_SPLITS and (
                e["persona"] in S.HELDOUT_PERSONAS or e["noise"] in S.NOISE_HELDOUT
            ):
                problems.append(f"{s['id']}[{e['i']}]: held-out style {e['persona']}/{e['noise']} in split {split}")
            if e["kind"] == "route":
                n_route += 1
                if e["target"] not in letters:
                    problems.append(f"{s['id']}[{e['i']}]: target outside the k conditions")
            else:
                if e["nm"] not in S.NM_KINDS:
                    problems.append(f"{s['id']}[{e['i']}]: unknown no_match kind")
                if e["nm"] == "sibling_removed" and (
                    s["k"] < 3
                    or e["intent"] not in letters
                    or e["sibling"] not in letters
                    or e["intent"] == e["sibling"]
                ):
                    problems.append(f"{s['id']}[{e['i']}]: invalid sibling_removed plan")
        if n_route < 3:
            problems.append(f"{s['id']}: fewer than 3 route examples")
    if problems:
        raise ValueError("invalid specs:\n  " + "\n  ".join(problems))


def summary(specs: list[dict]) -> dict:
    ex = [e for s in specs for e in s["examples"]]
    return {
        "scenarios": len(specs),
        "examples": len(ex),
        "domains": len({s["domain"] for s in specs}),
        "groups": len({s["group"] for s in specs}),
        "archetypes": dict(Counter(s["archetype"] for s in specs)),
        "k": dict(sorted(Counter(s["k"] for s in specs).items())),
        "condition_styles": dict(Counter(s["condition_style"] for s in specs)),
        "no_match_share": round(sum(e["kind"] == "no_match" for e in ex) / len(ex), 3),
        "no_match_kinds": dict(Counter(e["nm"] for e in ex if e["kind"] == "no_match")),
        "context": dict(Counter(e["context"] for e in ex)),
        "personas": dict(Counter(e["persona"] for e in ex)),
        "noise": dict(Counter(e["noise"] for e in ex)),
        "tags": dict(Counter(t for e in ex for t in e["tags"])),
        "heldout_persona_examples": sum(e["persona"] in S.HELDOUT_PERSONAS for e in ex),
    }


def match_scenario(scenario: dict, spec: dict) -> list[str]:
    """Problems of a generated scenario relative to its spec (empty list = it follows the spec)."""
    p: list[str] = []
    if scenario.get("id") != spec["id"]:
        p.append(f"id {scenario.get('id')!r} != {spec['id']!r}")
    letters = list(LETTERS[: spec["k"]])
    if list(scenario.get("conditions", {})) != letters:
        p.append(f"conditions must be exactly {letters}, got {list(scenario.get('conditions', {}))}")
    exs = scenario.get("examples", [])
    if len(exs) != len(spec["examples"]):
        return p + [f"{len(exs)} examples, spec has {len(spec['examples'])}"]
    for plan, ex in zip(spec["examples"], exs):
        tag = f"{spec['id']}[{plan['i']}]"
        if plan["kind"] == "route":
            if ex.get("gold") != plan["target"]:
                p.append(f"{tag}: gold {ex.get('gold')!r} != target {plan['target']!r}")
            if ex.get("cands", letters) != letters and sorted(ex.get("cands", letters)) != letters:
                p.append(f"{tag}: route example must show all candidates")
        else:
            if ex.get("gold") is not None or ex.get("nm") != plan["nm"]:
                p.append(f"{tag}: expected no_match kind {plan['nm']!r}")
            elif plan["nm"] == "sibling_removed":
                if ex.get("intent") != plan["intent"] or sorted(ex.get("cands", [])) != [
                    c for c in letters if c != plan["intent"]
                ]:
                    p.append(f"{tag}: sibling_removed must remove exactly {plan['intent']!r}")
            elif sorted(ex.get("cands", letters)) != letters:
                p.append(f"{tag}: this no_match kind keeps all candidates")
        ctx = ex.get("ctx", [])
        has_user = any(t[0] == "u" for t in ctx)
        if plan["context"] == "none" and ctx:
            p.append(f"{tag}: context must be empty")
        if plan["context"] == "first_turn" and (not ctx or has_user or ctx[-1][0] != "a"):
            p.append(f"{tag}: first_turn context = assistant turn(s) only")
        if plan["context"] == "later_turn" and (not has_user or ctx[-1][0] != "a"):
            p.append(f"{tag}: later_turn context = earlier user turn(s) and ends with an assistant turn")
    return p


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(prog=prog, description="Sample scenario specifications for a taxonomy split.")
    parser.add_argument("--split", choices=T.SPLITS, required=True)
    parser.add_argument("--count", type=int, required=True, help="number of scenarios")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prefix", required=True, help="scenario id prefix, e.g. v2")
    parser.add_argument(
        "--personas", choices=("eval", "train"), default="train", help="persona mix (held-out styles only for eval)"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    weights = S.EVAL_PERSONA_WEIGHTS if args.personas == "eval" else S.TRAIN_PERSONA_WEIGHTS
    specs = sample_specs(args.split, args.count, args.seed, weights, args.prefix)
    args.out.write_text(json.dumps(specs, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary(specs), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
