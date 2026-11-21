"""Statistics, artefact probes and the generated dataset card."""

from __future__ import annotations

from collections import Counter, defaultdict

from pathwiser.benchmark.candidates import Pool
from pathwiser.benchmark.schema import SPLITS, DropLog, SourceData

FEATURES = {
    # Pure surface style only. Digits are deliberately NOT a feature: they are semantic content
    # (calculator, flight booking, ...) and would only create false alarms.
    "ends_with_punct": lambda t: t.rstrip()[-1:] in ".!?",
    "has_uppercase": lambda t: any(ch.isupper() for ch in t),
}


def _frac_hit(scores: list[float], gold_idx: int) -> float:
    top = max(scores)
    tied = [i for i, s in enumerate(scores) if s >= top - 1e-12]
    return 1.0 / len(tied) if gold_idx in tied else 0.0


def compute_probes(examples: dict[str, list[dict]], pools: dict[tuple[str, str], Pool]) -> dict:
    """Cheap diagnostics that reveal shortcuts. They are NOT model baselines.

    lexical_overlap : top-1 accuracy of ranking candidates by TF-IDF cosine to the current message only
                      (ignores context). Far above chance => the set is solvable by word overlap.
    candidate_only  : how often the gold description is the longest one (no conversation needed).
    style           : conditions whose messages differ strongly from their dataset in surface style
                      (terminal punctuation / capitals / digits); a model could use this as a shortcut.
    """
    lex: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])  # hit, chance, n
    longest: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    style_rows: dict[str, dict[str, list[tuple[str, bool]]]] = defaultdict(lambda: defaultdict(list))
    for split in SPLITS:
        for ex in examples[split]:
            if ex["label_type"] != "route" or ex["construction"]["paired_with"] is not None:
                continue
            ds = ex["source"]["dataset"]
            meta, k = ex["candidate_meta"], len(ex["candidates"])
            gold_idx = ex["construction"]["gold_position"]
            if split in ("val", "test"):
                pool = pools[(ds, split)]
                v = pool.tfidf.vector(ex["current_message"])
                from pathwiser.benchmark.text import cosine

                scores = [cosine(v, pool.vecs[m["condition_uid"]]) for m in meta]
                row = lex[(ds, split)]
                row[0] += _frac_hit(scores, gold_idx)
                row[1] += 1.0 / k
                row[2] += 1
                lens = [len(c["description"].split()) for c in ex["candidates"]]
                row = longest[(ds, split)]
                row[0] += _frac_hit([float(x) for x in lens], gold_idx)
                row[1] += 1.0 / k
                row[2] += 1
            gold_uid = meta[gold_idx]["condition_uid"]
            for name, fn in FEATURES.items():
                style_rows[ds][name].append((gold_uid, bool(fn(ex["current_message"]))))
    probes = {"lexical_overlap_top1": {}, "gold_is_longest_description": {}, "style_artifacts": []}
    for (ds, split), (hit, chance, n) in sorted(lex.items()):
        probes["lexical_overlap_top1"][f"{ds}/{split}"] = {
            "accuracy": round(hit / n, 4),
            "chance": round(chance / n, 4),
            "n": int(n),
        }
    for (ds, split), (hit, chance, n) in sorted(longest.items()):
        probes["gold_is_longest_description"][f"{ds}/{split}"] = {
            "rate": round(hit / n, 4),
            "chance": round(chance / n, 4),
            "n": int(n),
        }
    for ds, feats in sorted(style_rows.items()):
        for name, rows in sorted(feats.items()):
            overall = sum(1 for _, b in rows if b) / len(rows)
            per: dict[str, list[bool]] = defaultdict(list)
            for uid, b in rows:
                per[uid].append(b)
            for uid, vals in sorted(per.items()):
                rate = sum(vals) / len(vals)
                if len(vals) >= 30 and abs(rate - overall) > 0.4:
                    probes["style_artifacts"].append(
                        {
                            "dataset": ds,
                            "feature": name,
                            "condition": uid,
                            "rate": round(rate, 3),
                            "dataset_rate": round(overall, 3),
                            "n": len(vals),
                        }
                    )
    return probes


def compute_stats(
    cfg: dict,
    sources: dict[str, SourceData],
    examples: dict[str, list[dict]],
    drops: DropLog,
    moments_kept: dict[str, Counter],
) -> dict:
    by = {s: Counter() for s in SPLITS}
    by_label = {s: Counter() for s in SPLITS}
    slices = {s: Counter() for s in SPLITS}
    kdist = {s: Counter() for s in SPLITS}
    kinds = {s: Counter() for s in SPLITS}
    variants = Counter()
    for s in SPLITS:
        for ex in examples[s]:
            ds = ex["source"]["dataset"]
            by[s][ds] += 1
            by_label[s][(ds, ex["label_type"])] += 1
            kdist[s][ex["construction"]["n_candidates"]] += 1
            kinds[s][ex["construction"]["kind"]] += 1
            for sl in ex["slices"]:
                slices[s][sl] += 1
            variants[(s, ex["construction"]["description_variant"])] += 1
    conds = {}
    for ds, sd in sources.items():
        c = Counter(x.partition for x in sd.conditions.values())
        conds[ds] = {
            "total": len(sd.conditions),
            **{p: c.get(p, 0) for p in ("train", "val", "test")},
            "families": len({x.family for x in sd.conditions.values()}),
            "equivalence_groups": len({x.group for x in sd.conditions.values()}),
        }
    return {
        "examples_per_split": {s: len(examples[s]) for s in SPLITS},
        "examples_by_split_dataset": {s: dict(sorted(by[s].items())) for s in SPLITS},
        "examples_by_split_dataset_label": {
            s: {f"{ds}/{lt}": n for (ds, lt), n in sorted(by_label[s].items())} for s in SPLITS
        },
        "slices": {s: dict(sorted(slices[s].items())) for s in SPLITS},
        "n_candidates_distribution": {s: {str(k): n for k, n in sorted(kdist[s].items())} for s in SPLITS},
        "construction_kinds": {s: dict(sorted(kinds[s].items())) for s in SPLITS},
        "description_variants": {f"{s}/{v}": n for (s, v), n in sorted(variants.items())},
        "conditions": conds,
        "moments_kept": {ds: dict(sorted(c.items())) for ds, c in sorted(moments_kept.items())},
        "dropped": drops.as_dict(),
    }


LIMITATIONS = [
    "Condition descriptions for Banking77, CLINC150, HWU64, STAR and ABCD are drafts written by an LLM for this project "
    "(`reviewed: false` in `data/curated/*.json`). Only SGD / SGD-X descriptions come from the source. Review them before trusting results.",
    "Banking77 / CLINC150 / HWU64 examples are single-turn with an EMPTY context (`context_origin: none`); no context is fabricated. "
    "Context-awareness can only be learned and measured on SGD, STAR and ABCD.",
    "SGD moments are turns where `active_intent` changes. The user message at such a turn may only imply the intent "
    "(e.g. 'Yes, go ahead.'); annotation noise has not been measured.",
    "`no_match` examples are constructed, not observed (except CLINC out-of-scope). `sibling_removed` examples can contain "
    "a partially valid sibling and are flagged `risk:sibling_partial_validity`; they have not been audited by hand.",
    "STAR: the wizard's `select_task` can precede the user's explicit request; low-information messages are filtered but timing noise remains. "
    "Turns followed by several different selected tasks (`multi_gold_turn`, a large share, see Discarded records) are dropped rather than guessed. "
    "Single-task STAR dialogues have no explicit routing event and are unused.",
    "Natural no_match is rare: CLINC out-of-scope is the only observed one. Its candidates are chosen by lexical similarity, "
    "so they are often not semantically close to the message.",
    "ABCD: the moment is the first substantive customer turn (heuristic); flow-level labels only (10 flows).",
    "Hard negatives use lexical (TF-IDF) similarity plus family structure, not semantic embeddings.",
    "`requires_context` is NOT computed. Only proxies exist (`ctx:*`, `msg:short` slices). Measure it with a context ablation.",
    "Held-out families (unseen conditions) were chosen by hand in the config; Banking77 families were authored for this project.",
    "HWU64 uses 63 labels: the 4 rare labels of the 68 raw ones and the catch-all `general_quirky` are excluded.",
    "STAR, HWU64 and the Banking77 validation split are deterministic hash splits, not official splits.",
    "Not yet included: the hand-written gold test set (T5), LLM-generated non-intent conditions (T4), `ambiguous` labels, multilingual data.",
    "SGD is CC BY-SA 4.0 (share-alike): derived data inherits that obligation. STAR/ABCD licenses shown are repository licenses.",
]


def render_card(cfg: dict, stats: dict, probes: dict, validation: dict, sources: dict[str, SourceData]) -> str:
    L: list[str] = []
    a = L.append
    a(f"# Pathwiser routing dataset {cfg['version']}")
    a("")
    a(
        "Generated by `pathwiser benchmark build`. Every number below is computed from the build; nothing is hand-edited."
    )
    a(f"Seed `{cfg['seed']}`, config hash `{cfg['_hash'][:16]}`.")
    a("")
    a("## What an example is")
    a("")
    a(
        "Given a conversation, the current user message, a workflow node and a set of candidate conditions "
        "(natural-language descriptions), pick the matching condition or none (`no_match`)."
    )
    a("")
    a("| field | meaning | model input |")
    a("|---|---|---|")
    for f, m, i in [
        ("context[]", "last turns before the message `{role, content}`", "yes"),
        ("current_message", "latest user message", "yes"),
        ("node", "workflow node `{name, description}` (+ `origin`)", "yes"),
        ("candidates[]", "`{cid, description}`; cid is positional and meaningless", "yes"),
        ("gold_cid / label_type", "target: a cid, or null for `no_match`", "target"),
        (
            "candidate_meta[]",
            "condition uid, family, equivalence group, role, negative strategy, description variant/origin/reviewed",
            "no",
        ),
        (
            "construction",
            "kind (route / other_family / sibling_removed / oos), label_risk, removed gold, pairing",
            "no",
        ),
        (
            "slices[]",
            "evaluation tags: tier, dataset, cond:seen|unseen, svc:seen|unseen, ctx:*, k:*, nm_kind:*, desc:*, risk:*",
            "no",
        ),
        ("source / split_group_id", "provenance and leakage unit", "no"),
        ("features", "message length, context length, first-turn flag", "no"),
    ]:
        a(f"| `{f}` | {m} | {i} |")
    a("")
    a("## Size")
    a("")
    a(
        "| split | examples | "
        + " | ".join(sorted({d for s in SPLITS for d in stats["examples_by_split_dataset"][s]}))
        + " |"
    )
    dss = sorted({d for s in SPLITS for d in stats["examples_by_split_dataset"][s]})
    a("|---|---|" + "---|" * len(dss))
    for s in SPLITS:
        a(
            f"| {s} | {stats['examples_per_split'][s]} | "
            + " | ".join(str(stats["examples_by_split_dataset"][s].get(d, 0)) for d in dss)
            + " |"
        )
    a("")
    a(
        "No-match share per split: "
        + ", ".join(
            f"{s} {100 * sum(n for k, n in stats['examples_by_split_dataset_label'][s].items() if k.endswith('/no_match')) / max(1, stats['examples_per_split'][s]):.1f}%"
            for s in SPLITS
        )
    )
    a("")
    a("## Conditions")
    a("")
    a("| dataset | conditions | train / val / test partition | families | equivalence groups |")
    a("|---|---|---|---|---|")
    for ds, c in sorted(stats["conditions"].items()):
        a(
            f"| {ds} | {c['total']} | {c['train']} / {c['val']} / {c['test']} | {c['families']} | {c['equivalence_groups']} |"
        )
    a("")
    a("Partition: `train` = usable everywhere, `val` = only val/test, `test` = only test (unseen conditions).")
    a("")
    a("## Evaluation slices (examples)")
    a("")
    keys = sorted({k for s in SPLITS for k in stats["slices"][s]})
    a("| slice | train | val | test |")
    a("|---|---|---|---|")
    for k in keys:
        a(f"| `{k}` | " + " | ".join(str(stats["slices"][s].get(k, 0)) for s in SPLITS) + " |")
    a("")
    a("## Artefact probes (diagnostics, not models)")
    a("")
    a("Lexical-overlap top-1 (message only, TF-IDF) versus chance:")
    a("")
    a("| dataset/split | accuracy | chance | n |")
    a("|---|---|---|---|")
    for k, v in probes["lexical_overlap_top1"].items():
        a(f"| {k} | {v['accuracy']} | {v['chance']} | {v['n']} |")
    a("")
    a(
        "Gold is the longest description (rate vs chance): "
        + "; ".join(f"{k} {v['rate']} vs {v['chance']}" for k, v in probes["gold_is_longest_description"].items())
    )
    a("")
    if probes["style_artifacts"]:
        a(
            f"Style artefacts flagged ({len(probes['style_artifacts'])}): conditions whose messages differ strongly in surface style from their dataset:"
        )
        for r in probes["style_artifacts"][:20]:
            a(
                f"- {r['dataset']} `{r['condition']}` {r['feature']}: {r['rate']} vs dataset {r['dataset_rate']} (n={r['n']})"
            )
    else:
        a("No style artefacts flagged.")
    a("")
    a("## Discarded records")
    a("")
    for ds, reasons in stats["dropped"].items():
        a(f"- **{ds}**: " + ", ".join(f"{r} = {n}" for r, n in reasons.items()))
    a("")
    a("## Validation")
    a("")
    a(f"{validation['n_checks']} invariant checks passed (0 failed).")
    for w in validation["warnings"]:
        a(f"- WARNING: {w}")
    a("")
    a("## Provenance and licenses")
    a("")
    for ds, sd in sorted(sources.items()):
        a(f"- **{ds}** ({sd.tier}): {sd.license}")
    a("")
    a("## Known limitations")
    a("")
    for lim in LIMITATIONS:
        a(f"- {lim}")
    a("")
    return "\n".join(L) + "\n"
