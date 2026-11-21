"""Hard invariants and leakage checks. Any violation aborts the build (nothing is published).

Warnings (distribution anomalies) are reported but do not abort.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from pathwiser.benchmark.schema import SPLIT_RANK, SPLITS, PipelineError, SourceData
from pathwiser.benchmark.text import norm_text

MAX_LISTED = 5


class Checker:
    def __init__(self) -> None:
        self.results: dict[str, dict] = {}

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        r = self.results.setdefault(name, {"violations": 0, "samples": []})
        if not ok:
            r["violations"] += 1
            if len(r["samples"]) < MAX_LISTED:
                r["samples"].append(detail)

    def failures(self) -> dict[str, dict]:
        return {k: v for k, v in self.results.items() if v["violations"]}


def validate(cfg: dict, sources: dict[str, SourceData], examples: dict[str, list[dict]]) -> dict:
    ck = Checker()
    warnings: list[str] = []
    max_k = max(int(k) for k in cfg["candidates"]["size_weights"])
    max_turns = cfg["context_max_turns"]
    conds = {uid: c for sd in sources.values() for uid, c in sd.conditions.items()}
    single_turn = {ds for ds, sd in sources.items() if sd.single_turn}
    seen_ids: set[str] = set()
    by_id: dict[str, dict] = {}
    groups_by_split: dict[str, set[str]] = {s: set() for s in SPLITS}
    texts: dict[tuple[str, str], set[str]] = defaultdict(set)  # (dataset, split) -> normalised texts
    text_counts: Counter = Counter()  # (dataset, split, normalised text) -> main examples

    for split in SPLITS:
        for ex in examples[split]:
            eid = ex["example_id"]
            ck.check("unique_example_id", eid not in seen_ids, eid)
            seen_ids.add(eid)
            by_id[eid] = ex
            ck.check("split_field_matches_file", ex["split"] == split, eid)
            cands, meta = ex["candidates"], ex["candidate_meta"]
            k = len(cands)
            ck.check("candidate_count_in_range", 2 <= k <= max_k, f"{eid} k={k}")
            ck.check(
                "meta_parallel_to_candidates",
                len(meta) == k and all(a["cid"] == b["cid"] for a, b in zip(cands, meta)),
                eid,
            )
            ck.check("cids_are_c0_to_ck", [c["cid"] for c in cands] == [f"c{i}" for i in range(k)], eid)
            descs = [norm_text(c["description"]) for c in cands]
            ck.check("descriptions_non_empty", all(descs), eid)
            ck.check("descriptions_unique_in_example", len(set(descs)) == len(descs), eid)
            ck.check(
                "equivalence_groups_unique_in_example", len({m["equivalence_group_id"] for m in meta}) == len(meta), eid
            )
            for m in meta:
                c = conds.get(m["condition_uid"])
                ck.check("candidate_condition_exists", c is not None, f"{eid} {m['condition_uid']}")
                if c is not None:
                    ck.check(
                        "candidate_partition_allowed_in_split",
                        SPLIT_RANK[c.partition] <= SPLIT_RANK[split],
                        f"{eid} {c.uid} partition={c.partition} split={split}",
                    )
            if ex["label_type"] == "route":
                gold = [m for m in meta if m["role"] == "gold"]
                ck.check("route_has_exactly_one_gold", len(gold) == 1, eid)
                if len(gold) == 1:
                    ck.check("gold_cid_consistent", ex["gold_cid"] == gold[0]["cid"], eid)
                    ck.check(
                        "gold_position_consistent", ex["construction"]["gold_position"] == int(gold[0]["cid"][1:]), eid
                    )
                    gc = conds.get(gold[0]["condition_uid"])
                    if gc is not None:
                        ck.check(
                            "gold_partition_allowed_in_split",
                            SPLIT_RANK[gc.partition] <= SPLIT_RANK[split],
                            f"{eid} {gc.uid}",
                        )
                        if split == "train":
                            ck.check("train_gold_is_train_partition", gc.partition == "train", eid)
            else:
                ck.check("no_match_has_no_gold", ex["gold_cid"] is None and all(m["role"] != "gold" for m in meta), eid)
                ck.check("no_match_gold_position_null", ex["construction"]["gold_position"] is None, eid)
                rid = ex["construction"]["removed_gold_uid"]
                if ex["construction"]["kind"] == "oos":
                    ck.check("oos_has_no_removed_gold", rid is None, eid)
                else:
                    rc = conds.get(rid) if rid else None
                    ck.check("removed_gold_known", rc is not None, eid)
                    if rc is not None:
                        ck.check(
                            "removed_gold_group_absent", all(m["equivalence_group_id"] != rc.group for m in meta), eid
                        )
                        if ex["construction"]["kind"] == "other_family":
                            ck.check(
                                "other_family_has_no_gold_family", all(m["family"] != rc.family for m in meta), eid
                            )
            ctx = ex["context"]
            ck.check("context_within_limit", len(ctx) <= max_turns, eid)
            ck.check(
                "context_roles_and_content",
                all(t["role"] in ("user", "assistant") and t["content"].strip() for t in ctx),
                eid,
            )
            ck.check("message_non_empty", bool(ex["current_message"].strip()), eid)
            ck.check("node_non_empty", bool(ex["node"]["description"].strip()), eid)
            if ex["node"]["origin"] == "sgd_service_schema":
                svcs = {conds[m["condition_uid"]].extra["service"] for m in meta}
                ck.check("single_service_node_matches_candidates", len(svcs) == 1, eid)
            if ex["source"]["dataset"] in single_turn:
                ck.check("single_turn_has_empty_context", not ctx, eid)
                key_text = norm_text(ex["current_message"])
                texts[(ex["source"]["dataset"], split)].add(key_text)
                text_counts[(ex["source"]["dataset"], split, key_text)] += 1
            # Leakage unit. Dialogues: the whole dialogue. Single-turn data grouped by annotator (HWU64): the
            # (annotator, condition) pair, because held-out conditions are routed by label and may legitimately
            # share an annotator with other splits; what must never leak is one annotator's paraphrases of the SAME condition.
            gkey = ex["split_group_id"]
            if ex["source"]["dataset"] in single_turn:
                if ex["label_type"] == "route":
                    gkey += "|" + next(m["condition_uid"] for m in meta if m["role"] == "gold")
                else:
                    gkey += "|" + str(ex["construction"]["removed_gold_uid"])
            groups_by_split[split].add(gkey)

    # paired (SGD-X held-out paraphrase) examples must mirror their main example exactly
    for ex in (e for s in SPLITS for e in examples[s]):
        pid = ex["construction"]["paired_with"]
        if pid is None:
            continue
        main = by_id.get(pid)
        ck.check("paired_main_exists", main is not None, ex["example_id"])
        if main is not None:
            same = (
                main["split"] == ex["split"]
                and main["current_message"] == ex["current_message"]
                and main["context"] == ex["context"]
                and main["gold_cid"] == ex["gold_cid"]
                and [m["condition_uid"] for m in main["candidate_meta"]]
                == [m["condition_uid"] for m in ex["candidate_meta"]]
            )
            ck.check("paired_mirrors_main", same, ex["example_id"])
            ck.check(
                "paired_uses_held_out_variant",
                ex["construction"]["description_variant"] in (cfg["sgdx"]["val_variant"], cfg["sgdx"]["test_variant"]),
                ex["example_id"],
            )
    train_variants = set(cfg["sgdx"]["train_variant_weights"])
    for ex in examples["train"]:
        ck.check(
            "train_never_uses_held_out_variant",
            ex["construction"]["description_variant"] in train_variants | {"orig"},
            ex["example_id"],
        )

    # dialogue / worker / utterance groups never cross splits
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        both = groups_by_split[a] & groups_by_split[b]
        by_ds = Counter(g.split(":", 1)[0] for g in both)
        ck.check(
            f"split_groups_disjoint_{a}_{b}",
            not both,
            f"{len(both)} shared groups by dataset {dict(by_ds)}, e.g. {sorted(both)[:3]}",
        )

    # single-turn texts never cross splits (within a dataset, and train vs later splits across datasets)
    for ds in sorted(single_turn):
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            both = texts[(ds, a)] & texts[(ds, b)]
            ck.check(f"single_turn_text_disjoint_{ds}_{a}_{b}", not both, f"{len(both)} shared e.g. {sorted(both)[:3]}")
    dup_texts = [k for k, n in text_counts.items() if n > 1]
    ck.check(
        "single_turn_text_unique_within_dataset_split",
        not dup_texts,
        f"{len(dup_texts)} duplicated, e.g. {sorted(dup_texts)[:3]}",
    )
    train_texts = set().union(*[texts[(ds, "train")] for ds in single_turn]) if single_turn else set()
    for ds in sorted(single_turn):
        for s in ("val", "test"):
            both = texts[(ds, s)] & train_texts
            ck.check(
                f"single_turn_text_not_in_any_train_{ds}_{s}", not both, f"{len(both)} shared e.g. {sorted(both)[:3]}"
            )

    # ---- soft checks
    sizes = {s: len(examples[s]) for s in SPLITS}
    for s, n in sizes.items():
        if n == 0:
            warnings.append(f"split '{s}' is empty")
    for s in SPLITS:
        pos: dict[int, Counter] = defaultdict(Counter)
        for ex in examples[s]:
            if ex["label_type"] == "route" and ex["construction"]["paired_with"] is None:
                pos[ex["construction"]["n_candidates"]][ex["construction"]["gold_position"]] += 1
        for k, cnt in sorted(pos.items()):
            n = sum(cnt.values())
            if n >= 500:
                dev = max(abs(cnt.get(i, 0) / n - 1 / k) for i in range(k))
                if dev > 0.05:
                    warnings.append(f"{s}: gold position non-uniform for k={k} (max deviation {dev:.3f}, n={n})")
    risky = sum(1 for s in SPLITS for ex in examples[s] if ex["construction"]["label_risk"] != "none")
    total = sum(sizes.values())
    if total and risky / total > 0.05:
        warnings.append(f"{100 * risky / total:.1f}% of examples carry label_risk (sibling_partial_validity)")
    unreviewed = sum(1 for c in conds.values() if not c.reviewed)
    if unreviewed:
        warnings.append(f"{unreviewed} condition descriptions are unreviewed LLM drafts (see data/curated/*)")

    failures = ck.failures()
    report = {
        "checks": {k: {"violations": v["violations"], "samples": v["samples"]} for k, v in sorted(ck.results.items())},
        "n_checks": len(ck.results),
        "n_failed": len(failures),
        "warnings": warnings,
    }
    if failures:
        lines = [f"  - {k}: {v['violations']} violation(s), e.g. {v['samples'][:2]}" for k, v in failures.items()]
        raise PipelineError("validation failed:\n" + "\n".join(lines))
    return report
