"""Condition partitions, split assignment and leakage-safe de-duplication.

Partition (per condition): ``train`` -> usable everywhere; ``val`` -> only val/test; ``test`` -> only test.
A condition family listed under ``heldout_families`` is never seen in training (unseen-condition evaluation).

Split assignment per source:
  official           the dataset's own splits (SGD).
  dialogue_routed    whole dialogues move to val/test if they contain a held-out gold condition, so a
                     dialogue is never split across train and evaluation. Otherwise the base split holds.
  partition_routed   single utterances: gold in a held-out family goes to val/test, otherwise base split.
"""

from __future__ import annotations

from pathwiser.benchmark.schema import SPLIT_RANK, SPLITS, Condition, DropLog, PipelineError, SourceData
from pathwiser.benchmark.text import norm_text


def assign_partitions(dataset: str, conditions: dict[str, Condition], heldout: dict | None) -> None:
    heldout = heldout or {}
    families = {c.family for c in conditions.values()}
    for part in ("val", "test"):
        for fam in heldout.get(part, []):
            if fam not in families:
                raise PipelineError(
                    f"{dataset}: held-out family '{fam}' ({part}) does not exist; known: {sorted(families)}"
                )
    val, test = set(heldout.get("val", [])), set(heldout.get("test", []))
    for c in conditions.values():
        c.partition = "test" if c.family in test else ("val" if c.family in val else "train")


def assign_splits(sd: SourceData) -> None:
    conds = sd.conditions
    if sd.split_mode == "official":
        for m in sd.moments:
            m.split = m.base_split
    elif sd.split_mode == "partition_routed":
        for m in sd.moments:
            rank = 0 if m.gold_uid is None else SPLIT_RANK[conds[m.gold_uid].partition]
            m.split = SPLITS[rank] if rank > 0 else m.base_split
    elif sd.split_mode == "dialogue_routed":
        worst: dict[str, int] = {}
        for m in sd.moments:
            rank = SPLIT_RANK[conds[m.gold_uid].partition]
            worst[m.group_id] = max(worst.get(m.group_id, 0), rank)
        for m in sd.moments:
            rank = worst[m.group_id]
            m.split = SPLITS[rank] if rank > 0 else m.base_split
    else:
        raise PipelineError(f"unknown split mode '{sd.split_mode}'")


def dedupe_single_turn(sources: list[SourceData], drops: DropLog) -> None:
    """Remove label conflicts and duplicates among single-turn sources (mutates ``sd.moments``).

    1. Within a dataset, the same normalised text with different golds is ambiguous -> all copies dropped.
    2. The earliest occurrence (train < val < test, then dataset, then id) is kept; a later copy in the
       same split is a duplicate, and a copy in a later split (of any single-turn dataset) would leak.
    """
    singles = [sd for sd in sources if sd.single_turn]
    for sd in singles:
        by_text: dict[str, set[str | None]] = {}
        for m in sd.moments:
            by_text.setdefault(norm_text(m.message), set()).add(m.gold_uid)
        conflict = {t for t, golds in by_text.items() if len(golds) > 1}
        kept = []
        for m in sd.moments:
            t = norm_text(m.message)
            if not t:
                drops.add(sd.dataset, "empty_message")
            elif t in conflict:
                drops.add(sd.dataset, "label_conflict")
            else:
                kept.append(m)
        sd.moments = kept
    order = sorted(
        ((SPLIT_RANK[m.split], sd.dataset, m.moment_id, m) for sd in singles for m in sd.moments),
        key=lambda x: (x[0], x[1], x[2]),
    )
    first_seen: dict[str, tuple[int, str]] = {}
    seen_in_dataset_split: set[tuple[str, int, str]] = set()  # (dataset, split rank, text)
    drop_ids: set[str] = set()
    for rank, ds, mid, m in order:
        t = norm_text(m.message)
        # A copy inside the same dataset and split is always a duplicate. This must be tracked separately from
        # ``first_seen``: when another dataset owns the first occurrence of a text, comparing only against it would
        # let two copies from the same dataset both survive.
        if (ds, rank, t) in seen_in_dataset_split:
            drops.add(ds, "duplicate_within_split")
            drop_ids.add(mid)
            continue
        seen_in_dataset_split.add((ds, rank, t))
        if t not in first_seen:
            first_seen[t] = (rank, ds)
            continue
        f_rank, f_ds = first_seen[t]
        if f_rank < rank:
            drops.add(ds, "duplicate_of_earlier_split" if f_ds == ds else "duplicate_of_other_dataset_earlier_split")
            drop_ids.add(mid)
    for sd in singles:
        sd.moments = [m for m in sd.moments if m.moment_id not in drop_ids]
