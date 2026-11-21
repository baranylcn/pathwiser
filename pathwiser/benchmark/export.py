"""Flat CSV views of the dataset for inspection in Excel / pandas / any spreadsheet tool.

The JSONL files stay the source of truth for training; these CSVs are derived, read-only views.
Files are UTF-8 with BOM (so Excel decodes them correctly) and use LF line endings.
Excel limits: 1,048,576 rows and 32,767 characters per cell; both are respected (long cells are truncated).
"""

from __future__ import annotations

import csv
from pathlib import Path

from pathwiser.benchmark.schema import SPLITS, Condition
from pathwiser.storage.hashing import sha256_file

CELL_LIMIT = 32000
ROLE_ABBR = {"user": "U", "assistant": "A"}

EXAMPLE_COLUMNS = [
    "example_id",
    "split",
    "tier",
    "dataset",
    "label_type",
    "nm_kind",
    "label_risk",
    "n_candidates",
    "gold_position",
    "cond_seen",
    "svc_seen",
    "ctx_kind",
    "current_message",
    "msg_words",
    "context",
    "n_context_turns",
    "node_name",
    "node_description",
    "gold_cid",
    "gold_description",
    "gold_condition_uid",
    "gold_family",
    "removed_gold_uid",
    "candidates",
    "candidate_families",
    "negative_strategies",
    "description_variant",
    "paired_with",
    "slices",
    "dialogue_id",
    "moment_id",
]
CANDIDATE_COLUMNS = [
    "example_id",
    "split",
    "dataset",
    "label_type",
    "cid",
    "is_gold",
    "description",
    "condition_uid",
    "family",
    "equivalence_group_id",
    "role",
    "negative_strategy",
    "description_variant",
    "description_origin",
    "reviewed",
]
CONDITION_COLUMNS = [
    "condition_uid",
    "dataset",
    "family",
    "equivalence_group_id",
    "partition",
    "description_orig",
    "n_description_variants",
    "description_origin",
    "reviewed",
]


def _cell(text: object) -> str:
    s = "" if text is None else str(text)
    return s if len(s) <= CELL_LIMIT else s[: CELL_LIMIT - 1] + "…"


def _slice_value(slices: list[str], prefix: str) -> str:
    return next((s.split(":", 1)[1] for s in slices if s.startswith(prefix + ":")), "")


def flatten_example(ex: dict) -> dict:
    meta = ex["candidate_meta"]
    cands = ex["candidates"]
    gold = next((m for m in meta if m["role"] == "gold"), None)
    gold_desc = next((c["description"] for c in cands if gold and c["cid"] == gold["cid"]), "")
    return {
        "example_id": ex["example_id"],
        "split": ex["split"],
        "tier": ex["tier"],
        "dataset": ex["source"]["dataset"],
        "label_type": ex["label_type"],
        "nm_kind": ex["construction"]["kind"] if ex["label_type"] == "no_match" else "",
        "label_risk": ex["construction"]["label_risk"],
        "n_candidates": ex["construction"]["n_candidates"],
        "gold_position": ex["construction"]["gold_position"],
        "cond_seen": _slice_value(ex["slices"], "cond"),
        "svc_seen": _slice_value(ex["slices"], "svc"),
        "ctx_kind": _slice_value(ex["slices"], "ctx"),
        "current_message": ex["current_message"],
        "msg_words": ex["features"]["msg_words"],
        "context": " | ".join(f"{ROLE_ABBR[t['role']]}: {t['content']}" for t in ex["context"]),
        "n_context_turns": ex["features"]["n_context_turns"],
        "node_name": ex["node"]["name"],
        "node_description": ex["node"]["description"],
        "gold_cid": ex["gold_cid"] or "",
        "gold_description": gold_desc,
        "gold_condition_uid": gold["condition_uid"] if gold else "",
        "gold_family": gold["family"] if gold else "",
        "removed_gold_uid": ex["construction"]["removed_gold_uid"] or "",
        "candidates": "\n".join(
            f"{'*' if gold and c['cid'] == gold['cid'] else ' '}{c['cid']}: {c['description']}" for c in cands
        ),
        "candidate_families": ", ".join(m["family"] for m in meta),
        "negative_strategies": ", ".join(m["negative_strategy"] or "GOLD" for m in meta),
        "description_variant": ex["construction"]["description_variant"],
        "paired_with": ex["construction"]["paired_with"] or "",
        "slices": ";".join(ex["slices"]),
        "dialogue_id": ex["source"]["dialogue_id"],
        "moment_id": ex["source"]["moment_id"],
    }


def _writer(path: Path, columns: list[str]):
    f = open(path, "w", encoding="utf-8-sig", newline="")  # noqa: SIM115 - the caller owns and closes the handle
    return f, csv.DictWriter(f, fieldnames=columns, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)


def export_csv(cfg: dict, examples: dict[str, list[dict]], conditions: list[Condition], out: Path) -> dict[str, dict]:
    ecfg = cfg["export"]
    files: dict[str, dict] = {}

    def record(name: str, rows: int) -> None:
        files[name] = {"rows": rows, "sha256": sha256_file(out / name)}

    n = 0
    f, w = _writer(out / "examples_flat.csv", EXAMPLE_COLUMNS)
    with f:
        w.writeheader()
        for split in SPLITS:
            for ex in examples[split]:
                w.writerow({k: _cell(v) for k, v in flatten_example(ex).items()})
                n += 1
    record("examples_flat.csv", n)

    # small, balanced sample: first N (hash-ordered => random but reproducible) main examples per dataset x split
    per = ecfg["sample_per_dataset_split"]
    taken: dict[tuple[str, str], int] = {}
    n = 0
    f, w = _writer(out / "examples_sample.csv", EXAMPLE_COLUMNS)
    with f:
        w.writeheader()
        for split in SPLITS:
            for ex in examples[split]:
                key = (ex["source"]["dataset"], split)
                if ex["construction"]["paired_with"] is None and taken.get(key, 0) < per:
                    taken[key] = taken.get(key, 0) + 1
                    w.writerow({k: _cell(v) for k, v in flatten_example(ex).items()})
                    n += 1
    record("examples_sample.csv", n)

    if ecfg["candidates_long"]:
        n = 0
        f, w = _writer(out / "candidates_long.csv", CANDIDATE_COLUMNS)
        with f:
            w.writeheader()
            for split in SPLITS:
                for ex in examples[split]:
                    desc = {c["cid"]: c["description"] for c in ex["candidates"]}
                    for m in ex["candidate_meta"]:
                        w.writerow(
                            {
                                "example_id": ex["example_id"],
                                "split": split,
                                "dataset": ex["source"]["dataset"],
                                "label_type": ex["label_type"],
                                "cid": m["cid"],
                                "is_gold": int(m["role"] == "gold"),
                                "description": _cell(desc[m["cid"]]),
                                "condition_uid": m["condition_uid"],
                                "family": m["family"],
                                "equivalence_group_id": m["equivalence_group_id"],
                                "role": m["role"],
                                "negative_strategy": m["negative_strategy"] or "",
                                "description_variant": m["description_variant"],
                                "description_origin": m["description_origin"],
                                "reviewed": int(m["reviewed"]),
                            }
                        )
                        n += 1
        record("candidates_long.csv", n)

    f, w = _writer(out / "conditions.csv", CONDITION_COLUMNS)
    with f:
        w.writeheader()
        for c in conditions:
            w.writerow(
                {
                    "condition_uid": c.uid,
                    "dataset": c.dataset,
                    "family": c.family,
                    "equivalence_group_id": c.group,
                    "partition": c.partition,
                    "description_orig": c.descriptions["orig"],
                    "n_description_variants": len(c.descriptions),
                    "description_origin": c.description_origin,
                    "reviewed": int(c.reviewed),
                }
            )
    record("conditions.csv", len(conditions))
    return files
