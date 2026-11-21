"""Business-context taxonomy for synthetic scenarios and its split by group.

Splits are made by *group* (industry cluster), never by example and never by domain inside a group, so that an evaluation domain is not
a close sibling of a training domain. Groups flagged ``near`` are adjacent to benchmark or ``realistic-v1`` topics and are training-only.
The assignment is made once with a seeded shuffle (``assign_splits``) and stored in the taxonomy file (``write_splits``), so the committed
file, not a random call at run time, is the source of truth.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from pathwiser.paths import REPO_ROOT

DEFAULT_PATH = REPO_ROOT / "data" / "curated" / "synth" / "taxonomy_v1.json"
SEED = 20261121
SPLIT_COUNTS = {
    "eval_v2": 6,
    "synth_test": 3,
    "synth_val": 3,
}  # number of far groups per split; everything else is train
SPLITS = ("eval_v2", "synth_test", "synth_val", "train")

# Topics of the benchmark datasets: nothing in the taxonomy may be one of these.
BENCHMARK_TOPICS = {
    "restaurants",
    "hotels",
    "flights",
    "trains",
    "buses",
    "rental_cars",
    "ride_sharing",
    "movies",
    "music",
    "events",
    "banks",
    "payments",
    "weather",
    "calendar",
    "alarm",
    "messaging",
    "media",
    "homes",
    "apartment",
    "doctor",
    "trivia",
    "small_talk",
    "order_issue",
    "shipping_issue",
    "subscription_inquiry",
    "account_access",
    "product_defect",
    "purchase_dispute",
}


def load(path: str | Path = DEFAULT_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def realistic_v1_blocklist(root: Path = REPO_ROOT) -> set[str]:
    """Scenario ids and domain labels of ``realistic-v1`` (read from its authored files)."""
    out: set[str] = set()
    for f in sorted((root / "data" / "curated" / "realistic_v1").glob("scenarios_*.json")):
        for sc in json.loads(f.read_text(encoding="utf-8")):
            out.add(sc["id"])
            out.add(sc["domain"])
    return out


def blocklist(root: Path = REPO_ROOT) -> set[str]:
    return set(BENCHMARK_TOPICS) | realistic_v1_blocklist(root)


def validate(tax: dict, blocked: set[str] | None = None) -> None:
    """Hard checks; raises ValueError listing every problem."""
    problems: list[str] = []
    blocked = blocklist() if blocked is None else blocked
    group_ids, domain_ids = set(), set()
    for g in tax["groups"]:
        if g["id"] in group_ids:
            problems.append(f"duplicate group {g['id']}")
        group_ids.add(g["id"])
        if not isinstance(g.get("near"), bool):
            problems.append(f"{g['id']}: 'near' must be true or false")
        if "split" in g and g["split"] not in SPLITS:
            problems.append(f"{g['id']}: unknown split {g['split']!r}")
        if g.get("near") and g.get("split", "train") != "train":
            problems.append(f"{g['id']}: a near group can only be used for training")
        if not 3 <= len(g["domains"]) <= 12:
            problems.append(f"{g['id']}: expected 3-12 domains, got {len(g['domains'])}")
        for d in g["domains"]:
            if d["id"] in domain_ids:
                problems.append(f"duplicate domain {d['id']}")
            domain_ids.add(d["id"])
            if not re.fullmatch(r"[a-z][a-z0-9_]*", d["id"]):
                problems.append(f"{d['id']}: ids are snake_case")
            if d["id"] in blocked or g["id"] in blocked:
                problems.append(f"{d['id']}: collides with a blocked benchmark / realistic-v1 name")
            if len(d.get("description", "").split()) < 5:
                problems.append(f"{d['id']}: description too short")
    if problems:
        raise ValueError("invalid taxonomy:\n  " + "\n  ".join(problems))


def assign_splits(tax: dict, seed: int = SEED, counts: dict[str, int] | None = None) -> dict[str, str]:
    """Group id -> split. Far groups are shuffled with ``seed`` and dealt to eval_v2, synth_test, synth_val; the rest and all near groups train."""
    counts = SPLIT_COUNTS if counts is None else counts
    far = sorted(g["id"] for g in tax["groups"] if not g["near"])
    if sum(counts.values()) >= len(far):
        raise ValueError("not enough far groups to leave any for training")
    random.Random(seed).shuffle(far)
    out, i = {}, 0
    for split, n in counts.items():
        for gid in far[i : i + n]:
            out[gid] = split
        i += n
    for g in tax["groups"]:
        out.setdefault(g["id"], "train")
    return out


def write_splits(path: str | Path = DEFAULT_PATH, seed: int = SEED) -> dict[str, str]:
    """Store the assignment in the file (text edit, so the hand-written layout is kept)."""
    path = Path(path)
    tax = load(path)
    splits = assign_splits(tax, seed)
    text = path.read_text(encoding="utf-8")
    for gid, split in splits.items():
        text, n = re.subn(
            rf'(\{{"id": "{gid}", "near": (?:true|false),)(?: "split": "[a-z_0-9]+",)?',
            rf'\1 "split": "{split}",',
            text,
        )
        if n != 1:
            raise ValueError(f"could not place the split of group {gid!r} ({n} matches)")
    path.write_text(text, encoding="utf-8", newline="\n")
    return splits


def domains_by_split(tax: dict, split: str) -> list[tuple[str, dict]]:
    """(group id, domain) pairs of a split, in file order. Requires the splits to have been written."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    out = []
    for g in tax["groups"]:
        if "split" not in g:
            raise ValueError("splits are not stored in the taxonomy; run write_splits()")
        if g["split"] == split:
            out += [(g["id"], d) for d in g["domains"]]
    return out
