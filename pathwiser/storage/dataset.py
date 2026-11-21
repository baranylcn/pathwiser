"""Loading of the model-ready dataset (``data/model/<version>``) written by ``pathwiser benchmark records``."""

from __future__ import annotations

import json
from pathlib import Path

from pathwiser.paths import DATA_DIR

SPLITS = ("train", "val", "test")
DEFAULT_DATA_DIR = DATA_DIR / "model" / "v1"


def load_split(split: str, data_dir: str | Path = DEFAULT_DATA_DIR) -> list[dict]:
    """All examples of one split, in file (= example id) order."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    path = Path(data_dir) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `pathwiser benchmark records` first")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_manifest(data_dir: str | Path = DEFAULT_DATA_DIR) -> dict:
    return json.loads((Path(data_dir) / "manifest.json").read_text(encoding="utf-8"))


def main_examples(records: list[dict]) -> list[dict]:
    """Examples that are not SGD-X paraphrase twins (those are reported separately)."""
    return [r for r in records if not r["meta"]["is_paired"]]


def collate_listwise(batch: list[dict], max_k: int = 10) -> dict:
    """Mini-batch -> framework-agnostic arrays (numpy). Conditions are padded to ``max_k`` with '' and masked."""
    import numpy as np

    conditions = [[""] * max_k for _ in batch]
    mask = np.zeros((len(batch), max_k), dtype=bool)
    for i, m in enumerate(batch):
        cs = m["input"]["conditions"]
        if len(cs) > max_k:
            raise ValueError(f"{m['example_id']}: {len(cs)} candidates exceed max_k={max_k}")
        conditions[i][: len(cs)] = cs
        mask[i, : len(cs)] = True
    return {
        "query": [m["input"]["query"] for m in batch],
        "conditions": conditions,  # (B, max_k) text
        "cond_mask": mask,  # (B, max_k) real candidate?
        "label": np.array([m["target"]["label"] for m in batch]),  # -1 => no_match
        "label_null": np.array([m["target"]["label_null"] for m in batch]),  # k => no_match
        "weight": np.array([m["weight"] for m in batch], dtype=np.float32),
    }


def explode_pairs(batch: list[dict]) -> dict:
    """Listwise -> pairwise: one row per (example, candidate); ``group`` is the example index in ``batch``."""
    import numpy as np

    query, condition, is_gold, group = [], [], [], []
    for i, m in enumerate(batch):
        for j, cond in enumerate(m["input"]["conditions"]):
            query.append(m["input"]["query"])
            condition.append(cond)
            is_gold.append(int(j == m["target"]["label"]))
            group.append(i)
    return {"query": query, "condition": condition, "is_gold": np.array(is_gold), "group": np.array(group)}
