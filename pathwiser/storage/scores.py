"""Persistent per-candidate scores: ``data/cache/scores/<data version>/<scorer>/<split>.npz``.

A stored file records the example ids, the candidate counts and a fingerprint of the scored texts. ``load_scores`` refuses to return
scores for records that differ in any of them, so scores cannot be attached to the wrong examples (for instance after an
example's text was edited). The cache is reproducible from the code and git-ignored.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from pathwiser.paths import DATA_DIR

DEFAULT_ROOT = DATA_DIR / "cache" / "scores"


class ScoreMismatch(ValueError):
    """Stored scores do not belong to the given records."""


def text_fingerprint(records: list[dict]) -> str:
    """SHA-256 over ids, messages, contexts and candidate texts (order-sensitive)."""
    h = hashlib.sha256()
    for r in records:
        h.update(r["example_id"].encode("utf-8"))
        h.update(b"\x1e")
        h.update(r["input"]["message"].encode("utf-8"))
        for t in r["input"]["context"]:
            h.update(b"\x1d" + t["content"].encode("utf-8"))
        for c in r["input"]["conditions"]:
            h.update(b"\x1f" + c.encode("utf-8"))
        h.update(b"\x1c")
    return h.hexdigest()


def score_path(root: str | Path, data_version: str, scorer: str, split: str) -> Path:
    return Path(root) / data_version / scorer / f"{split}.npz"


def save_scores(path: str | Path, records: list[dict], scores: list[np.ndarray], info: dict | None = None) -> Path:
    path = Path(path)
    sizes = np.array([len(r["input"]["conditions"]) for r in records], dtype=np.int32)
    if len(scores) != len(records) or any(len(s) != k for s, k in zip(scores, sizes)):
        raise ScoreMismatch("scores are not aligned with records")
    flat = (
        np.concatenate([np.asarray(s, dtype=np.float32) for s in scores]) if scores else np.zeros(0, dtype=np.float32)
    )
    if not np.isfinite(flat).all():
        raise ScoreMismatch("refusing to store non-finite scores")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
        np.savez_compressed(
            tmp,
            flat=flat,
            sizes=sizes,
            ids=np.array([r["example_id"] for r in records]),
            fingerprint=np.array(text_fingerprint(records)),
            info=np.array(json.dumps(info or {}, ensure_ascii=False)),
        )
    os.replace(tmp.name, path)
    return path


def load_scores(path: str | Path, records: list[dict]) -> list[np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        flat, sizes, ids, fp = z["flat"], z["sizes"], z["ids"], str(z["fingerprint"])
    want_sizes = np.array([len(r["input"]["conditions"]) for r in records], dtype=np.int32)
    if len(ids) != len(records) or not np.array_equal(sizes, want_sizes):
        raise ScoreMismatch(
            f"{path}: stored for {len(ids)} examples, got {len(records)} (or different candidate counts)"
        )
    if list(ids) != [r["example_id"] for r in records]:
        raise ScoreMismatch(f"{path}: example ids differ")
    if fp != text_fingerprint(records):
        raise ScoreMismatch(f"{path}: example texts changed since these scores were computed")
    return list(np.split(flat.astype(np.float64), np.cumsum(sizes)[:-1]))


def load_info(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["info"]))


def available(root: str | Path, data_version: str, scorer: str) -> list[str]:
    d = Path(root) / data_version / scorer
    return sorted(p.stem for p in d.glob("*.npz")) if d.exists() else []
