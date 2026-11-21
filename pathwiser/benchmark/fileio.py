"""JSON and text readers/writers that return content hashes (UTF-8, LF line endings)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _dumps(obj: object) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> tuple[int, str]:
    """Write rows as UTF-8 JSONL with LF endings. Returns (row_count, sha256)."""
    h = hashlib.sha256()
    n = 0
    with open(path, "wb") as f:
        for row in rows:
            b = _dumps(row)
            f.write(b)
            h.update(b)
            n += 1
    return n, h.hexdigest()


def write_json(path: Path, obj: object, indent: int = 2) -> str:
    data = (json.dumps(obj, ensure_ascii=False, indent=indent) + "\n").encode("utf-8")
    with open(path, "wb") as f:
        f.write(data)
    return hashlib.sha256(data).hexdigest()


def write_text(path: Path, text: str) -> str:
    data = text.encode("utf-8")
    with open(path, "wb") as f:
        f.write(data)
    return hashlib.sha256(data).hexdigest()


def iter_chunks(seq: Sequence, size: int) -> Iterator[Sequence]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]
