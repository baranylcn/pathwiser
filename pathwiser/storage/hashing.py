"""Content hashes used for provenance and cache validation."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def record_text_sha(record: dict) -> str:
    """Short fingerprint of what an annotator or a scorer sees: the message and the sorted candidate texts."""
    parts = [record["input"]["message"], *sorted(record["input"]["conditions"])]
    return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()[:16]
