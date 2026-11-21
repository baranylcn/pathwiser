"""Deterministic randomness: every draw derives from a SHA-256 of explicit keys, never from global state or ``hash()``."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence


def stable_int(*parts: object) -> int:
    """64-bit integer derived from the given parts (order-sensitive)."""
    payload = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def rng_for(seed: int, *parts: object) -> random.Random:
    """Independent RNG for ``(seed, *parts)``; unaffected by what else was generated."""
    return random.Random(stable_int(seed, *parts))


def stable_unit(seed: int, *parts: object) -> float:
    """Deterministic float in [0, 1) for ``(seed, *parts)``."""
    return (stable_int(seed, *parts) >> 11) / float(1 << 53)


def weighted_choice(rng: random.Random, items: Sequence[tuple[object, float]]) -> object:
    """Pick a key from ``[(key, weight), ...]`` (order of ``items`` matters for determinism)."""
    total = sum(w for _, w in items)
    x = rng.random() * total
    acc = 0.0
    for key, weight in items:
        acc += weight
        if x < acc:
            return key
    return items[-1][0]
