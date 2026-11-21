"""Deterministic character- and word-level noise for user messages.

An LLM's imitation of typing errors does not look like real errors, so the generator writes clean persona text and this module damages it
afterwards. Every function is a pure function of ``(text, seed_key)``: the same inputs always give the same output.

``apply_noise`` returns ``(text, applied)``. ``applied`` is False when the requested noise found nothing to change (for example a message
without any word that has a text-speak form); callers record the noise that was really applied, not only the one that was planned.
Numbers, identifiers and words with digits are never touched by the typo functions, so noise cannot change a reference number.
"""

from __future__ import annotations

import hashlib
import random
import re
from functools import partial

NOISE_TYPES = ("none", "keyboard_typos", "no_punct_lower", "text_speak", "autocorrect")

_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
NEIGHBORS: dict[str, str] = {}
for _r, _row in enumerate(_ROWS):
    for _i, _c in enumerate(_row):
        near = ""
        if _i > 0:
            near += _row[_i - 1]
        if _i < len(_row) - 1:
            near += _row[_i + 1]
        for dr in (-1, 1):
            if 0 <= _r + dr < len(_ROWS):
                other = _ROWS[_r + dr]
                near += "".join(other[j] for j in (_i - 1, _i, _i + 1) if 0 <= j < len(other))
        NEIGHBORS[_c] = near

TEXT_SPEAK = {
    "you": "u",
    "your": "ur",
    "you're": "ur",
    "are": "r",
    "to": "2",
    "too": "2",
    "tomorrow": "tmrw",
    "today": "2day",
    "tonight": "2nite",
    "please": "pls",
    "thanks": "thx",
    "because": "bc",
    "before": "b4",
    "with": "w/",
    "without": "w/o",
    "about": "abt",
    "between": "btwn",
    "message": "msg",
    "account": "acct",
    "appointment": "appt",
    "information": "info",
    "and": "&",
    "people": "ppl",
    "really": "rly",
    "okay": "k",
    "though": "tho",
    "want": "wanna",
    "going to": "gonna",
    "right now": "rn",
}
CONFUSIONS = [
    ("their", "there"),
    ("there", "their"),
    ("they're", "there"),
    ("its", "it's"),
    ("it's", "its"),
    ("your", "you're"),
    ("you're", "your"),
    ("then", "than"),
    ("than", "then"),
    ("lose", "loose"),
    ("loose", "lose"),
    ("affect", "effect"),
    ("effect", "affect"),
    ("to", "too"),
    ("too", "to"),
    ("whether", "weather"),
    ("weather", "whether"),
    ("accept", "except"),
    ("break", "brake"),
    ("week", "weak"),
    ("whole", "hole"),
    ("write", "right"),
    ("new", "knew"),
    ("buy", "by"),
    ("our", "are"),
]


def _rng(noise: str, seed_key: str) -> random.Random:
    return random.Random(int(hashlib.sha256(f"{noise}\x00{seed_key}".encode()).hexdigest()[:16], 16))


def _typo_word(word: str, rng: random.Random) -> str:
    """One keyboard-style error inside a word (never the first letter)."""
    i = rng.randrange(1, len(word))
    c = word[i]
    op = rng.choices(("sub", "drop", "double", "swap"), weights=(0.45, 0.20, 0.15, 0.20))[0]
    if op == "sub" and c.lower() in NEIGHBORS:
        rep = rng.choice(NEIGHBORS[c.lower()])
        return word[:i] + (rep.upper() if c.isupper() else rep) + word[i + 1 :]
    if op == "drop":
        return word[:i] + word[i + 1 :]
    if op == "double":
        return word[:i] + c + word[i:]
    if i < len(word) - 1:
        return word[:i] + word[i + 1] + word[i] + word[i + 2 :]
    return word[:i] + word[i - 1] + word[i:]


def keyboard_typos(text: str, seed_key: str) -> tuple[str, bool]:
    rng = _rng("keyboard_typos", seed_key)
    tokens = text.split(" ")
    cand = [
        i
        for i, t in enumerate(tokens)
        if len(re.sub(r"[^A-Za-z]", "", t)) >= 4
        and not re.search(r"\d", t)
        and t.isascii()
        and not t.startswith(("@", "#", "http"))
    ]
    if not cand:
        return text, False
    n = min(len(cand), 1 if len(tokens) < 8 else 2 if len(tokens) < 16 else 3)
    for i in rng.sample(cand, n):
        core = re.match(r"^(\W*)([A-Za-z']+)(\W*)$", tokens[i])
        if not core or len(core.group(2)) < 4:
            continue
        pre, w, post = core.groups()
        new = _typo_word(w, rng)
        if new != w:
            tokens[i] = pre + new + post
    out = " ".join(tokens)
    return out, out != text


def no_punct_lower(text: str, seed_key: str) -> tuple[str, bool]:
    rng = _rng("no_punct_lower", seed_key)
    out = text.lower()
    out = re.sub(r"(?<!\d)\.(?!\d)", "", out)  # keep decimals such as 3.5
    out = re.sub(r"[,;!?]", "", out)
    if rng.random() < 0.5:
        out = out.replace("'", "").replace("’", "")
    out = re.sub(r"\s+", " ", out).strip()
    return out, out != text


def text_speak(text: str, seed_key: str) -> tuple[str, bool]:
    rng = _rng("text_speak", seed_key)
    out = text
    hits = 0
    for long, short in sorted(TEXT_SPEAK.items(), key=lambda kv: -len(kv[0])):
        pattern = re.compile(rf"(?<![\w']){re.escape(long)}(?![\w'])", re.IGNORECASE)
        if pattern.search(out) and rng.random() < 0.75:
            out, n = pattern.subn(short, out, count=1)
            hits += n
    return out, hits > 0 and out != text


def _match_case(match: re.Match, word: str) -> str:
    return word.capitalize() if match.group(0)[0].isupper() else word


def autocorrect(text: str, seed_key: str) -> tuple[str, bool]:
    rng = _rng("autocorrect", seed_key)
    present = [(a, b) for a, b in CONFUSIONS if re.search(rf"(?<![\w']){re.escape(a)}(?![\w'])", text, re.IGNORECASE)]
    if not present:
        return text, False
    out = text
    for a, b in rng.sample(present, min(len(present), 2)):
        out = re.sub(
            rf"(?<![\w']){re.escape(a)}(?![\w'])",
            partial(_match_case, word=b),
            out,
            count=1,
            flags=re.IGNORECASE,
        )
    return out, out != text


_FUNCS = {
    "keyboard_typos": keyboard_typos,
    "no_punct_lower": no_punct_lower,
    "text_speak": text_speak,
    "autocorrect": autocorrect,
}


def apply_noise(text: str, noise: str, seed_key: str) -> tuple[str, bool]:
    """Damage ``text`` with the named noise; returns ``(new_text, applied)``."""
    if noise == "none":
        return text, False
    if noise not in _FUNCS:
        raise ValueError(f"unknown noise {noise!r}; expected one of {NOISE_TYPES}")
    return _FUNCS[noise](text, seed_key)
