"""Leakage checks of a synthetic set against the benchmark, ``realistic-v1`` and (optionally) other synthetic sets.

* exact: a message or condition text that already exists (case and surrounding space ignored),
* near messages: TF-IDF (word 1-2 grams) cosine similarity >= ``MSG_THRESHOLD`` to any reference message,
* near conditions: sentence-embedding cosine >= ``COND_THRESHOLD`` to any reference condition.

Exact matches are errors for an evaluation set. Near matches are reported with the matching reference text so each can be resolved by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

from pathwiser.paths import REPO_ROOT

MSG_THRESHOLD = 0.8
COND_THRESHOLD = 0.9


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def read_reference(*model_dirs: Path) -> tuple[list[str], list[str]]:
    """Unique messages and conditions of every ``*.jsonl`` split found in the given model-data directories."""
    msgs: dict[str, str] = {}
    conds: dict[str, str] = {}
    for d in model_dirs:
        for f in sorted(Path(d).glob("*.jsonl")):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    msgs.setdefault(norm(r["input"]["message"]), r["input"]["message"])
                    for c in r["input"]["conditions"]:
                        conds.setdefault(norm(c), c)
    return list(msgs.values()), list(conds.values())


def exact_leaks(texts: dict[str, str], reference: list[str]) -> list[dict]:
    ref = {norm(t) for t in reference}
    return [{"id": k, "text": v} for k, v in texts.items() if norm(v) in ref]


def duplicates_within(texts: dict[str, str]) -> list[dict]:
    seen: dict[str, str] = {}
    out = []
    for k, v in texts.items():
        n = norm(v)
        if n in seen:
            out.append({"id": k, "text": v, "same_as": seen[n]})
        seen.setdefault(n, k)
    return out


def tfidf_near(texts: dict[str, str], reference: list[str], threshold: float = MSG_THRESHOLD) -> list[dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not texts or not reference:
        return []
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True).fit(reference + list(texts.values()))
    ref = vec.transform(reference)
    ids = list(texts)
    q = vec.transform([texts[i] for i in ids])
    out = []
    for start in range(0, len(ids), 256):
        sims = (q[start : start + 256] @ ref.T).toarray()
        best = sims.argmax(axis=1)
        for row, j in enumerate(best):
            if sims[row, j] >= threshold:
                out.append(
                    {
                        "id": ids[start + row],
                        "text": texts[ids[start + row]],
                        "similarity": round(float(sims[row, j]), 3),
                        "reference": reference[j],
                    }
                )
    return out


def embedding_near(
    texts: dict[str, str],
    reference: list[str],
    threshold: float = COND_THRESHOLD,
    encoder=None,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> list[dict]:
    if not texts or not reference:
        return []
    if encoder is None:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(model)
    ids = list(texts)
    a = encoder.encode(
        [texts[i] for i in ids], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )
    b = encoder.encode(
        reference, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False, batch_size=128
    )
    sims = a @ b.T
    best = sims.argmax(axis=1)
    return [
        {"id": ids[i], "text": texts[ids[i]], "similarity": round(float(sims[i, j]), 3), "reference": reference[j]}
        for i, j in enumerate(best)
        if sims[i, j] >= threshold
    ]


def leakage_report(
    messages: dict[str, str],
    conditions: dict[str, str],
    reference_dirs: list[Path],
    encoder=None,
    use_embeddings: bool = True,
) -> dict:
    """``messages`` / ``conditions``: id -> text of the set under test (use the clean, pre-noise message)."""
    ref_msgs, ref_conds = read_reference(*reference_dirs)
    report = {
        "reference_messages": len(ref_msgs),
        "reference_conditions": len(ref_conds),
        "exact_messages": exact_leaks(messages, ref_msgs),
        "exact_conditions": exact_leaks(conditions, ref_conds),
        "duplicate_messages_within": duplicates_within(messages),
        "duplicate_conditions_within": duplicates_within(conditions),
        "near_messages": tfidf_near(messages, ref_msgs),
    }
    report["near_conditions"] = embedding_near(conditions, ref_conds, encoder=encoder) if use_embeddings else None
    report["embeddings_checked"] = use_embeddings
    return report


def default_reference_dirs(root: Path = REPO_ROOT) -> list[Path]:
    return [d for d in (root / "data" / "model" / "v1", root / "data" / "model" / "realistic-v1") if d.exists()]
