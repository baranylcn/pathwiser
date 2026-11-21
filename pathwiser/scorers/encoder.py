"""Pretrained sentence-transformer cosine scorer (no fine-tuning)."""

from __future__ import annotations

import numpy as np


class SentenceEncoder:
    """Pretrained sentence-transformer, no fine-tuning. ``query_mode``: "message" or "query" (node + context + message)."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        query_mode: str = "message",
        batch_size: int = 128,
        query_prefix: str = "",
        doc_prefix: str = "",
    ) -> None:
        if query_mode not in ("message", "query"):
            raise ValueError("query_mode must be 'message' or 'query'")
        self.model_name, self.query_mode, self.batch_size = model_name, query_mode, batch_size
        self.query_prefix, self.doc_prefix = (
            query_prefix,
            doc_prefix,
        )  # some retrievers need "query: " / "passage: " style prefixes
        self.name = f"st_{query_mode}"
        self.params = {"model": model_name, "query_mode": query_mode, "normalize_embeddings": True}
        if query_prefix or doc_prefix:
            self.params.update(query_prefix=query_prefix, doc_prefix=doc_prefix)
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._model.eval()
            self.params["device"] = str(self._model.device)
            self.params["n_parameters"] = int(sum(p.numel() for p in self._model.parameters()))
        return self._model

    def fit(self, train: list[dict], val: list[dict] | None = None) -> None:
        self._load()

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self._load().encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )

    def score(self, records: list[dict]) -> list[np.ndarray]:
        key = "message" if self.query_mode == "message" else "query"
        q_texts = sorted({r["input"][key] for r in records})
        q_row = {t: i for i, t in enumerate(q_texts)}
        c_texts = sorted({c for r in records for c in r["input"]["conditions"]})
        c_row = {t: i for i, t in enumerate(c_texts)}
        q_emb = self._encode([self.query_prefix + t for t in q_texts])
        c_emb = self._encode([self.doc_prefix + t for t in c_texts])
        return [c_emb[[c_row[c] for c in r["input"]["conditions"]]] @ q_emb[q_row[r["input"][key]]] for r in records]
