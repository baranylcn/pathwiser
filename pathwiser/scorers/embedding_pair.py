"""Frozen sentence embeddings, pair features and a logistic regression or LightGBM model."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from pathwiser.scorers.base import context_text
from pathwiser.scorers.training import (
    candidate_sizes,
    hardware,
    pair_labels_and_weights,
    relative_features,
    sample_examples,
    split_flat,
)


def embed_cached(model, model_name: str, texts: list[str], cache_dir: Path, batch_size: int = 128) -> np.ndarray:
    """Normalised embeddings of ``texts`` (row i = texts[i]), cached on disk as float16 (also used on first computation,
    so cached and uncached runs are numerically identical)."""
    key = hashlib.sha256((model_name + "\x00" + "\x00".join(texts)).encode("utf-8")).hexdigest()[:20]
    path = cache_dir / f"emb_{key}.npy"
    if path.exists():
        return np.load(path).astype(np.float32)
    emb = model.encode(
        texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    ).astype(np.float16)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb.astype(np.float32)


class EmbeddingPairModel:
    """Frozen sentence embeddings -> pair features -> logistic regression or LightGBM."""

    needs_train = True
    needs_val = False
    CHUNK = 4000  # examples scored at a time (bounds memory)

    def __init__(
        self,
        learner: str = "lr",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        train_examples: int | None = 30000,
        seed: int = 42,
        cache_dir: str | Path = "data/cache",
        query_prefix: str = "",
        doc_prefix: str = "",
    ) -> None:
        if learner not in ("lr", "lgbm"):
            raise ValueError("learner must be 'lr' or 'lgbm'")
        self.learner, self.model_name, self.train_examples, self.seed = learner, model_name, train_examples, seed
        self.query_prefix, self.doc_prefix = query_prefix, doc_prefix
        self.cache_dir = Path(cache_dir)
        self.name = f"st_{learner}"
        self.params = {
            "learner": learner,
            "embedding_model": model_name,
            "train_examples": train_examples,
            "seed": seed,
            "features": "cos(message,cond), cos(context,cond), listwise-relative cosines, message*cond, |message-cond|",
            "sampling": "weighted without replacement (training weights)",
        }
        if query_prefix or doc_prefix:
            self.params.update(query_prefix=query_prefix, doc_prefix=doc_prefix)
        if learner == "lr":
            self.params["C"] = 0.05
        else:
            self.params.update(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.5)
        self.info: dict = {}
        self._st = self.scaler = self.model = None

    def _encoder(self):
        if self._st is None:
            from sentence_transformers import SentenceTransformer

            self._st = SentenceTransformer(self.model_name)
            self._st.eval()
        return self._st

    def _features(self, records: list[dict]) -> np.ndarray:
        sizes = candidate_sizes(records)
        st = self._encoder()
        msgs = sorted({r["input"]["message"] for r in records})
        ctxs = sorted({t for t in (context_text(r) for r in records) if t})
        conds = sorted({c for r in records for c in r["input"]["conditions"]})
        qp, dp = self.query_prefix, self.doc_prefix
        e_msg = embed_cached(st, self.model_name, [qp + t for t in msgs], self.cache_dir)
        e_ctx = (
            embed_cached(st, self.model_name, [qp + t for t in ctxs], self.cache_dir)
            if ctxs
            else np.zeros((0, e_msg.shape[1]), np.float32)
        )
        e_cond = embed_cached(st, self.model_name, [dp + t for t in conds], self.cache_dir)
        msg_row, ctx_row = {t: i for i, t in enumerate(msgs)}, {t: i for i, t in enumerate(ctxs)}
        cond_row = {t: i for i, t in enumerate(conds)}
        m = e_msg[np.repeat([msg_row[r["input"]["message"]] for r in records], sizes)]
        e = e_cond[[cond_row[c] for r in records for c in r["input"]["conditions"]]]
        has_ctx = np.array([bool(context_text(r)) for r in records])
        c = np.zeros_like(m)
        if ctxs:
            rows = np.array([ctx_row.get(context_text(r), -1) for r in records])
            per_ex = np.where(rows[:, None] >= 0, e_ctx[np.maximum(rows, 0)], 0.0)
            c = np.repeat(per_ex, sizes, axis=0)
        cos = np.column_stack([(m * e).sum(1), (c * e).sum(1)])
        return np.hstack(
            [
                cos,
                relative_features(cos, sizes),
                np.column_stack([np.repeat(has_ctx.astype(np.float32), sizes), np.log(np.repeat(sizes, sizes))]),
                m * e,
                np.abs(m - e),
            ]
        ).astype(np.float32)

    def fit(self, train: list[dict], val: list[dict] | None = None) -> None:
        t0 = time.perf_counter()
        subset = sample_examples(train, self.train_examples, self.seed)
        x = np.vstack([self._features(subset[i : i + self.CHUNK]) for i in range(0, len(subset), self.CHUNK)])
        y, w = pair_labels_and_weights(subset)
        t_feat = time.perf_counter() - t0
        if self.learner == "lr":
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler().fit(x)
            self.model = LogisticRegression(C=self.params["C"], max_iter=500, random_state=self.seed)
            self.model.fit(self.scaler.transform(x), y, sample_weight=w)
        else:
            import lightgbm as lgb

            self.model = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.5,
                random_state=self.seed,
                n_jobs=8,
                verbose=-1,
            )
            self.model.fit(x, y, sample_weight=w)
        self.info = {
            "train_examples": len(subset),
            "train_pairs": int(len(y)),
            "feature_seconds": round(t_feat, 1),
            "train_seconds": round(time.perf_counter() - t0, 1),
            "n_features": int(x.shape[1]),
            "hardware": hardware(),
        }

    def _raw(self, x: np.ndarray) -> np.ndarray:
        if self.learner == "lr":
            return self.model.decision_function(self.scaler.transform(x))
        return self.model.predict(x, raw_score=True)

    def score(self, records: list[dict]) -> list[np.ndarray]:
        if self.model is None:
            raise RuntimeError("fit() must be called before score()")
        out: list[np.ndarray] = []
        for i in range(0, len(records), self.CHUNK):
            chunk = records[i : i + self.CHUNK]
            out.extend(split_flat(self._raw(self._features(chunk)), candidate_sizes(chunk)))
        return out
