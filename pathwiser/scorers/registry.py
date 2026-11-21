"""Names of the available scorers and their constructor."""

from __future__ import annotations

from pathwiser.scorers.base import Scorer
from pathwiser.scorers.encoder import SentenceEncoder
from pathwiser.scorers.lexical import LexicalOverlap, RandomScorer, TfidfCosine

DEFAULT_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

BASELINES = ("random", "lexical", "lexical_ctx", "tfidf", "tfidf_ctx", "st_message", "st_query")  # training-free
TRAINED_BASELINES = ("tfidf_lr", "st_lr", "st_lgbm", "cross_encoder")
ALL_BASELINES = BASELINES + TRAINED_BASELINES


def build(name: str, st_model: str = DEFAULT_ST_MODEL, params: dict | None = None) -> Scorer:
    """Instantiate a baseline by name. ``params`` overrides constructor arguments (e.g. ``{"train_examples": 4000}``)."""
    params = dict(params or {})
    if name in TRAINED_BASELINES:  # imported lazily: heavy optional dependencies
        if name == "tfidf_lr":
            from pathwiser.scorers.tfidf_lr import TfidfLogReg

            return TfidfLogReg(**params)
        if name in ("st_lr", "st_lgbm"):
            from pathwiser.scorers.embedding_pair import EmbeddingPairModel

            return EmbeddingPairModel(
                learner=name.split("_")[1], model_name=params.pop("model_name", st_model), **params
            )
        from pathwiser.scorers.cross_encoder import CrossEncoderScorer

        return CrossEncoderScorer(**params)
    registry = {
        "random": RandomScorer,
        "lexical": lambda: LexicalOverlap(False),
        "lexical_ctx": lambda: LexicalOverlap(True),
        "tfidf": lambda: TfidfCosine(False),
        "tfidf_ctx": lambda: TfidfCosine(True),
        "st_message": lambda: SentenceEncoder(st_model, "message", **params),
        "st_query": lambda: SentenceEncoder(st_model, "query", **params),
    }
    if name not in registry:
        raise KeyError(f"unknown baseline {name!r}; available: {', '.join(ALL_BASELINES)}")
    if params and name not in ("st_message", "st_query"):
        raise ValueError(f"{name} takes no parameters")
    return registry[name]()
