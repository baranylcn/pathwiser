"""Shared access to records and stored scores, JSON output and table formatting for the reports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pathwiser.decision.system import align
from pathwiser.evaluation.metrics import Prediction, predictions, slice_groups, summarize_decisions, tune
from pathwiser.storage.dataset import load_manifest, load_split
from pathwiser.storage.scores import available, load_scores, score_path

BASELINE_MODELS = ("lexical", "tfidf", "st_message", "tfidf_lr", "st_lr", "st_lgbm", "cross_encoder")


GENERALISATION = (
    "cond:seen",
    "cond:unseen",
    "seen_exact:exact",
    "seen_exact:equivalent_only",
    "svc:seen",
    "svc:unseen",
)


SLICES_SHOWN = GENERALISATION + (
    "ctx:none",
    "ctx:first_turn",
    "ctx:later_turn",
    "k:2-3",
    "k:4-6",
    "k:7+",
    "nm_kind:sibling_removed",
    "nm_kind:in_domain_uncovered",
    "nm_kind:oos",
    "nm_kind:other_family",
    "risk:sibling_partial_validity",
)


def to_jsonable(x):
    """JSON-safe: numpy scalars to python, NaN to None."""
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else round(float(x), 5)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(obj), indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


class Store:
    """Records and stored scores of the benchmark splits and of external sets."""

    def __init__(self, data_dir: Path, root: Path, extras: dict[str, Path]) -> None:
        self.data_dir, self.root, self.extras = data_dir, root, extras
        self.manifest = load_manifest(data_dir)
        self.version = self.manifest["version"]
        self.source_version = self.manifest["source_version"]
        self._records: dict[str, list[dict]] = {}

    def records(self, split: str) -> list[dict]:
        if split not in self._records:
            self._records[split] = (
                load_split("test", self.extras[split]) if split in self.extras else load_split(split, self.data_dir)
            )
        return self._records[split]

    def has(self, model: str, split: str) -> bool:
        return split in available(self.root, self.version, model)

    def scores(self, model: str, split: str) -> list[np.ndarray]:
        return load_scores(score_path(self.root, self.version, model, split), self.records(split))

    def main(self, split: str, models: list[str]) -> tuple[list[dict], dict[str, list[np.ndarray]]]:
        """Main (non-twin) examples of a split with the stored scores of ``models``."""
        recs = self.records(split)
        keep = np.array([not r["meta"]["is_paired"] for r in recs])
        return align(recs, {m: self.scores(m, split) for m in models}, keep)


def slice_metrics(preds: list[Prediction], abstain: np.ndarray, names=SLICES_SHOWN) -> dict:
    groups = slice_groups(preds)
    index = {p.example_id: i for i, p in enumerate(preds)}
    out = {}
    for name in names:
        if name in groups:
            ids = [index[p.example_id] for p in groups[name]]
            out[name] = summarize_decisions(groups[name], abstain[ids])
    return out


def tau_from_validation(store: Store, model: str) -> tuple[float, str]:
    if store.has(model, "val"):
        recs, sc = store.main("val", [model])
        return tune(predictions(recs, sc[model]))["tau"], "tuned on the stored validation scores"
    report = json.loads(
        (Path("reports") / "baselines" / store.source_version / f"{model}.json").read_text(encoding="utf-8")
    )
    return report["threshold"]["global"], "taken from the baseline result file (same weights, same validation tuning)"


def fmt(x, digits=3):
    return "–" if x is None else f"{x:.{digits}f}"


def ci(d: dict) -> str:
    return f"{fmt(d['value'])} [{fmt(d['lo'])}, {fmt(d['hi'])}]"


def fmt_diff(entry: dict, digits: int = 3) -> str:
    """``diff [lo, hi]`` of a paired-difference entry."""
    return f"{fmt(entry['diff'], digits)} [{fmt(entry['lo'], digits)}, {fmt(entry['hi'], digits)}]"
