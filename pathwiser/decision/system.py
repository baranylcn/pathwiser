"""A decision system: models, optional fusion and an abstention rule behind one ``fit`` / ``decide`` interface."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pathwiser.decision.abstention import FEATURE_SETS, AbstentionModel
from pathwiser.decision.fusion import Fusion
from pathwiser.evaluation.metrics import Prediction, predictions, tune


@dataclass(frozen=True)
class System:
    """A named combination: which models, fused or not, and which abstention rule (``tau``, ``relative`` or ``full``)."""

    models: tuple[str, ...]
    abstention: str = "tau"
    fuse: bool = False
    name: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("a system needs at least one model")
        if len(self.models) > 1 and not self.fuse:
            raise ValueError("several models need fuse=True")
        if self.abstention not in ("tau", *FEATURE_SETS):
            raise ValueError("abstention must be 'tau', 'relative' or 'full'")

    @property
    def label(self) -> str:
        return self.name or f"{'+'.join(self.models)}{'|fused' if self.fuse else ''}|{self.abstention}"

    @property
    def complexity(self) -> float:
        """Order used to break ties between statistically indistinguishable systems: simpler wins."""
        return (
            {"tau": 0, "relative": 1, "full": 2}[self.abstention]
            + (3 if self.fuse else 0)
            + 0.1 * (len(self.models) - 1)
        )


@dataclass
class Decision:
    preds: list[Prediction]
    abstain: np.ndarray
    probability: np.ndarray | None  # P(routing is correct); None for the plain threshold policy
    scores: list[np.ndarray]  # the scores the decision was made on (fused when the system fuses)


class Pipeline:
    def __init__(self, system: System, fusion_C: float = 1.0, abstention_C: float = 1.0) -> None:
        self.system, self.fusion_C, self.abstention_C = system, fusion_C, abstention_C

    def _scores(self, by_model: dict[str, list[np.ndarray]]) -> list[np.ndarray]:
        return self.fusion.transform(by_model) if self.system.fuse else by_model[self.system.models[0]]

    def fit(self, records: list[dict], by_model: dict[str, list[np.ndarray]]) -> Pipeline:
        """Fit on validation main examples (never on test, twins or an external set)."""
        if any(r["meta"]["is_paired"] for r in records):
            raise ValueError("fit on main examples only (SGD-X twins are reported separately)")
        labels = [r["target"]["label"] for r in records]
        if self.system.fuse:
            self.fusion = Fusion(self.system.models, self.fusion_C).fit(by_model, labels)
        scores = self._scores(by_model)
        preds = predictions(records, scores)
        correct = np.array([p.label >= 0 and p.arg == p.label for p in preds])
        nm = np.array([p.label < 0 for p in preds])
        if self.system.abstention == "tau":
            self.tau = tune(preds)["tau"]
        else:
            self.abstainer = AbstentionModel(self.system.abstention, self.abstention_C).fit(scores, correct, nm)
        return self

    def decide(self, records: list[dict], by_model: dict[str, list[np.ndarray]]) -> Decision:
        scores = self._scores(by_model)
        preds = predictions(records, scores)
        if self.system.abstention == "tau":
            return Decision(preds, np.array([p.top < self.tau for p in preds]), None, scores)
        p = self.abstainer.proba(scores)
        return Decision(preds, p < self.abstainer.threshold, p, scores)


def align(
    records: list[dict], by_model: dict[str, list[np.ndarray]], keep: np.ndarray
) -> tuple[list[dict], dict[str, list[np.ndarray]]]:
    idx = np.nonzero(keep)[0]
    return [records[i] for i in idx], {m: [s[i] for i in idx] for m, s in by_model.items()}
