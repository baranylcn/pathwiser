"""Plain data containers shared by all pipeline stages."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

SPLITS = ("train", "val", "test")
SPLIT_RANK = {"train": 0, "val": 1, "test": 2}


class PipelineError(RuntimeError):
    """Raised for configuration / input problems that must stop the build."""


@dataclass
class Condition:
    """A routing condition (branch) with one or more natural-language descriptions.

    ``descriptions`` maps a variant name to text; ``"orig"`` always exists.
    ``group`` is the equivalence group: conditions in one group mean the same thing and are
    never placed in the same candidate set. ``partition`` is the first split in which the
    condition may appear at all: train (everywhere), val (val+test), test (test only).
    """

    uid: str
    dataset: str
    family: str
    group: str
    descriptions: dict[str, str]
    description_origin: str
    reviewed: bool
    partition: str = "train"
    extra: dict = field(default_factory=dict)

    def describe(self, variant: str) -> str:
        return self.descriptions.get(variant) or self.descriptions["orig"]


@dataclass
class Moment:
    """A routing decision point: conversation context + current user message + gold condition."""

    moment_id: str
    dataset: str
    group_id: str  # leakage unit (dialogue / worker / utterance)
    base_split: str  # split before hold-out routing
    context: list[dict]  # [{"role": "user"|"assistant", "content": str}], already truncated
    message: str
    gold_uid: str | None  # None => out-of-scope example (no gold condition exists)
    turn_idx: int | None
    is_first_user_turn: bool
    n_context_total: int  # context turns before truncation
    context_origin: str  # "natural" | "none"
    extra: dict = field(default_factory=dict)
    split: str | None = None

    def to_record(self) -> dict:
        return {
            "moment_id": self.moment_id,
            "dataset": self.dataset,
            "group_id": self.group_id,
            "split": self.split,
            "context": self.context,
            "message": self.message,
            "gold_condition_uid": self.gold_uid,
            "turn_idx": self.turn_idx,
            "is_first_user_turn": self.is_first_user_turn,
            "n_context_total": self.n_context_total,
            "context_origin": self.context_origin,
            "extra": self.extra,
        }


@dataclass
class SourceData:
    """Output of one source loader."""

    dataset: str
    tier: str
    license: str
    conditions: dict[str, Condition]
    moments: list[Moment]
    split_mode: str  # official | dialogue_routed | partition_routed
    single_turn: bool = False


class DropLog:
    """Counts every discarded record by (dataset, reason) so nothing disappears silently."""

    def __init__(self) -> None:
        self.counts: Counter[tuple[str, str]] = Counter()

    def add(self, dataset: str, reason: str, n: int = 1) -> None:
        self.counts[(dataset, reason)] += n

    def as_dict(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for (ds, reason), n in sorted(self.counts.items()):
            out.setdefault(ds, {})[reason] = n
        return out
