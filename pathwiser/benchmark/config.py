"""Configuration loading, validation and hashing."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from pathwiser.benchmark.fileio import read_json
from pathwiser.benchmark.schema import PipelineError

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "dataset.json"
SGDX_VARIANTS = ("orig", "v1", "v2", "v3", "v4", "v5")
KNOWN_DATASETS = ("sgd", "banking77", "clinc150", "hwu64", "star", "abcd")
TIERS = {"sgd": "T1", "banking77": "T2", "clinc150": "T2", "hwu64": "T2", "star": "T3", "abcd": "T3"}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def config_hash(cfg: dict) -> str:
    # "paths" is environment, not content: moving a directory must not change the dataset identity.
    body = {k: v for k, v in cfg.items() if not k.startswith("_") and k != "paths"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _check_weights(name: str, weights: dict) -> None:
    if (
        not weights
        or any((not isinstance(w, (int, float))) or w < 0 for w in weights.values())
        or sum(weights.values()) <= 0
    ):
        raise PipelineError(f"config: '{name}' must be non-empty, non-negative and sum to > 0")


def load_config(path: str | Path | None = None) -> dict:
    cfg = read_json(DEFAULT_CONFIG)
    if path:
        cfg = _merge(cfg, read_json(Path(path)))
    validate(cfg)
    cfg["_hash"] = config_hash(cfg)
    return cfg


def validate(cfg: dict) -> None:
    if not isinstance(cfg.get("seed"), int):
        raise PipelineError("config: 'seed' must be an integer")
    if cfg["context_max_turns"] < 0:
        raise PipelineError("config: 'context_max_turns' must be >= 0")
    cand = cfg["candidates"]
    _check_weights("candidates.size_weights", cand["size_weights"])
    if any(int(k) < 2 for k in cand["size_weights"]):
        raise PipelineError("config: candidate set sizes must be >= 2")
    _check_weights("candidates.negative_strategy_weights", cand["negative_strategy_weights"])
    if set(cand["negative_strategy_weights"]) - {"family", "nearest", "random"}:
        raise PipelineError("config: unknown negative strategy")
    if cand["nearest_window_multiplier"] < 1:
        raise PipelineError("config: nearest_window_multiplier must be >= 1")
    nm = cfg["no_match"]
    if not 0 <= nm["fraction"] < 1:
        raise PipelineError("config: no_match.fraction must be in [0, 1)")
    _check_weights("no_match.kind_weights", nm["kind_weights"])
    if set(nm["kind_weights"]) - {"other_family", "sibling_removed"}:
        raise PipelineError("config: unknown no_match kind")
    sg = cfg["sgdx"]
    _check_weights("sgdx.train_variant_weights", sg["train_variant_weights"])
    for v in list(sg["train_variant_weights"]) + [sg["val_variant"], sg["test_variant"]]:
        if v not in SGDX_VARIANTS:
            raise PipelineError(f"config: unknown SGD-X variant '{v}'")
    held_out_variants = {sg["val_variant"], sg["test_variant"]}
    if held_out_variants & set(sg["train_variant_weights"]):
        raise PipelineError("config: SGD-X val/test variants must not be used for training (leakage)")
    if sg["val_variant"] == sg["test_variant"]:
        raise PipelineError("config: SGD-X val and test variants must differ")
    for ds, dcfg in cfg["datasets"].items():
        if ds not in KNOWN_DATASETS:
            raise PipelineError(f"config: unknown dataset '{ds}'")
        for key in ("split_fractions",):
            if key in dcfg:
                fr = dcfg[key]
                if abs(sum(fr.values()) - 1.0) > 1e-9 or set(fr) != {"train", "val", "test"}:
                    raise PipelineError(f"config: datasets.{ds}.{key} must have train/val/test summing to 1")
        ho = dcfg.get("heldout_families")
        if ho and set(ho.get("val", [])) & set(ho.get("test", [])):
            raise PipelineError(f"config: datasets.{ds}.heldout_families val/test overlap")
        if "val_fraction" in dcfg and not 0 < dcfg["val_fraction"] < 1:
            raise PipelineError(f"config: datasets.{ds}.val_fraction must be in (0, 1)")
    if not any(d.get("enabled") for d in cfg["datasets"].values()):
        raise PipelineError("config: no dataset enabled")
