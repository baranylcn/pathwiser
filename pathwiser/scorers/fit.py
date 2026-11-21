"""Fit or load one scorer and store its per-candidate scores for offline analysis.

    pathwiser scorers fit --baseline st_lgbm --splits val
    pathwiser scorers fit --baseline st_lgbm --splits test realistic --final --extra realistic=data/model/realistic-v1
    pathwiser scorers fit --baseline cross_encoder --load models/cross_encoder --splits val test --final
    pathwiser scorers fit --baseline cross_encoder --tag hn --param negatives=hard --param save_dir=models/ce_hn --splits val

Scores go to ``data/cache/scores/<data version>/<scorer>[__<tag>]/<split>.npz`` (see ``pathwiser.storage.scores``).
A trained scorer fits on the train split only. ``test`` is scored only with ``--final``. Extra sets (``--extra NAME=DIR`` reads
``DIR/test.jsonl``) are external evaluation sets: they are scored and never used to fit or tune. Existing score files are kept
unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from pathwiser.evaluation.metrics import predictions, summarize
from pathwiser.runtime import environment, parse_params
from pathwiser.scorers.registry import ALL_BASELINES, DEFAULT_ST_MODEL, build
from pathwiser.storage.dataset import DEFAULT_DATA_DIR, load_manifest, load_split, main_examples
from pathwiser.storage.scores import DEFAULT_ROOT, save_scores, score_path


def store_name(baseline: str, tag: str | None) -> str:
    return f"{baseline}__{tag}" if tag else baseline


def run(
    baseline: str,
    splits: list[str],
    data_dir: Path,
    root: Path,
    extras: dict[str, Path],
    final: bool = False,
    tag: str | None = None,
    params: dict | None = None,
    load: Path | None = None,
    force: bool = False,
    st_model: str = DEFAULT_ST_MODEL,
    log=print,
) -> dict[str, Path]:
    for s in splits:
        if s not in ("val", "test") and s not in extras:
            raise ValueError(f"unknown split {s!r}; use val, test or one of the --extra names {sorted(extras)}")
    if "test" in splits and not final:
        raise ValueError(
            "the test split is only scored with --final (protocol: it is read once, after the method is fixed)"
        )
    version = load_manifest(data_dir)["version"]
    name = store_name(baseline, tag)
    paths = {s: score_path(root, version, name, s) for s in splits}
    todo = [s for s in splits if force or not paths[s].exists()]
    for s in splits:
        if s not in todo:
            log(f"  {name}/{s}: scores already stored, skipping (use --force to recompute)")
    if not todo:
        return paths

    scorer = build(baseline, st_model, params)
    if hasattr(scorer, "log"):
        scorer.log = log
    fit_info: dict = {}
    t0 = time.perf_counter()
    if load is not None:
        if not hasattr(scorer, "load"):
            raise ValueError(f"{baseline} cannot be loaded from disk (it is cheap to refit; drop --load)")
        scorer.load(load)
        log(f"  loaded weights from {load}")
    else:
        train = main_examples(load_split("train", data_dir)) if getattr(scorer, "needs_train", False) else []
        val_for_fit = main_examples(load_split("val", data_dir)) if getattr(scorer, "needs_val", False) else None
        scorer.fit(train, val_for_fit)
        del train
        fit_info = {"fit_seconds": round(time.perf_counter() - t0, 1)}
    info_base = {
        "scorer": baseline,
        "tag": tag,
        "params": scorer.params,
        "training": getattr(scorer, "info", {}),
        "environment": environment(),
        "data_version": version,
        **fit_info,
    }
    for s in todo:
        records = load_split("test", extras[s]) if s in extras else load_split(s, data_dir)
        t1 = time.perf_counter()
        scores = scorer.score(records)
        seconds = time.perf_counter() - t1
        save_scores(
            paths[s],
            records,
            scores,
            {**info_base, "split": s, "n_examples": len(records), "score_seconds": round(seconds, 1)},
        )
        o = summarize([p for p in predictions(records, scores) if not p.paired])
        log(
            f"  {name}/{s}: {len(records)} examples scored in {seconds:.0f}s  top1={o['top1']} mrr={o['mrr']}  -> {paths[s]}"
        )
    return paths


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    p = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    p.add_argument("--baseline", required=True, choices=ALL_BASELINES)
    p.add_argument(
        "--splits", nargs="+", default=["val"], help="val, test (needs --final) and/or --extra names (default: val)"
    )
    p.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="external evaluation set (reads DIR/test.jsonl)",
    )
    p.add_argument("--final", action="store_true", help="allow scoring the test split")
    p.add_argument("--tag", help="variant name, stored as <baseline>__<tag>")
    p.add_argument("--param", action="append", metavar="KEY=VALUE", help="override a constructor argument (repeatable)")
    p.add_argument("--load", type=Path, metavar="DIR", help="load saved weights instead of training (cross_encoder)")
    p.add_argument("--force", action="store_true", help="recompute scores that are already stored")
    p.add_argument(
        "--data", type=Path, default=DEFAULT_DATA_DIR, help="model-ready data directory (default: data/model/v1)"
    )
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="score store root (default: data/cache/scores)")
    p.add_argument("--st-model", default=DEFAULT_ST_MODEL)
    a = p.parse_args(argv)
    extras = {}
    for item in a.extra:
        key, sep, value = item.partition("=")
        if not sep or key in ("val", "test", "train"):
            p.error(f"--extra expects NAME=DIR with a name other than val/test/train, got {item!r}")
        extras[key] = Path(value)
    try:
        run(
            a.baseline,
            a.splits,
            a.data,
            a.root,
            extras,
            a.final,
            a.tag,
            parse_params(a.param),
            a.load,
            a.force,
            a.st_model,
            log=lambda s: print(s, file=sys.stderr, flush=True),
        )
    except ValueError as e:
        p.error(str(e))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
