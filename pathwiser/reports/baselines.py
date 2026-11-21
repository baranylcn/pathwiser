"""Run baselines under the evaluation protocol and render the summary tables."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pathwiser.evaluation.metrics import evaluate_split, predictions, tune
from pathwiser.paths import REPORTS_DIR
from pathwiser.runtime import environment, parse_params
from pathwiser.scorers.registry import ALL_BASELINES, BASELINES, DEFAULT_ST_MODEL, TRAINED_BASELINES, build
from pathwiser.storage.dataset import DEFAULT_DATA_DIR, load_manifest, load_split, main_examples

HEADLINE = [
    ("top1", "Top-1"),
    ("top3", "Top-3"),
    ("mrr", "MRR"),
    ("decision_accuracy", "Decision acc"),
    ("no_match_precision", "no_match P"),
    ("no_match_recall", "no_match R"),
    ("false_routing_rate", "False routing"),
]


GENERALISATION = [
    "cond:seen",
    "seen_exact:exact",
    "seen_exact:equivalent_only",
    "cond:unseen",
    "svc:seen",
    "svc:unseen",
    "ctx:none",
    "ctx:first_turn",
    "ctx:later_turn",
    "k:2-3",
    "k:4-6",
    "k:7+",
    "nm_kind:other_family",
    "nm_kind:sibling_removed",
    "nm_kind:oos",
    "risk:sibling_partial_validity",
]


def _f(x) -> str:
    return "–" if x is None else f"{x:.3f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _split_table(results: list[dict], split: str) -> str:
    rows = []
    for r in results:
        s = r["splits"].get(split)
        if s:
            rows.append(
                [f"`{r['baseline']}`"]
                + [_f(s["overall"].get(k)) for k, _ in HEADLINE]
                + [f"{r['latency_ms_per_example'][split]:.2f}"]
            )
    return _table(["Baseline"] + [h for _, h in HEADLINE] + ["ms/example"], rows)


def _grid(results: list[dict], split: str, groups: list[str], metric: str, key: str) -> str:
    rows = []
    for r in results:
        s = r["splits"].get(split)
        if s:
            src = s[key]
            rows.append([f"`{r['baseline']}`"] + [_f(src[g].get(metric)) if g in src else "–" for g in groups])
    return _table(["Baseline"] + groups, rows)


def render_summary(results: list[dict]) -> str:
    if not results:
        return "# Baseline results\n\nNo results yet.\n"
    d = results[0]["data"]
    has_test = any("test" in r["splits"] for r in results)
    datasets = sorted(results[0]["splits"]["val"]["datasets"])
    L = [
        "# Baseline results",
        "",
        f"Data: model-ready `{d['version']}` · train `{d['train_sha256'][:12]}` · val `{d['val_sha256'][:12]}` · test `{d['test_sha256'][:12]}`",
        "",
        "**Protocol.** Scores come from training-free scorers and from trained ones (fitted on the train split only; `random` is the chance reference). The decision threshold "
        "τ (abstain = `no_match` when the best score < τ) is tuned on **validation main examples** to maximise decision "
        "accuracy, then applied unchanged to test. Validation numbers are therefore optimistic (τ was tuned on them); "
        "test numbers are the honest ones. SGD-X paraphrase twins are reported separately, never mixed into the totals. "
        "Scores are relevance scores, not calibrated probabilities.",
        "",
        "Reading guide: `Top-1/Top-3/MRR` are over examples that have a gold condition; `Decision acc` counts a "
        "`no_match` example as correct when the policy abstains; `False routing` = routed to a wrong condition, or routed "
        "although no condition applies, over all examples. "
        "Top-3 is trivial for k ≤ 3 (about 40% of examples), compare it against `random`.",
        "",
    ]
    L += ["## Validation (τ tuned on this split)", "", _split_table(results, "val"), ""]
    if has_test:
        L += ["## Test (τ from validation)", "", _split_table(results, "test"), ""]
    shown = "test" if has_test else "val"
    L += [f"## Top-1 by dataset ({shown})", "", _grid(results, shown, datasets, "top1", "datasets"), ""]
    L += [
        f"## Decision accuracy by dataset ({shown}, global τ)",
        "",
        _grid(results, shown, datasets, "decision_accuracy", "datasets"),
        "",
    ]
    groups = [
        g for g in GENERALISATION if any(g in r["splits"][shown]["slices"] for r in results if shown in r["splits"])
    ]
    L += [
        f"## Top-1 by slice ({shown})",
        "",
        "`cond:seen` = an equivalent of the gold condition exists in train; `seen_exact:*` splits it into the very same "
        "condition vs an equivalent-only one (different service or wording); `cond:unseen` = no equivalent in train.",
        "",
        _grid(results, shown, groups, "top1", "slices"),
        "",
    ]
    L += [
        f"## Decision accuracy by slice ({shown}, global τ)",
        "",
        _grid(results, shown, groups, "decision_accuracy", "slices"),
        "",
    ]
    rows = []
    for r in results:
        pc = r["splits"].get(shown, {}).get("paraphrase_consistency")
        if pc:
            rows.append(
                [
                    f"`{r['baseline']}`",
                    str(pc["n_pairs"]),
                    _f(pc.get("top1_main")),
                    _f(pc.get("top1_paraphrase")),
                    _f(pc.get("same_selected_candidate")),
                    _f(pc.get("same_abstain_decision")),
                ]
            )
    L += [
        f"## Paraphrase robustness: SGD-X twins ({shown})",
        "",
        "Same moment and candidates, condition descriptions replaced by held-out crowd paraphrases.",
        "",
        _table(["Baseline", "pairs", "Top-1 original", "Top-1 paraphrase", "Same pick", "Same abstain decision"], rows),
        "",
    ]
    cost_rows = [
        [
            f"`{r['baseline']}`",
            f"{r['threshold']['global']:.4f}",
            _f(r["threshold"]["val_decision_accuracy_at_global"]),
            f"{r['fit_seconds']:.1f}",
            str(r["params"].get("n_parameters", "–")),
        ]
        for r in results
    ]
    trained = [r for r in results if r.get("kind") == "trained"]
    if trained:
        rows = []
        for r in trained:
            t, prm = r.get("training", {}), r["params"]
            backbone = prm.get("resolved_init") or prm.get("embedding_model") or "–"
            rows.append(
                [
                    f"`{r['baseline']}`",
                    backbone,
                    str(t.get("train_examples", "–")),
                    str(t.get("train_pairs", "–")),
                    str(t.get("steps", "–")),
                    f"{t.get('train_seconds', 0) / 60:.1f}",
                    str(t.get("final_loss", "–")),
                ]
            )
        L += [
            "## Training (trained baselines)",
            "",
            "No hyper-parameter search: every setting was fixed before looking at validation results. Training uses the train split only; "
            "validation is used for the threshold, test is read once. Training budgets are limited by CPU-only hardware "
            "(`train examples` < full train split means a weighted sample, see `reports/baselines/<version>/<name>.json` for all parameters).",
            "",
            _table(["Baseline", "Backbone", "Train examples", "Train pairs", "Steps", "Train min", "Final loss"], rows),
            "",
        ]
    L += [
        "## Thresholds, fitting cost, size",
        "",
        "`ms/example` in the tables above is wall-clock scoring time per example on CPU, batched, and includes encoding "
        "the split's unique texts; it is a rough cost indicator, not a serving-latency benchmark.",
        "",
        _table(["Baseline", "τ (global)", "Val decision acc @ τ", "fit s", "parameters"], cost_rows),
        "",
    ]
    return "\n".join(L)


def run_baseline(
    name: str,
    data_dir: Path,
    out_dir: Path,
    final: bool,
    st_model: str = DEFAULT_ST_MODEL,
    params: dict | None = None,
    val_limit: int | None = None,
    log=print,
) -> dict:
    if final and val_limit:
        raise ValueError("--val-limit is a development aid and cannot be combined with --final")
    manifest = load_manifest(data_dir)
    scorer = build(name, st_model, params)
    if hasattr(scorer, "log"):
        scorer.log = log
    val = load_split("val", data_dir)
    if val_limit:
        val = val[:val_limit]
    t0 = time.perf_counter()
    train = main_examples(load_split("train", data_dir)) if getattr(scorer, "needs_train", False) else []
    scorer.fit(train, main_examples(val) if getattr(scorer, "needs_val", False) else None)
    fit_s = time.perf_counter() - t0
    del train

    result: dict = {
        "baseline": name,
        "kind": "trained" if name in TRAINED_BASELINES else "training-free",
        "params": scorer.params,
        "environment": environment(),
        "data": {
            "version": manifest["version"],
            "train_sha256": manifest["files"]["train.jsonl"]["sha256"],
            "val_sha256": manifest["files"]["val.jsonl"]["sha256"],
            "test_sha256": manifest["files"]["test.jsonl"]["sha256"],
            "preprocessing": manifest["preprocessing"],
        },
        "protocol": {
            "threshold_rule": "abstain (no_match) when the best score < tau",
            "threshold_objective": "decision_accuracy",
            "threshold_tuned_on": "val main examples",
            "test_evaluated": final,
            "val_limit": val_limit,
        },
        "fit_seconds": round(fit_s, 2),
        "training": getattr(scorer, "info", {}),
        "splits": {},
        "latency_ms_per_example": {},
    }
    tau = tau_by_dataset = None
    for split in ("val", "test") if final else ("val",):
        records = val if split == "val" else load_split(split, data_dir)
        t0 = time.perf_counter()
        scores = scorer.score(records)
        result["latency_ms_per_example"][split] = round((time.perf_counter() - t0) * 1000 / len(records), 3)
        if split == "val":
            main_preds = [p for p in predictions(records, scores) if not p.paired]
            tuned = tune(main_preds)
            tau = tuned["tau"]
            tau_by_dataset = {
                ds: tune([p for p in main_preds if p.dataset == ds])["tau"]
                for ds in sorted({p.dataset for p in main_preds})
            }
            result["threshold"] = {
                "global": tau,
                "per_dataset": tau_by_dataset,
                "val_decision_accuracy_at_global": round(tuned["decision_accuracy"], 4),
            }
        result["splits"][split] = evaluate_split(records, scores, tau, tau_by_dataset)
        o = result["splits"][split]["overall"]
        log(
            f"  {name:12s} {split:4s} top1={o['top1']:.3f} mrr={o['mrr']:.3f} decision_acc={o['decision_accuracy']:.3f} "
            f"nm_P={o['no_match_precision']} nm_R={o['no_match_recall']} false_routing={o['false_routing_rate']}"
        )
    target = out_dir / "dev" if val_limit else out_dir
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{name}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return result


def write_summary(out_dir: Path) -> Path:
    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("*.json"))]
    order = {n: i for i, n in enumerate(ALL_BASELINES)}
    results.sort(key=lambda r: order.get(r["baseline"], 99))
    path = out_dir / "summary.md"
    path.write_text(render_summary(results), encoding="utf-8", newline="\n")
    return path


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    p = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    p.add_argument("--baseline", choices=ALL_BASELINES, action="append", help="baseline to run (repeatable)")
    p.add_argument("--all", action="store_true", help="run every training-free baseline")
    p.add_argument("--trained", action="store_true", help="run every trained baseline")
    p.add_argument("--final", action="store_true", help="also evaluate the test split (use only for the final run)")
    p.add_argument("--summarize", action="store_true", help="only rebuild summary.md from the saved results")
    p.add_argument(
        "--param",
        action="append",
        metavar="KEY=VALUE",
        help="override a constructor argument of ONE baseline (repeatable)",
    )
    p.add_argument(
        "--val-limit", type=int, metavar="N", help="development aid: score only the first N validation examples"
    )
    p.add_argument("--data", default=str(DEFAULT_DATA_DIR), help="model-ready data directory (default: data/model/v1)")
    p.add_argument("--out", help="results directory (default: reports/baselines/<data version>)")
    p.add_argument("--st-model", default=DEFAULT_ST_MODEL, help="sentence-transformers model (default: %(default)s)")
    a = p.parse_args(argv)
    data_dir = Path(a.data)
    out_dir = Path(a.out) if a.out else REPORTS_DIR / "baselines" / load_manifest(data_dir)["source_version"]
    names = list(a.baseline or [])
    names += [n for n in (BASELINES if a.all else ()) + (TRAINED_BASELINES if a.trained else ()) if n not in names]
    params = parse_params(a.param)
    if params and len(names) != 1:
        p.error("--param needs exactly one baseline")
    if not names and not a.summarize:
        p.error("choose --baseline NAME, --all, --trained or --summarize")
    for name in names:
        print(f"[{name}]", file=sys.stderr, flush=True)
        run_baseline(
            name,
            data_dir,
            out_dir,
            a.final,
            a.st_model,
            params,
            a.val_limit,
            log=lambda s: print(s, file=sys.stderr, flush=True),
        )
    print(write_summary(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
