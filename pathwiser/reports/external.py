"""Baselines on an external evaluation set with the validation threshold and scenario-bootstrap intervals."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from pathwiser.evaluation.metrics import predictions, summarize_decisions
from pathwiser.evaluation.stats import cluster_bootstrap
from pathwiser.reports.store import (
    SLICES_SHOWN,
    Store,
    ci,
    fmt,
    fmt_diff,
    slice_metrics,
    tau_from_validation,
    write_json,
)


def external_report(store: Store, name: str, models: list[str], reference: str, n_boot: int, out_dir: Path) -> dict:
    recs = store.records(name)
    clusters = [r["meta"]["dialogue_id"] for r in recs]
    systems, rows = {}, {}
    for m in models:
        if not store.has(m, name):
            print(f"  skipping {m}: no stored scores for {name}", file=sys.stderr)
            continue
        tau, source = tau_from_validation(store, m)
        preds = predictions(recs, store.scores(m, name))
        abstain = np.array([p.top < tau for p in preds])
        systems[m] = (preds, abstain)
        bench = None
        bpath = Path("reports") / "baselines" / store.source_version / f"{m}.json"
        if bpath.exists():
            bench = json.loads(bpath.read_text(encoding="utf-8"))["splits"].get("test", {}).get("overall")
        rows[m] = {
            "tau": tau,
            "tau_source": source,
            "metrics": summarize_decisions(preds, abstain),
            "slices": slice_metrics(preds, abstain),
            "benchmark_test": bench,
        }
    boot = cluster_bootstrap(systems, clusters, reference=reference if reference in systems else None, n_boot=n_boot)
    result = {
        "set": name,
        "n_examples": len(recs),
        "n_clusters": boot["n_clusters"],
        "reference": reference,
        "protocol": "tau tuned on benchmark validation main examples, applied unchanged; nothing is fitted on this set",
        "models": rows,
        "bootstrap": {k: boot[k] for k in ("n_boot", "level", "systems", "vs_reference")},
    }
    write_json(out_dir / f"{name}_baselines.json", result)
    (out_dir / f"{name}_baselines.md").write_text(render_external(result), encoding="utf-8", newline="\n")
    return result


def render_external(r: dict) -> str:
    name, ref = r["set"], r["reference"]
    L = [
        f"# Baselines on `{name}` ({r['n_examples']} examples, {r['n_clusters']} scenarios)",
        "",
        f"{r['protocol']}. Intervals: 95% bootstrap over scenarios ({r['bootstrap']['n_boot']} resamples). The set is LLM-authored and "
        "not human-audited (see `docs/realistic-eval-set.md`); read it as a stress test, not as a measure of real traffic.",
        "",
        "## Transfer: benchmark test vs. this set",
        "",
        "| Baseline | Top-1 benchmark | Top-1 here | Decision acc. benchmark | Decision acc. here | False routing benchmark | False routing here |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m, v in r["models"].items():
        b, o = v["benchmark_test"] or {}, v["metrics"]
        L.append(
            f"| `{m}` | {fmt(b.get('top1'))} | {fmt(o['top1'])} | {fmt(b.get('decision_accuracy'))} | {fmt(o['decision_accuracy'])} "
            f"| {fmt(b.get('false_routing_rate'))} | {fmt(o['false_routing_rate'])} |"
        )
    L += [
        "",
        "## Metrics with intervals",
        "",
        "| Baseline | Top-1 | Decision acc. | False routing | Abstain rate | no_match recall |",
        "|---|---|---|---|---|---|",
    ]
    for m in r["models"]:
        s = r["bootstrap"]["systems"][m]
        L.append(
            f"| `{m}` | {ci(s['top1'])} | {ci(s['decision_accuracy'])} | {ci(s['false_routing_rate'])} | {ci(s['abstain_rate'])} | {ci(s['no_match_recall'])} |"
        )
    if r["bootstrap"]["vs_reference"]:
        L += [
            "",
            f"## Paired difference to `{ref}` (95% interval; an interval that excludes 0 is a real difference on this set)",
            "",
            "| Baseline | Δ Top-1 | Δ Decision acc. | Δ False routing |",
            "|---|---|---|---|",
        ]
        for m, d in r["bootstrap"]["vs_reference"].items():
            L.append(
                f"| `{m}` | {fmt_diff(d['top1'])} | {fmt_diff(d['decision_accuracy'])} | {fmt_diff(d['false_routing_rate'])} |"
            )
    for metric, title in (("top1", "Top-1"), ("decision_accuracy", "Decision accuracy")):
        names = [n for n in SLICES_SHOWN if any(n in v["slices"] for v in r["models"].values())]
        L += [
            "",
            f"## {title} by slice",
            "",
            "| Baseline | " + " | ".join(names) + " |",
            "|---|" + "---:|" * len(names),
        ]
        for m, v in r["models"].items():
            L.append(
                f"| `{m}` | "
                + " | ".join(fmt(v["slices"][n][metric]) if n in v["slices"] else "–" for n in names)
                + " |"
            )
    L += ["", "## Thresholds", "", "| Baseline | τ | source |", "|---|---:|---|"]
    L += [f"| `{m}` | {v['tau']:.3f} | {v['tau_source']} |" for m, v in r["models"].items()]
    return "\n".join(L) + "\n"
