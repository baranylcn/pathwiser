"""Command line of the reports: ``external``, ``dev``, ``final``, ``encoders``, ``diagnose`` and ``compare``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pathwiser.paths import REPORTS_DIR
from pathwiser.reports.diagnostics import threshold_diagnostic
from pathwiser.reports.encoders import compare_candidate, encoder_scan
from pathwiser.reports.external import external_report
from pathwiser.reports.final import final_report
from pathwiser.reports.selection import dev_selection
from pathwiser.reports.store import BASELINE_MODELS, Store
from pathwiser.storage.dataset import DEFAULT_DATA_DIR
from pathwiser.storage.scores import DEFAULT_ROOT


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    p = argparse.ArgumentParser(prog=prog, description="Reports built from stored scores.")
    p.add_argument("command", choices=("external", "dev", "final", "encoders", "diagnose", "compare"))
    p.add_argument(
        "--extra", action="append", default=[], metavar="NAME=DIR", help="external set (reads DIR/test.jsonl)"
    )
    p.add_argument("--models", nargs="+", default=list(BASELINE_MODELS))
    p.add_argument("--reference", default="st_message", help="reference baseline for paired differences (external)")
    p.add_argument(
        "--data", type=Path, default=DEFAULT_DATA_DIR, help="model-ready data directory (default: data/model/v1)"
    )
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="score store root (default: data/cache/scores)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument(
        "--out-dir",
        type=Path,
        help="final only: write the report here instead of reports/decision/<version> (selection.json is still read from there)",
    )
    a = p.parse_args(argv)
    extras = {}
    for item in a.extra:
        key, sep, value = item.partition("=")
        if not sep:
            p.error(f"--extra expects NAME=DIR, got {item!r}")
        extras[key] = Path(value)
    store = Store(a.data, a.root, extras)

    def log(s):
        return print(s, file=sys.stderr, flush=True)

    if a.command == "compare":
        if len(a.models) != 2:
            p.error("compare needs --models CANDIDATE REFERENCE")
        out = REPORTS_DIR / "encoders" / store.source_version
        r = compare_candidate(store, a.models[0], a.models[1], next(iter(extras)), a.n_boot, out)
        print(json.dumps(r["verdict"], indent=2))
        print(out / f"compare_{a.models[0]}_vs_{a.models[1]}.md")
    elif a.command == "diagnose":
        out = REPORTS_DIR / "decision" / store.source_version
        for m in a.models:
            r = threshold_diagnostic(store, m, next(iter(extras)), out)
            v, o, n = r["validation_tau"], r["oracle_tau_on_external_set"], r["never_abstain"]
            print(
                f"{m}: decision acc  val-tau {v['decision_accuracy']}  oracle-tau {o['decision_accuracy']}  never-abstain {n['decision_accuracy']}"
            )
    elif a.command == "encoders":
        out = REPORTS_DIR / "encoders" / store.source_version
        encoder_scan(store, a.models, a.reference, next(iter(extras)), a.n_boot, out)
        print(out / "zero_shot_scan.md")
    elif a.command == "external":
        for name in extras:
            out = REPORTS_DIR / "realistic" / store.source_version
            external_report(store, name, a.models, a.reference, a.n_boot, out)
            print(out / f"{name}_baselines.md")
    else:
        out = REPORTS_DIR / "decision" / store.source_version
        if a.command == "dev":
            sel = dev_selection(store, a.models, out, log)
            print(json.dumps({k: sel[k] for k in ("chosen", "best", "within_one_se")}, indent=2))
        else:
            target = a.out_dir or out
            final_report(store, next(iter(extras), None), a.n_boot, target, log, selection_dir=out)
            print(target / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
