"""Build the benchmark from ``data/raw``: ``python -m pathwiser.benchmark`` (also ``pathwiser benchmark build``)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pathwiser.benchmark.config import load_config
from pathwiser.benchmark.pipeline import build
from pathwiser.benchmark.schema import PipelineError


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    p = argparse.ArgumentParser(prog=prog, description="Build the Pathwiser routing dataset from data/raw/.")
    p.add_argument("--config", help="JSON file overriding configs/dataset.json (deep-merged)")
    p.add_argument("--out", help="output directory (default: <paths.out>/<version>)")
    p.add_argument("--root", default=".", help="repository root containing data/raw/ and data/curated/ (default: cwd)")
    p.add_argument("--skip-verify-raw", action="store_true", help="skip SHA-256 verification of data/raw/")
    p.add_argument(
        "--smoke",
        type=int,
        metavar="N",
        help="debug build: at most N moments per source/split, written to <version>-smoke",
    )
    args = p.parse_args(argv)
    try:
        cfg = load_config(args.config)
        res = build(
            cfg,
            Path(args.root).resolve(),
            Path(args.out).resolve() if args.out else None,
            verify=not args.skip_verify_raw,
            limit=args.smoke,
            log=lambda s: print(s, file=sys.stderr, flush=True),
        )
    except PipelineError as e:
        print(f"\nBUILD FAILED: {e}", file=sys.stderr)
        return 1
    print(res["dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
