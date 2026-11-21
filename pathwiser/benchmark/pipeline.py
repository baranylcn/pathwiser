"""End-to-end orchestration: raw data -> validated final dataset."""

from __future__ import annotations

import platform
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pathwiser.benchmark.candidates import Pool
from pathwiser.benchmark.card import compute_probes, compute_stats, render_card
from pathwiser.benchmark.examples import ExampleFactory
from pathwiser.benchmark.fileio import write_json, write_jsonl, write_text
from pathwiser.benchmark.schema import SPLIT_RANK, SPLITS, DropLog, PipelineError, SourceData
from pathwiser.benchmark.sources import LOADERS
from pathwiser.benchmark.splits import assign_partitions, assign_splits, dedupe_single_turn
from pathwiser.benchmark.validate import validate
from pathwiser.storage.hashing import sha256_file

RAW_DIRS = {
    "sgd": "sgd",
    "star": "star",
    "abcd": "abcd",
    "banking77": "banking77",
    "clinc150": "clinc150",
    "hwu64": "hwu64",
}


def verify_raw(raw: Path, datasets: list[str], log) -> str:
    """Verify raw files against data/raw/SHA256SUMS.txt (only the datasets in use). Returns the sums-file hash."""
    sums = raw / "SHA256SUMS.txt"
    if not sums.exists():
        raise PipelineError(f"{sums} not found; cannot verify raw data (use --skip-verify-raw to bypass)")
    wanted = tuple(f"./{RAW_DIRS[d]}/" for d in datasets)
    checked = 0
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, _, rel = line.partition(" *")
        if not rel.startswith(wanted):
            continue
        path = raw / rel[2:]
        if not path.exists():
            raise PipelineError(f"raw file missing: {rel}")
        if sha256_file(path) != digest:
            raise PipelineError(f"raw file changed (checksum mismatch): {rel}")
        checked += 1
    log(f"raw data verified ({checked} files)")
    return sha256_file(sums)


@dataclass
class Prepared:
    """Everything produced before anything is written (also used by tests)."""

    enabled: list[str]
    sources: dict[str, SourceData]
    pools: dict[tuple[str, str], Pool]
    examples: dict[str, list[dict]]
    drops: DropLog
    moments_kept: dict[str, Counter]
    raw_sums_hash: str | None


def prepare(cfg: dict, root: Path, verify: bool = True, limit: int | None = None, log=print) -> Prepared:
    paths = cfg["paths"]
    raw, curated = root / paths["raw"], root / paths["curated"]
    enabled = sorted(ds for ds, d in cfg["datasets"].items() if d.get("enabled"))
    drops = DropLog()

    raw_sums_hash = verify_raw(raw, enabled, log) if verify else None

    # 1. load ------------------------------------------------------------------------------
    sources: dict[str, SourceData] = {}
    for ds in enabled:
        t = time.time()
        sources[ds] = LOADERS[ds](cfg, raw, curated, drops, limit)
        log(
            f"loaded {ds}: {len(sources[ds].conditions)} conditions, {len(sources[ds].moments)} moments ({time.time() - t:.1f}s)"
        )

    # 2. partitions & splits ---------------------------------------------------------------
    for ds, sd in sources.items():
        if ds != "sgd":
            assign_partitions(ds, sd.conditions, cfg["datasets"][ds].get("heldout_families"))
        assign_splits(sd)
    dedupe_single_turn(list(sources.values()), drops)
    moments_kept: dict[str, Counter] = {ds: Counter(m.split for m in sd.moments) for ds, sd in sources.items()}
    log("moments kept: " + ", ".join(f"{ds}={sum(c.values())}" for ds, c in moments_kept.items()))

    # 3. pools -----------------------------------------------------------------------------
    pools: dict[tuple[str, str], Pool] = {}
    for ds, sd in sources.items():
        for split in SPLITS:
            members = [c for c in sd.conditions.values() if SPLIT_RANK[c.partition] <= SPLIT_RANK[split]]
            if members:
                pools[(ds, split)] = Pool(members)
        if (ds, "train") not in pools or len({c.group for c in pools[(ds, "train")].conditions}) < 2:
            raise PipelineError(f"{ds}: fewer than 2 distinct train conditions after hold-out; adjust heldout_families")

    # 4. examples --------------------------------------------------------------------------
    factory = ExampleFactory(cfg, sources, pools, drops)
    examples: dict[str, list[dict]] = {s: [] for s in SPLITS}
    t = time.time()
    for ds in enabled:
        for m in sorted(sources[ds].moments, key=lambda x: x.moment_id):
            if (ds, m.split) not in pools:
                drops.add(ds, "no_pool_for_split")
                continue
            for ex in factory.make(m):
                examples[ex["split"]].append(ex)
    for s in SPLITS:
        examples[s].sort(key=lambda e: e["example_id"])
    log("examples: " + ", ".join(f"{s}={len(examples[s])}" for s in SPLITS) + f" ({time.time() - t:.1f}s)")

    return Prepared(enabled, sources, pools, examples, drops, moments_kept, raw_sums_hash)


def build(
    cfg: dict, root: Path, out_dir: Path | None = None, verify: bool = True, limit: int | None = None, log=print
) -> dict:
    t0 = time.time()
    paths = cfg["paths"]
    curated = root / paths["curated"]
    final_dir = out_dir or (root / paths["out"] / (cfg["version"] + ("-smoke" if limit else "")))
    prep = prepare(cfg, root, verify=verify, limit=limit, log=log)
    enabled, sources, pools, examples = prep.enabled, prep.sources, prep.pools, prep.examples
    drops, moments_kept, raw_sums_hash = prep.drops, prep.moments_kept, prep.raw_sums_hash

    # 5. validate (aborts the build on any violation) ----------------------------------------
    validation = validate(cfg, sources, examples)
    log(f"validation: {validation['n_checks']} checks passed, {len(validation['warnings'])} warning(s)")

    # 6. report ----------------------------------------------------------------------------
    probes = compute_probes(examples, pools)
    stats = compute_stats(cfg, sources, examples, drops, moments_kept)
    card = render_card(cfg, stats, probes, validation, sources)

    # 7. write atomically: build in a temp dir, then swap ----------------------------------------
    tmp = final_dir.parent / (final_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    files: dict[str, dict] = {}
    for s in SPLITS:
        n, h = write_jsonl(tmp / f"{s}.jsonl", examples[s])
        files[f"{s}.jsonl"] = {"rows": n, "sha256": h}
    cond_rows = [
        {
            "condition_uid": c.uid,
            "dataset": c.dataset,
            "family": c.family,
            "equivalence_group_id": c.group,
            "partition": c.partition,
            "descriptions": c.descriptions,
            "description_origin": c.description_origin,
            "reviewed": c.reviewed,
            "extra": c.extra,
        }
        for ds in enabled
        for c in sorted(sources[ds].conditions.values(), key=lambda x: x.uid)
    ]
    n, h = write_jsonl(tmp / "conditions.jsonl", cond_rows)
    files["conditions.jsonl"] = {"rows": n, "sha256": h}
    mom_rows = (m.to_record() for ds in enabled for m in sorted(sources[ds].moments, key=lambda x: x.moment_id))
    n, h = write_jsonl(tmp / "moments.jsonl", mom_rows)
    files["moments.jsonl"] = {"rows": n, "sha256": h}
    if cfg["export"]["csv"]:
        from pathwiser.benchmark.export import export_csv

        files.update(
            export_csv(
                cfg,
                examples,
                [c for ds in enabled for c in sorted(sources[ds].conditions.values(), key=lambda x: x.uid)],
                tmp,
            )
        )
    files["stats.json"] = {"sha256": write_json(tmp / "stats.json", {"stats": stats, "probes": probes})}
    files["validation_report.json"] = {"sha256": write_json(tmp / "validation_report.json", validation)}
    files["DATASET_CARD.md"] = {"sha256": write_text(tmp / "DATASET_CARD.md", card)}
    curated_hashes = {
        p.name: sha256_file(p)
        for p in sorted(curated.glob("*.json"))
        if p.stem in {"banking77_conditions", "clinc150_conditions", "hwu64_conditions", "star_tasks", "abcd_flows"}
    }
    manifest = {
        "version": cfg["version"],
        "seed": cfg["seed"],
        "config_hash": cfg["_hash"],
        "datasets": enabled,
        "smoke_build_limit": limit,
        "raw_sha256sums_hash": raw_sums_hash,
        "curated_files_sha256": curated_hashes,
        "files": files,
    }
    write_json(tmp / "manifest.json", manifest)
    # build_info is the only non-deterministic file (time, platform); it is excluded from the manifest
    write_json(
        tmp / "build_info.json",
        {
            "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "duration_seconds": round(time.time() - t0, 1),
        },
    )
    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp.rename(final_dir)
    log(f"wrote {final_dir} in {time.time() - t0:.1f}s")
    return {"dir": str(final_dir), "stats": stats, "validation": validation, "manifest": manifest}
