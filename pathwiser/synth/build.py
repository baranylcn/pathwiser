"""Build model-ready evaluation sets from authored scenarios.

A set is described by an entry of ``SETS``. The build checks the scenarios (against their specifications when the set has any), applies the
planned noise, converts to listwise records, attaches the blind-verification status, runs the leakage checks and writes
``data/model/<version>/{<split>.jsonl, manifest.json}``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

from pathwiser.benchmark.records import ModelDataConfig
from pathwiser.paths import DATA_DIR
from pathwiser.storage.hashing import sha256_file
from pathwiser.synth import check, leakage
from pathwiser.synth import styles as S
from pathwiser.synth.prepare import apply_drops, prepare
from pathwiser.synth.scenarios import build_records, load_scenarios, validate_scenarios
from pathwiser.synth.verify import attach_verification

SETS = {
    "realistic_v1": {
        "src": DATA_DIR / "curated" / "realistic_v1",
        "out": DATA_DIR / "model" / "realistic-v1",
        "version": "realistic-v1",
        "id_prefix": "rl",
        "dataset": "realistic",
        "tier": "R",
        "model_split": "test",
        "specs": False,
        "criteria": False,
        "purpose": "external stress test; never used for training, tuning or threshold selection",
        "generator": "claude-sonnet-5-5",
        "references": ["v1"],
    },
    "realistic_v2": {
        "src": DATA_DIR / "curated" / "realistic_v2",
        "out": DATA_DIR / "model" / "realistic-v2",
        "version": "realistic-v2",
        "id_prefix": "r2",
        "dataset": "realistic_v2",
        "tier": "R2",
        "model_split": "test",
        "specs": True,
        "criteria": True,
        "purpose": "fresh external evaluation set; never used to fit, tune or select",
        "generator": "claude-opus (subagents)",
        "references": ["v1", "realistic-v1"],
    },
}


def acceptance(manifest: dict) -> dict:
    """The acceptance criteria of ``realistic-v2`` (fixed in docs/synthetic-data.md before the data was generated)."""
    verified = manifest["quality"]["blind_verified"]
    personas = manifest["persona_counts"]
    leak = manifest["leakage"]
    checks = {
        "at_least_75_scenarios": manifest["n_scenarios"] >= 75,
        "at_least_350_examples": manifest["n_examples"] >= 350,
        "heldout_personas_at_least_30_each": all(personas.get(p, 0) >= 30 for p in S.HELDOUT_PERSONAS),
        "no_exact_leakage": not (
            leak["exact_messages"] or leak["exact_conditions"] or leak["duplicate_messages_within"]
        ),
        "near_duplicate_flags_resolved": leak["near_messages_flagged"] == 0
        and leak["near_conditions_flagged"] in (0, None),
        "all_examples_blind_verified": verified["status_counts"].get("blind_verified", 0) == manifest["n_examples"],
        "every_disagreement_resolved": verified["blind_verified_examples"] > 0
        and verified["blind_verified_examples"] == verified["all_passes_agree_with_intended"],
    }
    checks["all"] = all(checks.values())
    return checks


def _read_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _write_atomic(directory: Path, name: str, text: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False, newline="\n") as tmp:
        tmp.write(text)
    Path(tmp.name).replace(directory / name)


def _style_summary(meta: list[dict]) -> dict:
    return {
        "persona_counts": dict(Counter(m["persona"] for m in meta)),
        "style_split_counts": dict(Counter(m["style_split"] for m in meta)),
        "noise_planned_counts": dict(Counter(m["noise_planned"] for m in meta)),
        "noise_applied_counts": dict(Counter(m["noise_applied"] for m in meta)),
        "tag_counts": dict(Counter(t for m in meta for t in m["tags"])),
    }


def build_set(
    name: str, log=print, use_embeddings: bool = True, encoder=None, reference_dirs: list[Path] | None = None
) -> dict:
    cfg = SETS[name]
    src, out = Path(cfg["src"]), Path(cfg["out"])
    scenarios, hashes = load_scenarios(src)
    dropped_path, verification_path = src / "dropped.json", src / "blind_verification.json"
    dropped = _read_json(dropped_path, [])
    specs = None
    if cfg["specs"]:
        specs = json.loads((src / "specs.json").read_text(encoding="utf-8"))
        scenarios, specs = apply_drops(scenarios, specs, dropped)
        problems = check.validate_generated(scenarios, specs)
        if problems:
            raise ValueError("generated scenarios do not follow their specs:\n  " + "\n  ".join(problems[:30]))
        scenarios = prepare(
            scenarios,
            specs,
            id_prefix=cfg["id_prefix"],
            dataset=cfg["dataset"],
            tier=cfg["tier"],
            model_split=cfg["model_split"],
        )
    else:
        validate_scenarios(scenarios)
    model_cfg = ModelDataConfig()
    records = build_records(scenarios, model_cfg)
    blind = attach_verification(records, _read_json(verification_path, None))
    for f in (src / "specs.json", src / "generation.json", verification_path, dropped_path):
        if f.exists():
            hashes[f.name] = sha256_file(f)

    messages = {r["example_id"]: r["meta"].get("clean_message", r["input"]["message"]) for r in records}
    conditions = {f"{sc['id']}.{c}": t for sc in scenarios for c, t in sc["conditions"].items()}
    refs = (
        reference_dirs
        if reference_dirs is not None
        else [DATA_DIR / "model" / r for r in cfg["references"] if (DATA_DIR / "model" / r).exists()]
    )
    leak = leakage.leakage_report(messages, conditions, refs, encoder=encoder, use_embeddings=use_embeddings)
    meta = [r["meta"] for r in records]
    manifest = {
        "version": cfg["version"],
        "purpose": cfg["purpose"],
        "quality": {
            "origin": "llm_generated",
            "generator": cfg["generator"],
            "human_audited": False,
            "blind_verified": blind,
            "note": "not 'gold' until audited by a human",
        },
        "source_files_sha256": hashes,
        "n_scenarios": len(scenarios),
        "n_examples": len(records),
        "dropped_examples": [{k: d[k] for k in ("scenario", "i", "reason")} for d in dropped],
        "label_counts": dict(Counter(m["label_type"] for m in meta)),
        "nm_kind_counts": dict(Counter(m["nm_kind"] for m in meta if m["nm_kind"])),
        "candidate_size_counts": dict(sorted(Counter(m["n_candidates"] for m in meta).items())),
        "context_counts": dict(Counter(next(s for s in m["slices"] if s.startswith("ctx:")) for m in meta)),
        "domains": sorted({sc["domain"] for sc in scenarios}),
        "options": {"preprocessing": model_cfg.__dict__},
        "leakage": {
            "reference_messages": leak["reference_messages"],
            "reference_conditions": leak["reference_conditions"],
            "exact_messages": leak["exact_messages"],
            "exact_conditions": leak["exact_conditions"],
            "duplicate_messages_within": leak["duplicate_messages_within"],
            "duplicate_conditions_within": leak["duplicate_conditions_within"],
            "near_messages_flagged": len(leak["near_messages"]),
            "near_messages": leak["near_messages"],
            "near_conditions_flagged": None if leak["near_conditions"] is None else len(leak["near_conditions"]),
            "near_conditions": leak["near_conditions"],
            "embeddings_checked": leak["embeddings_checked"],
        },
    }
    if specs is not None:
        generation = _read_json(src / "generation.json", {})
        manifest.update(
            _style_summary(meta),
            sampler_version=generation.get("sampler_version"),
            archetype_counts=dict(Counter(sp["archetype"] for sp in specs)),
            groups=sorted({sp["group"] for sp in specs}),
        )
    if cfg["criteria"]:
        manifest["acceptance"] = acceptance(manifest)
    out.mkdir(parents=True, exist_ok=True)
    _write_atomic(
        out,
        f"{cfg['model_split']}.jsonl",
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records),
    )
    _write_atomic(out, "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    status = f"  acceptance: {manifest['acceptance']['all']}" if cfg["criteria"] else ""
    log(f"{cfg['version']}: {len(records)} examples from {len(scenarios)} scenarios -> {out}{status}")
    return manifest


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(prog=prog, description="Build an evaluation set from authored scenarios.")
    parser.add_argument("--set", choices=sorted(SETS), required=True)
    parser.add_argument(
        "--no-embeddings", action="store_true", help="skip the condition embedding near-duplicate check"
    )
    args = parser.parse_args(argv)
    manifest = build_set(args.set, use_embeddings=not args.no_embeddings)
    print(json.dumps({"n_examples": manifest["n_examples"], "acceptance": manifest.get("acceptance")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
