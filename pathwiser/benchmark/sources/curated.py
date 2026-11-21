"""Loader for hand/LLM-authored condition description files in data/curated/."""

from __future__ import annotations

from pathlib import Path

from pathwiser.benchmark.fileio import read_json
from pathwiser.benchmark.schema import PipelineError


def load_curated(path: Path, expected: set[str], dataset: str) -> tuple[dict, dict[str, dict]]:
    """Return (meta, conditions). The label set must match ``expected`` exactly (no silent gaps)."""
    if not path.exists():
        raise PipelineError(f"{dataset}: curated description file missing: {path}")
    doc = read_json(path)
    meta, conds = doc.get("_meta"), doc.get("conditions")
    if not isinstance(meta, dict) or not isinstance(conds, dict):
        raise PipelineError(f"{dataset}: {path.name} must contain '_meta' and 'conditions'")
    for key in ("description_origin", "reviewed"):
        if key not in meta:
            raise PipelineError(f"{dataset}: {path.name} _meta lacks '{key}'")
    missing = sorted(expected - set(conds))
    extra = sorted(set(conds) - expected)
    if missing or extra:
        raise PipelineError(f"{dataset}: curated labels mismatch data. missing={missing[:10]} unexpected={extra[:10]}")
    for label, entry in conds.items():
        desc = entry.get("description", "")
        if not isinstance(desc, str) or len(desc.split()) < 3:
            raise PipelineError(f"{dataset}: description for '{label}' is empty or too short")
    return meta, conds
