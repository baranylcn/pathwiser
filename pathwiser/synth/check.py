"""Checks of generated scenarios against their specifications.

    pathwiser sets check --specs specs.json --scenarios scenarios.json

Exit code 0 means no problems. The checks are mechanical (structure, conformance to the spec, description style, copying);
whether a label is *right* is decided by blind annotators, not here.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from pathwiser.synth import spec as SP
from pathwiser.synth.scenarios import ScenarioError, validate_scenarios

NAME = re.compile(r"[a-z][a-z0-9_]*")


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9'/&#$%+-]+", text)


def style_problems(scenario: dict, spec: dict) -> list[str]:
    """Condition descriptions must follow the requested style; nodes and messages must have sane sizes."""
    p: list[str] = []
    sid = spec["id"]
    node = scenario.get("node") or {}
    if not NAME.fullmatch(str(node.get("name", ""))):
        p.append(f"{sid}: node.name must be snake_case")
    if len(_words(str(node.get("description", "")))) < 4:
        p.append(f"{sid}: node.description is too short")
    style = spec["condition_style"]
    for letter, text in scenario.get("conditions", {}).items():
        n = len(_words(text))
        ok = {
            "imperative_terse": 2 <= n <= 7,
            "user_wants": 6 <= n <= 20,
            "question": text.strip().endswith("?") and n >= 5,
            "detailed": n >= 12,
            "internal_jargon": 2 <= n <= 14,
        }[style]
        if not ok:
            p.append(f"{sid}.{letter}: {n} words does not fit style {style!r}: {text!r}")
    conds = scenario.get("conditions", {})
    for plan, ex in zip(spec["examples"], scenario.get("examples", [])):
        tag = f"{sid}[{plan['i']}]"
        n = len(_words(str(ex.get("msg", ""))))
        if not 1 <= n <= 80:
            p.append(f"{tag}: message has {n} words")
        for turn in ex.get("ctx", []):
            if not 1 <= len(_words(str(turn[1]))) <= 80:
                p.append(f"{tag}: a context turn has {len(_words(str(turn[1])))} words")
        why = str(ex.get("why", ""))
        if not why.strip() or len(_words(why)) > 25:
            p.append(f"{tag}: 'why' must be a short note (1-25 words)")
        gold = ex.get("gold")
        if gold in conds:
            a, b = {w.lower() for w in _words(ex.get("msg", ""))}, {w.lower() for w in _words(conds[gold])}
            if a and b and len(a & b) / len(a | b) > 0.75:
                p.append(f"{tag}: the message copies the wording of its condition")
    return p


def validate_generated(scenarios: list[dict], specs: list[dict]) -> list[str]:
    """All problems of a batch: one scenario per spec, conformance, structure, style."""
    problems: list[str] = []
    by_id = {s["id"]: s for s in scenarios}
    if len(by_id) != len(scenarios):
        problems.append("duplicate scenario ids in the output")
    spec_ids = {s["id"] for s in specs}
    for missing in sorted(spec_ids - set(by_id)):
        problems.append(f"{missing}: no scenario written")
    for extra in sorted(set(by_id) - spec_ids):
        problems.append(f"{extra}: scenario without a spec")
    for spec in specs:
        sc = by_id.get(spec["id"])
        if sc is None:
            continue
        problems += SP.match_scenario(sc, spec)
        problems += style_problems(sc, spec)
    structural = [dict(sc, domain=sc.get("domain", "x")) for sc in scenarios if sc["id"] in spec_ids]
    try:
        validate_scenarios(structural)
    except ScenarioError as e:
        problems += [line.strip() for line in str(e).splitlines()[1:]]
    return problems


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    ap.add_argument("--specs", type=Path, required=True)
    ap.add_argument("--scenarios", type=Path, required=True)
    a = ap.parse_args(argv)
    specs = json.loads(a.specs.read_text(encoding="utf-8"))
    scenarios = json.loads(a.scenarios.read_text(encoding="utf-8"))
    problems = validate_generated(scenarios, specs)
    if problems:
        print(f"{len(problems)} problem(s):")
        print("\n".join(f"  - {x}" for x in problems))
        return 1
    print(f"OK: {len(scenarios)} scenarios follow their specifications")
    return 0


if __name__ == "__main__":
    sys.exit(main())
