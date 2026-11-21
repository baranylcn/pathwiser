"""Blind verification of synthetic examples by annotators that did not write them.

``make_blind`` writes anonymous item files (no labels, no persona, shuffled order, a different candidate order in pass B) plus a private key;
the annotators answer with a letter or NONE; ``build_verification`` turns the answers into the record stored next to the data
(``blind_verification.json``, texts instead of letters, fingerprinted with the text that was shown) and ``disagreements`` lists the examples
on which any annotator differs from the intended label so that they can be adjudicated (rewritten and re-verified, or dropped).

Agreement between model annotators measures that a label is consistent and unambiguous to models; it is not a human audit.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from pathwiser.storage.hashing import record_text_sha

LETTERS = "ABCDEFGHIJ"


def make_blind(
    records: list[dict], out_dir: Path, seed: int, prefix_passes: tuple[str, ...] = ("A", "B"), pieces: int = 2
) -> dict:
    """Write ``<out_dir>/blind/<Pass><n>.json`` and ``<out_dir>/key/key.json``. Every example is in exactly one file of each pass."""
    out_dir = Path(out_dir)
    (out_dir / "blind").mkdir(parents=True, exist_ok=True)
    (out_dir / "key").mkdir(parents=True, exist_ok=True)
    (out_dir / "answers").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    key: dict = {}
    files: dict[str, int] = {}
    for pi, p in enumerate(prefix_passes):
        order = list(range(len(records)))
        rng.shuffle(order)
        size = -(-len(order) // pieces)
        for piece in range(pieces):
            name = f"{p}{piece + 1}"
            items = []
            for n, i in enumerate(order[piece * size : (piece + 1) * size]):
                r = records[i]
                conds = list(enumerate(r["input"]["conditions"]))
                if pi % 2 == 1:
                    conds = conds[::-1]  # pass B: reversed candidate order
                item_id = f"{name}-{n:03d}"
                key[item_id] = {
                    "example_id": r["example_id"],
                    "letters": {LETTERS[j]: orig for j, (orig, _) in enumerate(conds)},
                }
                items.append(
                    {
                        "id": item_id,
                        "node": f"{r['input']['node']['name'].replace('_', ' ')}: {r['input']['node']['description']}",
                        "context": [
                            f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['content']}"
                            for t in r["input"]["context"]
                        ],
                        "message": r["input"]["message"],
                        "conditions": {LETTERS[j]: text for j, (_, text) in enumerate(conds)},
                    }
                )
            (out_dir / "blind" / f"{name}.json").write_text(
                json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            files[name] = len(items)
    (out_dir / "key" / "key.json").write_text(json.dumps(key, ensure_ascii=False), encoding="utf-8")
    return files


def collect(out_dir: Path, names: list[str]) -> dict[str, dict]:
    """example_id -> {pass letter: {"pick": text | "NONE", "alt": text | None, "note": str}} from the answer files of ``names``."""
    out_dir = Path(out_dir)
    key = json.loads((out_dir / "key" / "key.json").read_text(encoding="utf-8"))
    result: dict[str, dict] = {}
    for name in names:
        blind = {i["id"]: i for i in json.loads((out_dir / "blind" / f"{name}.json").read_text(encoding="utf-8"))}
        answers = json.loads((out_dir / "answers" / f"{name}.json").read_text(encoding="utf-8"))
        seen = set()
        for a in answers:
            item = blind[a["id"]]
            conds = item["conditions"]
            if a["pick"] != "NONE" and a["pick"] not in conds:
                raise ValueError(f"{a['id']}: pick {a['pick']!r} is not a candidate letter")
            seen.add(a["id"])
            result.setdefault(key[a["id"]]["example_id"], {})[name[0]] = {
                "pick": "NONE" if a["pick"] == "NONE" else conds[a["pick"]],
                "alt": conds[a["alt"]] if a.get("alt") in conds else None,
                "note": a.get("note", ""),
            }
        missing = set(blind) - seen
        if missing:
            raise ValueError(f"{name}: {len(missing)} items were not answered")
    return result


def build_verification(records: list[dict], answers: dict[str, dict], annotators: dict[str, str], note: str) -> dict:
    """The record stored in the repo. ``verified_sha`` fingerprints the text the annotators saw (current text of ``records``)."""
    by_id = {r["example_id"]: r for r in records}
    unknown = set(answers) - set(by_id)
    if unknown:
        raise ValueError(f"answers for unknown examples: {sorted(unknown)[:3]}")
    examples = {}
    for eid, passes in answers.items():
        examples[eid] = {
            "verified_sha": record_text_sha(by_id[eid]),
            "passes": {p: {"annotator": annotators[p], **v} for p, v in sorted(passes.items())},
        }
    return {"examples": examples, "note": note}


def merge_verification(old: dict, new: dict, label: str) -> dict:
    """Add a later annotation round. Entries of ``new`` replace those of ``old`` for the same example (those examples were edited and
    re-annotated); the annotation rounds are kept in ``rounds``."""
    rounds = list(old.get("rounds", [{"label": "round 1", "examples_annotated": len(old["examples"])}]))
    rounds.append({"label": label, "examples_annotated": len(new["examples"])})
    merged = dict(old, examples={**old["examples"], **new["examples"]}, rounds=rounds)
    merged["note"] = old.get("note", "") + f" | {label}: {new.get('note', '')}"
    return merged


def attach_verification(records: list[dict], verification: dict | None) -> dict:
    """Attach ``meta.verification`` from the blind-annotation record; returns an aggregate summary.

    An example counts as verified only if its current text fingerprint equals the one that was annotated;
    examples edited after annotation are reported as ``changed_after_verification``.
    """
    entries = (verification or {}).get("examples", {})
    status = Counter()
    agree = Counter()
    for r in records:
        e = entries.get(r["example_id"])
        label = r["target"]["label"]
        intended = "NONE" if label < 0 else r["input"]["conditions"][label]
        if e is None:
            v = {"status": "unverified"}
        elif e["verified_sha"] != record_text_sha(r):
            v = {"status": "changed_after_verification"}
        else:
            picks = [p["pick"] for p in e["passes"].values()]
            v = {
                "status": "blind_verified",
                "passes": len(picks),
                "all_agree_with_intended": all(p == intended for p in picks),
                "alt_flagged": any(p.get("alt") for p in e["passes"].values()),
            }
            agree[(r["meta"]["label_type"], v["all_agree_with_intended"])] += 1
        r["meta"]["verification"] = v
        status[v["status"]] += 1
    ver = [r["meta"]["verification"] for r in records if r["meta"]["verification"]["status"] == "blind_verified"]
    return {
        "status_counts": dict(status),
        "blind_verified_examples": len(ver),
        "all_passes_agree_with_intended": sum(v["all_agree_with_intended"] for v in ver),
        "alt_flagged": sum(v["alt_flagged"] for v in ver),
        "agreement_by_label_type": {
            t: f"{agree[(t, True)]}/{agree[(t, True)] + agree[(t, False)]}" for t in ("route", "no_match")
        },
        "note": (verification or {}).get("note"),
    }


def disagreements(records: list[dict], verification: dict) -> list[dict]:
    """Examples on which at least one annotator differs from the intended label (or that nobody annotated)."""
    out = []
    entries = verification["examples"]
    for r in records:
        label = r["target"]["label"]
        intended = "NONE" if label < 0 else r["input"]["conditions"][label]
        e = entries.get(r["example_id"])
        if e is None:
            out.append({"example_id": r["example_id"], "reason": "not annotated"})
            continue
        if record_text_sha(r) != e["verified_sha"]:
            out.append({"example_id": r["example_id"], "reason": "changed after annotation"})
            continue
        wrong = {p: v["pick"] for p, v in e["passes"].items() if v["pick"] != intended}
        if wrong:
            out.append(
                {
                    "example_id": r["example_id"],
                    "reason": "disagreement",
                    "intended": intended,
                    "picks": wrong,
                    "message": r["input"]["message"],
                    "context": [t["content"] for t in r["input"]["context"]],
                    "alts": {p: v["alt"] for p, v in e["passes"].items() if v.get("alt")},
                    "notes": {p: v["note"] for p, v in e["passes"].items() if v.get("note")},
                }
            )
    return out


def agreement_summary(records: list[dict], verification: dict) -> dict:
    entries = verification["examples"]
    n = len(records)
    full = n - len(disagreements(records, verification))
    by_pass: dict[str, list[int]] = {}
    for r in records:
        label = r["target"]["label"]
        intended = "NONE" if label < 0 else r["input"]["conditions"][label]
        for p, v in entries.get(r["example_id"], {}).get("passes", {}).items():
            by_pass.setdefault(p, [0, 0])
            by_pass[p][0] += v["pick"] == intended
            by_pass[p][1] += 1
    return {
        "examples": n,
        "all_annotators_agree": full,
        "share": round(full / n, 4) if n else None,
        "per_pass": {p: {"agree": a, "total": t, "share": round(a / t, 4)} for p, (a, t) in sorted(by_pass.items())},
    }


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    """``pathwiser sets verify --work DIR --records test.jsonl --out blind_verification.json --annotators A=model B=model``"""
    import argparse

    ap = argparse.ArgumentParser(prog=prog, description=main.__doc__)
    ap.add_argument("--work", type=Path, required=True, help="directory made by make_blind (blind/, key/, answers/)")
    ap.add_argument(
        "--records", type=Path, required=True, help="the model-ready records that were shown to the annotators"
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="verification record to write (data/curated/<set>/blind_verification.json)",
    )
    ap.add_argument(
        "--annotators", nargs="+", required=True, metavar="PASS=MODEL", help="e.g. A=claude-sonnet-5-5 B=claude-fable"
    )
    ap.add_argument("--names", nargs="+", default=["A1", "A2", "B1", "B2"], help="answer files to read")
    ap.add_argument(
        "--note",
        default="Blind annotation by independent model passes (anonymous ids, shuffled items, no labels); NOT a human audit.",
    )
    ap.add_argument("--disagreements", type=Path, help="write the list of disagreements here (JSON)")
    ap.add_argument(
        "--merge-round", metavar="LABEL", help="merge into the existing --out file as a later round with this label"
    )
    a = ap.parse_args(argv)
    recs = [json.loads(line) for line in a.records.read_text(encoding="utf-8").splitlines()]
    ver = build_verification(recs, collect(a.work, a.names), dict(x.split("=", 1) for x in a.annotators), a.note)
    if a.merge_round:
        ver = merge_verification(json.loads(a.out.read_text(encoding="utf-8")), ver, a.merge_round)
    a.out.write_text(
        json.dumps(ver, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    bad = disagreements(recs, ver)
    if a.disagreements:
        a.disagreements.write_text(json.dumps(bad, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(agreement_summary(recs, ver), indent=1))
    print(f"{len(bad)} example(s) need adjudication")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
