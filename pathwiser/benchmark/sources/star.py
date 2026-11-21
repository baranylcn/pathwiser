"""STAR loader: task selection in multi-task dialogues.

Condition = one of the 24 STAR tasks (descriptions authored in data/curated/star_tasks.json).
Moment = the user utterance right before a wizard ``select_task`` event; gold = the selected task.
Only ``CompletionLevel == Complete`` dialogues are used. Single-task dialogues contain no explicit
task selection and therefore yield no moments.
"""

from __future__ import annotations

import glob
from pathlib import Path

from pathwiser.benchmark.fileio import read_json
from pathwiser.benchmark.schema import Condition, DropLog, Moment, SourceData
from pathwiser.benchmark.seeding import stable_unit
from pathwiser.benchmark.sources.curated import load_curated
from pathwiser.benchmark.text import GREETING_ONLY, clean_text, n_words


def load(cfg: dict, raw: Path, curated: Path, drops: DropLog, limit: int | None) -> SourceData:
    dcfg = cfg["datasets"]["star"]
    root = raw / "star"
    task_ids = sorted(p.name for p in (root / "tasks").iterdir() if p.is_dir())
    meta, cur = load_curated(curated / "star_tasks.json", set(task_ids), "star")
    conds = {
        f"star:{t}": Condition(
            uid=f"star:{t}",
            dataset="star",
            family=t.split("_")[0],
            group=f"star:{t}",
            descriptions={"orig": cur[t]["description"].strip()},
            description_origin=meta["description_origin"],
            reviewed=bool(meta["reviewed"]),
            extra={"task": t},
        )
        for t in task_ids
    }
    max_turns = cfg["context_max_turns"]
    fr = dcfg["split_fractions"]
    files = sorted(glob.glob(str(root / "dialogues" / "*.json")), key=lambda p: int(Path(p).stem))
    moments: list[Moment] = []
    for path in files:
        if limit is not None and len(moments) >= limit:
            break
        d = read_json(Path(path))
        if d.get("CompletionLevel") != "Complete":
            drops.add("star", "incomplete_dialogue")
            continue
        did = str(d["DialogueID"])
        u = stable_unit(cfg["seed"], "star-split", did)
        base = "train" if u < fr["train"] else ("val" if u < fr["train"] + fr["val"] else "test")
        moments.extend(_extract(d, did, base, conds, dcfg["min_message_words"], max_turns, drops))
    return SourceData("star", "T3", cfg["licenses"]["star"], conds, moments, "dialogue_routed", single_turn=False)


def _extract(
    d: dict, did: str, base: str, conds: dict[str, Condition], min_words: int, max_turns: int, drops: DropLog
) -> list[Moment]:
    hist: list[dict] = []
    last_user: int | None = None
    selects: dict[int, list[tuple[int, str]]] = {}  # hist index of the user turn -> [(event_idx, task)]
    for ei, e in enumerate(d["Events"]):
        action, agent = e["Action"], e["Agent"]
        if action == "utter" and agent == "User":
            text = clean_text(e.get("Text") or "")
            if text:
                hist.append({"role": "user", "content": text})
                last_user = len(hist) - 1
        elif (action == "utter" and agent == "Wizard") or action == "pick_suggestion":
            text = clean_text(e.get("Text") or "")
            if text:
                hist.append({"role": "assistant", "content": text})
        elif action == "select_task":
            if last_user is None:
                drops.add("star", "select_before_any_user_turn")
            elif len(hist) - 1 != last_user:
                drops.add("star", "wizard_turn_between_user_and_select")
            else:
                selects.setdefault(last_user, []).append((ei, e["Task"]))
    out: list[Moment] = []
    for ui in sorted(selects):
        tasks = sorted({t for _, t in selects[ui]})
        if len(tasks) > 1:
            drops.add("star", "multi_gold_turn")
            continue
        uid = f"star:{tasks[0]}"
        if uid not in conds:
            drops.add("star", "unknown_task")
            continue
        msg = hist[ui]["content"]
        if n_words(msg) < min_words or GREETING_ONLY.match(msg):
            drops.add("star", "low_information_message")
            continue
        out.append(
            Moment(
                moment_id=f"star:{did}:{selects[ui][0][0]}",
                dataset="star",
                group_id=f"star:{did}",
                base_split=base,
                context=hist[:ui][-max_turns:] if max_turns else [],
                message=msg,
                gold_uid=uid,
                turn_idx=ui,
                is_first_user_turn=not any(h["role"] == "user" for h in hist[:ui]),
                n_context_total=ui,
                context_origin="natural" if ui else "none",
                extra={"domains": d.get("Scenario", {}).get("Domains")},
            )
        )
    return out
