"""ABCD loader: flow identification in customer-service conversations.

Condition = one of the 10 ABCD flows (descriptions derived from the source guidelines).
Moment = the first *substantive* customer turn (>= N words, not a bare greeting/yes/no) within the
first few customer turns; gold = the conversation's flow. Conversations without such a turn are dropped.
Splits are the official ones (dev -> val).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from pathwiser.benchmark.schema import Condition, DropLog, Moment, PipelineError, SourceData
from pathwiser.benchmark.sources.curated import load_curated
from pathwiser.benchmark.text import GREETING_ONLY, clean_text, n_words

SPLIT_MAP = (("train", "train"), ("dev", "val"), ("test", "test"))


def load(cfg: dict, raw: Path, curated: Path, drops: DropLog, limit: int | None) -> SourceData:
    dcfg = cfg["datasets"]["abcd"]
    data_dir = raw / "abcd" / "data"
    with gzip.open(data_dir / "abcd_v1.1.json.gz", "rt", encoding="utf-8") as f:
        data = json.load(f)
    with open(data_dir / "ontology.json", encoding="utf-8") as f:
        flows = sorted(json.load(f)["intents"]["flows"])
    meta, cur = load_curated(curated / "abcd_flows.json", set(flows), "abcd")
    conds = {
        f"abcd:{fl}": Condition(
            uid=f"abcd:{fl}",
            dataset="abcd",
            family=fl,
            group=f"abcd:{fl}",
            descriptions={"orig": cur[fl]["description"].strip()},
            description_origin=meta["description_origin"],
            reviewed=bool(meta["reviewed"]),
            extra={"flow": fl},
        )
        for fl in flows
    }
    max_turns = cfg["context_max_turns"]
    moments: list[Moment] = []
    for key, split in SPLIT_MAP:
        produced = 0
        for c in data[key]:
            if limit is not None and produced >= limit:
                break
            flow = c["scenario"]["flow"]
            uid = f"abcd:{flow}"
            if uid not in conds:
                raise PipelineError(f"abcd: flow '{flow}' not in ontology")
            turns = [
                (("user" if s == "customer" else "assistant"), clean_text(t)) for s, t in c["original"] if s != "action"
            ]
            found = None
            ordinal = 0
            for i, (role, text) in enumerate(turns):
                if role != "user":
                    continue
                ordinal += 1
                if ordinal > dcfg["max_customer_turns_scanned"]:
                    break
                if n_words(text) >= dcfg["min_message_words"] and not GREETING_ONLY.match(text):
                    found = (i, ordinal, text)
                    break
            if found is None:
                drops.add("abcd", "no_substantive_customer_turn_early")
                continue
            i, ordinal, text = found
            ctx = [{"role": r, "content": t} for r, t in turns[:i] if t]
            moments.append(
                Moment(
                    moment_id=f"abcd:{c['convo_id']}",
                    dataset="abcd",
                    group_id=f"abcd:{c['convo_id']}",
                    base_split=split,
                    context=ctx[-max_turns:] if max_turns else [],
                    message=text,
                    gold_uid=uid,
                    turn_idx=i,
                    is_first_user_turn=(ordinal == 1),
                    n_context_total=len(ctx),
                    context_origin="natural" if ctx else "none",
                    extra={"customer_turn_ordinal": ordinal},
                )
            )
            produced += 1
    return SourceData("abcd", "T3", cfg["licenses"]["abcd"], conds, moments, "dialogue_routed", single_turn=False)
