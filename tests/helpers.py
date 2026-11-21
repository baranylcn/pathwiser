"""Shared builders for the tests: toy records, a deterministic fake encoder, synthetic worlds and scenarios."""

from __future__ import annotations

import hashlib
import random

import numpy as np

from pathwiser.synth import spec as SP
from pathwiser.synth import styles as S

TOPICS = {
    "flight": "search for cheap airline flights to a destination city",
    "hotel": "reserve a hotel room for several nights",
    "music": "play a song by my favourite artist on the speaker",
    "weather": "tell me the weather forecast for tomorrow",
    "bank": "check the balance of my bank account",
    "pizza": "order a pizza for delivery to my home",
}


MESSAGES = {
    "flight": "can you find me a cheap airline flight to the city of paris",
    "hotel": "i need a hotel room for three nights please",
    "music": "play my favourite song by that artist",
    "weather": "what is the weather forecast for tomorrow",
    "bank": "what is the balance in my bank account",
    "pizza": "i want to order a pizza delivered to my home",
}


def synth(n: int, seed: int, k: int = 4, no_match: float = 0.2, weight: float = 1.0) -> list[dict]:
    rng = random.Random(seed)
    names = sorted(TOPICS)
    out = []
    for i in range(n):
        topic = rng.choice(names)
        others = rng.sample([t for t in names if t != topic], k)
        if rng.random() < no_match:
            conds, label = others, -1
        else:
            conds = others[: k - 1] + [topic]
            rng.shuffle(conds)
            label = conds.index(topic)
        conds = [TOPICS[c] for c in conds]
        ctx = [{"role": "assistant", "content": "How can I help you today?"}] if i % 2 else []
        out.append(
            {
                "example_id": f"ex{seed}_{i:04d}",
                "split": "train",
                "input": {"message": MESSAGES[topic], "context": ctx, "conditions": conds, "query": MESSAGES[topic]},
                "target": {"label": label, "label_null": label if label >= 0 else k, "has_match": int(label >= 0)},
                "weight": weight,
                "meta": {"dataset": "toy", "is_paired": False, "paired_with": None, "slices": []},
            }
        )
    return out


class FakeEncoder:
    """Deterministic bag-of-hashed-words embedding: cosine grows with word overlap."""

    dim = 64

    def encode(self, texts, batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, int(hashlib.sha256(w.encode()).hexdigest()[:6], 16) % self.dim] += 1.0
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norm, 1e-9)


def make_world(seed: int = 0, offsets=(0.0, 3.0, -3.0), n_per: int = 300, k: int = 4, no_match: float = 0.25):
    """Three 'datasets' whose absolute score level differs (domain shift); inside each, the gold candidate stands out by a margin,
    and no_match examples have no standout. Two models: ``a`` is informative, ``b`` is noisier."""
    rng = np.random.default_rng(seed)
    records, a, b = [], [], []
    for d, off in enumerate(offsets):
        for i in range(n_per):
            nm = rng.random() < no_match
            label = -1 if nm else int(rng.integers(k))
            base = off + rng.normal(0, 0.6, size=k)
            sa = base.copy()
            sb = off + rng.normal(0, 1.2, size=k)
            if label >= 0:
                sa[label] += 2.5
                sb[label] += 1.0
            records.append(
                {
                    "example_id": f"d{d}_{i}",
                    "input": {"conditions": [f"c{j}" for j in range(k)]},
                    "target": {"label": label},
                    "meta": {"dataset": f"ds{d}", "is_paired": False, "paired_with": None, "slices": []},
                }
            )
            a.append(sa)
            b.append(sb)
    return records, {"a": a, "b": b}


def fake_scenario(spec: dict) -> dict:
    """A scenario that follows ``spec`` exactly (what a perfect generator would return)."""
    letters = list(SP.LETTERS[: spec["k"]])
    exs = []
    for plan in spec["examples"]:
        ctx = {
            "none": [],
            "first_turn": [["a", "How can I help?"]],
            "later_turn": [["u", "hi there"], ["a", "Hello, what do you need?"]],
        }[plan["context"]]
        ex = {"ctx": ctx, "msg": f"message {spec['id']} {plan['i']}"}
        if plan["kind"] == "route":
            ex["gold"] = plan["target"]
        else:
            ex.update(gold=None, nm=plan["nm"])
            if plan["nm"] == "sibling_removed":
                ex.update(intent=plan["intent"], cands=[c for c in letters if c != plan["intent"]])
        exs.append(ex)
    return {"id": spec["id"], "conditions": {c: f"condition {c}" for c in letters}, "examples": exs}


CONDITION_TEXT = {
    "imperative_terse": "Reschedule a visit",
    "user_wants": "The customer wants to move an existing booking to a new date",
    "question": "Is the user asking about the status of an existing job?",
    "detailed": "The user wants to change something about a job that is already booked, for example moving the date or changing the address",
    "internal_jargon": "RESCH req on open WO",
}


def good_scenario(spec: dict) -> dict:
    sc = fake_scenario(spec)
    sc["node"] = {"name": "demo_router", "description": "Decide what the person wants to do."}
    text = CONDITION_TEXT[spec["condition_style"]]
    sc["conditions"] = {
        c: f"{text} {i}" if spec["condition_style"] != "question" else text.replace("job", f"job {i}")
        for i, c in enumerate(sc["conditions"])
    }
    for i, ex in enumerate(sc["examples"]):
        ex["msg"] = f"I would like some help with my thing number {spec['id']} {i} thanks"
        ex["why"] = "plain request"
    return sc


def make_batch(n: int = 12, seed: int = 21):
    specs = SP.sample_specs("eval_v2", n, seed, S.EVAL_PERSONA_WEIGHTS, "t")
    scenarios = []
    for spec in specs:
        sc = good_scenario(spec)
        for i, ex in enumerate(sc["examples"]):  # sentences that noise functions can act on
            ex["msg"] = (
                f"Hello, please could you tell me tomorrow about the appointment number {spec['id']} {i} because I really want to know"
            )
            ex["ctx"] = [[r, f"{t} please tell me before tomorrow" if r == "u" else t] for r, t in ex["ctx"]]
        scenarios.append(sc)
    return specs, scenarios


def full_records(world_records: list[dict], prefix: str, paired: bool = False) -> list[dict]:
    out = []
    for i, r in enumerate(world_records):
        k = len(r["input"]["conditions"])
        out.append(
            {
                "example_id": f"{prefix}{r['example_id']}",
                "input": {"message": f"message {prefix}{i}", "context": [], "conditions": r["input"]["conditions"]},
                "target": r["target"],
                "meta": {
                    "dataset": r["meta"]["dataset"],
                    "is_paired": paired,
                    "paired_with": None,
                    "dialogue_id": f"{prefix}{i // 5}",
                    "slices": [
                        "ctx:none",
                        f"k:{'2-3' if k <= 3 else '4-6'}",
                        "label:route" if r["target"]["label"] >= 0 else "label:no_match",
                    ],
                },
            }
        )
    return out
