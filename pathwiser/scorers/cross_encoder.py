"""Cross-encoder scorer: message, context and one condition description are encoded jointly.

Every candidate is scored independently. Training is listwise: the candidate scores of an example plus one learned scalar
"none of them" logit go through a softmax, so ``no_match`` examples teach the model where to abstain. That logit is only a
training device; at inference the validation-tuned threshold rule decides, as for every other scorer.

Input: ``<message> [SEP] <newest turn> [SEP] <older turn> ...`` paired with the condition description, so truncation drops the
oldest context and never the message. The workflow node text is not part of the input.

Training options:
* ``train_examples``: weighted sample of the train split (the default budget is sized for CPU-only training).
* ``negatives="hard"``: always include the ``n_hard`` wrong candidates that a frozen bi-encoder finds most similar to the message.
* ``checkpoint_dir``: save the training state every ``checkpoint_every`` steps; ``fit`` resumes after an interruption with the
  same data order and optimiser state.
* ``monitor_examples``: Top-1 and MRR on validation examples at 25/50/75/100% of training (diagnostic only).
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np

from pathwiser.scorers.training import candidate_sizes, hardware, sample_examples, split_flat

DEFAULT_INIT = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FALLBACK_INIT = "sentence-transformers/all-MiniLM-L6-v2"
HARD_NEGATIVE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIGNATURE_KEYS = (
    "init",
    "max_len",
    "epochs",
    "batch_examples",
    "lr",
    "train_examples",
    "max_train_candidates",
    "max_context_turns",
    "seed",
    "negatives",
    "n_hard",
    "hard_negative_model",
)


def format_query(record: dict, sep: str, max_context_turns: int) -> str:
    """Message first, then the context newest turn first: tail truncation drops the oldest context, never the message."""
    msg = record["input"]["message"]
    turns = record["input"]["context"][-max_context_turns:] if max_context_turns > 0 else []
    if not turns:
        return msg
    return f"{msg} {sep} " + f" {sep} ".join(f"{t['role']}: {t['content']}" for t in reversed(turns))


def build_group(
    record: dict, max_candidates: int, rng: random.Random, hard_order: list[int] | None = None, n_hard: int = 2
) -> tuple[list[int], int]:
    """Candidates used for one training example and the target position inside them.

    Route example: the gold candidate plus up to ``max_candidates - 1`` negatives; target = gold position.
    no_match example: up to ``max_candidates`` negatives; target = ``len(chosen)`` (the "none" slot).
    Negatives are random, unless ``hard_order`` (candidate indices, most message-similar first) is given: then the first
    ``n_hard`` wrong candidates of that order are always included and only the remaining slots are random.
    """
    k, label = len(record["input"]["conditions"]), record["target"]["label"]
    others = [j for j in range(k) if j != label]
    need = min(len(others), max_candidates - 1 if label >= 0 else max_candidates)
    if hard_order is None:
        negatives = rng.sample(others, need)
    else:
        hard = [j for j in hard_order if j != label][: min(n_hard, need)]
        rest = [j for j in others if j not in hard]
        negatives = hard + rng.sample(rest, need - len(hard))
    chosen = negatives + [label] if label >= 0 else negatives
    rng.shuffle(chosen)
    return chosen, (chosen.index(label) if label >= 0 else len(chosen))


class CrossEncoderScorer:
    needs_train = True
    needs_val = False
    name = "cross_encoder"

    def __init__(
        self,
        init: str = DEFAULT_INIT,
        max_len: int = 128,
        epochs: int = 1,
        batch_examples: int = 16,
        lr: float = 3e-5,
        train_examples: int | None = 8000,
        max_train_candidates: int = 4,
        max_context_turns: int = 4,
        seed: int = 42,
        eval_batch_pairs: int = 128,
        save_dir: str | None = "models/cross_encoder",
        threads: int | None = None,
        negatives: str = "random",
        n_hard: int = 2,
        hard_negative_model: str = HARD_NEGATIVE_MODEL,
        checkpoint_dir: str | None = None,
        checkpoint_every: int = 100,
        monitor_examples: int = 0,
    ) -> None:
        if negatives not in ("random", "hard"):
            raise ValueError("negatives must be 'random' or 'hard'")
        if checkpoint_every < 1:
            raise ValueError("checkpoint_every must be >= 1")
        self.init, self.max_len, self.epochs, self.batch_examples, self.lr = init, max_len, epochs, batch_examples, lr
        self.train_examples, self.max_train_candidates, self.max_context_turns = (
            train_examples,
            max_train_candidates,
            max_context_turns,
        )
        self.seed, self.eval_batch_pairs, self.save_dir, self.threads = seed, eval_batch_pairs, save_dir, threads
        self.negatives, self.n_hard, self.hard_negative_model = negatives, n_hard, hard_negative_model
        self.checkpoint_dir, self.checkpoint_every, self.monitor_examples = (
            checkpoint_dir,
            checkpoint_every,
            monitor_examples,
        )
        self.needs_val = monitor_examples > 0
        self.params = {
            "init": init,
            "max_len": max_len,
            "epochs": epochs,
            "batch_examples": batch_examples,
            "lr": lr,
            "train_examples": train_examples,
            "max_train_candidates": max_train_candidates,
            "max_context_turns": max_context_turns,
            "seed": seed,
            "negatives": negatives,
            "n_hard": n_hard if negatives == "hard" else None,
            "hard_negative_model": hard_negative_model if negatives == "hard" else None,
            "loss": "listwise softmax over candidates + learned none logit",
            "optimizer": "AdamW (wd 0.01), 6% warmup then linear decay, grad clip 1.0",
            "sampling": "weighted without replacement (training weights)",
            "input": "message [SEP] context newest-first; no node text",
        }
        self.info: dict = {}
        self.log = lambda s: None
        self._tok = self._model = None
        self._null = 0.0

    # -- model loading (overridden in tests with a tiny random model)
    def _load_model(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        err = None
        for name in dict.fromkeys([self.init, FALLBACK_INIT]):
            try:
                tok = AutoTokenizer.from_pretrained(name)
                model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1)
                self.params["resolved_init"] = name
                return tok, model
            except OSError as e:  # not cached and not downloadable
                err = e
                self.log(f"  could not load {name}: {type(e).__name__}")
        raise err

    def _load_pretrained(self, directory: Path):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        return AutoTokenizer.from_pretrained(directory), AutoModelForSequenceClassification.from_pretrained(directory)

    def _load_encoder(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.hard_negative_model)

    def _threads(self) -> int:
        return self.threads or os.cpu_count() or 4

    def _signature(self) -> str:
        return json.dumps({k: self.params.get(k) for k in SIGNATURE_KEYS}, sort_keys=True)

    # -- hard negatives
    def _hard_orders(self, subset: list[dict]) -> list[list[int]]:
        """Per example, candidate indices ordered by cosine similarity to the message (most similar first), frozen bi-encoder."""
        enc = self._load_encoder()
        messages = sorted({r["input"]["message"] for r in subset})
        conds = sorted({c for r in subset for c in r["input"]["conditions"]})
        m_emb = enc.encode(
            messages, batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        c_emb = enc.encode(
            conds, batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        m_row = {t: i for i, t in enumerate(messages)}
        c_row = {t: i for i, t in enumerate(conds)}
        orders = []
        for r in subset:
            sims = c_emb[[c_row[c] for c in r["input"]["conditions"]]] @ m_emb[m_row[r["input"]["message"]]]
            orders.append([int(j) for j in np.argsort(-sims, kind="stable")])
        return orders

    # -- checkpoints
    def _ckpt_path(self) -> Path | None:
        return Path(self.checkpoint_dir) / "ckpt.pt" if self.checkpoint_dir else None

    def _save_ckpt(self, state: dict) -> None:
        import torch

        path = self._ckpt_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)

    def _load_ckpt(self) -> dict | None:
        import torch

        path = self._ckpt_path()
        if path is None or not path.exists():
            return None
        state = torch.load(path, map_location="cpu", weights_only=False)  # written by this class only
        if state.get("signature") != self._signature():
            self.log(f"  ignoring checkpoint {path}: it was written with different training settings")
            return None
        return state

    def _after_step(self, step: int) -> None:
        """Hook called after every optimiser step (tests use it to simulate an interruption)."""

    # -- training
    def fit(self, train: list[dict], val: list[dict] | None = None) -> None:
        import torch
        import torch.nn.functional as F

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.set_num_threads(self._threads())
        tok, model = self._load_model()
        subset = sample_examples(train, self.train_examples, self.seed)
        hard_orders = self._hard_orders(subset) if self.negatives == "hard" else None
        monitor_set = (
            [r for r in (val or []) if not r["meta"]["is_paired"]][: self.monitor_examples]
            if self.monitor_examples
            else []
        )
        B = self.batch_examples
        steps = -(-len(subset) // B) * self.epochs
        warm = max(1, int(0.06 * steps))
        monitor_at = {max(1, round(steps * f)) for f in (0.25, 0.5, 0.75)} | {steps} if monitor_set else set()
        null = torch.nn.Parameter(torch.zeros(()))
        opt = torch.optim.AdamW(
            [
                {"params": list(model.parameters()), "lr": self.lr, "weight_decay": 0.01},
                {"params": [null], "lr": 1e-2, "weight_decay": 0.0},
            ]
        )
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm)))
        )
        params = list(model.parameters()) + [null]
        rng = random.Random(self.seed)
        sep = tok.sep_token
        step, n_pairs, win_loss, win_hit, win_n, prior_elapsed = 0, 0, 0.0, 0, 0, 0.0
        curve: list[dict] = []
        val_curve: list[dict] = []
        start_epoch, first_b, resume_state = 0, 0, None
        ckpt = self._load_ckpt()
        if ckpt is not None:
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])
            sched.load_state_dict(ckpt["sched"])
            with torch.no_grad():
                null.copy_(ckpt["null"])
            torch.set_rng_state(ckpt["torch_rng"])
            step, n_pairs, (win_loss, win_hit, win_n) = ckpt["step"], ckpt["n_pairs"], ckpt["win"]
            curve, val_curve, prior_elapsed = ckpt["curve"], ckpt["val_curve"], ckpt["elapsed"]
            start_epoch, first_b, resume_state = ckpt["epoch"], ckpt["next_b"], ckpt
            self.log(f"  resumed from checkpoint at step {step}/{steps}")
        model.train()
        t0 = time.perf_counter() - prior_elapsed
        for epoch in range(start_epoch, self.epochs):
            if resume_state is not None and epoch == start_epoch:
                rng.setstate(resume_state["epoch_rng"])  # recreate this epoch's order, then continue where we stopped
                epoch_rng = resume_state["epoch_rng"]
                order = list(range(len(subset)))
                rng.shuffle(order)
                rng.setstate(resume_state["rng"])
            else:
                epoch_rng = rng.getstate()
                order = list(range(len(subset)))
                rng.shuffle(order)
                first_b = 0
            for b in range(first_b, len(order), B):
                a_texts, b_texts, spans, targets = [], [], [], []
                for i in order[b : b + B]:
                    r = subset[i]
                    chosen, target = build_group(
                        r,
                        self.max_train_candidates,
                        rng,
                        hard_orders[i] if hard_orders is not None else None,
                        self.n_hard,
                    )
                    q = format_query(r, sep, self.max_context_turns)
                    a_texts += [q] * len(chosen)
                    b_texts += [r["input"]["conditions"][j] for j in chosen]
                    spans.append(len(chosen))
                    targets.append(target)
                enc = tok(
                    a_texts,
                    b_texts,
                    padding=True,
                    truncation="only_first",
                    max_length=self.max_len,
                    return_tensors="pt",
                )
                logits = model(**enc).logits.squeeze(-1)
                loss, start = 0.0, 0
                for span, target in zip(spans, targets):
                    z = torch.cat([logits[start : start + span], null.view(1)])
                    loss = loss + F.cross_entropy(z.unsqueeze(0), torch.tensor([target]))
                    win_hit += int(z.argmax().item() == target)
                    start += span
                loss = loss / len(spans)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                n_pairs += len(a_texts)
                win_loss += loss.item() * len(spans)
                win_n += len(spans)
                if step % 25 == 0 or step == steps:
                    curve.append(
                        {"step": step, "loss": round(win_loss / win_n, 4), "train_acc": round(win_hit / win_n, 4)}
                    )
                    el = time.perf_counter() - t0
                    self.log(
                        f"  step {step}/{steps} loss={curve[-1]['loss']:.3f} acc={curve[-1]['train_acc']:.3f} "
                        f"{n_pairs / max(el, 1e-9):.1f} pairs/s elapsed={el / 60:.1f} min eta={(steps - step) * el / step / 60:.1f} min"
                    )
                    win_loss, win_hit, win_n = 0.0, 0, 0
                if step in monitor_at:
                    val_curve.append(
                        {"step": step, "fraction": round(step / steps, 3), **self._monitor(model, tok, monitor_set)}
                    )
                    self.log(
                        f"  monitor @ {step}/{steps}: top1={val_curve[-1]['top1']:.3f} mrr={val_curve[-1]['mrr']:.3f} "
                        f"on {val_curve[-1]['n_route']} validation examples"
                    )
                if self.checkpoint_dir and step % self.checkpoint_every == 0 and step < steps:
                    self._save_ckpt(
                        {
                            "signature": self._signature(),
                            "model": model.state_dict(),
                            "opt": opt.state_dict(),
                            "sched": sched.state_dict(),
                            "null": null.detach().clone(),
                            "step": step,
                            "epoch": epoch,
                            "next_b": b + B,
                            "epoch_rng": epoch_rng,
                            "rng": rng.getstate(),
                            "torch_rng": torch.get_rng_state(),
                            "n_pairs": n_pairs,
                            "win": (win_loss, win_hit, win_n),
                            "curve": curve,
                            "val_curve": val_curve,
                            "elapsed": time.perf_counter() - t0,
                        }
                    )
                self._after_step(step)
        model.eval()
        self._tok, self._model, self._null = tok, model, null.detach().item()
        secs = time.perf_counter() - t0
        self.info = {
            "train_examples": len(subset),
            "epochs": self.epochs,
            "steps": step,
            "train_pairs": n_pairs,
            "train_seconds": round(secs, 1),
            "pairs_per_second": round(n_pairs / max(secs, 1e-9), 2),
            "null_logit": round(null.detach().item(), 4),
            "final_loss": curve[-1]["loss"] if curve else None,
            "loss_curve": curve,
            "validation_curve": val_curve,
            "negatives": self.negatives,
            "resumed": ckpt is not None,
            "hardware": hardware(),
        }
        if self.save_dir:
            out = Path(self.save_dir)
            out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out)
            tok.save_pretrained(out)
            (out / "training.json").write_text(
                json.dumps({"params": self.params, "info": self.info}, indent=2) + "\n", encoding="utf-8"
            )
        if self.checkpoint_dir and self._ckpt_path().exists():
            self._ckpt_path().unlink()  # finished: a stale checkpoint must never be resumed

    def _monitor(self, model, tok, records: list[dict]) -> dict:
        prev = self._tok, self._model
        self._tok, self._model = tok, model
        try:
            scores = self.score(records)
        finally:
            self._tok, self._model = prev
            model.train()
        route = [(r["target"]["label"], s) for r, s in zip(records, scores) if r["target"]["label"] >= 0]
        ranks = [int(np.nonzero(np.argsort(-s, kind="stable") == y)[0][0]) + 1 for y, s in route]
        return {
            "n_route": len(route),
            "top1": round(float(np.mean([x == 1 for x in ranks])), 4),
            "mrr": round(float(np.mean([1 / x for x in ranks])), 4),
        }

    # -- scoring only: weights written by ``fit`` (``save_dir``)
    def load(self, directory: str | Path) -> None:
        """Load weights saved by a previous ``fit`` instead of training; the scoring settings come from its ``training.json``."""
        d = Path(directory)
        meta = json.loads((d / "training.json").read_text(encoding="utf-8"))
        tok, model = self._load_pretrained(d)
        model.eval()
        saved = meta["params"]
        self.max_len, self.max_context_turns = saved["max_len"], saved["max_context_turns"]
        self.params = dict(saved, loaded_from=str(d))
        self.info = meta["info"]
        self._tok, self._model, self._null = tok, model, float(meta["info"].get("null_logit", 0.0))

    # -- inference
    def score(self, records: list[dict]) -> list[np.ndarray]:
        import torch

        if self._model is None:
            raise RuntimeError("fit() or load() must be called before score()")
        torch.set_num_threads(self._threads())
        tok, model = self._tok, self._model
        sep = tok.sep_token
        a_texts, b_texts = [], []
        for r in records:
            q = format_query(r, sep, self.max_context_turns)
            for c in r["input"]["conditions"]:
                a_texts.append(q)
                b_texts.append(c)
        ids: list[list[int]] = []
        types: list[list[int]] = []
        for i in range(0, len(a_texts), 4096):
            enc = tok(a_texts[i : i + 4096], b_texts[i : i + 4096], truncation="only_first", max_length=self.max_len)
            ids.extend(enc["input_ids"])
            types.extend(enc["token_type_ids"] if "token_type_ids" in enc else [[0] * len(x) for x in enc["input_ids"]])
        n = len(ids)
        order = np.argsort([len(x) for x in ids], kind="stable")  # similar lengths together: little padding
        out = np.empty(n, dtype=np.float32)
        pad = tok.pad_token_id or 0
        t0 = time.perf_counter()
        model.eval()
        with torch.inference_mode():
            for bi, start in enumerate(range(0, n, self.eval_batch_pairs)):
                idx = order[start : start + self.eval_batch_pairs]
                width = max(len(ids[i]) for i in idx)
                inp = torch.full((len(idx), width), pad, dtype=torch.long)
                mask = torch.zeros((len(idx), width), dtype=torch.long)
                tt = torch.zeros((len(idx), width), dtype=torch.long)
                for row, i in enumerate(idx):
                    L = len(ids[i])
                    inp[row, :L] = torch.tensor(ids[i])
                    mask[row, :L] = 1
                    tt[row, :L] = torch.tensor(types[i])
                out[idx] = (
                    model(input_ids=inp, attention_mask=mask, token_type_ids=tt).logits.squeeze(-1).float().numpy()
                )
                if bi % 100 == 0:
                    el = time.perf_counter() - t0
                    self.log(
                        f"  scored {start + len(idx)}/{n} pairs ({(start + len(idx)) / max(el, 1e-9):.0f} pairs/s)"
                    )
        return split_flat(out, candidate_sizes(records))
