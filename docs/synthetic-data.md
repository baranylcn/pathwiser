# Synthetic scenario data: design

Status: the evaluation set `realistic-v2` is built and blind-verified (results below). Generating training data with the same
machinery has not started. The acceptance criteria below were fixed before any data was generated.

Commands: `pathwiser sets specs` (sample specifications), `pathwiser sets prompt` (generation prompt), `pathwiser sets check`
(conformance of generated scenarios), `pathwiser sets verify` (blind annotations to a verification record), `pathwiser sets build`
(evaluation set from authored scenarios).

## Why

The benchmark's training split has 88,790 examples but only **302 distinct conditions in 62 families**. The central research question is
generalisation to conditions never seen in training, and the model sees each condition text ~290 times. Models lose 11-21 Top-1 points on
the LLM-authored stress test `realistic-v1`. The lever is condition diversity (many scenarios, few examples each), not example count.

## Units and splits

* **Taxonomy** (`data/curated/synth/taxonomy_v1.json`): 33 groups x 6 domains = 198 business contexts. A *domain* is a specific business
  context (e.g. `dairy_cooperative`), a *group* an industry cluster (e.g. `agriculture`).
* **Blocklist.** Nothing may reuse a benchmark topic (restaurants, hotels, flights, trains, buses, rental cars, ride sharing, movies, music,
  events, banks/cards/payments, weather, calendar/alarms, messaging, apartments/homes, doctor visits, order/shipping/subscription/account
  support, small talk, ...) or one of the 65 business contexts of `realistic-v1`. Groups that are adjacent to those are flagged `near`.
* **Splits are made by group**, never by example or by domain inside a group: `eval_v2` (6 groups), `synth_test` (3), `synth_val` (3), `train`
  (the rest). Only `far` groups can be eval/test/val; `near` groups are training only. A split is fixed by a seeded shuffle and stored in the file.
* `realistic-v1` stays out of every training set until `realistic-v2` has been measured.

## Scenario and example specification (made by code, not by the LLM)

A scenario = workflow node + k candidate conditions (k from 2 to 10, benchmark-like) + ~5 examples. A **sampler** fixes, with a seed:
domain, node archetype (`intent_router`, `triage`, `confirmation`, `post_answer_followup`, `slot_answer_type`, `handoff`), k, condition
description style (imperative, "the user wants ...", question, long with examples, internal jargon), and per example: gold condition (a letter),
`no_match` kind (`sibling_removed`, `in_domain_uncovered`, `oos`; ~22% of examples), context kind (none / first turn / later turn),
whether the context decides the label, difficulty tags (indirect, negation, distractor), persona and noise.
The LLM only writes text for a given spec. Labels are correct by construction (the message is written for a chosen condition).

## Personas and noise

Persona = how a person writes. Noise = character-level damage applied **by code** afterwards (deterministic, seeded), because an LLM's
imitation of typos does not look like real typing errors. The clean persona text is kept in `meta.clean_message`.

| | seen in training | held out (evaluation and `synth_test` only) |
|---|---|---|
| Personas | neutral, formal, casual, frustrated, anxious, confused, hinting (indirect), lay description, jargon, multi-topic | slang_heavy (colloquial, mild profanity, no hate), rambling_polite (long, tangents), non_native (L2 grammar), fragment_texter |
| Noise | none, keyboard typos, no punctuation / lower case | text speak (u, 2day, tmrw, pls), autocorrect-style word swaps |

About half of the examples are plain neutral/casual with no noise; extreme styles are a minority, otherwise the data becomes a caricature.
Held-out styles never appear in `train`, so style generalisation is measurable (slices `persona:*`, `style_split:seen|heldout`).
Language is a field (`meta.language = "en"`); Turkish and other languages are a later, separate step with native generation (not translation).

## Quality control

1. Structural validation (same hard checks as `realistic-v1`) plus: sibling_removed needs an existing removed condition and a near sibling left;
   persona/noise tags must match the spec.
2. **Blind verification by two independent annotators** that did not write the data (anonymous ids, shuffled items, reversed candidate order in one
   pass, no labels). The generator and the verifiers are different models. Items on which an annotator disagrees with the intended label are
   adjudicated: rewritten (and re-verified) or dropped, never silently kept.
3. Leakage: no exact duplicate message or condition against the benchmark or `realistic-v1`; TF-IDF near-duplicate messages (cosine >= 0.8)
   and condition embedding near-duplicates (cosine >= 0.9) are listed and resolved.
4. Shortcut probes (as notebook 01): gold-position and length bias, accuracy of the lexical baselines (must be clearly above chance but far from solved).
5. Distribution report: domains/groups, k, no_match kinds, context kinds, personas, noise, tags against the plan.

## Acceptance criteria for `realistic-v2` (fixed in advance)

* >= 75 scenarios from `eval_v2` groups only, >= 350 examples, 4 held-out personas present with >= 30 examples each.
* First-round blind agreement of both annotators with the intended label >= 96%; every disagreement resolved by rewrite or drop.
* Zero exact leakage; every near-duplicate flag resolved.
* Not called "gold" unless a human audit is recorded (`human_audited` in the manifest).
* It is an evaluation set: never used to fit, tune or select. It is read only after the method to be evaluated is fixed.

## Limits stated up front

LLM-authored text, even with many personas, is not human text; the human-written benchmark stays the reality anchor (a gain on synthetic
sets that hurts the benchmark is not a gain). Simulated slang is not real slang. Annotator agreement measures consistency, not truth.

---

# Results: `realistic-v2` (Phase 0, built 2026-11-21)

Build: `pathwiser sets build --set realistic_v2` from `data/curated/realistic_v2/` (specs, authored scenarios, blind-verification record,
dropped examples) to `data/model/realistic-v2/` (`test.jsonl`, `manifest.json`). It is an evaluation set: never used to fit, tune or select.

| | |
|---|---|
| Size | **434 examples** in 80 scenarios (342 route, 92 `no_match`: 39 sibling_removed, 36 in_domain_uncovered, 17 oos) |
| Domains | 36 business contexts in 6 groups that are in no training split: arts_culture, community_religion, construction_trades, energy_industry, media_production, space_defense |
| Candidate set size | 2: 50, 3: 105, 4: 114, 5: 68, 6: 37, 7: 20, 8: 18, 9: 2, 10: 20 |
| Archetypes (scenarios) | intent_router 40, slot_answer_type 10, confirmation 8, post_answer_followup 8, triage 8, handoff 6 |
| Context | none 144, first turn 122, later turn 168; tags: indirect 89, context_dependent 79, distractor 32, negation 15 |
| Personas | 14 (neutral 83 ... jargon 9); held-out styles: slang_heavy 38, non_native 38, fragment_texter 38, rambling_polite 37; 157 examples (36%) carry a held-out persona or noise |
| Noise planned -> applied | keyboard typos 86/86, no punctuation 79 -> 73, text speak 27 -> 15, autocorrect 9 -> 6 (a message with nothing to abbreviate or swap stays clean; the applied noise is what is recorded) |
| Generator | claude-opus (one subagent per batch of <= 10 specifications; it saw only its own specification) |
| Annotators | two blind passes by claude-sonnet-5-5 (different prompt wording, reversed candidate order in pass B) |

## Acceptance criteria (fixed before generation) - all met

>= 75 scenarios, >= 350 examples, >= 30 examples for each held-out persona, zero exact leakage, no unresolved near-duplicate, every example
blind-verified, every disagreement resolved by rewrite or drop. Leakage was checked against 130,472 benchmark / `realistic-v1` messages and 1,065
conditions (exact, TF-IDF cosine >= 0.8, embedding cosine >= 0.9): no hit remains.

## Adjudication history (kept in the repository)

* **Round 1:** 436 examples written. Both annotators equal to the intended label on 426 (**97.7%**; passes 98.9% and 98.4%). All 342 route examples
  agreed; all 10 disagreements were `no_match` examples (6 sibling_removed, 3 in_domain_uncovered, 1 oos). Original texts: `adjudication_original.json`.
* **Round 2:** all 10 were rewritten (or, for one scenario, a condition made explicit); 14 examples re-annotated by two fresh passes: 11 agreed.
* **Round 3:** one of the three remaining was rewritten again and then agreed. Two examples stayed contested after two to three rewrites and were
  **dropped** (`dropped.json`, with the reason): a binary "live staff or continue with the bot" node, and a three-option node in which annotators
  read every answer without information, including "I don't know", as "declines to share". Example ids of the other examples did not change.
* Also resolved before verification: one near-duplicate message (a generic refusal that CLINC150 contains) and one condition text that two scenarios shared.
* The checker found a real bug on the way: dropping examples with `zip()` could silently truncate a scenario that did not match its specification.
  Fixed and covered by a regression test.

## Shortcut probes

No gold-position bias (positions follow the uniform shuffle), no length bias (gold is the longest candidate 38.0% vs 37.3% expected, shortest 38.0% vs 39.1%;
always choosing the longest = 0.251, chance 0.257), low word overlap between message and gold condition (0.047 vs 0.032 for the others).
Training-free difficulty (Top-1): lexical 0.401, TF-IDF 0.380, MiniLM zero-shot cosine 0.643 (`realistic-v1`: 0.343 / 0.307 / 0.734); trained TF-IDF + LR 0.439,
the 8,000-example cross-encoder 0.637.

## What this set does not show

* It is **not human-audited** (`human_audited: false`) and must not be called gold. Annotator agreement is agreement between two passes of the same model
  family; it measures consistency, not truth. Independence from the generator (Opus) is kept, independence between the two passes is weaker than with two
  different models (the instruction was to use Sonnet 5.5 only).
* **Selection toward agreeable examples.** Examples were repaired or dropped until Sonnet-family annotators agreed. Genuine human ambiguity, especially in
  `sibling_removed` and `in_domain_uncovered` cases, is therefore under-represented, and the set may be easier for models than real traffic. The two dropped
  examples are both of that kind.
* **Stereotyped style.** LLM slang is repetitive ("freaking" occurs in 14 messages); simulated slang is not real slang. Noise is character-level and mild.
* Per-slice numbers rest on 17 to 60 examples; report intervals (scenario-bootstrap), not point values.
* **Scored:** the fixed systems were evaluated on it without any change; results and their interpretation are in [decision-layer.md](decision-layer.md) (selected system: Top-1 0.693, decision accuracy 0.523).
