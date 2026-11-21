# Realistic evaluation set (`realistic-v1`)

An **external stress test** for the routing models: 344 decision examples in workflow settings that none of the six
benchmark datasets cover. It is never used for training, tuning or threshold selection.

| | |
|---|---|
| Examples | 344 (274 route, 70 `no_match`) from 65 scenarios (47 domain labels: HR, insurance, IT, clinic, SaaS billing, utilities, DMV, ...) |
| `no_match` kinds | `sibling_removed` 31 (the true branch is removed, a close sibling stays), `in_domain_uncovered` 23, `oos` 16 |
| Candidate set size | 3 to 10: k 2-3 = 43, k 4-6 = 252, k 7+ = 49 |
| Context | none 210, first user turn 68, later turn 66 (includes short context-dependent replies such as "price first please") |
| Language | English |
| Source | `data/curated/realistic_v1/scenarios_*.json` (authored) -> `pathwiser sets build --set realistic_v1` -> `data/model/realistic-v1/test.jsonl` |

## How it was made

1. **Authored by an LLM (Claude Sonnet 5.5), label by construction**: a scenario fixes the node and its candidate conditions,
   then each message is written for a chosen condition. `no_match` examples are written for a condition that is removed from the
   candidates (`sibling_removed`), for something the workflow does not handle (`in_domain_uncovered`) or for something unrelated (`oos`).
2. **Blind verification**: every example was labelled by two independent annotators who saw only the node, context,
   message and the lettered candidates (anonymous ids, shuffled item order, reversed candidate order in pass B, no labels):
   pass A = default model, pass B = Claude Opus. Records are in `data/curated/realistic_v1/blind_verification.json`.
   First round: 340 of 344 examples had both passes equal to the intended label (98.8%). The four others were genuinely ambiguous
   (a removed branch whose sibling was a valid reading, or an off-topic question that fit "customer asks a question"); they were rewritten
   and re-labelled by fresh blind passes, after which all 344 agree. The final 100% is therefore post-hoc for those four items;
   the first-round figure is the honest estimate.
3. **Leakage check**: no exact message or condition text is shared with the benchmark (`manifest.json`). Near-duplicate leakage
   has NOT been checked.

## What this set can and cannot tell you

- It is **not human-audited** and must not be called "gold". `manifest.json` carries `human_audited: false`.
- Agreement between model annotators measures that the labels are *consistent and unambiguous to LLMs*; it does not measure that real
  users would write or label these the same way. Author and annotators share model-family biases.
- Messages were written by an LLM: style is cleaner than real chat logs, and models may profit from stylistic similarity with LLM-drafted
  condition descriptions (the same caveat applies to the benchmark's non-SGD descriptions).
- Indirect phrasing, negation and context-dependent short replies were added on purpose, so the difficulty mix is designed, not a sample of real traffic.
- 23 examples carry an annotator note that a second branch is also arguable (`meta.verification.alt_flagged`); most are the designed
  `sibling_removed` cases. They are kept, not removed.
- Small: slice-level numbers (e.g. 31 `sibling_removed`) have wide confidence intervals; report them with that caveat.

Every record has `meta.verification` (`status`, number of passes, agreement with the intended label, alternative flagged).
A change to an example's text after verification makes its status `changed_after_verification` until it is re-annotated.
