# Data pipeline

Two deterministic stages turn raw public datasets into model-ready records. Pure Python standard library, no installs.
The same inputs and options give byte-identical outputs on any machine (independent of `PYTHONHASHSEED`).

```bash
pathwiser benchmark build                   # stage 1: data/raw + data/curated -> data/processed/v1   (about 2 min)
pathwiser benchmark records                 # stage 2: data/processed/v1 -> data/model/v1             (about 1 min)

pathwiser benchmark build --smoke 200       # quick debug build -> data/processed/v1-smoke/
pathwiser benchmark build --config my.json  # override any value of configs/dataset.json (deep-merged)
pathwiser benchmark build --skip-verify-raw # skip the SHA-256 check of data/raw/
pathwiser benchmark records --no-node       # ablation: leave the workflow node out of the query
```

If any invariant is violated a stage **aborts and publishes nothing** (it writes to `<out>.tmp` and swaps on success).

## Stage 1: benchmark (`pathwiser/benchmark/`)

`verify raw` → `load sources` → `partitions & splits` → `dedupe` → `pools` → `examples` → `validate` → `card` → `write`

| step | module | what it guarantees |
|---|---|---|
| load | `sources/*.py` (`abcd`, `curated`, `intents`, `sgd`, `star`) | one loader per dataset; every discarded record is counted (`DropLog`), nothing vanishes silently |
| partitions / splits | `splits.py` | held-out condition families are never seen in training; a dialogue is never split across splits |
| dedupe | `splits.py` | single-turn label conflicts removed; one copy per text within a dataset and split; no cross-split leakage |
| candidates | `candidates.py` | gold + hard negatives (family / TF-IDF-nearest / random); no equivalent or identical-description candidates |
| examples | `examples.py` | node, slices, provenance, SGD-X paired held-out paraphrase examples |
| validate | `validate.py` | 50 hard checks (structure, leakage, partition integrity, pairing) |
| card | `card.py` | statistics, artefact probes and `DATASET_CARD.md` |
| write | `export.py` | atomic publication, manifest with the SHA-256 of every output (`pipeline.py` orchestrates all steps) |

Inputs: `data/raw/` (see its README; MultiWOZ is kept locally but unused) and `data/curated/` (authored condition descriptions).

Outputs, `data/processed/<version>/`: `train|val|test.jsonl`, `conditions.jsonl`, `moments.jsonl`, CSV views
(`examples_flat.csv`, `examples_sample.csv`, `candidates_long.csv`, `conditions.csv`), `manifest.json` (config hash and
SHA-256 of every output), `stats.json`, `validation_report.json`, `DATASET_CARD.md` (composition, slices, probes,
**limitations**) and `build_info.json` (time/platform; the only non-deterministic file).

## Stage 2: model-ready records (`pathwiser/benchmark/records.py`)

Turns every benchmark example into one self-contained listwise record: normalised text, a bounded conversation context,
the rendered `query`, candidate descriptions (never ids), targets and training weights. The record layout is documented in
`data/model/<version>/SCHEMA.md`. Rules that keep results honest:

- `meta` (dataset, tier, family, slices, ids, ...) is for analysis only and must never be fed to a model.
- Training weights exist only in `train` (dataset balancing `n^-0.5`, risky `sibling_removed` labels halved; mean 1).
- SGD-X paraphrase twins (`meta.is_paired`) exist only in val/test and are reported separately.

## Design decisions worth knowing

- **Unseen conditions** are defined per condition `partition` (train / val / test). A condition's partition is the first split
  where it may appear; pools grow with the split (train pool = train conditions only). The slice `cond:seen` means an
  *equivalent* of the gold condition was in train, which can be a different service or wording; `meta.gold_seen_exact`
  separates the very same condition from an equivalent-only one.
- **SGD ids are not unique across splits** (`1_00000` exists in several); all SGD ids are namespaced by split directory.
- **SGD-X**: train samples descriptions from {orig, v1, v2, v3}; val/test get an extra *paired* example per moment with the
  held-out variant (v4 / v5) so paraphrase robustness is measured on identical candidate sets.
- **no_match** kinds: `other_family`, `sibling_removed` (flagged `label_risk`), `oos` (CLINC out-of-scope).
- Candidate ids (`c0..`) are positional and carry no meaning; gold position is uniform by construction.
- The same utterance may appear in two different datasets *within one split* (separate candidate pools); it never crosses splits.
- HWU64 text is lower-cased and de-punctuated (the `general_*` rows only exist as raw `answer`, whose trailing '.'
  would otherwise reveal the label).
