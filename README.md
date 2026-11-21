# Pathwiser

[![CI](https://github.com/baranylcn/pathwiser/actions/workflows/ci.yml/badge.svg)](https://github.com/baranylcn/pathwiser/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)

**Dynamic routing for conversational workflows, with machine learning instead of hand-written conditions.**

A workflow node has several outgoing branches, each described in plain language ("The user wants to cancel an existing order").
Given the conversation, the current message and those descriptions, Pathwiser picks the branch that fits, or abstains (`no_match`)
when none does.

```
score(conversation, node, condition description)  ->  relevance score  ->  route | no_match
```

This is **dynamic candidate selection, not fixed-label intent classification**. The candidate conditions are supplied at inference time and
the model reads their descriptions, so it should work on branches it has never seen. Whether it really does is the research question
behind this repository, and every result below is reported with that in mind.

> **Status: research stage (v0.1).** The data pipeline, scorers, decision layer and evaluation tooling are implemented and tested.
> A packaged inference API (`engine.route(...)`), a REST service and an SDK are planned but **not built yet**. See [Roadmap](#roadmap).

## Results

Systems are compared on a benchmark built from six public datasets (SGD, STAR, ABCD, Banking77, CLINC150, HWU64; 19,567 test
examples, conditions and services held out from training) and on an external stress test of 344 LLM-authored examples from business
contexts none of those datasets cover. Details, slices and intervals: [reports/](reports/) and [docs/decision-layer.md](docs/decision-layer.md).

| System | Benchmark Top-1 | Benchmark decision acc. | External Top-1 | External decision acc. |
|---|---:|---:|---:|---:|
| random | 0.242 | 0.227 | | |
| TF-IDF cosine | 0.588 | 0.559 | 0.307 | |
| MiniLM cosine (zero-shot) | 0.846 | 0.725 | 0.734 | 0.549 |
| MiniLM + LightGBM | 0.875 | 0.783 | 0.759 | 0.605 |
| cross-encoder (CPU budget, 8k examples) | 0.854 | 0.783 | 0.646 | |
| **fusion of two LightGBM scorers (MiniLM, bge-small) + threshold** | **0.888** | **0.802** | **0.792** | **0.657** |

Decision accuracy counts a `no_match` example as correct when the system abstains. Intervals are in the reports; on the external set they
are about +/-0.05, so differences between the top systems there are not conclusive.

What the results support, and what they do not:

* Frozen sentence embeddings with a small trained scoring layer beat lexical baselines by a wide margin and keep most of their accuracy
  on services and conditions never seen in training.
* The abstention rule is not the bottleneck: calibrated and learned abstention policies did not beat one global threshold. The loss on
  unfamiliar domains (Top-1 0.875 to 0.759) is a ranking problem.
* The external set is LLM-authored, blind-verified by model annotators and **not human-audited**. A second set, `realistic-v2`
  (434 examples, 36 unseen business contexts, 14 writing styles), is built and has not been scored with the fixed systems yet.
* The benchmark's `no_match` examples are largely constructed, and non-SGD condition descriptions are unreviewed drafts
  ([dataset card](data/processed/v1/DATASET_CARD.md)).

## Install

Python 3.10 or newer.

```bash
git clone https://github.com/baranylcn/pathwiser
cd pathwiser
pip install -e .                    # core: numpy, scikit-learn
pip install -e ".[train]"           # + sentence-transformers, lightgbm (embedding scorers, cross-encoder)
pip install -e ".[notebooks]"       # + pandas, matplotlib, jupyter
pip install -e ".[dev]"             # + ruff
```

## Quick start

The third-party datasets are not redistributed. Fetch them as described in [data/raw/README.md](data/raw/README.md); the pipeline verifies
them against `SHA256SUMS.txt`. Everything runs from the repository root through one command:

```bash
pathwiser benchmark build                    # data/raw + data/curated -> data/processed/v1   (validated benchmark)
pathwiser benchmark records                  # data/processed/v1 -> data/model/v1             (model-ready records)
pathwiser sets build --set realistic_v1      # authored scenarios -> data/model/realistic-v1  (external stress test)
pathwiser sets build --set realistic_v2      # authored scenarios -> data/model/realistic-v2

pathwiser baselines run --all --final        # training-free baselines under the evaluation protocol
pathwiser baselines run --trained            # trained baselines on validation

pathwiser scorers fit --baseline st_lgbm --splits val test --final     # score once, store the scores
pathwiser report dev                         # choose a decision system (validation only)
pathwiser report final                       # evaluate the selected system (reads test once)
```

`pathwiser --help` lists every command. The evaluation rules behind these commands (test is read once, external sets are never used to
tune) are in [docs/evaluation-protocol.md](docs/evaluation-protocol.md).

## Repository layout

```
pathwiser/
  benchmark/    raw datasets -> validated routing benchmark -> model-ready records
  storage/      dataset loading, content hashing, persistent score store
  scorers/      lexical, embedding, LightGBM, logistic-regression and cross-encoder scorers
  decision/     threshold rule, score fusion, learned abstention, system selection
  evaluation/   metrics and cluster-bootstrap statistics shared by every model
  reports/      baseline, external-set, final and encoder-comparison reports
  synth/        synthetic scenario data: taxonomy, personas, noise, specifications, checks, blind verification
  cli.py        the `pathwiser` command
configs/        dataset configuration (versioned input of the pipeline)
data/           raw -> processed -> model, authored scenarios and condition descriptions (see data/README.md)
docs/           pipeline, evaluation protocol, decision layer, evaluation sets, synthetic data
notebooks/      data analysis and baseline analysis (import the package, contain no logic)
reports/        results of tracked runs (small JSON and summary.md)
tests/          unit and end-to-end tests
```

## Documentation

| | |
|---|---|
| [docs/dataset-pipeline.md](docs/dataset-pipeline.md) | the two deterministic data stages and their guarantees |
| [docs/evaluation-protocol.md](docs/evaluation-protocol.md) | splits, metrics, slices, what may and may not be tuned |
| [docs/decision-layer.md](docs/decision-layer.md) | route / no_match policies, selection, results |
| [docs/realistic-eval-set.md](docs/realistic-eval-set.md) | the external stress test `realistic-v1` and its limits |
| [docs/synthetic-data.md](docs/synthetic-data.md) | synthetic scenario design and the evaluation set `realistic-v2` |
| [data/README.md](data/README.md) | what lives where under `data/` |

## Roadmap

1. Score `realistic-v2` with the fixed systems.
2. Generate training scenarios with the synthetic-data machinery and test whether condition diversity improves unseen-condition accuracy.
3. Human audit of a sample of the LLM-authored sets.
4. Packaged inference engine with a typed Python API, then a REST service.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Code: [MIT](LICENSE). The datasets keep the licenses of their sources, listed in [data/raw/README.md](data/raw/README.md); note that
SGD is CC BY-SA 4.0 (share-alike).
