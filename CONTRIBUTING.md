# Contributing

Pathwiser is a research-stage project: results must be reproducible and claims must be backed by an experiment. Contributions that
follow that bar are welcome.

## Setup

```bash
git clone https://github.com/baranylcn/pathwiser
cd pathwiser
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                                # add ,train for embeddings / LightGBM / cross-encoder work
```

Python 3.10 or newer. The data pipeline and the test suite need only `numpy` and `scikit-learn`; tests that need optional packages
skip themselves when those are missing.

## Checks

```bash
ruff check pathwiser tests
ruff format --check pathwiser tests
python -m unittest discover -s tests -t .
```

CI runs the same three commands. A pull request needs all of them green.

## Project rules

* **Code lives in `pathwiser/`, never in notebooks.** Notebooks import the package and only present results.
* **Data flows one way:** `data/raw` to `data/processed` to `data/model`. Generated files are reproducible and git-ignored, except the small
  provenance files listed in `.gitignore`. Never commit third-party raw data.
* **Scoring and deciding stay separate.** A scorer returns relevance scores; `pathwiser/decision/` turns scores into `route` or `no_match`.
* **Evaluation protocol** ([docs/evaluation-protocol.md](docs/evaluation-protocol.md)): fit on `train`, tune on `val`, read `test`
  once at the end, never tune on an external set. Do not change a method in response to test numbers.
* **Honest reporting.** Report intervals and failure slices, not only aggregates. LLM-authored sets are not "gold" unless a human audit
  is recorded.
* **Determinism.** The same inputs and options give identical outputs; seeds are explicit.
* **Dependencies.** Keep the core dependency list minimal; heavy packages are optional extras.

## Commits

Conventional Commits, subject line only, without a scope: `type: summary`, lowercase, imperative, no trailing period,
about 72 characters (for example `feat: export model-ready listwise records`).
Types: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `build`, `ci`, `style`, `chore`. One concern per commit.

## Adding a scorer

1. Implement the scorer interface in `pathwiser/scorers/` (see `base.py`) and register it in `registry.py`.
2. Add tests under `tests/scorers/`.
3. Fit and store its scores with `pathwiser scorers fit`, then compare it with the existing systems using `pathwiser report ...`.

## Reporting problems

Open an issue with the command you ran, the data version and the full error. For data problems, include the example id.
