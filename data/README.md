# data/

Everything the project reads or generates. Each directory has one job and a one-way flow:

```
raw/  ──►  processed/<version>/  ──►  model/<version>/
(3rd party)   (benchmark)             (what a model consumes)
curated/ ──►  (authored condition descriptions feed the benchmark)
```

| Directory | Contents | Written by | In git |
|---|---|---|---|
| `raw/` | Public datasets exactly as downloaded (SGD, SGD-X, STAR, ABCD, Banking77, CLINC150, HWU64; MultiWOZ kept but unused). Never edited. | you (see `raw/README.md`) | `README.md`, `SHA256SUMS.txt` only |
| `curated/` | Condition descriptions authored for this project (`reviewed: false` = unreviewed draft), `realistic_v1/` and `realistic_v2/`: authored scenarios, specifications and blind-annotation records of the two LLM-authored evaluation sets (see [realistic-eval-set](../docs/realistic-eval-set.md), [synthetic-data](../docs/synthetic-data.md)), and `synth/taxonomy_v1.json`: business contexts and their split by group. | by hand / LLM-authored | yes |
| `processed/<version>/` | The validated routing benchmark: examples with provenance, conditions, moments, CSV views for inspection. | `pathwiser benchmark build` | provenance files only (manifest, dataset card, stats, validation report) |
| `cache/` | Frozen sentence-embedding matrices (`emb_*.npy`) and stored per-candidate scores (`scores/`). Safe to delete; reproducible. | `pathwiser scorers fit` | no |
| `model/<version>/` | Model-ready listwise records: query text, candidate descriptions, targets, weights. | `pathwiser benchmark records` (benchmark), `pathwiser sets build` (`realistic-v1`, `realistic-v2`) | `manifest.json`, `SCHEMA.md` only |

Large files are git-ignored because they are byte-for-byte reproducible from `raw/` + `curated/` + `configs/`
(every output records the SHA-256 of its inputs in `manifest.json`).

Treat a version as immutable once results refer to it: a change to the data or to a preprocessing option gets a new version name in
`configs/dataset.json`, so results in `reports/` always point at exactly one dataset.
