# data/raw

Raw, **unmodified** copies of source datasets. Nothing in this directory is edited by hand or by scripts.
Derived data belongs elsewhere (e.g. `data/processed/`).

Retrieved: 2026-11-11 via `git clone --depth 1`, `.git` directories removed.
Integrity: see `SHA256SUMS.txt` (verify with `sha256sum -c SHA256SUMS.txt` from this directory).

| Dir | Source | Commit | Repo-level license | Kept |
|---|---|---|---|---|
| `sgd/` | https://github.com/google-research-datasets/dstc8-schema-guided-dialogue | `e852981ae34990f4358979625854259302feaa78` | CC BY-SA 4.0 | `train/ dev/ test/ sgd_x/` + docs |
| `star/` | https://github.com/RasaHQ/STAR | `3058975092cdd09798e4a3a9e4ae135755824545` | MIT | `dialogues/ tasks/ apis/` |
| `abcd/` | https://github.com/asappresearch/abcd | `6b8700ce67c6b37b062dd7a60abc76d7ef832a97` | MIT | `data/` (code excluded) |
| `multiwoz/` | https://github.com/budzianowski/multiwoz | `fe0c8e65cfcd8462bd33c86e35f21addc84ca82b` | MIT | `data/MultiWOZ_2.2/` only |
| `banking77/` | https://github.com/PolyAI-LDN/task-specific-datasets | `57ec275d8078af65b7731c2a98be812d844a6d6b` | CC BY 4.0 | `banking_data/` |
| `clinc150/` | https://github.com/clinc/oos-eval | `828f8093932c8fe6ca7936c3d2e52903b1c523de` | CC BY 3.0 | `data/` |
| `hwu64/` | https://github.com/xliuhw/NLU-Evaluation-Data | `f6071b496b17d71e6eb43f543af0707f4ff30557` | CC BY 4.0 | `AnnotatedData/ Collected-Original-Data/ AnnotationGuidelines/` |

## Notes
- Licenses above are the **repository LICENSE files**. The license of the underlying data may differ
  (e.g. MultiWOZ is MIT in this repo; verify before any redistribution). SGD is **share-alike**.
- `hwu64/`: the upstream repo contains very deep `CrossValidation/` paths that fail to check out on Windows
  (path length). They were intentionally excluded; the main annotated CSVs are kept.
- `sgd/sgd_x/` contains only the paraphrased **schemas** (variants v1-v5). SGD-X dialogues are the original
  SGD dialogues paired with a variant schema (see `generate_sgdx_dialogues.py` upstream, not copied).
- Not included (not needed yet): MultiWOZ 1.0/2.0/2.1 zips, DialogStudio, CLAMBER.
- Size is ~1.07 GB, so the data is not tracked in git. Only this README and `SHA256SUMS.txt` are.

## Reproducing this directory

For each row: `git clone <source>`, `git checkout <commit>`, copy the directories listed under *Kept* into `data/raw/<dir>/`, drop the `.git`
directory. Then verify from this directory with `sha256sum -c SHA256SUMS.txt` (the pipeline also verifies it before every build unless
`--skip-verify-raw` is given).
