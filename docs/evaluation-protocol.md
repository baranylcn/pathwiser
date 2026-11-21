# Evaluation protocol

The research question is generalisation to conditions the model has never seen, so an aggregate score is not enough.
These rules apply to every model, baseline or not (`pathwiser/evaluation/`, `pathwiser/reports/`).

1. **Splits.** `train` fits, `val` tunes (thresholds, hyper-parameters, model choice), `test` is read **once**, at the end.
   Baseline runs touch test only with `--final`.
2. **Decision policy.** A model produces one relevance score per candidate; the rule in `pathwiser/decision/threshold.py`
   routes to the best candidate or abstains (`no_match`) when the best score is below a threshold τ.
   τ is tuned on validation main examples (objective: decision accuracy) and applied unchanged to test.
   Scores are not probabilities and are never reported as such.
3. **Main vs. paraphrase twins.** SGD-X twins are a second measurement of the same moment. Totals use main examples only;
   twins get their own table (Top-1 original vs. paraphrase, same pick, same abstain decision).
4. **Slices, not just totals.** Every report includes dataset, `cond:seen` / `seen_exact:*` / `cond:unseen`, `svc:*`,
   context kind, candidate-set size, `no_match` kind and label risk.
5. **Reference points.** `random` shows chance (Top-1 ≈ 0.25, decision accuracy ≈ the no_match share).
   Lexical baselines show how much of the data is solvable by word overlap alone; read neural scores against them.
6. **Metrics.** Top-1 / Top-3 / MRR over examples with a gold condition (Top-3 is trivial for k ≤ 3);
   decision accuracy, `no_match` precision / recall / F1 and the false routing rate over all examples; latency per example
   and parameter count where applicable. Calibration error (ECE) and the Brier score are reported only for systems whose outputs are calibrated probabilities.
7. **Reproducibility.** Every result file records the data version and SHA-256 of train/val/test, the preprocessing
   options, library versions, parameters, threshold and timings.
8. **Trained models.** They fit on `train` only (never on val/test, never on twins). Hyper-parameters are fixed before looking at
   validation results unless a run is explicitly a tuning run; the training budget (examples, epochs, wall-clock) and the hardware
   are recorded and reported next to the score, because a budget-limited model is a lower bound, not a verdict on the method.
   Pointwise models (everything in `pathwiser/scorers/`) score each candidate independently from relations between texts, never
   from condition identity; listwise information enters only through relative features or the listwise training loss.
   Training-time devices (e.g. the cross-encoder's learned "none" logit) are not used at inference: the shared threshold rule decides.
9. **Development vs. final runs.** `--val-limit N` runs are development aids (output in `reports/**/dev/`, git-ignored).
   A `--final` run reads the test split once; after it, a method is not changed in response to test numbers.
10. **Stored scores.** Models are scored once; thresholds, policies and reports are built from the stored scores
    (`pathwiser scorers fit`, `pathwiser/storage/scores.py`). A score file records example ids, candidate counts and a fingerprint
    of the texts and refuses to load for records that differ. `test` is only scored with `--final`.
11. **Decision-layer selection.** Policies are fitted and selected on validation main examples only, by leave-one-dataset-out
    cross-validation with a preset criterion, a generalisation guard on the unseen-service slice and a parsimony tie-break
    (`pathwiser/decision/selection.py`, [decision-layer.md](decision-layer.md), which also records the one amendment of the rule and why).
    The selection is written to `reports/decision/<version>/selection.json` before test or an external set is evaluated, and
    `pathwiser report final` only reads it.
12. **External sets** (e.g. `realistic-v1`) are never used to fit, tune or select; they are scored and reported with scenario-bootstrap
    confidence intervals and paired differences. Oracle diagnostics on an external set are labelled as such and never selected from.

Commands:

```bash
pathwiser baselines run --baseline tfidf        # develop on validation only
pathwiser baselines run --all --final           # final protocol, writes reports/baselines/<version>/
pathwiser baselines run --trained               # trained baselines on validation
pathwiser baselines run --baseline cross_encoder --final --param train_examples=8000
pathwiser baselines run --summarize             # rebuild summary.md from the saved JSON files

# score once, analyse offline (items 10-12)
pathwiser scorers fit --baseline st_lgbm --splits val test realistic --final --extra realistic=data/model/realistic-v1
pathwiser report external --extra realistic=data/model/realistic-v1   # baselines on the external set
pathwiser report dev                                                  # choose a decision system (validation only)
pathwiser report final --extra realistic=data/model/realistic-v1      # evaluate the pre-selected system
```
