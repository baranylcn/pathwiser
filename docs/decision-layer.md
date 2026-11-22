# Decision layer

How per-candidate scores become `route` or `no_match`, which designs were compared, and what the evidence supports.

Code: [`pathwiser/decision/`](../pathwiser/decision) (`threshold`, `fusion`, `abstention`, `selection`, `system`).
Every number below is computed from stored scores (`data/cache/scores`, see [evaluation-protocol.md](evaluation-protocol.md))
by `pathwiser report ...`. The reports are in `reports/decision/v1/`, `reports/realistic/v1/` and `reports/encoders/v1/`.

## Design

* **Scoring and deciding are separate.** A scorer returns one relevance score per candidate. The decision layer sees only scores.
* **Threshold rule** (`decision/threshold.py`): route to the best candidate unless its score is below `tau`. `tau` maximises decision
  accuracy on validation; ties prefer fewer abstentions.
* **Fusion** (`decision/fusion.py`): a logistic regression over standardised scores of several scorers, fitted on out-of-fold validation scores.
* **Learned abstention** (`decision/abstention.py`): a logistic regression over listwise score features (best score, margin, spread,
  candidate count, ...), either scale-free (`relative`) or with the absolute scores added (`full`).
* **Selection** (`decision/selection.py`): leave-one-dataset-out cross-validation on the validation split. Fit on five datasets, decide on
  the sixth; six datasets stand in for a domain shift. Criterion: macro decision accuracy.

Selection rule (`select_system`): systems within one standard error of the best stay. Of those, systems whose out-of-fold Top-1 on
the unseen-service slice is more than 0.02 below the best candidate's are removed (`GUARD_TOLERANCE`), unless that removes all of them.
The least complex remaining system wins.

The guard was added after the external set had been looked at: without it a margin of 0.0003 inside the one-standard-error band picked a model
that is significantly worse on unseen services on validation. It was derived from validation slices only, but read it with that in mind.
The test split was read once for the final systems.

## What was compared (validation, leave-one-dataset-out)

| System | macro decision acc. | s.e. |
|---|---:|---:|
| `st_message` + threshold (zero-shot MiniLM) | 0.735 | 0.026 |
| `st_lr` + threshold | 0.761 | 0.023 |
| `st_lgbm` + threshold | 0.800 | 0.017 |
| `st_lgbm` + `relative` features | 0.786 | 0.014 |
| `st_lgbm` + `full` features | 0.802 | 0.017 |
| `st_lgbm` (bge-small) + threshold | 0.803 | 0.019 |
| fusion of `st_lgbm` (MiniLM) and `st_lgbm` (bge-small) + threshold | **0.820** | 0.015 |
| same fusion + `full` abstention | 0.827 | 0.017 |

The learned abstention models are indistinguishable from the plain threshold, and scale-free features alone are worse than the threshold.
The gain comes from better scores (fusion of two encoders), not from a cleverer policy. 21 candidates were compared; the maximum over
many candidates is optimistic.

The cross-encoder was not part of the selection (its validation scores were not stored).

## Selected system

**Fusion of `st_lgbm` (MiniLM-L6) and `st_lgbm` (bge-small) + global threshold.** Results (details and slices:
[`reports/decision/v1/summary.md`](../reports/decision/v1/summary.md)):

| Test (19,567 examples) | Top-1 | Decision acc. | no_match F1 | False routing |
|---|---:|---:|---:|---:|
| `st_message` + threshold (zero-shot) | 0.846 | 0.725 | 0.613 | 0.173 |
| `st_lgbm` + threshold | 0.875 | 0.783 | 0.660 | 0.159 |
| **fusion + threshold** | **0.888** | **0.802** | **0.683** | **0.149** |
| paired difference to `st_lgbm` | +0.013 [0.010, 0.016] | +0.018 [0.015, 0.021] | | -0.010 [-0.013, -0.007] |

The fusion is better than `st_lgbm` on every generalisation slice of test (Top-1): unseen services 0.847 vs 0.828, unseen conditions
0.868 vs 0.854, equivalent-only 0.859 vs 0.852.

| External set `realistic-v1` (344 examples) | Top-1 | Decision acc. | False routing |
|---|---:|---:|---:|
| `st_message` + threshold | 0.734 | 0.549 | 0.142 |
| `st_lgbm` + threshold | 0.759 | 0.605 | 0.192 |
| **fusion + threshold** | **0.792** | **0.657** | 0.172 |
| paired difference to `st_lgbm` | +0.033 [-0.011, 0.078] | +0.052 [0.009, 0.095] | -0.020 [-0.050, 0.012] |

Intervals are 95% scenario-cluster bootstraps. They are wide: the set is small, LLM-authored and not human-audited.

## Is the threshold the bottleneck?

No. A threshold tuned on the external set itself (an oracle that must never be used for a real result) bounds the possible gain:

| `realistic-v1`, decision accuracy | validation threshold | oracle threshold | never abstain |
|---|---:|---:|---:|
| `st_lgbm` | 0.605 | 0.654 | 0.605 |
| `st_message` (zero-shot MiniLM) | 0.549 | 0.625 | 0.584 |
| zero-shot `bge-small` | 0.538 | 0.666 | 0.631 |

For the trained model even a perfect threshold gains 4.9 points. The loss on unfamiliar domains (Top-1 0.875 to 0.759) is mainly a
ranking problem. For zero-shot encoders the benchmark threshold does not transfer at all (cosine scores shift with the domain), which is
a reason to keep a trained scoring layer on top of frozen embeddings.

## Encoder choice

Zero-shot scan (`reports/encoders/v1/zero_shot_scan.md`):

| Encoder | val Top-1 | `realistic-v1` Top-1 | difference to MiniLM-L6 |
|---|---:|---:|---|
| MiniLM-L6 | 0.854 | 0.734 | reference |
| bge-small | 0.860 | 0.792 | +0.058 [0.011, 0.103] |
| gte-small | 0.864 | 0.766 | +0.033 [-0.018, 0.081] |
| MiniLM-L12 | 0.855 | 0.723 | -0.011 [-0.068, 0.043] |
| e5-small | 0.834 | 0.730 | -0.004 [-0.067, 0.058] |
| multi-qa-MiniLM-L6 | 0.849 | 0.679 | -0.055 [-0.104, -0.007] |

Five comparisons and one marginally significant gain, so this is a lead and not a decision. It was tested for the trained model
(`reports/encoders/v1/compare_st_lgbm__bge_small_vs_st_lgbm.md`; the adoption rule is `ADOPTION_RULE` in `reports/encoders.py`):

| `st_lgbm` on | validation Top-1 | validation unseen-service Top-1 | `realistic-v1` Top-1 | `realistic-v1` decision acc. |
|---|---:|---:|---:|---:|
| MiniLM-L6 | 0.893 | 0.902 | 0.759 | 0.605 |
| bge-small | 0.892 | 0.855 | 0.792 | 0.639 |
| difference | -0.001 | -0.047 [-0.060, -0.033] | +0.033 [-0.025, 0.087] | +0.035 [-0.020, 0.089] |

The evidence for replacing MiniLM by bge-small is mixed (it wins on seen conditions and `banking77`, loses on SGD unseen services on
validation). The validation deficit on unseen services did **not** reproduce on test (bge 0.830 vs MiniLM 0.828); it rests on few
services, so examples are strongly correlated and a bootstrap over dialogues understates the uncertainty. Fusing both models is better
than either alone, on validation, on test and on the external set.

## Second external set: `realistic-v2`

The fixed systems were scored on `realistic-v2` (434 examples, 80 scenarios, 36 business contexts in six industry groups absent from every
training split, 14 writing styles) without changing anything: same fitted models, same validation thresholds
([`reports/decision/v1/realistic_v2/summary.md`](../reports/decision/v1/realistic_v2/summary.md),
[`reports/realistic/v1/realistic_v2_baselines.md`](../reports/realistic/v1/realistic_v2_baselines.md)).

| `realistic-v2` | Top-1 | Decision acc. | False routing | no_match recall |
|---|---:|---:|---:|---:|
| `st_message` + threshold (zero-shot) | 0.643 [0.579, 0.697] | 0.408 [0.355, 0.454] | 0.175 | 0.696 |
| `st_lgbm` + threshold | 0.658 [0.602, 0.707] | 0.525 [0.471, 0.577] | 0.244 | 0.565 |
| cross-encoder (CPU budget) | 0.637 [0.588, 0.683] | 0.495 [0.449, 0.545] | 0.214 | 0.598 |
| **fusion + threshold (selected)** | **0.693** [0.636, 0.741] | 0.523 [0.472, 0.572] | 0.235 | 0.522 |
| paired difference, fusion minus `st_lgbm` | +0.035 [-0.006, 0.072] | -0.002 [-0.042, 0.035] | -0.009 [-0.049, 0.032] | |

What this shows:

* **The absolute level is low.** On business contexts the models have never seen, the selected system picks the right condition for 69% of
  the route examples and decides correctly on 52% of all examples; 24% of all examples are routed wrongly. The benchmark numbers
  (0.888 / 0.802) overstate what to expect on new domains by a wide margin, and `realistic-v1` (0.792 / 0.657) was the easier of the two
  external sets.
* **The fusion's advantage did not replicate.** Top-1 moves in the same direction as before (+0.033 on `realistic-v1`, +0.035 here) but is
  not significant on either set alone. The decision-accuracy gain over `st_lgbm` is +0.018 on the benchmark test and +0.052 on
  `realistic-v1` (interval above zero) but -0.002 here (interval around zero). The evidence for the fusion is therefore: clearly better on
  the benchmark, possibly better in ranking elsewhere, and no demonstrated gain in decisions on new domains. The selection rule was not
  changed because of this.
* **Training does help the decision, compared with the zero-shot encoder:** `st_lgbm` beats `st_message` by +0.117 [0.074, 0.162] in
  decision accuracy on this set (+0.055 on `realistic-v1`), mostly because the zero-shot threshold abstains far too often (56% of examples).
* **Writing style is not the main loss.** An exploratory slice (not part of the report, no intervals): `st_lgbm` Top-1 on route examples is
  0.610 for the held-out styles (n=118) and 0.683 for the seen styles (n=224). Even `neutral` examples reach only 0.725. Rambling,
  long-winded messages are the weakest persona (0.435, n=23). The gap is mainly about unfamiliar conditions and domains, not about slang
  or typos.

## Cross-encoder

Fine-tuning helps a little on `realistic-v1` (zero-shot ms-marco cross-encoder Top-1 0.617, fine-tuned on 8,000 examples 0.646), but its
starting point is weak for this task (web-search relevance, short condition descriptions) and it stays well below the bi-encoder based
models (0.759). It is budget-limited (CPU training) and was not scaled.

## Conclusions

1. The decision layer is not the bottleneck; the plain threshold stays. Calibrated probabilities are not needed for routing quality
   (`AbstentionModel` outputs probabilities if they are needed later).
2. The loss on unfamiliar domains is a ranking problem. Combining two different frozen encoders under the trained scoring layer helps;
   replacing MiniLM by bge-small alone does not. More diverse training data is the other lever (see [synthetic-data.md](synthetic-data.md)).
3. The external evidence rests on two LLM-authored, not human-audited sets (344 and 434 examples). Every system loses a lot on new
   business contexts (decision accuracy about 0.52 on `realistic-v2`), and the fusion's decision advantage is not reproduced there.

## Reproduce

```bash
pathwiser scorers fit --baseline st_lgbm --splits val test realistic --final --extra realistic=data/model/realistic-v1
pathwiser scorers fit --baseline st_lgbm --tag bge_small --st-model BAAI/bge-small-en-v1.5 \
    --param "query_prefix=Represent this sentence for searching relevant passages: " \
    --splits val test realistic --final --extra realistic=data/model/realistic-v1
pathwiser report dev                                        # choose a decision system (validation only)
pathwiser report final --extra realistic=data/model/realistic-v1
pathwiser report external --extra realistic=data/model/realistic-v1
pathwiser report encoders --extra realistic=data/model/realistic-v1
```

Scoring the fixed systems on `realistic-v2` (refitting `st_lr`/`st_lgbm` needs about 2 GB of free RAM; the same fits as above with
`--splits realistic_v2 --extra realistic_v2=data/model/realistic-v2`, and `--load models/cross_encoder` for the cross-encoder):

```bash
pathwiser report external --extra realistic_v2=data/model/realistic-v2
pathwiser report final --extra realistic_v2=data/model/realistic-v2 --out-dir reports/decision/v1/realistic_v2
```

Stored scores are tied to the exact example list. After the set changed (two examples were dropped), scores of the earlier version were
refused and had to be recomputed with `--force`.
