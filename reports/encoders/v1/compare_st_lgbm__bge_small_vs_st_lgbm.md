# `st_lgbm__bge_small` vs `st_lgbm`

Rule fixed before the candidate was trained:

- validation_not_worse: leave-one-dataset-out macro decision accuracy >= reference - 1 standard error of the reference, AND validation Top-1 >= reference - 0.005
- external_gain: paired external-set difference to the reference: Top-1 > 0 AND decision accuracy >= 0 (point estimates)
- confirmed_if: the 95% scenario-bootstrap interval of the Top-1 difference excludes 0; otherwise the adoption is only tentative
- flag: unseen-service validation Top-1 more than 0.02 below the reference is reported as a warning
- test: not read for this comparison; read once, for the adopted model only

A candidate (same model family, different frozen encoder) replaces the reference only if it does not lose on the benchmark validation split and gains on the external set. Fixed before the candidate was trained or scored.

## Validation (benchmark)

| | Top-1 | MRR | unseen-service Top-1 | LODO macro decision acc. (s.e.) |
|---|---:|---:|---:|---:|
| `st_lgbm__bge_small` | 0.892 | 0.942 | 0.855 | 0.803 (0.018) |
| `st_lgbm` | 0.893 | 0.942 | 0.902 | 0.800 (0.017) |

## External set `realistic` (95% scenario-bootstrap intervals; threshold from validation)

| | Top-1 | Decision acc. | False routing |
|---|---|---|---|
| `st_lgbm__bge_small` | 0.792 [0.746, 0.834] | 0.639 [0.595, 0.686] | 0.134 [0.100, 0.171] |
| `st_lgbm` | 0.759 [0.704, 0.816] | 0.605 [0.548, 0.659] | 0.192 [0.156, 0.227] |

Paired difference (candidate minus reference): Top-1 0.033 [-0.025, 0.087]; decision accuracy 0.035 [-0.020, 0.089]; false routing -0.058 [-0.097, -0.017].

## Verdict

- validation not worse: **True**
- external gain: **True**
- adopt: **True** (tentative)
- unseen-service warning: True
