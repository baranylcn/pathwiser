# Decision layer: results

Chosen on validation only (leave-one-dataset-out): **`st_lgbm+st_lgbm__bge_small|fused|tau`**. Rule: highest macro decision accuracy under leave-one-dataset-out; among systems within one standard error of the best, drop those whose out-of-fold Top-1 on svc:unseen is more than 0.02 below the best candidate's; of the rest, the least complex. 21 candidates were compared; the maximum over many candidates is optimistic; the selection itself is computed from validation only, but the generalisation guard and the candidate list were chosen after the external set had been looked at (docs/decision-layer.md); the test split was not used.

## Test (benchmark)

| System | Top-1 | Decision acc. | no_match F1 | False routing | Abstain rate | ECE |
|---|---:|---:|---:|---:|---:|---:|
| `st_lgbm+st_lgbm__bge_small\|fused\|tau` (chosen) | 0.888 | 0.802 | 0.683 | 0.149 | 0.193 | – |
| `st_message\|tau` | 0.846 | 0.725 | 0.613 | 0.173 | 0.248 | – |
| `st_lgbm\|tau` | 0.875 | 0.783 | 0.660 | 0.159 | 0.197 | – |
| `st_lgbm__bge_small\|full` | 0.876 | 0.784 | 0.667 | 0.149 | 0.215 | 0.026 |

Generalisation slices on test (Top-1 / decision accuracy; `cond:unseen` = no equivalent of the gold condition in train, `svc:unseen` = service not in train):

| System | cond:seen | cond:unseen | seen_exact:exact | seen_exact:equivalent_only | svc:seen | svc:unseen |
|---|---:|---:|---:|---:|---:|---:|
| `st_lgbm+st_lgbm__bge_small\|fused\|tau` | 0.900 / 0.873 | 0.868 / 0.817 | 0.926 / 0.900 | 0.859 / 0.829 | 0.938 / 0.916 | 0.847 / 0.820 |
| `st_message\|tau` | 0.847 / 0.780 | 0.845 / 0.696 | 0.871 / 0.795 | 0.808 / 0.757 | 0.879 / 0.859 | 0.783 / 0.737 |
| `st_lgbm\|tau` | 0.887 / 0.852 | 0.854 / 0.800 | 0.909 / 0.876 | 0.852 / 0.814 | 0.922 / 0.895 | 0.828 / 0.793 |
| `st_lgbm__bge_small\|full` | 0.888 / 0.846 | 0.857 / 0.785 | 0.919 / 0.885 | 0.837 / 0.782 | 0.931 / 0.901 | 0.830 / 0.782 |

Paired differences on test to `st_lgbm|tau` (95% bootstrap over 19567 dialogues/examples, 400 resamples):

| System | Δ Top-1 | Δ Decision acc. | Δ False routing |
|---|---|---|---|
| `st_lgbm+st_lgbm__bge_small\|fused\|tau` | 0.013 [0.010, 0.016] | 0.018 [0.015, 0.021] | -0.010 [-0.013, -0.007] |
| `st_message\|tau` | -0.029 [-0.031, -0.026] | -0.059 [-0.064, -0.055] | 0.013 [0.010, 0.017] |
| `st_lgbm__bge_small\|full` | 0.001 [-0.003, 0.005] | 0.000 [-0.004, 0.004] | -0.011 [-0.014, -0.007] |

## External set `realistic_v2` (95% scenario-bootstrap intervals)

| System | Top-1 | Decision acc. | False routing | Abstain rate | no_match recall |
|---|---|---|---|---|---|
| `st_lgbm+st_lgbm__bge_small\|fused\|tau` (chosen) | 0.693 [0.636, 0.741] | 0.523 [0.472, 0.572] | 0.235 [0.195, 0.278] | 0.352 [0.299, 0.408] | 0.522 [0.420, 0.617] |
| `st_message\|tau` | 0.643 [0.579, 0.697] | 0.408 [0.355, 0.454] | 0.175 [0.132, 0.227] | 0.565 [0.500, 0.632] | 0.696 [0.602, 0.784] |
| `st_lgbm\|tau` | 0.658 [0.602, 0.707] | 0.525 [0.471, 0.577] | 0.244 [0.201, 0.292] | 0.350 [0.292, 0.409] | 0.565 [0.463, 0.667] |
| `st_lgbm__bge_small\|full` | 0.649 [0.598, 0.700] | 0.484 [0.439, 0.530] | 0.242 [0.199, 0.289] | 0.378 [0.324, 0.433] | 0.489 [0.379, 0.594] |

Paired difference to the reference system `st_lgbm|tau`:

| System | Δ Top-1 | Δ Decision acc. | Δ False routing |
|---|---|---|---|
| `st_lgbm+st_lgbm__bge_small\|fused\|tau` | 0.035 [-0.006, 0.072] | -0.002 [-0.042, 0.035] | -0.009 [-0.049, 0.032] |
| `st_message\|tau` | -0.015 [-0.040, 0.009] | -0.117 [-0.162, -0.074] | -0.069 [-0.101, -0.034] |
| `st_lgbm__bge_small\|full` | -0.009 [-0.058, 0.041] | -0.042 [-0.091, 0.007] | -0.002 [-0.056, 0.053] |
