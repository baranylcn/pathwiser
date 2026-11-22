# Baselines on `realistic_v2` (434 examples, 80 scenarios)

tau tuned on benchmark validation main examples, applied unchanged; nothing is fitted on this set. Intervals: 95% bootstrap over scenarios (2000 resamples). The set is LLM-authored and not human-audited (see `docs/realistic-eval-set.md`); read it as a stress test, not as a measure of real traffic.

## Transfer: benchmark test vs. this set

| Baseline | Top-1 benchmark | Top-1 here | Decision acc. benchmark | Decision acc. here | False routing benchmark | False routing here |
|---|---:|---:|---:|---:|---:|---:|
| `lexical` | 0.616 | 0.401 | 0.579 | 0.297 | 0.228 | 0.246 |
| `tfidf` | 0.588 | 0.380 | 0.559 | 0.302 | 0.210 | 0.171 |
| `st_message` | 0.846 | 0.643 | 0.725 | 0.408 | 0.173 | 0.175 |
| `tfidf_lr` | 0.619 | 0.439 | 0.584 | 0.343 | 0.212 | 0.330 |
| `st_lr` | 0.866 | 0.655 | 0.758 | 0.505 | 0.182 | 0.304 |
| `st_lgbm` | 0.875 | 0.658 | 0.783 | 0.525 | 0.159 | 0.244 |
| `cross_encoder` | 0.854 | 0.637 | 0.783 | 0.495 | 0.155 | 0.214 |

## Metrics with intervals

| Baseline | Top-1 | Decision acc. | False routing | Abstain rate | no_match recall |
|---|---|---|---|---|---|
| `lexical` | 0.401 [0.345, 0.456] | 0.297 [0.256, 0.340] | 0.246 [0.203, 0.292] | 0.567 [0.507, 0.622] | 0.522 [0.419, 0.622] |
| `tfidf` | 0.380 [0.324, 0.437] | 0.302 [0.264, 0.343] | 0.171 [0.136, 0.207] | 0.670 [0.613, 0.723] | 0.674 [0.582, 0.768] |
| `st_message` | 0.643 [0.579, 0.697] | 0.408 [0.355, 0.454] | 0.175 [0.132, 0.227] | 0.565 [0.500, 0.632] | 0.696 [0.602, 0.784] |
| `tfidf_lr` | 0.439 [0.385, 0.491] | 0.343 [0.297, 0.388] | 0.330 [0.284, 0.372] | 0.410 [0.350, 0.470] | 0.391 [0.290, 0.489] |
| `st_lr` | 0.655 [0.603, 0.703] | 0.505 [0.457, 0.546] | 0.304 [0.259, 0.354] | 0.288 [0.237, 0.340] | 0.457 [0.358, 0.553] |
| `st_lgbm` | 0.658 [0.602, 0.707] | 0.525 [0.471, 0.577] | 0.244 [0.201, 0.292] | 0.350 [0.292, 0.409] | 0.565 [0.463, 0.667] |
| `cross_encoder` | 0.637 [0.588, 0.683] | 0.495 [0.449, 0.545] | 0.214 [0.181, 0.251] | 0.417 [0.368, 0.467] | 0.598 [0.505, 0.695] |

## Paired difference to `st_message` (95% interval; an interval that excludes 0 is a real difference on this set)

| Baseline | Δ Top-1 | Δ Decision acc. | Δ False routing |
|---|---|---|---|
| `lexical` | -0.243 [-0.319, -0.158] | -0.111 [-0.163, -0.055] | 0.071 [0.021, 0.120] |
| `tfidf` | -0.263 [-0.336, -0.182] | -0.106 [-0.154, -0.053] | -0.005 [-0.055, 0.042] |
| `tfidf_lr` | -0.205 [-0.280, -0.127] | -0.065 [-0.124, 0.000] | 0.154 [0.082, 0.216] |
| `st_lr` | 0.012 [-0.018, 0.043] | 0.097 [0.058, 0.135] | 0.129 [0.088, 0.170] |
| `st_lgbm` | 0.015 [-0.009, 0.040] | 0.117 [0.074, 0.162] | 0.069 [0.034, 0.101] |
| `cross_encoder` | -0.006 [-0.063, 0.053] | 0.088 [0.037, 0.138] | 0.039 [-0.016, 0.091] |

## Top-1 by slice

| Baseline | ctx:none | ctx:first_turn | ctx:later_turn | k:2-3 | k:4-6 | k:7+ | nm_kind:sibling_removed | nm_kind:in_domain_uncovered | nm_kind:oos | risk:sibling_partial_validity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lexical` | 0.455 | 0.379 | 0.370 | 0.588 | 0.328 | 0.229 | – | – | – | – |
| `tfidf` | 0.420 | 0.349 | 0.370 | 0.570 | 0.306 | 0.208 | – | – | – | – |
| `st_message` | 0.696 | 0.621 | 0.614 | 0.614 | 0.689 | 0.542 | – | – | – | – |
| `tfidf_lr` | 0.500 | 0.456 | 0.370 | 0.570 | 0.411 | 0.229 | – | – | – | – |
| `st_lr` | 0.670 | 0.641 | 0.653 | 0.632 | 0.678 | 0.625 | – | – | – | – |
| `st_lgbm` | 0.696 | 0.660 | 0.622 | 0.614 | 0.700 | 0.604 | – | – | – | – |
| `cross_encoder` | 0.714 | 0.670 | 0.543 | 0.658 | 0.639 | 0.583 | – | – | – | – |

## Decision accuracy by slice

| Baseline | ctx:none | ctx:first_turn | ctx:later_turn | k:2-3 | k:4-6 | k:7+ | nm_kind:sibling_removed | nm_kind:in_domain_uncovered | nm_kind:oos | risk:sibling_partial_validity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lexical` | 0.396 | 0.287 | 0.220 | 0.387 | 0.251 | 0.233 | 0.410 | 0.500 | 0.824 | 0.410 |
| `tfidf` | 0.389 | 0.271 | 0.250 | 0.394 | 0.260 | 0.217 | 0.538 | 0.722 | 0.882 | 0.538 |
| `st_message` | 0.528 | 0.328 | 0.363 | 0.368 | 0.429 | 0.433 | 0.564 | 0.694 | 1.000 | 0.564 |
| `tfidf_lr` | 0.410 | 0.344 | 0.286 | 0.439 | 0.297 | 0.267 | 0.282 | 0.333 | 0.765 | 0.282 |
| `st_lr` | 0.569 | 0.459 | 0.482 | 0.477 | 0.530 | 0.483 | 0.333 | 0.444 | 0.765 | 0.333 |
| `st_lgbm` | 0.590 | 0.500 | 0.488 | 0.484 | 0.571 | 0.467 | 0.436 | 0.528 | 0.941 | 0.436 |
| `cross_encoder` | 0.562 | 0.484 | 0.446 | 0.484 | 0.502 | 0.500 | 0.513 | 0.611 | 0.765 | 0.513 |

## Thresholds

| Baseline | τ | source |
|---|---:|---|
| `lexical` | 0.035 | tuned on the stored validation scores |
| `tfidf` | 0.025 | tuned on the stored validation scores |
| `st_message` | 0.278 | tuned on the stored validation scores |
| `tfidf_lr` | -0.768 | tuned on the stored validation scores |
| `st_lr` | -0.843 | tuned on the stored validation scores |
| `st_lgbm` | -0.589 | tuned on the stored validation scores |
| `cross_encoder` | 0.200 | taken from the baseline result file (same weights, same validation tuning) |
