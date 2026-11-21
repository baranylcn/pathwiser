# Baselines on `realistic` (344 examples, 65 scenarios)

tau tuned on benchmark validation main examples, applied unchanged; nothing is fitted on this set. Intervals: 95% bootstrap over scenarios (2000 resamples). The set is LLM-authored and not human-audited (see `docs/realistic-eval-set.md`); read it as a stress test, not as a measure of real traffic.

## Transfer: benchmark test vs. this set

| Baseline | Top-1 benchmark | Top-1 here | Decision acc. benchmark | Decision acc. here | False routing benchmark | False routing here |
|---|---:|---:|---:|---:|---:|---:|
| `lexical` | 0.616 | 0.343 | 0.579 | 0.349 | 0.228 | 0.137 |
| `tfidf` | 0.588 | 0.307 | 0.559 | 0.311 | 0.210 | 0.087 |
| `st_message` | 0.846 | 0.734 | 0.725 | 0.549 | 0.173 | 0.142 |
| `tfidf_lr` | 0.619 | 0.383 | 0.584 | 0.366 | 0.212 | 0.160 |
| `st_lr` | 0.866 | 0.734 | 0.758 | 0.576 | 0.182 | 0.212 |
| `st_lgbm` | 0.875 | 0.759 | 0.783 | 0.605 | 0.159 | 0.192 |
| `cross_encoder` | 0.854 | 0.646 | 0.783 | 0.500 | 0.155 | 0.183 |

## Metrics with intervals

| Baseline | Top-1 | Decision acc. | False routing | Abstain rate | no_match recall |
|---|---|---|---|---|---|
| `lexical` | 0.343 [0.288, 0.395] | 0.349 [0.306, 0.390] | 0.137 [0.097, 0.177] | 0.683 [0.628, 0.743] | 0.829 [0.735, 0.914] |
| `tfidf` | 0.307 [0.248, 0.362] | 0.311 [0.273, 0.348] | 0.087 [0.055, 0.121] | 0.776 [0.723, 0.829] | 0.857 [0.768, 0.931] |
| `st_message` | 0.734 [0.678, 0.793] | 0.549 [0.489, 0.608] | 0.142 [0.109, 0.178] | 0.445 [0.388, 0.501] | 0.671 [0.545, 0.778] |
| `tfidf_lr` | 0.383 [0.336, 0.431] | 0.366 [0.324, 0.406] | 0.160 [0.122, 0.197] | 0.642 [0.587, 0.700] | 0.829 [0.738, 0.913] |
| `st_lr` | 0.734 [0.680, 0.788] | 0.576 [0.526, 0.626] | 0.212 [0.169, 0.257] | 0.311 [0.259, 0.365] | 0.486 [0.368, 0.594] |
| `st_lgbm` | 0.759 [0.704, 0.816] | 0.605 [0.548, 0.659] | 0.192 [0.156, 0.227] | 0.314 [0.268, 0.359] | 0.543 [0.426, 0.653] |
| `cross_encoder` | 0.646 [0.593, 0.703] | 0.500 [0.453, 0.548] | 0.183 [0.135, 0.238] | 0.477 [0.418, 0.534] | 0.786 [0.682, 0.877] |

## Paired difference to `st_message` (95% interval; an interval that excludes 0 is a real difference on this set)

| Baseline | Δ Top-1 | Δ Decision acc. | Δ False routing |
|---|---|---|---|
| `lexical` | -0.391 [-0.458, -0.325] | -0.201 [-0.255, -0.144] | -0.006 [-0.055, 0.041] |
| `tfidf` | -0.427 [-0.504, -0.354] | -0.238 [-0.293, -0.179] | -0.055 [-0.103, -0.009] |
| `tfidf_lr` | -0.350 [-0.416, -0.286] | -0.183 [-0.237, -0.126] | 0.017 [-0.035, 0.064] |
| `st_lr` | 0.000 [-0.040, 0.043] | 0.026 [-0.018, 0.070] | 0.070 [0.026, 0.116] |
| `st_lgbm` | 0.025 [-0.011, 0.065] | 0.055 [0.009, 0.106] | 0.049 [0.012, 0.086] |
| `cross_encoder` | -0.088 [-0.145, -0.030] | -0.049 [-0.102, 0.006] | 0.041 [-0.012, 0.099] |

## Top-1 by slice

| Baseline | ctx:none | ctx:first_turn | ctx:later_turn | k:2-3 | k:4-6 | k:7+ | nm_kind:sibling_removed | nm_kind:in_domain_uncovered | nm_kind:oos | risk:sibling_partial_validity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lexical` | 0.345 | 0.409 | 0.273 | 0.353 | 0.348 | 0.309 | – | – | – | – |
| `tfidf` | 0.317 | 0.348 | 0.242 | 0.353 | 0.298 | 0.309 | – | – | – | – |
| `st_message` | 0.775 | 0.697 | 0.682 | 0.647 | 0.748 | 0.738 | – | – | – | – |
| `tfidf_lr` | 0.359 | 0.409 | 0.409 | 0.412 | 0.399 | 0.286 | – | – | – | – |
| `st_lr` | 0.789 | 0.697 | 0.651 | 0.588 | 0.752 | 0.762 | – | – | – | – |
| `st_lgbm` | 0.789 | 0.727 | 0.727 | 0.647 | 0.788 | 0.714 | – | – | – | – |
| `cross_encoder` | 0.676 | 0.667 | 0.561 | 0.529 | 0.687 | 0.548 | – | – | – | – |

## Decision accuracy by slice

| Baseline | ctx:none | ctx:first_turn | ctx:later_turn | k:2-3 | k:4-6 | k:7+ | nm_kind:sibling_removed | nm_kind:in_domain_uncovered | nm_kind:oos | risk:sibling_partial_validity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lexical` | 0.419 | 0.324 | 0.151 | 0.279 | 0.365 | 0.327 | 0.774 | 0.783 | 1.000 | 0.774 |
| `tfidf` | 0.391 | 0.250 | 0.121 | 0.279 | 0.314 | 0.327 | 0.806 | 0.826 | 1.000 | 0.806 |
| `st_message` | 0.619 | 0.515 | 0.364 | 0.372 | 0.575 | 0.571 | 0.581 | 0.565 | 1.000 | 0.581 |
| `tfidf_lr` | 0.419 | 0.338 | 0.227 | 0.302 | 0.381 | 0.347 | 0.806 | 0.783 | 0.938 | 0.806 |
| `st_lr` | 0.591 | 0.559 | 0.545 | 0.535 | 0.583 | 0.571 | 0.419 | 0.435 | 0.688 | 0.419 |
| `st_lgbm` | 0.619 | 0.588 | 0.576 | 0.465 | 0.639 | 0.551 | 0.516 | 0.478 | 0.688 | 0.516 |
| `cross_encoder` | 0.543 | 0.485 | 0.379 | 0.395 | 0.528 | 0.449 | 0.710 | 0.826 | 0.875 | 0.710 |

## Thresholds

| Baseline | τ | source |
|---|---:|---|
| `lexical` | 0.035 | tuned on the stored validation scores |
| `tfidf` | 0.025 | tuned on the stored validation scores |
| `st_message` | 0.278 | tuned on the stored validation scores |
| `tfidf_lr` | -0.768 | tuned on the stored validation scores |
| `st_lr` | -0.843 | tuned on the stored validation scores |
| `st_lgbm` | -0.589 | tuned on the stored validation scores |
| `cross_encoder` | 0.200 | taken from the first-round result file (same weights, same validation tuning) |
