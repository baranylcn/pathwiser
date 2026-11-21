# Baseline results

Data: model-ready `v1-model` · train `7b63435290e2` · val `da0758fd6499` · test `eeb434b789bf`

**Protocol.** Scores come from training-free scorers and from trained ones (fitted on the train split only; `random` is the chance reference). The decision threshold τ (abstain = `no_match` when the best score < τ) is tuned on **validation main examples** to maximise decision accuracy, then applied unchanged to test. Validation numbers are therefore optimistic (τ was tuned on them); test numbers are the honest ones. SGD-X paraphrase twins are reported separately, never mixed into the totals. Scores are relevance scores, not calibrated probabilities.

Reading guide: `Top-1/Top-3/MRR` are over examples that have a gold condition; `Decision acc` counts a `no_match` example as correct when the policy abstains; `False routing` = routed to a wrong condition, or routed although no condition applies, over all examples. Top-3 is trivial for k ≤ 3 (about 40% of examples), compare it against `random`.

## Validation (τ tuned on this split)

| Baseline | Top-1 | Top-3 | MRR | Decision acc | no_match P | no_match R | False routing | ms/example |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `random` | 0.242 | 0.689 | 0.495 | 0.203 | 0.204 | 0.988 | 0.012 | 0.03 |
| `lexical` | 0.663 | 0.907 | 0.792 | 0.625 | 0.458 | 0.671 | 0.213 | 0.04 |
| `lexical_ctx` | 0.653 | 0.915 | 0.789 | 0.605 | 0.480 | 0.567 | 0.270 | 0.06 |
| `tfidf` | 0.641 | 0.891 | 0.774 | 0.610 | 0.434 | 0.719 | 0.199 | 0.03 |
| `tfidf_ctx` | 0.653 | 0.900 | 0.785 | 0.614 | 0.448 | 0.693 | 0.212 | 0.09 |
| `st_message` | 0.854 | 0.985 | 0.918 | 0.749 | 0.565 | 0.630 | 0.152 | 2.44 |
| `st_query` | 0.693 | 0.941 | 0.818 | 0.587 | 0.450 | 0.403 | 0.312 | 9.85 |
| `tfidf_lr` | 0.676 | 0.916 | 0.802 | 0.628 | 0.453 | 0.673 | 0.206 | 0.10 |
| `st_lr` | 0.872 | 0.989 | 0.930 | 0.770 | 0.625 | 0.561 | 0.161 | 0.10 |
| `st_lgbm` | 0.893 | 0.992 | 0.942 | 0.807 | 0.679 | 0.638 | 0.132 | 0.13 |
| `cross_encoder` | 0.887 | 0.992 | 0.939 | 0.820 | 0.747 | 0.723 | 0.130 | 31.85 |

## Test (τ from validation)

| Baseline | Top-1 | Top-3 | MRR | Decision acc | no_match P | no_match R | False routing | ms/example |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `random` | 0.242 | 0.680 | 0.492 | 0.227 | 0.228 | 0.989 | 0.011 | 0.03 |
| `lexical` | 0.616 | 0.886 | 0.760 | 0.579 | 0.442 | 0.673 | 0.228 | 0.04 |
| `lexical_ctx` | 0.606 | 0.889 | 0.755 | 0.561 | 0.451 | 0.601 | 0.273 | 0.04 |
| `tfidf` | 0.588 | 0.862 | 0.736 | 0.559 | 0.415 | 0.722 | 0.210 | 0.02 |
| `tfidf_ctx` | 0.595 | 0.867 | 0.742 | 0.562 | 0.422 | 0.701 | 0.220 | 0.07 |
| `st_message` | 0.846 | 0.985 | 0.915 | 0.725 | 0.587 | 0.642 | 0.173 | 2.44 |
| `st_query` | 0.621 | 0.923 | 0.773 | 0.519 | 0.447 | 0.380 | 0.375 | 11.85 |
| `tfidf_lr` | 0.619 | 0.876 | 0.759 | 0.584 | 0.431 | 0.681 | 0.212 | 0.11 |
| `st_lr` | 0.866 | 0.986 | 0.926 | 0.758 | 0.671 | 0.541 | 0.182 | 3.84 |
| `st_lgbm` | 0.875 | 0.990 | 0.931 | 0.783 | 0.710 | 0.616 | 0.159 | 0.13 |
| `cross_encoder` | 0.854 | 0.986 | 0.919 | 0.783 | 0.725 | 0.716 | 0.155 | 32.80 |

## Top-1 by dataset (test)

| Baseline | abcd | banking77 | clinc150 | hwu64 | sgd | star |
|---|---:|---:|---:|---:|---:|---:|
| `random` | 0.239 | 0.244 | 0.242 | 0.247 | 0.240 | 0.227 |
| `lexical` | 0.595 | 0.683 | 0.656 | 0.606 | 0.569 | 0.699 |
| `lexical_ctx` | 0.595 | 0.683 | 0.656 | 0.606 | 0.547 | 0.600 |
| `tfidf` | 0.551 | 0.642 | 0.630 | 0.578 | 0.544 | 0.720 |
| `tfidf_ctx` | 0.556 | 0.642 | 0.629 | 0.585 | 0.561 | 0.707 |
| `st_message` | 0.643 | 0.849 | 0.944 | 0.876 | 0.809 | 0.795 |
| `st_query` | 0.386 | 0.766 | 0.743 | 0.712 | 0.505 | 0.448 |
| `tfidf_lr` | 0.534 | 0.685 | 0.650 | 0.578 | 0.597 | 0.733 |
| `st_lr` | 0.647 | 0.859 | 0.938 | 0.898 | 0.848 | 0.840 |
| `st_lgbm` | 0.684 | 0.864 | 0.948 | 0.904 | 0.853 | 0.877 |
| `cross_encoder` | 0.609 | 0.855 | 0.929 | 0.884 | 0.837 | 0.843 |

## Decision accuracy by dataset (test, global τ)

| Baseline | abcd | banking77 | clinc150 | hwu64 | sgd | star |
|---|---:|---:|---:|---:|---:|---:|
| `random` | 0.207 | 0.196 | 0.292 | 0.208 | 0.202 | 0.207 |
| `lexical` | 0.518 | 0.600 | 0.633 | 0.574 | 0.542 | 0.652 |
| `lexical_ctx` | 0.518 | 0.600 | 0.633 | 0.574 | 0.500 | 0.519 |
| `tfidf` | 0.497 | 0.570 | 0.613 | 0.549 | 0.523 | 0.681 |
| `tfidf_ctx` | 0.498 | 0.569 | 0.613 | 0.554 | 0.532 | 0.650 |
| `st_message` | 0.514 | 0.715 | 0.811 | 0.694 | 0.710 | 0.696 |
| `st_query` | 0.312 | 0.629 | 0.610 | 0.615 | 0.420 | 0.378 |
| `tfidf_lr` | 0.501 | 0.603 | 0.638 | 0.577 | 0.549 | 0.654 |
| `st_lr` | 0.535 | 0.740 | 0.818 | 0.801 | 0.743 | 0.751 |
| `st_lgbm` | 0.585 | 0.755 | 0.855 | 0.818 | 0.763 | 0.800 |
| `cross_encoder` | 0.481 | 0.758 | 0.860 | 0.811 | 0.777 | 0.776 |

## Top-1 by slice (test)

`cond:seen` = an equivalent of the gold condition exists in train; `seen_exact:*` splits it into the very same condition vs an equivalent-only one (different service or wording); `cond:unseen` = no equivalent in train.

| Baseline | cond:seen | seen_exact:exact | seen_exact:equivalent_only | cond:unseen | svc:seen | svc:unseen | ctx:none | ctx:first_turn | ctx:later_turn | k:2-3 | k:4-6 | k:7+ | nm_kind:other_family | nm_kind:sibling_removed | nm_kind:oos | risk:sibling_partial_validity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `random` | 0.239 | 0.238 | 0.239 | 0.247 | 0.239 | 0.240 | 0.241 | 0.235 | 0.244 | 0.392 | 0.209 | 0.114 | – | – | – | – |
| `lexical` | 0.617 | 0.682 | 0.510 | 0.615 | 0.731 | 0.509 | 0.615 | 0.604 | 0.620 | 0.703 | 0.603 | 0.531 | – | – | – | – |
| `lexical_ctx` | 0.600 | 0.662 | 0.499 | 0.615 | 0.676 | 0.499 | 0.615 | 0.603 | 0.584 | 0.696 | 0.593 | 0.516 | – | – | – | – |
| `tfidf` | 0.588 | 0.661 | 0.470 | 0.587 | 0.689 | 0.491 | 0.585 | 0.554 | 0.600 | 0.673 | 0.575 | 0.503 | – | – | – | – |
| `tfidf_ctx` | 0.600 | 0.664 | 0.497 | 0.586 | 0.704 | 0.508 | 0.586 | 0.558 | 0.623 | 0.678 | 0.581 | 0.517 | – | – | – | – |
| `st_message` | 0.847 | 0.871 | 0.808 | 0.845 | 0.879 | 0.783 | 0.878 | 0.652 | 0.803 | 0.896 | 0.842 | 0.791 | – | – | – | – |
| `st_query` | 0.593 | 0.659 | 0.487 | 0.668 | 0.486 | 0.512 | 0.703 | 0.397 | 0.467 | 0.737 | 0.603 | 0.510 | – | – | – | – |
| `tfidf_lr` | 0.630 | 0.691 | 0.531 | 0.602 | 0.744 | 0.543 | 0.613 | 0.535 | 0.648 | 0.695 | 0.610 | 0.541 | – | – | – | – |
| `st_lr` | 0.877 | 0.899 | 0.842 | 0.847 | 0.921 | 0.822 | 0.892 | 0.661 | 0.839 | 0.914 | 0.861 | 0.813 | – | – | – | – |
| `st_lgbm` | 0.887 | 0.909 | 0.852 | 0.854 | 0.922 | 0.828 | 0.898 | 0.695 | 0.850 | 0.916 | 0.872 | 0.828 | – | – | – | – |
| `cross_encoder` | 0.865 | 0.902 | 0.805 | 0.837 | 0.938 | 0.799 | 0.887 | 0.611 | 0.819 | 0.908 | 0.847 | 0.799 | – | – | – | – |

## Decision accuracy by slice (test, global τ)

| Baseline | cond:seen | seen_exact:exact | seen_exact:equivalent_only | cond:unseen | svc:seen | svc:unseen | ctx:none | ctx:first_turn | ctx:later_turn | k:2-3 | k:4-6 | k:7+ | nm_kind:other_family | nm_kind:sibling_removed | nm_kind:oos | risk:sibling_partial_validity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `random` | 0.003 | 0.003 | 0.003 | 0.004 | 0.002 | 0.003 | 0.240 | 0.208 | 0.198 | 0.234 | 0.229 | 0.214 | 0.990 | 0.990 | 0.984 | 0.989 |
| `lexical` | 0.563 | 0.628 | 0.456 | 0.534 | 0.687 | 0.459 | 0.582 | 0.521 | 0.583 | 0.625 | 0.576 | 0.527 | 0.715 | 0.526 | 0.761 | 0.466 |
| `lexical_ctx` | 0.559 | 0.613 | 0.472 | 0.535 | 0.642 | 0.473 | 0.582 | 0.521 | 0.517 | 0.617 | 0.555 | 0.501 | 0.620 | 0.465 | 0.761 | 0.424 |
| `tfidf` | 0.523 | 0.596 | 0.405 | 0.489 | 0.636 | 0.429 | 0.558 | 0.495 | 0.570 | 0.594 | 0.557 | 0.515 | 0.760 | 0.601 | 0.774 | 0.545 |
| `tfidf_ctx` | 0.540 | 0.600 | 0.443 | 0.491 | 0.652 | 0.458 | 0.559 | 0.495 | 0.582 | 0.600 | 0.560 | 0.518 | 0.736 | 0.577 | 0.778 | 0.523 |
| `st_message` | 0.780 | 0.795 | 0.757 | 0.696 | 0.859 | 0.737 | 0.748 | 0.522 | 0.701 | 0.762 | 0.719 | 0.687 | 0.652 | 0.459 | 0.923 | 0.398 |
| `st_query` | 0.516 | 0.591 | 0.395 | 0.631 | 0.329 | 0.431 | 0.584 | 0.316 | 0.389 | 0.604 | 0.509 | 0.425 | 0.410 | 0.257 | 0.470 | 0.225 |
| `tfidf_lr` | 0.567 | 0.628 | 0.467 | 0.535 | 0.693 | 0.478 | 0.585 | 0.501 | 0.593 | 0.632 | 0.582 | 0.523 | 0.719 | 0.543 | 0.767 | 0.493 |
| `st_lr` | 0.838 | 0.861 | 0.801 | 0.794 | 0.891 | 0.782 | 0.781 | 0.551 | 0.735 | 0.797 | 0.753 | 0.716 | 0.578 | 0.387 | 0.657 | 0.341 |
| `st_lgbm` | 0.852 | 0.876 | 0.814 | 0.800 | 0.895 | 0.793 | 0.805 | 0.596 | 0.761 | 0.816 | 0.781 | 0.746 | 0.649 | 0.446 | 0.782 | 0.382 |
| `cross_encoder` | 0.830 | 0.868 | 0.769 | 0.759 | 0.917 | 0.764 | 0.815 | 0.491 | 0.754 | 0.827 | 0.776 | 0.742 | 0.770 | 0.496 | 0.877 | 0.418 |

## Paraphrase robustness: SGD-X twins (test)

Same moment and candidates, condition descriptions replaced by held-out crowd paraphrases.

| Baseline | pairs | Top-1 original | Top-1 paraphrase | Same pick | Same abstain decision |
|---|---:|---:|---:|---:|---:|
| `random` | 11781 | 0.240 | 0.238 | 0.240 | 0.973 |
| `lexical` | 11781 | 0.569 | 0.503 | 0.568 | 0.768 |
| `lexical_ctx` | 11781 | 0.547 | 0.496 | 0.522 | 0.838 |
| `tfidf` | 11781 | 0.544 | 0.475 | 0.580 | 0.731 |
| `tfidf_ctx` | 11781 | 0.561 | 0.506 | 0.526 | 0.744 |
| `st_message` | 11781 | 0.809 | 0.758 | 0.691 | 0.907 |
| `st_query` | 11781 | 0.505 | 0.523 | 0.582 | 0.824 |
| `tfidf_lr` | 11781 | 0.597 | 0.530 | 0.459 | 0.735 |
| `st_lr` | 11781 | 0.848 | 0.781 | 0.713 | 0.852 |
| `st_lgbm` | 11781 | 0.853 | 0.787 | 0.713 | 0.855 |
| `cross_encoder` | 11781 | 0.837 | 0.783 | 0.738 | 0.886 |

## Training (trained baselines)

No hyper-parameter search: every setting was fixed before looking at validation results. Training uses the train split only; validation is used for the threshold, test is read once. Training budgets are limited by CPU-only hardware (`train examples` < full train split means a weighted sample, see `reports/baselines/<version>/<name>.json` for all parameters).

| Baseline | Backbone | Train examples | Train pairs | Steps | Train min | Final loss |
|---|---:|---:|---:|---:|---:|---:|
| `tfidf_lr` | – | 88790 | 449700 | – | 0.2 | – |
| `st_lr` | sentence-transformers/all-MiniLM-L6-v2 | 30000 | 151413 | – | 0.2 | – |
| `st_lgbm` | sentence-transformers/all-MiniLM-L6-v2 | 30000 | 151413 | – | 1.0 | – |
| `cross_encoder` | cross-encoder/ms-marco-MiniLM-L-6-v2 | 8000 | 28855 | 500 | 39.5 | 0.4668 |

## Thresholds, fitting cost, size

`ms/example` in the tables above is wall-clock scoring time per example on CPU, batched, and includes encoding the split's unique texts; it is a rough cost indicator, not a serving-latency benchmark.

| Baseline | τ (global) | Val decision acc @ τ | fit s | parameters |
|---|---:|---:|---:|---:|
| `random` | 0.9973 | 0.203 | 0.0 | – |
| `lexical` | 0.0345 | 0.625 | 0.0 | – |
| `lexical_ctx` | 0.0257 | 0.605 | 0.0 | – |
| `tfidf` | 0.0246 | 0.610 | 6.2 | – |
| `tfidf_ctx` | 0.0203 | 0.614 | 9.0 | – |
| `st_message` | 0.2782 | 0.749 | 5.3 | 22713216 |
| `st_query` | 0.2751 | 0.587 | 5.0 | 22713216 |
| `tfidf_lr` | -0.7681 | 0.628 | 20.5 | – |
| `st_lr` | -0.8427 | 0.770 | 15.2 | – |
| `st_lgbm` | -0.5890 | 0.807 | 61.6 | – |
| `cross_encoder` | 0.1996 | 0.820 | 2384.6 | – |
