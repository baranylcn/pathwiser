# Zero-shot encoder scan

zero-shot cosine of message and condition embeddings; no fitting; test not read. Validation = selection evidence; `realistic` = confirmation (95% scenario-bootstrap intervals, paired difference to `st_message`).

| Encoder | val Top-1 | val macro Top-1 (datasets) | val Top-1 unseen service | realistic Top-1 | Δ vs `st_message` (realistic) |
|---|---:|---:|---:|---|---|
| `st_message` | 0.854 | 0.847 | 0.849 | 0.734 [0.678, 0.793] | (reference) |
| `st_message__bge_small` | 0.860 | 0.852 | 0.830 | 0.792 [0.743, 0.838] | 0.058 [0.011, 0.103] |
| `st_message__gte_small` | 0.864 | 0.858 | 0.813 | 0.766 [0.716, 0.814] | 0.033 [-0.018, 0.081] |
| `st_message__minilm_l12` | 0.855 | 0.840 | 0.864 | 0.723 [0.670, 0.775] | -0.011 [-0.068, 0.043] |
| `st_message__e5_small` | 0.834 | 0.829 | 0.785 | 0.730 [0.677, 0.781] | -0.004 [-0.067, 0.058] |
| `st_message__multiqa_l6` | 0.849 | 0.843 | 0.814 | 0.679 [0.623, 0.731] | -0.055 [-0.104, -0.007] |
