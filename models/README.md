# models/

Trained model artifacts (weights, tokenizer, `training.json` with parameters and loss curve), one directory per model, for example
`models/cross_encoder/`. Written by `pathwiser baselines run` and `pathwiser scorers fit` (`--param save_dir=models/<name>`).

Weights are git-ignored (large, and reproducible from `data/model/` plus the parameters recorded in `reports/`); only this README is
tracked. Results computed with a model live in `reports/` and name the exact data hashes.
