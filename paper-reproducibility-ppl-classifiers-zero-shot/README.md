# Paper Reproducibility: PPL Classifiers (Zero-Shot, Pre-extracted Features)

Train statistical classifiers (LR, XGBoost) on **pre-extracted** GPT-2 perplexity
features. No GPU needed -- perplexity extraction was already done and the feature
files live in `gdrive-data/fe/`.

## Goal

Reproduce the paper's perplexity-based classification results by reusing the
existing `.csv.features.gzip` files from previous extraction runs, skipping the
expensive GPU step entirely.

## Scope

- **Zero-shot only**: train on EFCAMDAT-train, test on EFCAMDAT-test, CELVA-SP, KUPA-KEYS
- **No perplexity re-extraction**: all features are read directly from `gdrive-data/fe/`
- **No AL training**: the artificial learner models were already fine-tuned
- **Classifiers**: Logistic Regression + XGBoost
- **Feature configurations**: 4 experiments (see `feature-files.md`)

## Documents

| File | Contents |
|---|---|
| `README.md` | This file |
| `feature-files.md` | Inventory of pre-extracted feature files with paths and rationale |
| `commands.md` | Full pipeline commands to run end-to-end |
