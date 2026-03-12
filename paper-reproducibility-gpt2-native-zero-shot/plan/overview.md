# GPT-2 Native Perplexity Zero-Shot Experiment

## Goal

Train LR and XGBoost classifiers on GPT-2 (native, pre-trained) perplexity
features extracted from EFCAMDAT training data, then evaluate zero-shot on:
- EFCAMDAT test (in-domain)
- CELVA-SP (cross-corpus)
- KUPA-KEYS (cross-corpus)

## Pipeline

1. Extract aggregate perplexity features from pre-trained GPT-2 for all datasets
2. Convert perplexity CSVs to features_dense.csv (numeric-only) format
3. Train LR and XGBoost on EFCAMDAT-train features
4. Predict on all 3 test sets with both classifiers
5. Generate evaluation reports and summary

## Data

Source: `/home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits/`

| Dataset | File | Rows | Role |
|---|---|---|---|
| EFCAMDAT train | norm-EFCAMDAT-train.csv | 80k | Classifier training |
| EFCAMDAT test | norm-EFCAMDAT-test.csv | 20k | In-domain test |
| CELVA-SP | norm-CELVA-SP.csv | 1,742 | Cross-corpus test |
| KUPA-KEYS | norm-KUPA-KEYS.csv | 1,006 | Cross-corpus test |

All CSVs: schema `writing_id, l1, cefr_level, text`

## Constraints

- CPU only (no GPU)
- CANNOT modify anything in src/
- All outputs go into this reproducibility folder
- GPT-2 is auto-downloaded from HuggingFace Hub
