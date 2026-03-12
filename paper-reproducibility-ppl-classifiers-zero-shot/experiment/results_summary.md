# Experiment Results Summary: experiment

**Experiment Directory:** `/home/b/p/cefr-classification/minimal-cefr/paper-reproducibility-ppl-classifiers-zero-shot/experiment`
**Generated:** /home/b/p/cefr-classification/minimal-cefr

## Overview

- **Total Models:** 6
- **Datasets Evaluated:** 3
- **Prediction Strategies:** 2
- **Total Evaluations:** 36

## Top 10 Models by Accuracy

### Strategy: Argmax

| Rank | Model | Dataset | Accuracy | TF-IDF Config | Classifier |
|------|-------|---------|----------|---------------|------------|
| 1 | `ppl_native_only_xgboost` | norm-KUPA-KEYS | 0.5400 | N/A | xgboost |
| 2 | `ppl_native_general_xgboost` | norm-KUPA-KEYS | 0.5300 | N/A | xgboost |
| 3 | `ppl_all_models_xgboost` | norm-KUPA-KEYS | 0.5200 | N/A | xgboost |
| 4 | `ppl_all_models_logistic` | norm-KUPA-KEYS | 0.4900 | N/A | logistic |
| 5 | `ppl_native_only_logistic` | norm-KUPA-KEYS | 0.4300 | N/A | logistic |
| 6 | `ppl_native_general_logistic` | norm-KUPA-KEYS | 0.4000 | N/A | logistic |
| 7 | `ppl_all_models_logistic` | norm-EFCAMDAT-test | 0.3400 | N/A | logistic |
| 8 | `ppl_all_models_xgboost` | norm-EFCAMDAT-test | 0.3400 | N/A | xgboost |
| 9 | `ppl_native_general_xgboost` | norm-EFCAMDAT-test | 0.3400 | N/A | xgboost |
| 10 | `ppl_native_only_xgboost` | norm-EFCAMDAT-test | 0.3400 | N/A | xgboost |

### Strategy: Rounded Avg

| Rank | Model | Dataset | Accuracy | TF-IDF Config | Classifier |
|------|-------|---------|----------|---------------|------------|
| 1 | `ppl_native_only_xgboost` | norm-KUPA-KEYS | 0.5600 | N/A | xgboost |
| 2 | `ppl_all_models_xgboost` | norm-KUPA-KEYS | 0.5400 | N/A | xgboost |
| 3 | `ppl_native_general_xgboost` | norm-KUPA-KEYS | 0.5400 | N/A | xgboost |
| 4 | `ppl_all_models_logistic` | norm-KUPA-KEYS | 0.5000 | N/A | logistic |
| 5 | `ppl_native_only_logistic` | norm-KUPA-KEYS | 0.4300 | N/A | logistic |
| 6 | `ppl_native_general_logistic` | norm-KUPA-KEYS | 0.4100 | N/A | logistic |
| 7 | `ppl_all_models_logistic` | norm-EFCAMDAT-test | 0.3400 | N/A | logistic |
| 8 | `ppl_all_models_xgboost` | norm-EFCAMDAT-test | 0.3400 | N/A | xgboost |
| 9 | `ppl_native_general_xgboost` | norm-EFCAMDAT-test | 0.3400 | N/A | xgboost |
| 10 | `ppl_native_only_xgboost` | norm-EFCAMDAT-test | 0.3400 | N/A | xgboost |

## Top 10 Models by Adjacent Accuracy

### Strategy: Argmax

| Rank | Model | Dataset | Adjacent Accuracy | TF-IDF Config | Classifier |
|------|-------|---------|----------|---------------|------------|
| 1 | `ppl_all_models_xgboost` | norm-KUPA-KEYS | 0.9700 | N/A | xgboost |
| 2 | `ppl_native_general_xgboost` | norm-KUPA-KEYS | 0.9700 | N/A | xgboost |
| 3 | `ppl_native_only_xgboost` | norm-KUPA-KEYS | 0.9700 | N/A | xgboost |
| 4 | `ppl_all_models_logistic` | norm-KUPA-KEYS | 0.9500 | N/A | logistic |
| 5 | `ppl_native_general_logistic` | norm-KUPA-KEYS | 0.9400 | N/A | logistic |
| 6 | `ppl_native_only_logistic` | norm-KUPA-KEYS | 0.9400 | N/A | logistic |
| 7 | `ppl_native_general_xgboost` | norm-EFCAMDAT-test | 0.7400 | N/A | xgboost |
| 8 | `ppl_native_only_xgboost` | norm-EFCAMDAT-test | 0.7400 | N/A | xgboost |
| 9 | `ppl_all_models_logistic` | norm-EFCAMDAT-test | 0.7300 | N/A | logistic |
| 10 | `ppl_all_models_xgboost` | norm-EFCAMDAT-test | 0.7300 | N/A | xgboost |

### Strategy: Rounded Avg

| Rank | Model | Dataset | Adjacent Accuracy | TF-IDF Config | Classifier |
|------|-------|---------|----------|---------------|------------|
| 1 | `ppl_all_models_xgboost` | norm-KUPA-KEYS | 0.9800 | N/A | xgboost |
| 2 | `ppl_native_general_xgboost` | norm-KUPA-KEYS | 0.9800 | N/A | xgboost |
| 3 | `ppl_native_only_xgboost` | norm-KUPA-KEYS | 0.9800 | N/A | xgboost |
| 4 | `ppl_all_models_logistic` | norm-KUPA-KEYS | 0.9600 | N/A | logistic |
| 5 | `ppl_native_general_logistic` | norm-KUPA-KEYS | 0.9400 | N/A | logistic |
| 6 | `ppl_native_only_logistic` | norm-KUPA-KEYS | 0.9400 | N/A | logistic |
| 7 | `ppl_native_general_xgboost` | norm-EFCAMDAT-test | 0.7400 | N/A | xgboost |
| 8 | `ppl_native_only_xgboost` | norm-EFCAMDAT-test | 0.7400 | N/A | xgboost |
| 9 | `ppl_all_models_logistic` | norm-EFCAMDAT-test | 0.7300 | N/A | logistic |
| 10 | `ppl_all_models_xgboost` | norm-EFCAMDAT-test | 0.7300 | N/A | xgboost |

## Top 10 Models by Macro F1-Score

### Strategy: Argmax

| Rank | Model | Dataset | Macro F1-Score | TF-IDF Config | Classifier |
|------|-------|---------|----------|---------------|------------|
| 1 | `ppl_all_models_logistic` | norm-KUPA-KEYS | 0.2800 | N/A | logistic |
| 2 | `ppl_all_models_xgboost` | norm-KUPA-KEYS | 0.2800 | N/A | xgboost |
| 3 | `ppl_native_only_xgboost` | norm-KUPA-KEYS | 0.2500 | N/A | xgboost |
| 4 | `ppl_native_general_xgboost` | norm-KUPA-KEYS | 0.2400 | N/A | xgboost |
| 5 | `ppl_all_models_logistic` | norm-EFCAMDAT-test | 0.2000 | N/A | logistic |
| 6 | `ppl_all_models_xgboost` | norm-EFCAMDAT-test | 0.2000 | N/A | xgboost |
| 7 | `ppl_native_general_logistic` | norm-EFCAMDAT-test | 0.2000 | N/A | logistic |
| 8 | `ppl_native_general_logistic` | norm-KUPA-KEYS | 0.2000 | N/A | logistic |
| 9 | `ppl_native_general_xgboost` | norm-EFCAMDAT-test | 0.2000 | N/A | xgboost |
| 10 | `ppl_native_only_logistic` | norm-EFCAMDAT-test | 0.2000 | N/A | logistic |

### Strategy: Rounded Avg

| Rank | Model | Dataset | Macro F1-Score | TF-IDF Config | Classifier |
|------|-------|---------|----------|---------------|------------|
| 1 | `ppl_all_models_xgboost` | norm-KUPA-KEYS | 0.2900 | N/A | xgboost |
| 2 | `ppl_all_models_logistic` | norm-KUPA-KEYS | 0.2700 | N/A | logistic |
| 3 | `ppl_native_only_xgboost` | norm-KUPA-KEYS | 0.2500 | N/A | xgboost |
| 4 | `ppl_native_general_xgboost` | norm-KUPA-KEYS | 0.2400 | N/A | xgboost |
| 5 | `ppl_all_models_logistic` | norm-EFCAMDAT-test | 0.2000 | N/A | logistic |
| 6 | `ppl_all_models_xgboost` | norm-EFCAMDAT-test | 0.2000 | N/A | xgboost |
| 7 | `ppl_native_general_logistic` | norm-EFCAMDAT-test | 0.2000 | N/A | logistic |
| 8 | `ppl_native_general_logistic` | norm-KUPA-KEYS | 0.2000 | N/A | logistic |
| 9 | `ppl_native_general_xgboost` | norm-EFCAMDAT-test | 0.2000 | N/A | xgboost |
| 10 | `ppl_native_only_logistic` | norm-EFCAMDAT-test | 0.2000 | N/A | logistic |

## Top 10 Models by Weighted F1-Score

### Strategy: Argmax

| Rank | Model | Dataset | Weighted F1-Score | TF-IDF Config | Classifier |
|------|-------|---------|----------|---------------|------------|
| 1 | `ppl_all_models_xgboost` | norm-KUPA-KEYS | 0.4900 | N/A | xgboost |
| 2 | `ppl_native_only_xgboost` | norm-KUPA-KEYS | 0.4800 | N/A | xgboost |
| 3 | `ppl_all_models_logistic` | norm-KUPA-KEYS | 0.4700 | N/A | logistic |
| 4 | `ppl_native_general_xgboost` | norm-KUPA-KEYS | 0.4700 | N/A | xgboost |
| 5 | `ppl_native_only_logistic` | norm-KUPA-KEYS | 0.3900 | N/A | logistic |
| 6 | `ppl_native_general_logistic` | norm-KUPA-KEYS | 0.3800 | N/A | logistic |
| 7 | `ppl_all_models_logistic` | norm-EFCAMDAT-test | 0.3400 | N/A | logistic |
| 8 | `ppl_all_models_xgboost` | norm-EFCAMDAT-test | 0.3400 | N/A | xgboost |
| 9 | `ppl_native_general_xgboost` | norm-EFCAMDAT-test | 0.3400 | N/A | xgboost |
| 10 | `ppl_native_only_xgboost` | norm-EFCAMDAT-test | 0.3400 | N/A | xgboost |

### Strategy: Rounded Avg

| Rank | Model | Dataset | Weighted F1-Score | TF-IDF Config | Classifier |
|------|-------|---------|----------|---------------|------------|
| 1 | `ppl_all_models_xgboost` | norm-KUPA-KEYS | 0.5100 | N/A | xgboost |
| 2 | `ppl_native_only_xgboost` | norm-KUPA-KEYS | 0.4900 | N/A | xgboost |
| 3 | `ppl_all_models_logistic` | norm-KUPA-KEYS | 0.4700 | N/A | logistic |
| 4 | `ppl_native_general_xgboost` | norm-KUPA-KEYS | 0.4700 | N/A | xgboost |
| 5 | `ppl_native_general_logistic` | norm-KUPA-KEYS | 0.3900 | N/A | logistic |
| 6 | `ppl_native_only_logistic` | norm-KUPA-KEYS | 0.3900 | N/A | logistic |
| 7 | `ppl_all_models_logistic` | norm-EFCAMDAT-test | 0.3400 | N/A | logistic |
| 8 | `ppl_all_models_xgboost` | norm-EFCAMDAT-test | 0.3400 | N/A | xgboost |
| 9 | `ppl_native_general_xgboost` | norm-EFCAMDAT-test | 0.3400 | N/A | xgboost |
| 10 | `ppl_native_only_xgboost` | norm-EFCAMDAT-test | 0.3400 | N/A | xgboost |

## Top 10 Models by Micro F1-Score

### Strategy: Argmax

| Rank | Model | Dataset | Micro F1-Score | TF-IDF Config | Classifier |
|------|-------|---------|----------|---------------|------------|
| 1 | `ppl_native_only_xgboost` | norm-KUPA-KEYS | 0.5400 | N/A | xgboost |
| 2 | `ppl_native_general_xgboost` | norm-KUPA-KEYS | 0.5300 | N/A | xgboost |
| 3 | `ppl_all_models_xgboost` | norm-KUPA-KEYS | 0.5200 | N/A | xgboost |
| 4 | `ppl_all_models_logistic` | norm-KUPA-KEYS | 0.4900 | N/A | logistic |
| 5 | `ppl_native_only_logistic` | norm-KUPA-KEYS | 0.4300 | N/A | logistic |
| 6 | `ppl_native_general_logistic` | norm-KUPA-KEYS | 0.4000 | N/A | logistic |
| 7 | `ppl_all_models_logistic` | norm-EFCAMDAT-test | 0.3400 | N/A | logistic |
| 8 | `ppl_all_models_xgboost` | norm-EFCAMDAT-test | 0.3400 | N/A | xgboost |
| 9 | `ppl_native_general_xgboost` | norm-EFCAMDAT-test | 0.3400 | N/A | xgboost |
| 10 | `ppl_native_only_xgboost` | norm-EFCAMDAT-test | 0.3400 | N/A | xgboost |

### Strategy: Rounded Avg

| Rank | Model | Dataset | Micro F1-Score | TF-IDF Config | Classifier |
|------|-------|---------|----------|---------------|------------|
| 1 | `ppl_native_only_xgboost` | norm-KUPA-KEYS | 0.5600 | N/A | xgboost |
| 2 | `ppl_all_models_xgboost` | norm-KUPA-KEYS | 0.5400 | N/A | xgboost |
| 3 | `ppl_native_general_xgboost` | norm-KUPA-KEYS | 0.5400 | N/A | xgboost |
| 4 | `ppl_all_models_logistic` | norm-KUPA-KEYS | 0.5000 | N/A | logistic |
| 5 | `ppl_native_only_logistic` | norm-KUPA-KEYS | 0.4300 | N/A | logistic |
| 6 | `ppl_native_general_logistic` | norm-KUPA-KEYS | 0.4100 | N/A | logistic |
| 7 | `ppl_all_models_logistic` | norm-EFCAMDAT-test | 0.3400 | N/A | logistic |
| 8 | `ppl_all_models_xgboost` | norm-EFCAMDAT-test | 0.3400 | N/A | xgboost |
| 9 | `ppl_native_general_xgboost` | norm-EFCAMDAT-test | 0.3400 | N/A | xgboost |
| 10 | `ppl_native_only_xgboost` | norm-EFCAMDAT-test | 0.3400 | N/A | xgboost |

## Performance by Dataset

### norm-CELVA-SP

- **Best Accuracy:** 0.2400 (`ppl_all_models_xgboost`)
- **Best Adjacent Accuracy:** 0.6600 (`ppl_native_general_xgboost`)
- **Models Evaluated:** 6

### norm-EFCAMDAT-test

- **Best Accuracy:** 0.3400 (`ppl_all_models_logistic`)
- **Best Adjacent Accuracy:** 0.7400 (`ppl_native_general_xgboost`)
- **Models Evaluated:** 6

### norm-KUPA-KEYS

- **Best Accuracy:** 0.5400 (`ppl_native_only_xgboost`)
- **Best Adjacent Accuracy:** 0.9700 (`ppl_all_models_xgboost`)
- **Models Evaluated:** 6

## TF-IDF Configuration Analysis
