# Experiment Results Summary: zero-shot-2

**Experiment Directory:** `data/experiments/zero-shot-2`
**Generated:** /home/b/p/cefr-classification/minimal-cefr

## Overview

- **Total Models:** 48
- **Datasets Evaluated:** 4
- **Prediction Strategies:** 2
- **Total Evaluations:** 384

## Top 10 Models by Accuracy

### Strategy: Argmax

| Rank | Model | Dataset | Accuracy | TF-IDF Config | Classifier |
|------|-------|---------|----------|---------------|------------|
| 1 | `norm-EFCAMDAT-train_xgboost_10b21d1b_tfidf_grouped` | norm-EFCAMDAT-train | 0.9900 | N/A | xgboost |
| 2 | `norm-EFCAMDAT-train_logistic_063988a0_tfidf` | norm-EFCAMDAT-train | 0.9800 | N/A | logistic |
| 3 | `norm-EFCAMDAT-train_logistic_10b21d1b_tfidf_grouped` | norm-EFCAMDAT-train | 0.9800 | N/A | logistic |
| 4 | `norm-EFCAMDAT-train_xgboost_01733819_tfidf` | norm-EFCAMDAT-train | 0.9800 | N/A | xgboost |
| 5 | `norm-EFCAMDAT-train_xgboost_063988a0_tfidf` | norm-EFCAMDAT-train | 0.9800 | N/A | xgboost |
| 6 | `norm-EFCAMDAT-train_xgboost_10b21d1b_tfidf` | norm-EFCAMDAT-train | 0.9800 | N/A | xgboost |
| 7 | `norm-EFCAMDAT-train_xgboost_1387f6a3_tfidf` | norm-EFCAMDAT-train | 0.9800 | N/A | xgboost |
| 8 | `norm-EFCAMDAT-train_xgboost_252cd532_tfidf` | norm-EFCAMDAT-train | 0.9800 | N/A | xgboost |
| 9 | `norm-EFCAMDAT-train_xgboost_336a6205_tfidf` | norm-EFCAMDAT-train | 0.9800 | N/A | xgboost |
| 10 | `norm-EFCAMDAT-train_xgboost_336a6205_tfidf_grouped` | norm-EFCAMDAT-train | 0.9800 | N/A | xgboost |

### Strategy: Rounded Avg

| Rank | Model | Dataset | Accuracy | TF-IDF Config | Classifier |
|------|-------|---------|----------|---------------|------------|
| 1 | `norm-EFCAMDAT-train_xgboost_10b21d1b_tfidf_grouped` | norm-EFCAMDAT-train | 0.9800 | N/A | xgboost |
| 2 | `norm-EFCAMDAT-train_xgboost_336a6205_tfidf_grouped` | norm-EFCAMDAT-train | 0.9800 | N/A | xgboost |
| 3 | `norm-EFCAMDAT-train_logistic_10b21d1b_tfidf_grouped` | norm-EFCAMDAT-train | 0.9700 | N/A | logistic |
| 4 | `norm-EFCAMDAT-train_xgboost_01733819_tfidf` | norm-EFCAMDAT-train | 0.9700 | N/A | xgboost |
| 5 | `norm-EFCAMDAT-train_xgboost_063988a0_tfidf` | norm-EFCAMDAT-train | 0.9700 | N/A | xgboost |
| 6 | `norm-EFCAMDAT-train_xgboost_10b21d1b_tfidf` | norm-EFCAMDAT-train | 0.9700 | N/A | xgboost |
| 7 | `norm-EFCAMDAT-train_xgboost_1387f6a3_tfidf` | norm-EFCAMDAT-train | 0.9700 | N/A | xgboost |
| 8 | `norm-EFCAMDAT-train_xgboost_252cd532_tfidf` | norm-EFCAMDAT-train | 0.9700 | N/A | xgboost |
| 9 | `norm-EFCAMDAT-train_xgboost_4827cafe_tfidf` | norm-EFCAMDAT-train | 0.9700 | N/A | xgboost |
| 10 | `norm-EFCAMDAT-train_xgboost_57c46ca5_tfidf_grouped` | norm-EFCAMDAT-train | 0.9700 | N/A | xgboost |

## Top 10 Models by Adjacent Accuracy

### Strategy: Argmax

| Rank | Model | Dataset | Adjacent Accuracy | TF-IDF Config | Classifier |
|------|-------|---------|----------|---------------|------------|
| 1 | `norm-EFCAMDAT-train_logistic_005ebc16_tfidf` | norm-EFCAMDAT-train | 0.9900 | N/A | logistic |
| 2 | `norm-EFCAMDAT-train_logistic_01733819_tfidf` | norm-EFCAMDAT-test | 0.9900 | N/A | logistic |
| 3 | `norm-EFCAMDAT-train_logistic_01733819_tfidf` | norm-EFCAMDAT-train | 0.9900 | N/A | logistic |
| 4 | `norm-EFCAMDAT-train_logistic_063988a0_tfidf` | norm-EFCAMDAT-test | 0.9900 | N/A | logistic |
| 5 | `norm-EFCAMDAT-train_logistic_10b21d1b_tfidf` | norm-EFCAMDAT-test | 0.9900 | N/A | logistic |
| 6 | `norm-EFCAMDAT-train_logistic_10b21d1b_tfidf` | norm-EFCAMDAT-train | 0.9900 | N/A | logistic |
| 7 | `norm-EFCAMDAT-train_logistic_10b21d1b_tfidf_grouped` | norm-EFCAMDAT-test | 0.9900 | N/A | logistic |
| 8 | `norm-EFCAMDAT-train_logistic_1387f6a3_tfidf` | norm-EFCAMDAT-test | 0.9900 | N/A | logistic |
| 9 | `norm-EFCAMDAT-train_logistic_1387f6a3_tfidf` | norm-EFCAMDAT-train | 0.9900 | N/A | logistic |
| 10 | `norm-EFCAMDAT-train_logistic_252cd532_tfidf` | norm-EFCAMDAT-test | 0.9900 | N/A | logistic |

### Strategy: Rounded Avg

| Rank | Model | Dataset | Adjacent Accuracy | TF-IDF Config | Classifier |
|------|-------|---------|----------|---------------|------------|
| 1 | `norm-EFCAMDAT-train_logistic_005ebc16_tfidf` | norm-EFCAMDAT-test | 0.9900 | N/A | logistic |
| 2 | `norm-EFCAMDAT-train_logistic_005ebc16_tfidf` | norm-EFCAMDAT-train | 0.9900 | N/A | logistic |
| 3 | `norm-EFCAMDAT-train_logistic_10b21d1b_tfidf` | norm-EFCAMDAT-test | 0.9900 | N/A | logistic |
| 4 | `norm-EFCAMDAT-train_logistic_10b21d1b_tfidf` | norm-EFCAMDAT-train | 0.9900 | N/A | logistic |
| 5 | `norm-EFCAMDAT-train_logistic_1b7f653c_tfidf` | norm-EFCAMDAT-test | 0.9900 | N/A | logistic |
| 6 | `norm-EFCAMDAT-train_logistic_1b7f653c_tfidf` | norm-EFCAMDAT-train | 0.9900 | N/A | logistic |
| 7 | `norm-EFCAMDAT-train_logistic_336a6205_tfidf` | norm-EFCAMDAT-test | 0.9900 | N/A | logistic |
| 8 | `norm-EFCAMDAT-train_logistic_336a6205_tfidf` | norm-EFCAMDAT-train | 0.9900 | N/A | logistic |
| 9 | `norm-EFCAMDAT-train_logistic_336a6205_tfidf_grouped` | norm-EFCAMDAT-test | 0.9900 | N/A | logistic |
| 10 | `norm-EFCAMDAT-train_logistic_53d29bb1_tfidf_grouped` | norm-EFCAMDAT-test | 0.9900 | N/A | logistic |

## Top 10 Models by Macro F1-Score

### Strategy: Argmax

| Rank | Model | Dataset | Macro F1-Score | TF-IDF Config | Classifier |
|------|-------|---------|----------|---------------|------------|
| 1 | `norm-EFCAMDAT-train_xgboost_10b21d1b_tfidf_grouped` | norm-EFCAMDAT-train | 0.9900 | N/A | xgboost |
| 2 | `norm-EFCAMDAT-train_xgboost_01733819_tfidf` | norm-EFCAMDAT-train | 0.9800 | N/A | xgboost |
| 3 | `norm-EFCAMDAT-train_xgboost_063988a0_tfidf` | norm-EFCAMDAT-train | 0.9800 | N/A | xgboost |
| 4 | `norm-EFCAMDAT-train_xgboost_10b21d1b_tfidf` | norm-EFCAMDAT-train | 0.9800 | N/A | xgboost |
| 5 | `norm-EFCAMDAT-train_xgboost_1387f6a3_tfidf` | norm-EFCAMDAT-train | 0.9800 | N/A | xgboost |
| 6 | `norm-EFCAMDAT-train_xgboost_252cd532_tfidf` | norm-EFCAMDAT-train | 0.9800 | N/A | xgboost |
| 7 | `norm-EFCAMDAT-train_xgboost_336a6205_tfidf` | norm-EFCAMDAT-train | 0.9800 | N/A | xgboost |
| 8 | `norm-EFCAMDAT-train_xgboost_336a6205_tfidf_grouped` | norm-EFCAMDAT-train | 0.9800 | N/A | xgboost |
| 9 | `norm-EFCAMDAT-train_xgboost_4827cafe_tfidf` | norm-EFCAMDAT-train | 0.9800 | N/A | xgboost |
| 10 | `norm-EFCAMDAT-train_xgboost_57c46ca5_tfidf_grouped` | norm-EFCAMDAT-train | 0.9800 | N/A | xgboost |

### Strategy: Rounded Avg

| Rank | Model | Dataset | Macro F1-Score | TF-IDF Config | Classifier |
|------|-------|---------|----------|---------------|------------|
| 1 | `norm-EFCAMDAT-train_xgboost_10b21d1b_tfidf_grouped` | norm-EFCAMDAT-train | 0.9700 | N/A | xgboost |
| 2 | `norm-EFCAMDAT-train_xgboost_336a6205_tfidf_grouped` | norm-EFCAMDAT-train | 0.9700 | N/A | xgboost |
| 3 | `norm-EFCAMDAT-train_xgboost_e89a99e6_tfidf_grouped` | norm-EFCAMDAT-train | 0.9700 | N/A | xgboost |
| 4 | `norm-EFCAMDAT-train_logistic_10b21d1b_tfidf_grouped` | norm-EFCAMDAT-train | 0.9600 | N/A | logistic |
| 5 | `norm-EFCAMDAT-train_xgboost_57c46ca5_tfidf_grouped` | norm-EFCAMDAT-train | 0.9600 | N/A | xgboost |
| 6 | `norm-EFCAMDAT-train_xgboost_db3a2b11_tfidf_grouped` | norm-EFCAMDAT-train | 0.9600 | N/A | xgboost |
| 7 | `norm-EFCAMDAT-train_logistic_063988a0_tfidf` | norm-EFCAMDAT-train | 0.9500 | N/A | logistic |
| 8 | `norm-EFCAMDAT-train_xgboost_01733819_tfidf` | norm-EFCAMDAT-train | 0.9500 | N/A | xgboost |
| 9 | `norm-EFCAMDAT-train_xgboost_063988a0_tfidf` | norm-EFCAMDAT-train | 0.9500 | N/A | xgboost |
| 10 | `norm-EFCAMDAT-train_xgboost_10b21d1b_tfidf` | norm-EFCAMDAT-train | 0.9500 | N/A | xgboost |

## Top 10 Models by Weighted F1-Score

### Strategy: Argmax

| Rank | Model | Dataset | Weighted F1-Score | TF-IDF Config | Classifier |
|------|-------|---------|----------|---------------|------------|
| 1 | `norm-EFCAMDAT-train_xgboost_10b21d1b_tfidf_grouped` | norm-EFCAMDAT-train | 0.9900 | N/A | xgboost |
| 2 | `norm-EFCAMDAT-train_logistic_063988a0_tfidf` | norm-EFCAMDAT-train | 0.9800 | N/A | logistic |
| 3 | `norm-EFCAMDAT-train_logistic_10b21d1b_tfidf_grouped` | norm-EFCAMDAT-train | 0.9800 | N/A | logistic |
| 4 | `norm-EFCAMDAT-train_xgboost_01733819_tfidf` | norm-EFCAMDAT-train | 0.9800 | N/A | xgboost |
| 5 | `norm-EFCAMDAT-train_xgboost_063988a0_tfidf` | norm-EFCAMDAT-train | 0.9800 | N/A | xgboost |
| 6 | `norm-EFCAMDAT-train_xgboost_10b21d1b_tfidf` | norm-EFCAMDAT-train | 0.9800 | N/A | xgboost |
| 7 | `norm-EFCAMDAT-train_xgboost_1387f6a3_tfidf` | norm-EFCAMDAT-train | 0.9800 | N/A | xgboost |
| 8 | `norm-EFCAMDAT-train_xgboost_252cd532_tfidf` | norm-EFCAMDAT-train | 0.9800 | N/A | xgboost |
| 9 | `norm-EFCAMDAT-train_xgboost_336a6205_tfidf` | norm-EFCAMDAT-train | 0.9800 | N/A | xgboost |
| 10 | `norm-EFCAMDAT-train_xgboost_336a6205_tfidf_grouped` | norm-EFCAMDAT-train | 0.9800 | N/A | xgboost |

### Strategy: Rounded Avg

| Rank | Model | Dataset | Weighted F1-Score | TF-IDF Config | Classifier |
|------|-------|---------|----------|---------------|------------|
| 1 | `norm-EFCAMDAT-train_xgboost_10b21d1b_tfidf_grouped` | norm-EFCAMDAT-train | 0.9800 | N/A | xgboost |
| 2 | `norm-EFCAMDAT-train_xgboost_336a6205_tfidf_grouped` | norm-EFCAMDAT-train | 0.9800 | N/A | xgboost |
| 3 | `norm-EFCAMDAT-train_logistic_10b21d1b_tfidf_grouped` | norm-EFCAMDAT-train | 0.9700 | N/A | logistic |
| 4 | `norm-EFCAMDAT-train_xgboost_01733819_tfidf` | norm-EFCAMDAT-train | 0.9700 | N/A | xgboost |
| 5 | `norm-EFCAMDAT-train_xgboost_063988a0_tfidf` | norm-EFCAMDAT-train | 0.9700 | N/A | xgboost |
| 6 | `norm-EFCAMDAT-train_xgboost_10b21d1b_tfidf` | norm-EFCAMDAT-train | 0.9700 | N/A | xgboost |
| 7 | `norm-EFCAMDAT-train_xgboost_1387f6a3_tfidf` | norm-EFCAMDAT-train | 0.9700 | N/A | xgboost |
| 8 | `norm-EFCAMDAT-train_xgboost_252cd532_tfidf` | norm-EFCAMDAT-train | 0.9700 | N/A | xgboost |
| 9 | `norm-EFCAMDAT-train_xgboost_4827cafe_tfidf` | norm-EFCAMDAT-train | 0.9700 | N/A | xgboost |
| 10 | `norm-EFCAMDAT-train_xgboost_57c46ca5_tfidf_grouped` | norm-EFCAMDAT-train | 0.9700 | N/A | xgboost |

## Performance by Dataset

### norm-CELVA-SP

- **Best Accuracy:** 0.3200 (`norm-EFCAMDAT-train_xgboost_063988a0_tfidf`)
- **Best Adjacent Accuracy:** 0.7900 (`norm-EFCAMDAT-train_xgboost_10b21d1b_tfidf_grouped`)
- **Models Evaluated:** 48

### norm-EFCAMDAT-test

- **Best Accuracy:** 0.9600 (`norm-EFCAMDAT-train_logistic_01733819_tfidf`)
- **Best Adjacent Accuracy:** 0.9900 (`norm-EFCAMDAT-train_logistic_01733819_tfidf`)
- **Models Evaluated:** 48

### norm-EFCAMDAT-train

- **Best Accuracy:** 0.9900 (`norm-EFCAMDAT-train_xgboost_10b21d1b_tfidf_grouped`)
- **Best Adjacent Accuracy:** 0.9900 (`norm-EFCAMDAT-train_logistic_005ebc16_tfidf`)
- **Models Evaluated:** 48

### norm-KUPA-KEYS

- **Best Accuracy:** 0.4900 (`norm-EFCAMDAT-train_logistic_53d29bb1_tfidf_grouped`)
- **Best Adjacent Accuracy:** 0.9500 (`norm-EFCAMDAT-train_logistic_063988a0_tfidf`)
- **Models Evaluated:** 48

## TF-IDF Configuration Analysis
