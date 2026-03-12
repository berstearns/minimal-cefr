# Evaluation Report: norm-CELVA-SP

**Classifier**: ppl_all_models_logistic
**Dataset**: norm-CELVA-SP
**Samples**: 1742
**Classes in test set**: A1, A2, B1, B2, C1, C2

## Strategy 1: Argmax Predictions

Standard argmax strategy: predict class with highest probability.

### CEFR Classification Report

```
    precision    recall  f1-score   support

A1       0.59      0.08      0.15       157
A2       0.40      0.07      0.11       511
B1       0.29      0.13      0.18       609
B2       0.23      0.67      0.34       353
C1       0.07      0.22      0.10       100
C2       0.00      0.00      0.00        12

macro avg       0.26      0.20      0.15      1742
weighted avg       0.32      0.22      0.19      1742

accuracy      0.22      1742
adjacent accuracy      0.65      1742
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          A1       0.59      0.08      0.15       157
          A2       0.40      0.07      0.11       511
          B1       0.29      0.13      0.18       609
          B2       0.23      0.67      0.34       353
          C1       0.07      0.22      0.10       100
          C2       0.00      0.00      0.00        12

    accuracy                           0.22      1742
   macro avg       0.26      0.20      0.15      1742
weighted avg       0.32      0.22      0.19      1742
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

```
[[ 13  27  51  51  15   0]
 [  5  34 113 266  93   0]
 [  4  16  80 394 115   0]
 [  0   6  28 237  82   0]
 [  0   1   6  71  22   0]
 [  0   0   0   7   5   0]]
```

### Calibration Report

```
CEFR CALIBRATION REPORT (Soft Probabilities)
================================================================================

Expected Calibration Error (ECE): 0.72

Confidence Bins:
Range                     Count     Accuracy   Confidence          Gap
--------------------------------------------------------------------------------
(0.30, 0.40]                  4         0.75         0.37        -0.38
(0.40, 0.50]                 17         0.24         0.47         0.23
(0.50, 0.60]                 54         0.20         0.55         0.35
(0.60, 0.70]                 70         0.20         0.65         0.45
(0.70, 0.80]                 73         0.23         0.76         0.53
(0.80, 0.90]                127         0.24         0.86         0.62
(0.90, 1.00]               1397         0.22         0.99         0.77

Per-Class Calibration:
Class               Avg Prob     Accuracy      Count
--------------------------------------------------------------------------------
A1                      0.85         0.59         22
A2                      0.83         0.40         84
B1                      0.87         0.29        278
B2                      0.96         0.23       1026
C1                      0.94         0.07        332
```

## Strategy 2: Rounded Average Predictions

Regression-style strategy: calculate expected class index from probabilities, round to nearest integer, map back to class label.

### CEFR Classification Report

```
    precision    recall  f1-score   support

A1       0.60      0.08      0.14       157
A2       0.43      0.07      0.12       511
B1       0.30      0.14      0.19       609
B2       0.23      0.67      0.34       353
C1       0.07      0.22      0.10       100
C2       0.00      0.00      0.00        12

macro avg       0.27      0.20      0.15      1742
weighted avg       0.33      0.23      0.19      1742

accuracy      0.23      1742
adjacent accuracy      0.65      1742
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          A1       0.60      0.08      0.14       157
          A2       0.43      0.07      0.12       511
          B1       0.30      0.14      0.19       609
          B2       0.23      0.67      0.34       353
          C1       0.07      0.22      0.10       100
          C2       0.00      0.00      0.00        12

    accuracy                           0.23      1742
   macro avg       0.27      0.20      0.15      1742
weighted avg       0.33      0.23      0.19      1742
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

```
[[ 12  27  54  50  14   0]
 [  4  37 110 271  89   0]
 [  4  16  85 392 112   0]
 [  0   6  27 238  82   0]
 [  0   1   6  71  22   0]
 [  0   0   0   7   5   0]]
```

