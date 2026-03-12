# Evaluation Report: norm-CELVA-SP

**Classifier**: ppl_native_general_xgboost
**Dataset**: norm-CELVA-SP
**Samples**: 1742
**Classes in test set**: A1, A2, B1, B2, C1, C2

## Strategy 1: Argmax Predictions

Standard argmax strategy: predict class with highest probability.

### CEFR Classification Report

```
    precision    recall  f1-score   support

A1       0.54      0.14      0.22       157
A2       0.46      0.09      0.14       511
B1       0.32      0.12      0.18       609
B2       0.24      0.74      0.36       353
C1       0.07      0.19      0.10       100
C2       0.00      0.00      0.00        12

macro avg       0.27      0.21      0.17      1742
weighted avg       0.35      0.24      0.20      1742

accuracy      0.24      1742
adjacent accuracy      0.66      1742
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          A1       0.54      0.14      0.22       157
          A2       0.46      0.09      0.14       511
          B1       0.32      0.12      0.18       609
          B2       0.24      0.74      0.36       353
          C1       0.07      0.19      0.10       100
          C2       0.00      0.00      0.00        12

    accuracy                           0.24      1742
   macro avg       0.27      0.21      0.17      1742
weighted avg       0.35      0.24      0.20      1742
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

```
[[ 22  29  42  53  11   0]
 [ 11  44  95 284  77   0]
 [  7  17  76 402 107   0]
 [  0   5  21 261  66   0]
 [  0   1   4  76  19   0]
 [  1   0   0   7   4   0]]
```

### Calibration Report

```
CEFR CALIBRATION REPORT (Soft Probabilities)
================================================================================

Expected Calibration Error (ECE): 0.50

Confidence Bins:
Range                     Count     Accuracy   Confidence          Gap
--------------------------------------------------------------------------------
(0.30, 0.40]                  8         0.25         0.37         0.12
(0.40, 0.50]                 65         0.22         0.46         0.25
(0.50, 0.60]                309         0.20         0.55         0.35
(0.60, 0.70]                317         0.26         0.65         0.39
(0.70, 0.80]                322         0.24         0.75         0.51
(0.80, 0.90]                404         0.26         0.85         0.59
(0.90, 1.00]                317         0.25         0.94         0.69

Per-Class Calibration:
Class               Avg Prob     Accuracy      Count
--------------------------------------------------------------------------------
A1                      0.75         0.54         41
A2                      0.64         0.46         96
B1                      0.69         0.32        238
B2                      0.78         0.24       1083
C1                      0.67         0.07        284
```

## Strategy 2: Rounded Average Predictions

Regression-style strategy: calculate expected class index from probabilities, round to nearest integer, map back to class label.

### CEFR Classification Report

```
    precision    recall  f1-score   support

A1       0.60      0.11      0.19       157
A2       0.44      0.08      0.13       511
B1       0.32      0.14      0.19       609
B2       0.24      0.76      0.37       353
C1       0.07      0.17      0.10       100
C2       0.00      0.00      0.00        12

macro avg       0.28      0.21      0.16      1742
weighted avg       0.35      0.25      0.20      1742

accuracy      0.25      1742
adjacent accuracy      0.67      1742
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          A1       0.60      0.11      0.19       157
          A2       0.44      0.08      0.13       511
          B1       0.32      0.14      0.19       609
          B2       0.24      0.76      0.37       353
          C1       0.07      0.17      0.10       100
          C2       0.00      0.00      0.00        12

    accuracy                           0.25      1742
   macro avg       0.28      0.21      0.16      1742
weighted avg       0.35      0.25      0.20      1742
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

```
[[ 18  30  46  55   8   0]
 [  8  40 103 295  65   0]
 [  4  17  83 411  94   0]
 [  0   3  22 269  59   0]
 [  0   0   5  78  17   0]
 [  0   1   0   7   4   0]]
```

