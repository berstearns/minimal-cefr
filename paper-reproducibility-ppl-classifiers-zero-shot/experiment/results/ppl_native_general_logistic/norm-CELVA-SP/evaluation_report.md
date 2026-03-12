# Evaluation Report: norm-CELVA-SP

**Classifier**: ppl_native_general_logistic
**Dataset**: norm-CELVA-SP
**Samples**: 1742
**Classes in test set**: A1, A2, B1, B2, C1, C2

## Strategy 1: Argmax Predictions

Standard argmax strategy: predict class with highest probability.

### CEFR Classification Report

```
    precision    recall  f1-score   support

A1       0.57      0.10      0.17       157
A2       0.42      0.07      0.12       511
B1       0.25      0.06      0.10       609
B2       0.25      0.74      0.37       353
C1       0.04      0.15      0.06       100
C2       0.00      0.00      0.00        12

macro avg       0.25      0.19      0.14      1742
weighted avg       0.32      0.21      0.16      1742

accuracy      0.21      1742
adjacent accuracy      0.61      1742
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          A1       0.57      0.10      0.17       157
          A2       0.42      0.07      0.12       511
          B1       0.25      0.06      0.10       609
          B2       0.25      0.74      0.37       353
          C1       0.04      0.15      0.06       100
          C2       0.00      0.00      0.00        12

    accuracy                           0.21      1742
   macro avg       0.25      0.19      0.14      1742
weighted avg       0.32      0.21      0.16      1742
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

```
[[ 16  28  36  52  25   0]
 [  9  35  63 253 151   0]
 [  3  16  37 399 154   0]
 [  0   3  10 260  80   0]
 [  0   0   0  85  15   0]
 [  0   1   0   9   2   0]]
```

### Calibration Report

```
CEFR CALIBRATION REPORT (Soft Probabilities)
================================================================================

Expected Calibration Error (ECE): 0.60

Confidence Bins:
Range                     Count     Accuracy   Confidence          Gap
--------------------------------------------------------------------------------
(0.20, 0.30]                  1         0.00         0.27         0.27
(0.30, 0.40]                 13         0.23         0.38         0.15
(0.40, 0.50]                 95         0.26         0.46         0.20
(0.50, 0.60]                224         0.19         0.55         0.36
(0.60, 0.70]                208         0.17         0.65         0.48
(0.70, 0.80]                196         0.19         0.75         0.56
(0.80, 0.90]                212         0.17         0.85         0.69
(0.90, 1.00]                793         0.23         0.98         0.75

Per-Class Calibration:
Class               Avg Prob     Accuracy      Count
--------------------------------------------------------------------------------
A1                      0.74         0.57         28
A2                      0.60         0.42         83
B1                      0.62         0.25        146
B2                      0.86         0.25       1058
C1                      0.80         0.04        427
```

## Strategy 2: Rounded Average Predictions

Regression-style strategy: calculate expected class index from probabilities, round to nearest integer, map back to class label.

### CEFR Classification Report

```
    precision    recall  f1-score   support

A1       0.56      0.09      0.15       157
A2       0.40      0.05      0.09       511
B1       0.24      0.07      0.11       609
B2       0.24      0.74      0.36       353
C1       0.04      0.14      0.06       100
C2       0.00      0.00      0.00        12

macro avg       0.25      0.18      0.13      1742
weighted avg       0.30      0.20      0.15      1742

accuracy      0.20      1742
adjacent accuracy      0.61      1742
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          A1       0.56      0.09      0.15       157
          A2       0.40      0.05      0.09       511
          B1       0.24      0.07      0.11       609
          B2       0.24      0.74      0.36       353
          C1       0.04      0.14      0.06       100
          C2       0.00      0.00      0.00        12

    accuracy                           0.20      1742
   macro avg       0.25      0.18      0.13      1742
weighted avg       0.30      0.20      0.15      1742
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

```
[[ 14  25  40  55  23   0]
 [  8  25  76 263 139   0]
 [  3  10  41 412 143   0]
 [  0   2  11 262  78   0]
 [  0   0   0  86  14   0]
 [  0   0   1   9   2   0]]
```

