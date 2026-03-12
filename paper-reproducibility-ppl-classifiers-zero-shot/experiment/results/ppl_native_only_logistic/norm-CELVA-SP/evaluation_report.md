# Evaluation Report: norm-CELVA-SP

**Classifier**: ppl_native_only_logistic
**Dataset**: norm-CELVA-SP
**Samples**: 1742
**Classes in test set**: A1, A2, B1, B2, C1, C2

## Strategy 1: Argmax Predictions

Standard argmax strategy: predict class with highest probability.

### CEFR Classification Report

```
    precision    recall  f1-score   support

A1       0.58      0.11      0.19       157
A2       0.45      0.07      0.13       511
B1       0.27      0.09      0.13       609
B2       0.25      0.73      0.37       353
C1       0.03      0.11      0.04       100
C2       0.00      0.00      0.00        12

macro avg       0.26      0.19      0.14      1742
weighted avg       0.33      0.22      0.18      1742

accuracy      0.22      1742
adjacent accuracy      0.62      1742
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          A1       0.58      0.11      0.19       157
          A2       0.45      0.07      0.13       511
          B1       0.27      0.09      0.13       609
          B2       0.25      0.73      0.37       353
          C1       0.03      0.11      0.04       100
          C2       0.00      0.00      0.00        12

    accuracy                           0.22      1742
   macro avg       0.26      0.19      0.14      1742
weighted avg       0.33      0.22      0.18      1742
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

```
[[ 18  24  40  50  25   0]
 [  8  38  81 243 141   0]
 [  5  16  52 391 145   0]
 [  0   5  16 258  74   0]
 [  0   0   3  86  11   0]
 [  0   1   0   8   3   0]]
```

### Calibration Report

```
CEFR CALIBRATION REPORT (Soft Probabilities)
================================================================================

Expected Calibration Error (ECE): 0.59

Confidence Bins:
Range                     Count     Accuracy   Confidence          Gap
--------------------------------------------------------------------------------
(0.30, 0.40]                  4         0.25         0.38         0.13
(0.40, 0.50]                 82         0.26         0.47         0.21
(0.50, 0.60]                245         0.18         0.55         0.37
(0.60, 0.70]                239         0.19         0.65         0.46
(0.70, 0.80]                211         0.16         0.75         0.59
(0.80, 0.90]                162         0.13         0.85         0.72
(0.90, 1.00]                799         0.26         0.98         0.72

Per-Class Calibration:
Class               Avg Prob     Accuracy      Count
--------------------------------------------------------------------------------
A1                      0.76         0.58         31
A2                      0.62         0.45         84
B1                      0.63         0.27        192
B2                      0.87         0.25       1036
C1                      0.79         0.03        399
```

## Strategy 2: Rounded Average Predictions

Regression-style strategy: calculate expected class index from probabilities, round to nearest integer, map back to class label.

### CEFR Classification Report

```
    precision    recall  f1-score   support

A1       0.61      0.11      0.18       157
A2       0.45      0.06      0.11       511
B1       0.27      0.08      0.13       609
B2       0.24      0.74      0.36       353
C1       0.03      0.11      0.05       100
C2       0.00      0.00      0.00        12

macro avg       0.27      0.18      0.14      1742
weighted avg       0.33      0.21      0.17      1742

accuracy      0.21      1742
adjacent accuracy      0.62      1742
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          A1       0.61      0.11      0.18       157
          A2       0.45      0.06      0.11       511
          B1       0.27      0.08      0.13       609
          B2       0.24      0.74      0.36       353
          C1       0.03      0.11      0.05       100
          C2       0.00      0.00      0.00        12

    accuracy                           0.21      1742
   macro avg       0.27      0.18      0.14      1742
weighted avg       0.33      0.21      0.17      1742
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

```
[[ 17  23  40  52  25   0]
 [  8  33  78 260 132   0]
 [  3  13  51 405 137   0]
 [  0   3  17 260  73   0]
 [  0   0   2  87  11   0]
 [  0   1   0   8   3   0]]
```

