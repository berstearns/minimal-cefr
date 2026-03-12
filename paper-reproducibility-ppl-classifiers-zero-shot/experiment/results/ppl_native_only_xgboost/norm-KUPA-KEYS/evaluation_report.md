# Evaluation Report: norm-KUPA-KEYS

**Classifier**: ppl_native_only_xgboost
**Dataset**: norm-KUPA-KEYS
**Samples**: 1006
**Classes in test set**: B1, B2, C1, C2

## Strategy 1: Argmax Predictions

Standard argmax strategy: predict class with highest probability.

### CEFR Classification Report

```
    precision    recall  f1-score   support

B1       0.00      0.00      0.00       109
B2       0.58      0.82      0.68       570
C1       0.39      0.25      0.31       312
C2       0.00      0.00      0.00        15

macro avg       0.24      0.27      0.25      1006
weighted avg       0.45      0.54      0.48      1006

accuracy      0.54      1006
adjacent accuracy      0.97      1006
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          B1       0.00      0.00      0.00       109
          B2       0.58      0.82      0.68       570
          C1       0.39      0.25      0.31       312
          C2       0.00      0.00      0.00        15

    accuracy                           0.54      1006
   macro avg       0.24      0.27      0.25      1006
weighted avg       0.45      0.54      0.48      1006
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['B1', 'B2', 'C1', 'C2']

```
[[  0  93  16   0]
 [  2 468 100   0]
 [  0 234  78   0]
 [  0  10   5   0]]
```

### Calibration Report

```
CEFR CALIBRATION REPORT (Soft Probabilities)
================================================================================

Expected Calibration Error (ECE): 0.22

Confidence Bins:
Range                     Count     Accuracy   Confidence          Gap
--------------------------------------------------------------------------------
(0.30, 0.40]                  3         0.00         0.39         0.39
(0.40, 0.50]                 26         0.35         0.48         0.14
(0.50, 0.60]                145         0.50         0.55         0.05
(0.60, 0.70]                142         0.56         0.65         0.09
(0.70, 0.80]                206         0.50         0.75         0.25
(0.80, 0.90]                271         0.59         0.85         0.25
(0.90, 1.00]                213         0.57         0.94         0.37

Per-Class Calibration:
Class               Avg Prob     Accuracy      Count
--------------------------------------------------------------------------------
B1                      0.65         0.00          2
B2                      0.79         0.58        805
C1                      0.67         0.39        199
```

## Strategy 2: Rounded Average Predictions

Regression-style strategy: calculate expected class index from probabilities, round to nearest integer, map back to class label.

### CEFR Classification Report

```
    precision    recall  f1-score   support

B1       0.00      0.00      0.00       109
B2       0.59      0.87      0.70       570
C1       0.43      0.22      0.30       312
C2       0.00      0.00      0.00        15

macro avg       0.25      0.27      0.25      1006
weighted avg       0.47      0.56      0.49      1006

accuracy      0.56      1006
adjacent accuracy      0.98      1006
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          B1       0.00      0.00      0.00       109
          B2       0.59      0.87      0.70       570
          C1       0.43      0.22      0.30       312
          C2       0.00      0.00      0.00        15

    accuracy                           0.56      1006
   macro avg       0.25      0.27      0.25      1006
weighted avg       0.47      0.56      0.49      1006
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['B1', 'B2', 'C1', 'C2']

```
[[  0  97  12   0]
 [  1 494  75   0]
 [  1 241  70   0]
 [  0  10   5   0]]
```

