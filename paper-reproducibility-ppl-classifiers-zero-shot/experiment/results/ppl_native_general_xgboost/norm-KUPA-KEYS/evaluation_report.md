# Evaluation Report: norm-KUPA-KEYS

**Classifier**: ppl_native_general_xgboost
**Dataset**: norm-KUPA-KEYS
**Samples**: 1006
**Classes in test set**: B1, B2, C1, C2

## Strategy 1: Argmax Predictions

Standard argmax strategy: predict class with highest probability.

### CEFR Classification Report

```
    precision    recall  f1-score   support

B1       0.33      0.01      0.02       109
B2       0.58      0.81      0.67       570
C1       0.35      0.23      0.28       312
C2       0.00      0.00      0.00        15

macro avg       0.32      0.26      0.24      1006
weighted avg       0.47      0.53      0.47      1006

accuracy      0.53      1006
adjacent accuracy      0.97      1006
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          B1       0.33      0.01      0.02       109
          B2       0.58      0.81      0.67       570
          C1       0.35      0.23      0.28       312
          C2       0.00      0.00      0.00        15

    accuracy                           0.53      1006
   macro avg       0.32      0.26      0.24      1006
weighted avg       0.47      0.53      0.47      1006
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['B1', 'B2', 'C1', 'C2']

```
[[  1  89  19   0]
 [  2 459 109   0]
 [  0 239  73   0]
 [  0   7   8   0]]
```

### Calibration Report

```
CEFR CALIBRATION REPORT (Soft Probabilities)
================================================================================

Expected Calibration Error (ECE): 0.24

Confidence Bins:
Range                     Count     Accuracy   Confidence          Gap
--------------------------------------------------------------------------------
(0.40, 0.50]                 27         0.48         0.48        -0.00
(0.50, 0.60]                150         0.49         0.55         0.06
(0.60, 0.70]                157         0.51         0.65         0.14
(0.70, 0.80]                188         0.52         0.75         0.23
(0.80, 0.90]                262         0.55         0.85         0.30
(0.90, 1.00]                222         0.55         0.94         0.39

Per-Class Calibration:
Class               Avg Prob     Accuracy      Count
--------------------------------------------------------------------------------
B1                      0.54         0.33          3
B2                      0.79         0.58        794
C1                      0.69         0.35        209
```

## Strategy 2: Rounded Average Predictions

Regression-style strategy: calculate expected class index from probabilities, round to nearest integer, map back to class label.

### CEFR Classification Report

```
    precision    recall  f1-score   support

B1       0.00      0.00      0.00       109
B2       0.58      0.83      0.68       570
C1       0.37      0.22      0.28       312
C2       0.00      0.00      0.00        15

macro avg       0.24      0.26      0.24      1006
weighted avg       0.44      0.54      0.47      1006

accuracy      0.54      1006
adjacent accuracy      0.98      1006
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          B1       0.00      0.00      0.00       109
          B2       0.58      0.83      0.68       570
          C1       0.37      0.22      0.28       312
          C2       0.00      0.00      0.00        15

    accuracy                           0.54      1006
   macro avg       0.24      0.26      0.24      1006
weighted avg       0.44      0.54      0.47      1006
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['B1', 'B2', 'C1', 'C2']

```
[[  0  94  15   0]
 [  1 474  95   0]
 [  0 243  69   0]
 [  0   8   7   0]]
```

