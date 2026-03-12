# Evaluation Report: norm-KUPA-KEYS

**Classifier**: ppl_all_models_xgboost
**Dataset**: norm-KUPA-KEYS
**Samples**: 1006
**Classes in test set**: B1, B2, C1, C2

## Strategy 1: Argmax Predictions

Standard argmax strategy: predict class with highest probability.

### CEFR Classification Report

```
    precision    recall  f1-score   support

B1       0.17      0.01      0.02       109
B2       0.62      0.57      0.59       570
C1       0.41      0.63      0.50       312
C2       0.00      0.00      0.00        15

macro avg       0.30      0.30      0.28      1006
weighted avg       0.50      0.52      0.49      1006

accuracy      0.52      1006
adjacent accuracy      0.97      1006
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          B1       0.17      0.01      0.02       109
          B2       0.62      0.57      0.59       570
          C1       0.41      0.63      0.50       312
          C2       0.00      0.00      0.00        15

    accuracy                           0.52      1006
   macro avg       0.30      0.30      0.28      1006
weighted avg       0.50      0.52      0.49      1006
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['B1', 'B2', 'C1', 'C2']

```
[[  1  81  27   0]
 [  5 323 242   0]
 [  0 114 198   0]
 [  0   1  14   0]]
```

### Calibration Report

```
CEFR CALIBRATION REPORT (Soft Probabilities)
================================================================================

Expected Calibration Error (ECE): 0.28

Confidence Bins:
Range                     Count     Accuracy   Confidence          Gap
--------------------------------------------------------------------------------
(0.30, 0.40]                  2         0.50         0.38        -0.12
(0.40, 0.50]                 23         0.43         0.47         0.04
(0.50, 0.60]                123         0.37         0.56         0.19
(0.60, 0.70]                138         0.50         0.65         0.15
(0.70, 0.80]                151         0.46         0.75         0.30
(0.80, 0.90]                201         0.51         0.85         0.34
(0.90, 1.00]                368         0.61         0.96         0.35

Per-Class Calibration:
Class               Avg Prob     Accuracy      Count
--------------------------------------------------------------------------------
B1                      0.59         0.17          6
B2                      0.79         0.62        519
C1                      0.82         0.41        481
```

## Strategy 2: Rounded Average Predictions

Regression-style strategy: calculate expected class index from probabilities, round to nearest integer, map back to class label.

### CEFR Classification Report

```
    precision    recall  f1-score   support

B1       0.33      0.01      0.02       109
B2       0.63      0.61      0.62       570
C1       0.43      0.63      0.51       312
C2       0.00      0.00      0.00        15

macro avg       0.35      0.31      0.29      1006
weighted avg       0.53      0.54      0.51      1006

accuracy      0.54      1006
adjacent accuracy      0.98      1006
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          B1       0.33      0.01      0.02       109
          B2       0.63      0.61      0.62       570
          C1       0.43      0.63      0.51       312
          C2       0.00      0.00      0.00        15

    accuracy                           0.54      1006
   macro avg       0.35      0.31      0.29      1006
weighted avg       0.53      0.54      0.51      1006
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['B1', 'B2', 'C1', 'C2']

```
[[  1  85  23   0]
 [  2 345 223   0]
 [  0 116 196   0]
 [  0   1  14   0]]
```

