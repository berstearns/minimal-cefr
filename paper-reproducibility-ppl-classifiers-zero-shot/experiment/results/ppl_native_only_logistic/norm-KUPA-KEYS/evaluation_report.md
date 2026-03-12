# Evaluation Report: norm-KUPA-KEYS

**Classifier**: ppl_native_only_logistic
**Dataset**: norm-KUPA-KEYS
**Samples**: 1006
**Classes in test set**: B1, B2, C1, C2

## Strategy 1: Argmax Predictions

Standard argmax strategy: predict class with highest probability.

### CEFR Classification Report

```
    precision    recall  f1-score   support

B1       0.00      0.00      0.00       109
B2       0.53      0.63      0.57       570
C1       0.22      0.23      0.22       312
C2       0.00      0.00      0.00        15

macro avg       0.19      0.21      0.20      1006
weighted avg       0.37      0.43      0.39      1006

accuracy      0.43      1006
adjacent accuracy      0.94      1006
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          B1       0.00      0.00      0.00       109
          B2       0.53      0.63      0.57       570
          C1       0.22      0.23      0.22       312
          C2       0.00      0.00      0.00        15

    accuracy                           0.43      1006
   macro avg       0.19      0.21      0.20      1006
weighted avg       0.37      0.43      0.39      1006
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['B1', 'B2', 'C1', 'C2']

```
[[  0  67  42   0]
 [  1 358 211   0]
 [  0 241  71   0]
 [  0  15   0   0]]
```

### Calibration Report

```
CEFR CALIBRATION REPORT (Soft Probabilities)
================================================================================

Expected Calibration Error (ECE): 0.41

Confidence Bins:
Range                     Count     Accuracy   Confidence          Gap
--------------------------------------------------------------------------------
(0.40, 0.50]                 15         0.60         0.49        -0.11
(0.50, 0.60]                133         0.46         0.55         0.09
(0.60, 0.70]                120         0.40         0.65         0.25
(0.70, 0.80]                140         0.41         0.75         0.33
(0.80, 0.90]                119         0.39         0.85         0.46
(0.90, 1.00]                479         0.43         0.98         0.55

Per-Class Calibration:
Class               Avg Prob     Accuracy      Count
--------------------------------------------------------------------------------
B1                      0.59         0.00          1
B2                      0.87         0.53        681
C1                      0.74         0.22        324
```

## Strategy 2: Rounded Average Predictions

Regression-style strategy: calculate expected class index from probabilities, round to nearest integer, map back to class label.

### CEFR Classification Report

```
    precision    recall  f1-score   support

B1       0.00      0.00      0.00       109
B2       0.53      0.65      0.58       570
C1       0.21      0.21      0.21       312
C2       0.00      0.00      0.00        15

macro avg       0.18      0.21      0.20      1006
weighted avg       0.36      0.43      0.39      1006

accuracy      0.43      1006
adjacent accuracy      0.94      1006
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          B1       0.00      0.00      0.00       109
          B2       0.53      0.65      0.58       570
          C1       0.21      0.21      0.21       312
          C2       0.00      0.00      0.00        15

    accuracy                           0.43      1006
   macro avg       0.18      0.21      0.20      1006
weighted avg       0.36      0.43      0.39      1006
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['B1', 'B2', 'C1', 'C2']

```
[[  0  68  41   0]
 [  1 370 199   0]
 [  0 248  64   0]
 [  0  15   0   0]]
```

