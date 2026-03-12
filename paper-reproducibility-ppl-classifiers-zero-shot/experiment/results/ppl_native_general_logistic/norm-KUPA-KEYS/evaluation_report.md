# Evaluation Report: norm-KUPA-KEYS

**Classifier**: ppl_native_general_logistic
**Dataset**: norm-KUPA-KEYS
**Samples**: 1006
**Classes in test set**: B1, B2, C1, C2

## Strategy 1: Argmax Predictions

Standard argmax strategy: predict class with highest probability.

### CEFR Classification Report

```
    precision    recall  f1-score   support

B1       0.00      0.00      0.00       109
B2       0.52      0.55      0.53       570
C1       0.24      0.30      0.27       312
C2       0.00      0.00      0.00        15

macro avg       0.19      0.21      0.20      1006
weighted avg       0.37      0.40      0.38      1006

accuracy      0.40      1006
adjacent accuracy      0.94      1006
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          B1       0.00      0.00      0.00       109
          B2       0.52      0.55      0.53       570
          C1       0.24      0.30      0.27       312
          C2       0.00      0.00      0.00        15

    accuracy                           0.40      1006
   macro avg       0.19      0.21      0.20      1006
weighted avg       0.37      0.40      0.38      1006
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['B1', 'B2', 'C1', 'C2']

```
[[  0  60  49   0]
 [  0 312 258   0]
 [  0 217  95   0]
 [  0  15   0   0]]
```

### Calibration Report

```
CEFR CALIBRATION REPORT (Soft Probabilities)
================================================================================

Expected Calibration Error (ECE): 0.44

Confidence Bins:
Range                     Count     Accuracy   Confidence          Gap
--------------------------------------------------------------------------------
(0.40, 0.50]                  7         0.14         0.48         0.34
(0.50, 0.60]                 89         0.38         0.54         0.16
(0.60, 0.70]                122         0.41         0.65         0.24
(0.70, 0.80]                139         0.42         0.75         0.33
(0.80, 0.90]                145         0.42         0.85         0.43
(0.90, 1.00]                504         0.40         0.97         0.57

Per-Class Calibration:
Class               Avg Prob     Accuracy      Count
--------------------------------------------------------------------------------
B2                      0.88         0.52        604
C1                      0.79         0.24        402
```

## Strategy 2: Rounded Average Predictions

Regression-style strategy: calculate expected class index from probabilities, round to nearest integer, map back to class label.

### CEFR Classification Report

```
    precision    recall  f1-score   support

B1       0.00      0.00      0.00       109
B2       0.52      0.56      0.54       570
C1       0.24      0.29      0.26       312
C2       0.00      0.00      0.00        15

macro avg       0.19      0.21      0.20      1006
weighted avg       0.37      0.41      0.39      1006

accuracy      0.41      1006
adjacent accuracy      0.94      1006
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          B1       0.00      0.00      0.00       109
          B2       0.52      0.56      0.54       570
          C1       0.24      0.29      0.26       312
          C2       0.00      0.00      0.00        15

    accuracy                           0.41      1006
   macro avg       0.19      0.21      0.20      1006
weighted avg       0.37      0.41      0.39      1006
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['B1', 'B2', 'C1', 'C2']

```
[[  0  62  47   0]
 [  0 321 249   0]
 [  0 220  92   0]
 [  0  15   0   0]]
```

