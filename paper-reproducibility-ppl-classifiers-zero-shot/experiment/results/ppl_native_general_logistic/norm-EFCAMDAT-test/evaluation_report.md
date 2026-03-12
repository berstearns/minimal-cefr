# Evaluation Report: norm-EFCAMDAT-test

**Classifier**: ppl_native_general_logistic
**Dataset**: norm-EFCAMDAT-test
**Samples**: 20001
**Classes in test set**: A1, A2, B1, B2, C1

## Strategy 1: Argmax Predictions

Standard argmax strategy: predict class with highest probability.

### CEFR Classification Report

```
    precision    recall  f1-score   support

A1       0.47      0.46      0.47      9437
A2       0.29      0.28      0.29      5937
B1       0.16      0.16      0.16      3240
B2       0.05      0.05      0.05      1112
C1       0.02      0.04      0.03       275

macro avg       0.20      0.20      0.20     20001
weighted avg       0.34      0.33      0.33     20001

accuracy      0.33     20001
adjacent accuracy      0.72     20001
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          A1       0.47      0.46      0.47      9437
          A2       0.29      0.28      0.29      5937
          B1       0.16      0.16      0.16      3240
          B2       0.05      0.05      0.05      1112
          C1       0.02      0.04      0.03       275

    accuracy                           0.33     20001
   macro avg       0.20      0.20      0.20     20001
weighted avg       0.34      0.33      0.33     20001
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['A1', 'A2', 'B1', 'B2', 'C1']

```
[[4353 2719 1568  575  222]
 [2736 1680 1005  371  145]
 [1526  930  520  193   71]
 [ 509  326  185   59   33]
 [ 129   82   39   14   11]]
```

### Calibration Report

```
CEFR CALIBRATION REPORT (Soft Probabilities)
================================================================================

Expected Calibration Error (ECE): 0.43

Confidence Bins:
Range                     Count     Accuracy   Confidence          Gap
--------------------------------------------------------------------------------
(0.20, 0.30]                  3         0.00         0.28         0.28
(0.30, 0.40]                150         0.19         0.37         0.18
(0.40, 0.50]               1382         0.23         0.46         0.24
(0.50, 0.60]               2941         0.27         0.55         0.28
(0.60, 0.70]               3062         0.29         0.65         0.36
(0.70, 0.80]               3168         0.30         0.75         0.45
(0.80, 0.90]               3757         0.35         0.85         0.50
(0.90, 1.00]               5538         0.42         0.95         0.53

Per-Class Calibration:
Class               Avg Prob     Accuracy      Count
--------------------------------------------------------------------------------
A1                      0.84         0.47       9253
A2                      0.70         0.29       5737
B1                      0.66         0.16       3317
B2                      0.66         0.05       1212
C1                      0.74         0.02        482
```

## Strategy 2: Rounded Average Predictions

Regression-style strategy: calculate expected class index from probabilities, round to nearest integer, map back to class label.

### CEFR Classification Report

```
    precision    recall  f1-score   support

A1       0.47      0.43      0.45      9437
A2       0.29      0.30      0.30      5937
B1       0.16      0.17      0.16      3240
B2       0.05      0.06      0.05      1112
C1       0.03      0.04      0.03       275

macro avg       0.20      0.20      0.20     20001
weighted avg       0.34      0.32      0.33     20001

accuracy      0.32     20001
adjacent accuracy      0.72     20001
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          A1       0.47      0.43      0.45      9437
          A2       0.29      0.30      0.30      5937
          B1       0.16      0.17      0.16      3240
          B2       0.05      0.06      0.05      1112
          C1       0.03      0.04      0.03       275

    accuracy                           0.32     20001
   macro avg       0.20      0.20      0.20     20001
weighted avg       0.34      0.32      0.33     20001
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['A1', 'A2', 'B1', 'B2', 'C1']

```
[[4044 2910 1662  636  185]
 [2570 1795 1042  408  122]
 [1435  988  541  215   61]
 [ 473  357  190   63   29]
 [ 119   89   42   14   11]]
```

