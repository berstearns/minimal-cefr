# Evaluation Report: norm-EFCAMDAT-test

**Classifier**: ppl_native_only_xgboost
**Dataset**: norm-EFCAMDAT-test
**Samples**: 20001
**Classes in test set**: A1, A2, B1, B2, C1

## Strategy 1: Argmax Predictions

Standard argmax strategy: predict class with highest probability.

### CEFR Classification Report

```
    precision    recall  f1-score   support

A1       0.47      0.48      0.47      9437
A2       0.30      0.29      0.30      5937
B1       0.16      0.17      0.16      3240
B2       0.06      0.05      0.05      1112
C1       0.04      0.02      0.03       275

macro avg       0.20      0.20      0.20     20001
weighted avg       0.34      0.34      0.34     20001

accuracy      0.34     20001
adjacent accuracy      0.74     20001
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          A1       0.47      0.48      0.47      9437
          A2       0.30      0.29      0.30      5937
          B1       0.16      0.17      0.16      3240
          B2       0.06      0.05      0.05      1112
          C1       0.04      0.02      0.03       275

    accuracy                           0.34     20001
   macro avg       0.20      0.20      0.20     20001
weighted avg       0.34      0.34      0.34     20001
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['A1', 'A2', 'B1', 'B2', 'C1']

```
[[4511 2787 1628  439   72]
 [2833 1748 1017  296   43]
 [1598  919  537  160   26]
 [ 527  342  178   53   12]
 [ 133   83   42   11    6]]
```

### Calibration Report

```
CEFR CALIBRATION REPORT (Soft Probabilities)
================================================================================

Expected Calibration Error (ECE): 0.51

Confidence Bins:
Range                     Count     Accuracy   Confidence          Gap
--------------------------------------------------------------------------------
(0.20, 0.30]                  1         0.00         0.30         0.30
(0.30, 0.40]                 34         0.24         0.38         0.14
(0.40, 0.50]                409         0.24         0.47         0.23
(0.50, 0.60]               1595         0.27         0.55         0.28
(0.60, 0.70]               1767         0.26         0.65         0.40
(0.70, 0.80]               2289         0.28         0.75         0.47
(0.80, 0.90]               3262         0.29         0.85         0.57
(0.90, 1.00]              10644         0.40         0.97         0.57

Per-Class Calibration:
Class               Avg Prob     Accuracy      Count
--------------------------------------------------------------------------------
A1                      0.92         0.47       9602
A2                      0.81         0.30       5879
B1                      0.76         0.16       3402
B2                      0.75         0.06        959
C1                      0.72         0.04        159
```

## Strategy 2: Rounded Average Predictions

Regression-style strategy: calculate expected class index from probabilities, round to nearest integer, map back to class label.

### CEFR Classification Report

```
    precision    recall  f1-score   support

A1       0.47      0.47      0.47      9437
A2       0.30      0.31      0.30      5937
B1       0.15      0.16      0.16      3240
B2       0.06      0.05      0.05      1112
C1       0.04      0.02      0.02       275

macro avg       0.20      0.20      0.20     20001
weighted avg       0.34      0.34      0.34     20001

accuracy      0.34     20001
adjacent accuracy      0.74     20001
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          A1       0.47      0.47      0.47      9437
          A2       0.30      0.31      0.30      5937
          B1       0.15      0.16      0.16      3240
          B2       0.06      0.05      0.05      1112
          C1       0.04      0.02      0.02       275

    accuracy                           0.34     20001
   macro avg       0.20      0.20      0.20     20001
weighted avg       0.34      0.34      0.34     20001
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['A1', 'A2', 'B1', 'B2', 'C1']

```
[[4446 2849 1651  433   58]
 [2773 1819 1025  281   39]
 [1566  962  529  161   22]
 [ 520  353  176   54    9]
 [ 131   85   42   12    5]]
```

