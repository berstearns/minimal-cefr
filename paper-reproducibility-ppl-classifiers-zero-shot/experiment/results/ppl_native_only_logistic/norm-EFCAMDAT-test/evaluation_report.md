# Evaluation Report: norm-EFCAMDAT-test

**Classifier**: ppl_native_only_logistic
**Dataset**: norm-EFCAMDAT-test
**Samples**: 20001
**Classes in test set**: A1, A2, B1, B2, C1

## Strategy 1: Argmax Predictions

Standard argmax strategy: predict class with highest probability.

### CEFR Classification Report

```
    precision    recall  f1-score   support

A1       0.47      0.46      0.46      9437
A2       0.29      0.28      0.29      5937
B1       0.16      0.16      0.16      3240
B2       0.05      0.05      0.05      1112
C1       0.03      0.05      0.04       275

macro avg       0.20      0.20      0.20     20001
weighted avg       0.34      0.33      0.33     20001

accuracy      0.33     20001
adjacent accuracy      0.72     20001
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          A1       0.47      0.46      0.46      9437
          A2       0.29      0.28      0.29      5937
          B1       0.16      0.16      0.16      3240
          B2       0.05      0.05      0.05      1112
          C1       0.03      0.05      0.04       275

    accuracy                           0.33     20001
   macro avg       0.20      0.20      0.20     20001
weighted avg       0.34      0.33      0.33     20001
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['A1', 'A2', 'B1', 'B2', 'C1']

```
[[4328 2763 1565  557  224]
 [2734 1685 1027  341  150]
 [1507  948  522  181   82]
 [ 502  339  182   54   35]
 [ 128   76   47   10   14]]
```

### Calibration Report

```
CEFR CALIBRATION REPORT (Soft Probabilities)
================================================================================

Expected Calibration Error (ECE): 0.40

Confidence Bins:
Range                     Count     Accuracy   Confidence          Gap
--------------------------------------------------------------------------------
(0.20, 0.30]                  2         0.00         0.28         0.28
(0.30, 0.40]                156         0.24         0.37         0.14
(0.40, 0.50]               1769         0.24         0.47         0.23
(0.50, 0.60]               3468         0.26         0.55         0.29
(0.60, 0.70]               3545         0.28         0.65         0.37
(0.70, 0.80]               3308         0.30         0.75         0.45
(0.80, 0.90]               3480         0.39         0.85         0.47
(0.90, 1.00]               4273         0.45         0.95         0.50

Per-Class Calibration:
Class               Avg Prob     Accuracy      Count
--------------------------------------------------------------------------------
A1                      0.83         0.47       9199
A2                      0.66         0.29       5811
B1                      0.61         0.16       3343
B2                      0.62         0.05       1143
C1                      0.73         0.03        505
```

## Strategy 2: Rounded Average Predictions

Regression-style strategy: calculate expected class index from probabilities, round to nearest integer, map back to class label.

### CEFR Classification Report

```
    precision    recall  f1-score   support

A1       0.47      0.42      0.44      9437
A2       0.29      0.30      0.29      5937
B1       0.16      0.17      0.16      3240
B2       0.05      0.06      0.05      1112
C1       0.03      0.05      0.04       275

macro avg       0.20      0.20      0.20     20001
weighted avg       0.34      0.32      0.33     20001

accuracy      0.32     20001
adjacent accuracy      0.72     20001
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          A1       0.47      0.42      0.44      9437
          A2       0.29      0.30      0.29      5937
          B1       0.16      0.17      0.16      3240
          B2       0.05      0.06      0.05      1112
          C1       0.03      0.05      0.04       275

    accuracy                           0.32     20001
   macro avg       0.20      0.20      0.20     20001
weighted avg       0.34      0.32      0.33     20001
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['A1', 'A2', 'B1', 'B2', 'C1']

```
[[3987 2943 1663  654  190]
 [2534 1773 1100  405  125]
 [1384 1013  554  221   68]
 [ 461  362  196   62   31]
 [ 122   79   48   13   13]]
```

