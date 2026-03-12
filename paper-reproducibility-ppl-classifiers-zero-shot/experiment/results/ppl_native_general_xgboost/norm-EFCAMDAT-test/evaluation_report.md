# Evaluation Report: norm-EFCAMDAT-test

**Classifier**: ppl_native_general_xgboost
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
B1       0.15      0.16      0.16      3240
B2       0.06      0.05      0.06      1112
C1       0.05      0.03      0.03       275

macro avg       0.21      0.20      0.20     20001
weighted avg       0.34      0.34      0.34     20001

accuracy      0.34     20001
adjacent accuracy      0.74     20001
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          A1       0.47      0.48      0.47      9437
          A2       0.30      0.29      0.30      5937
          B1       0.15      0.16      0.16      3240
          B2       0.06      0.05      0.06      1112
          C1       0.05      0.03      0.03       275

    accuracy                           0.34     20001
   macro avg       0.21      0.20      0.20     20001
weighted avg       0.34      0.34      0.34     20001
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['A1', 'A2', 'B1', 'B2', 'C1']

```
[[4502 2781 1632  452   70]
 [2827 1747 1028  296   39]
 [1599  937  516  166   22]
 [ 526  352  165   60    9]
 [ 134   83   41   10    7]]
```

### Calibration Report

```
CEFR CALIBRATION REPORT (Soft Probabilities)
================================================================================

Expected Calibration Error (ECE): 0.52

Confidence Bins:
Range                     Count     Accuracy   Confidence          Gap
--------------------------------------------------------------------------------
(0.30, 0.40]                 41         0.10         0.38         0.28
(0.40, 0.50]                386         0.23         0.47         0.24
(0.50, 0.60]               1511         0.27         0.55         0.28
(0.60, 0.70]               1602         0.26         0.65         0.39
(0.70, 0.80]               2106         0.27         0.75         0.48
(0.80, 0.90]               2987         0.28         0.85         0.57
(0.90, 1.00]              11368         0.40         0.97         0.57

Per-Class Calibration:
Class               Avg Prob     Accuracy      Count
--------------------------------------------------------------------------------
A1                      0.93         0.47       9588
A2                      0.82         0.30       5900
B1                      0.78         0.15       3382
B2                      0.76         0.06        984
C1                      0.73         0.05        147
```

## Strategy 2: Rounded Average Predictions

Regression-style strategy: calculate expected class index from probabilities, round to nearest integer, map back to class label.

### CEFR Classification Report

```
    precision    recall  f1-score   support

A1       0.47      0.47      0.47      9437
A2       0.30      0.30      0.30      5937
B1       0.15      0.16      0.16      3240
B2       0.06      0.06      0.06      1112
C1       0.05      0.02      0.03       275

macro avg       0.21      0.20      0.20     20001
weighted avg       0.34      0.34      0.34     20001

accuracy      0.34     20001
adjacent accuracy      0.74     20001
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          A1       0.47      0.47      0.47      9437
          A2       0.30      0.30      0.30      5937
          B1       0.15      0.16      0.16      3240
          B2       0.06      0.06      0.06      1112
          C1       0.05      0.02      0.03       275

    accuracy                           0.34     20001
   macro avg       0.21      0.20      0.20     20001
weighted avg       0.34      0.34      0.34     20001
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['A1', 'A2', 'B1', 'B2', 'C1']

```
[[4428 2860 1651  441   57]
 [2776 1802 1038  286   35]
 [1571  965  523  163   18]
 [ 515  361  168   62    6]
 [ 130   85   43   11    6]]
```

