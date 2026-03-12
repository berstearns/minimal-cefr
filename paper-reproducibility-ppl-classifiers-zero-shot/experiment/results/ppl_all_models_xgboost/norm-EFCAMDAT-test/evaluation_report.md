# Evaluation Report: norm-EFCAMDAT-test

**Classifier**: ppl_all_models_xgboost
**Dataset**: norm-EFCAMDAT-test
**Samples**: 20001
**Classes in test set**: A1, A2, B1, B2, C1

## Strategy 1: Argmax Predictions

Standard argmax strategy: predict class with highest probability.

### CEFR Classification Report

```
    precision    recall  f1-score   support

A1       0.47      0.47      0.47      9437
A2       0.29      0.29      0.29      5937
B1       0.15      0.16      0.15      3240
B2       0.05      0.05      0.05      1112
C1       0.03      0.02      0.02       275

macro avg       0.20      0.20      0.20     20001
weighted avg       0.34      0.34      0.34     20001

accuracy      0.34     20001
adjacent accuracy      0.73     20001
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          A1       0.47      0.47      0.47      9437
          A2       0.29      0.29      0.29      5937
          B1       0.15      0.16      0.15      3240
          B2       0.05      0.05      0.05      1112
          C1       0.03      0.02      0.02       275

    accuracy                           0.34     20001
   macro avg       0.20      0.20      0.20     20001
weighted avg       0.34      0.34      0.34     20001
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['A1', 'A2', 'B1', 'B2', 'C1']

```
[[4427 2809 1583  516  102]
 [2802 1733  994  338   70]
 [1577  936  504  189   34]
 [ 515  349  171   60   17]
 [ 129   84   43   13    6]]
```

### Calibration Report

```
CEFR CALIBRATION REPORT (Soft Probabilities)
================================================================================

Expected Calibration Error (ECE): 0.64

Confidence Bins:
Range                     Count     Accuracy   Confidence          Gap
--------------------------------------------------------------------------------
(0.30, 0.40]                 14         0.14         0.37         0.23
(0.40, 0.50]                 60         0.20         0.47         0.27
(0.50, 0.60]                268         0.22         0.55         0.33
(0.60, 0.70]                297         0.27         0.65         0.38
(0.70, 0.80]                330         0.22         0.75         0.53
(0.80, 0.90]                580         0.28         0.85         0.57
(0.90, 1.00]              18452         0.34         0.99         0.65

Per-Class Calibration:
Class               Avg Prob     Accuracy      Count
--------------------------------------------------------------------------------
A1                      0.99         0.47       9450
A2                      0.97         0.29       5911
B1                      0.95         0.15       3295
B2                      0.93         0.05       1116
C1                      0.91         0.03        229
```

## Strategy 2: Rounded Average Predictions

Regression-style strategy: calculate expected class index from probabilities, round to nearest integer, map back to class label.

### CEFR Classification Report

```
    precision    recall  f1-score   support

A1       0.47      0.47      0.47      9437
A2       0.29      0.29      0.29      5937
B1       0.15      0.16      0.15      3240
B2       0.05      0.05      0.05      1112
C1       0.03      0.02      0.02       275

macro avg       0.20      0.20      0.20     20001
weighted avg       0.34      0.34      0.34     20001

accuracy      0.34     20001
adjacent accuracy      0.73     20001
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          A1       0.47      0.47      0.47      9437
          A2       0.29      0.29      0.29      5937
          B1       0.15      0.16      0.15      3240
          B2       0.05      0.05      0.05      1112
          C1       0.03      0.02      0.02       275

    accuracy                           0.34     20001
   macro avg       0.20      0.20      0.20     20001
weighted avg       0.34      0.34      0.34     20001
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['A1', 'A2', 'B1', 'B2', 'C1']

```
[[4422 2815 1593  511   96]
 [2791 1744 1005  330   67]
 [1572  939  508  187   34]
 [ 515  349  173   60   15]
 [ 128   85   43   13    6]]
```

