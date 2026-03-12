# Evaluation Report: norm-EFCAMDAT-test

**Classifier**: ppl_all_models_logistic
**Dataset**: norm-EFCAMDAT-test
**Samples**: 20001
**Classes in test set**: A1, A2, B1, B2, C1

## Strategy 1: Argmax Predictions

Standard argmax strategy: predict class with highest probability.

### CEFR Classification Report

```
    precision    recall  f1-score   support

A1       0.47      0.47      0.47      9437
A2       0.30      0.29      0.29      5937
B1       0.16      0.16      0.16      3240
B2       0.05      0.05      0.05      1112
C1       0.03      0.03      0.03       275

macro avg       0.20      0.20      0.20     20001
weighted avg       0.34      0.34      0.34     20001

accuracy      0.34     20001
adjacent accuracy      0.73     20001
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          A1       0.47      0.47      0.47      9437
          A2       0.30      0.29      0.29      5937
          B1       0.16      0.16      0.16      3240
          B2       0.05      0.05      0.05      1112
          C1       0.03      0.03      0.03       275

    accuracy                           0.34     20001
   macro avg       0.20      0.20      0.20     20001
weighted avg       0.34      0.34      0.34     20001
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['A1', 'A2', 'B1', 'B2', 'C1']

```
[[4421 2790 1565  532  129]
 [2784 1739  971  349   94]
 [1569  931  506  194   40]
 [ 516  342  175   58   21]
 [ 128   81   44   14    8]]
```

### Calibration Report

```
CEFR CALIBRATION REPORT (Soft Probabilities)
================================================================================

Expected Calibration Error (ECE): 0.64

Confidence Bins:
Range                     Count     Accuracy   Confidence          Gap
--------------------------------------------------------------------------------
(0.30, 0.40]                  4         0.00         0.39         0.39
(0.40, 0.50]                 32         0.44         0.47         0.03
(0.50, 0.60]                209         0.32         0.55         0.23
(0.60, 0.70]                243         0.27         0.65         0.38
(0.70, 0.80]                276         0.27         0.75         0.48
(0.80, 0.90]                422         0.35         0.86         0.51
(0.90, 1.00]              18815         0.34         1.00         0.66

Per-Class Calibration:
Class               Avg Prob     Accuracy      Count
--------------------------------------------------------------------------------
A1                      0.99         0.47       9418
A2                      0.98         0.30       5883
B1                      0.97         0.16       3261
B2                      0.97         0.05       1147
C1                      0.96         0.03        292
```

## Strategy 2: Rounded Average Predictions

Regression-style strategy: calculate expected class index from probabilities, round to nearest integer, map back to class label.

### CEFR Classification Report

```
    precision    recall  f1-score   support

A1       0.47      0.47      0.47      9437
A2       0.30      0.29      0.30      5937
B1       0.16      0.16      0.16      3240
B2       0.05      0.05      0.05      1112
C1       0.03      0.03      0.03       275

macro avg       0.20      0.20      0.20     20001
weighted avg       0.34      0.34      0.34     20001

accuracy      0.34     20001
adjacent accuracy      0.73     20001
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          A1       0.47      0.47      0.47      9437
          A2       0.30      0.29      0.30      5937
          B1       0.16      0.16      0.16      3240
          B2       0.05      0.05      0.05      1112
          C1       0.03      0.03      0.03       275

    accuracy                           0.34     20001
   macro avg       0.20      0.20      0.20     20001
weighted avg       0.34      0.34      0.34     20001
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['A1', 'A2', 'B1', 'B2', 'C1']

```
[[4394 2819 1569  531  124]
 [2775 1750  973  346   93]
 [1566  929  510  196   39]
 [ 516  341  174   61   20]
 [ 128   81   45   13    8]]
```

