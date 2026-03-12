# Evaluation Report: norm-CELVA-SP

**Classifier**: ppl_native_only_xgboost
**Dataset**: norm-CELVA-SP
**Samples**: 1742
**Classes in test set**: A1, A2, B1, B2, C1, C2

## Strategy 1: Argmax Predictions

Standard argmax strategy: predict class with highest probability.

### CEFR Classification Report

```
    precision    recall  f1-score   support

A1       0.51      0.13      0.20       157
A2       0.43      0.07      0.12       511
B1       0.31      0.14      0.19       609
B2       0.24      0.75      0.37       353
C1       0.08      0.20      0.11       100
C2       0.00      0.00      0.00        12

macro avg       0.26      0.21      0.17      1742
weighted avg       0.33      0.24      0.20      1742

accuracy      0.24      1742
adjacent accuracy      0.66      1742
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          A1       0.51      0.13      0.20       157
          A2       0.43      0.07      0.12       511
          B1       0.31      0.14      0.19       609
          B2       0.24      0.75      0.37       353
          C1       0.08      0.20      0.11       100
          C2       0.00      0.00      0.00        12

    accuracy                           0.24      1742
   macro avg       0.26      0.21      0.17      1742
weighted avg       0.33      0.24      0.20      1742
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

```
[[ 20  28  43  55  11   0]
 [ 11  37 113 281  69   0]
 [  7  16  83 403 100   0]
 [  0   6  24 266  57   0]
 [  0   0   6  74  20   0]
 [  1   0   0  10   1   0]]
```

### Calibration Report

```
CEFR CALIBRATION REPORT (Soft Probabilities)
================================================================================

Expected Calibration Error (ECE): 0.50

Confidence Bins:
Range                     Count     Accuracy   Confidence          Gap
--------------------------------------------------------------------------------
(0.30, 0.40]                  5         0.40         0.38        -0.02
(0.40, 0.50]                 70         0.24         0.47         0.23
(0.50, 0.60]                287         0.18         0.55         0.36
(0.60, 0.70]                299         0.23         0.65         0.42
(0.70, 0.80]                350         0.25         0.75         0.50
(0.80, 0.90]                404         0.28         0.85         0.57
(0.90, 1.00]                327         0.26         0.94         0.68

Per-Class Calibration:
Class               Avg Prob     Accuracy      Count
--------------------------------------------------------------------------------
A1                      0.81         0.51         39
A2                      0.66         0.43         87
B1                      0.70         0.31        269
B2                      0.78         0.24       1089
C1                      0.66         0.08        258
```

## Strategy 2: Rounded Average Predictions

Regression-style strategy: calculate expected class index from probabilities, round to nearest integer, map back to class label.

### CEFR Classification Report

```
    precision    recall  f1-score   support

A1       0.54      0.12      0.20       157
A2       0.44      0.08      0.13       511
B1       0.31      0.14      0.19       609
B2       0.24      0.77      0.37       353
C1       0.09      0.19      0.12       100
C2       0.00      0.00      0.00        12

macro avg       0.27      0.22      0.17      1742
weighted avg       0.34      0.25      0.21      1742

accuracy      0.25      1742
adjacent accuracy      0.67      1742
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          A1       0.54      0.12      0.20       157
          A2       0.44      0.08      0.13       511
          B1       0.31      0.14      0.19       609
          B2       0.24      0.77      0.37       353
          C1       0.09      0.19      0.12       100
          C2       0.00      0.00      0.00        12

    accuracy                           0.25      1742
   macro avg       0.27      0.22      0.17      1742
weighted avg       0.34      0.25      0.21      1742
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

```
[[ 19  27  48  53  10   0]
 [ 10  40 112 296  53   0]
 [  6  16  86 420  81   0]
 [  0   6  24 272  51   0]
 [  0   0   6  75  19   0]
 [  0   1   0  10   1   0]]
```

