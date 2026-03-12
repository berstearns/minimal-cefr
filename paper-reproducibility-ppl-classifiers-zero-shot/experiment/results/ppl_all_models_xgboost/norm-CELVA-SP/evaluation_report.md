# Evaluation Report: norm-CELVA-SP

**Classifier**: ppl_all_models_xgboost
**Dataset**: norm-CELVA-SP
**Samples**: 1742
**Classes in test set**: A1, A2, B1, B2, C1, C2

## Strategy 1: Argmax Predictions

Standard argmax strategy: predict class with highest probability.

### CEFR Classification Report

```
    precision    recall  f1-score   support

A1       0.49      0.13      0.20       157
A2       0.49      0.09      0.16       511
B1       0.33      0.15      0.20       609
B2       0.22      0.60      0.32       353
C1       0.12      0.45      0.19       100
C2       0.00      0.00      0.00        12

macro avg       0.28      0.24      0.18      1742
weighted avg       0.36      0.24      0.21      1742

accuracy      0.24      1742
adjacent accuracy      0.65      1742
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          A1       0.49      0.13      0.20       157
          A2       0.49      0.09      0.16       511
          B1       0.33      0.15      0.20       609
          B2       0.22      0.60      0.32       353
          C1       0.12      0.45      0.19       100
          C2       0.00      0.00      0.00        12

    accuracy                           0.24      1742
   macro avg       0.28      0.24      0.18      1742
weighted avg       0.36      0.24      0.21      1742
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

```
[[ 20  24  54  50   9   0]
 [ 11  47 100 265  88   0]
 [  8  16  90 371 124   0]
 [  1   7  22 211 112   0]
 [  0   1   4  50  45   0]
 [  1   0   0   3   8   0]]
```

### Calibration Report

```
CEFR CALIBRATION REPORT (Soft Probabilities)
================================================================================

Expected Calibration Error (ECE): 0.53

Confidence Bins:
Range                     Count     Accuracy   Confidence          Gap
--------------------------------------------------------------------------------
(0.30, 0.40]                 13         0.23         0.37         0.14
(0.40, 0.50]                 73         0.23         0.47         0.23
(0.50, 0.60]                269         0.23         0.55         0.32
(0.60, 0.70]                260         0.20         0.65         0.46
(0.70, 0.80]                308         0.21         0.75         0.54
(0.80, 0.90]                334         0.28         0.85         0.58
(0.90, 1.00]                485         0.25         0.95         0.70

Per-Class Calibration:
Class               Avg Prob     Accuracy      Count
--------------------------------------------------------------------------------
A1                      0.78         0.49         41
A2                      0.74         0.49         95
B1                      0.76         0.33        270
B2                      0.79         0.22        950
C1                      0.72         0.12        386
```

## Strategy 2: Rounded Average Predictions

Regression-style strategy: calculate expected class index from probabilities, round to nearest integer, map back to class label.

### CEFR Classification Report

```
    precision    recall  f1-score   support

A1       0.50      0.10      0.17       157
A2       0.48      0.10      0.17       511
B1       0.34      0.15      0.21       609
B2       0.22      0.63      0.33       353
C1       0.13      0.42      0.19       100
C2       0.00      0.00      0.00        12

macro avg       0.28      0.23      0.18      1742
weighted avg       0.36      0.24      0.21      1742

accuracy      0.24      1742
adjacent accuracy      0.66      1742
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          A1       0.50      0.10      0.17       157
          A2       0.48      0.10      0.17       511
          B1       0.34      0.15      0.21       609
          B2       0.22      0.63      0.33       353
          C1       0.13      0.42      0.19       100
          C2       0.00      0.00      0.00        12

    accuracy                           0.24      1742
   macro avg       0.28      0.23      0.18      1742
weighted avg       0.36      0.24      0.21      1742
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

```
[[ 16  29  54  51   7   0]
 [  8  51  99 285  68   0]
 [  7  17  90 390 105   0]
 [  1   7  19 223 103   0]
 [  0   1   4  53  42   0]
 [  0   1   0   3   8   0]]
```

