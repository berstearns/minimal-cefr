# Evaluation Report: norm-KUPA-KEYS

**Classifier**: ppl_all_models_logistic
**Dataset**: norm-KUPA-KEYS
**Samples**: 1006
**Classes in test set**: B1, B2, C1, C2

## Strategy 1: Argmax Predictions

Standard argmax strategy: predict class with highest probability.

### CEFR Classification Report

```
    precision    recall  f1-score   support

B1       0.20      0.11      0.14       109
B2       0.57      0.66      0.61       570
C1       0.36      0.33      0.34       312
C2       0.00      0.00      0.00        15

macro avg       0.28      0.28      0.28      1006
weighted avg       0.46      0.49      0.47      1006

accuracy      0.49      1006
adjacent accuracy      0.95      1006
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          B1       0.20      0.11      0.14       109
          B2       0.57      0.66      0.61       570
          C1       0.36      0.33      0.34       312
          C2       0.00      0.00      0.00        15

    accuracy                           0.49      1006
   macro avg       0.28      0.28      0.28      1006
weighted avg       0.46      0.49      0.47      1006
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['B1', 'B2', 'C1', 'C2']

```
[[ 12  70  27   0]
 [ 39 378 153   0]
 [  8 201 103   0]
 [  0  11   4   0]]
```

### Calibration Report

```
CEFR CALIBRATION REPORT (Soft Probabilities)
================================================================================

Expected Calibration Error (ECE): 0.46

Confidence Bins:
Range                     Count     Accuracy   Confidence          Gap
--------------------------------------------------------------------------------
(0.30, 0.40]                  1         1.00         0.36        -0.64
(0.40, 0.50]                  3         0.67         0.48        -0.19
(0.50, 0.60]                 28         0.39         0.55         0.16
(0.60, 0.70]                 30         0.33         0.65         0.31
(0.70, 0.80]                 41         0.49         0.76         0.27
(0.80, 0.90]                 74         0.46         0.86         0.40
(0.90, 1.00]                829         0.50         0.99         0.49

Per-Class Calibration:
Class               Avg Prob     Accuracy      Count
--------------------------------------------------------------------------------
B1                      0.81         0.20         59
B2                      0.96         0.57        660
C1                      0.94         0.36        287
```

## Strategy 2: Rounded Average Predictions

Regression-style strategy: calculate expected class index from probabilities, round to nearest integer, map back to class label.

### CEFR Classification Report

```
    precision    recall  f1-score   support

B1       0.18      0.08      0.11       109
B2       0.58      0.68      0.62       570
C1       0.36      0.33      0.35       312
C2       0.00      0.00      0.00        15

macro avg       0.28      0.27      0.27      1006
weighted avg       0.46      0.50      0.47      1006

accuracy      0.50      1006
adjacent accuracy      0.96      1006
```

### Standard Classification Report

```
              precision    recall  f1-score   support

          B1       0.18      0.08      0.11       109
          B2       0.58      0.68      0.62       570
          C1       0.36      0.33      0.35       312
          C2       0.00      0.00      0.00        15

    accuracy                           0.50      1006
   macro avg       0.28      0.27      0.27      1006
weighted avg       0.46      0.50      0.47      1006
```

### Confusion Matrix

Labels (rows=true, cols=pred): ['B1', 'B2', 'C1', 'C2']

```
[[  9  73  27   0]
 [ 33 388 149   0]
 [  7 202 103   0]
 [  0  11   4   0]]
```

