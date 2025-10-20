# CEFR-Specific Metrics Documentation

This document explains the custom CEFR classification and calibration metrics implemented in the pipeline.

## Overview

Instead of using `imbalanced-learn`'s `classification_report_imbalanced`, we implement custom CEFR-specific metrics that account for the ordered nature of CEFR levels (A1, A2, B1, B2, C1, C2) and provide probability calibration analysis.

## Custom Metrics

### 1. CEFR Classification Report (Multiclass)

**Function:** `cefr_classification_report(y_true, y_pred, labels, target_names, digits=2)`

This report provides standard classification metrics plus CEFR-specific evaluations:

#### Standard Metrics (per class):
- **Precision**: Proportion of predicted labels that are correct
- **Recall**: Proportion of true labels that were correctly predicted
- **F1-Score**: Harmonic mean of precision and recall
- **Support**: Number of samples in each class

#### Aggregate Metrics:
- **Macro Average**: Unweighted mean across classes
- **Weighted Average**: Weighted by class support

#### CEFR-Specific Metrics:
- **Accuracy**: Standard classification accuracy
- **Adjacent Accuracy**: Proportion of predictions that are either exact or off by one level
  - For ordered CEFR levels, predicting B1 when truth is B2 is less severe than predicting A1
  - This metric tolerates off-by-one errors (e.g., A2 predicted as A1 or B1)
  - Calculated as: `mean(|true_level_idx - pred_level_idx| <= 1)`

#### Example Output:
```
================================================================================
CEFR CLASSIFICATION REPORT (Multiclass)
================================================================================
              precision    recall  f1-score   support

          A1       0.85      0.90      0.87        10
          A2       0.80      0.75      0.77        10
          B1       0.75      0.80      0.77        10
          B2       0.82      0.78      0.80         9
          C1       0.88      0.85      0.86        10
          C2       0.90      0.95      0.92        11

   macro avg       0.83      0.84      0.83        60
weighted avg       0.83      0.84      0.83        60

    accuracy       0.83        60
adjacent accuracy  0.92        60
```

### 2. CEFR Calibration Report (Soft Probabilities)

**Function:** `cefr_calibration_report(y_true, y_pred_proba, labels, target_names, n_bins=10, digits=2)`

This report evaluates how well predicted probabilities match actual outcomes, crucial for confidence estimation in CEFR classification.

#### Metrics:

##### Expected Calibration Error (ECE)
- Measures the difference between predicted confidence and actual accuracy
- Lower is better (0 = perfectly calibrated)
- Formula: `ECE = Σ |accuracy_in_bin - confidence_in_bin| * proportion_in_bin`
- Bins predictions by confidence level and compares confidence to accuracy

##### Confidence Bins Analysis
For each confidence interval (e.g., 0.0-0.1, 0.1-0.2, ..., 0.9-1.0):
- **Count**: Number of predictions in this bin
- **Accuracy**: Actual accuracy of predictions in this bin
- **Confidence**: Average predicted probability in this bin
- **Gap**: Difference between confidence and accuracy (calibration error)

A well-calibrated model should have:
- Predictions with 70% confidence should be correct ~70% of the time
- Small gaps across all bins
- Low ECE overall

##### Per-Class Calibration
For each CEFR level:
- **Avg Prob**: Average predicted probability when this class was predicted
- **Accuracy**: Actual accuracy when this class was predicted
- **Count**: Number of times this class was predicted

#### Example Output:
```
================================================================================
CEFR CALIBRATION REPORT (Soft Probabilities)
================================================================================

Expected Calibration Error (ECE): 0.08

Confidence Bins:
Range                     Count     Accuracy   Confidence          Gap
--------------------------------------------------------------------------------
(0.00, 0.10]                  0         0.00         0.00         0.00
(0.10, 0.20]                  2         0.50         0.18        -0.32
(0.20, 0.30]                  5         0.60         0.25        -0.35
(0.30, 0.40]                  8         0.75         0.35        -0.40
(0.40, 0.50]                 12         0.67         0.45        -0.22
(0.50, 0.60]                 10         0.80         0.55        -0.25
(0.60, 0.70]                  8         0.88         0.65        -0.23
(0.70, 0.80]                  7         0.86         0.75        -0.11
(0.80, 0.90]                  5         0.80         0.85         0.05
(0.90, 1.00]                  3         1.00         0.95        -0.05

Per-Class Calibration:
Class           Avg Prob     Accuracy      Count
--------------------------------------------------------------------------------
A1                  0.82         0.90         10
A2                  0.68         0.75          8
B1                  0.71         0.80         10
B2                  0.75         0.78          9
C1                  0.79         0.85         12
C2                  0.88         0.95         11
```

## When to Use Each Report

### CEFR Classification Report
- **Always printed** for all predictions
- Use to evaluate overall classification performance
- Pay attention to adjacent accuracy for CEFR-ordered evaluation
- Lower adjacent accuracy compared to exact accuracy indicates boundary confusion

### CEFR Calibration Report
- **Only printed** when classifier supports `predict_proba`
- Available for: Logistic Regression, Random Forest, XGBoost, Multinomial Naive Bayes
- Not available for: Linear SVM (without probability calibration)
- Use to evaluate confidence/probability reliability
- Important when using probabilities for decision-making or ranking

## Implementation Details

### Adjacent Accuracy Calculation

```python
# Map labels to ordered indices
cefr_order = {label: idx for idx, label in enumerate(labels)}
# e.g., {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5}

# Convert labels to indices
y_true_idx = np.array([cefr_order[y] for y in y_true])
y_pred_idx = np.array([cefr_order[y] for y in y_pred])

# Check if predictions are within 1 level
adjacent_correct = np.abs(y_true_idx - y_pred_idx) <= 1

# Calculate proportion correct
adjacent_accuracy = np.mean(adjacent_correct)
```

### ECE Calculation

```python
# For each confidence bin
for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
    # Find predictions in this bin
    in_bin = (confidence > bin_lower) & (confidence <= bin_upper)

    if any(in_bin):
        # Calculate accuracy and average confidence in bin
        accuracy_in_bin = mean(y_true[in_bin] == y_pred[in_bin])
        avg_confidence_in_bin = mean(confidence[in_bin])

        # Add weighted calibration error to ECE
        proportion_in_bin = mean(in_bin)
        ece += abs(avg_confidence_in_bin - accuracy_in_bin) * proportion_in_bin
```

## Comparison with Imbalanced-Learn

### Why We Don't Use `classification_report_imbalanced`

1. **CEFR-Specific Needs**: We need adjacent accuracy metric specific to ordered CEFR levels
2. **Calibration Analysis**: We need probability calibration reporting for confidence estimation
3. **Custom Interface**: Full control over report format and metrics
4. **No Extra Dependency**: Removes need for `imbalanced-learn` package

### What We Keep

- Standard precision, recall, F1-score calculations (using sklearn)
- Similar report formatting and API interface
- Support for custom labels and target names

## Usage in Code

### Basic Usage

```python
from src.predict import cefr_classification_report, cefr_calibration_report

# After making predictions
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)

# Print classification report
print(cefr_classification_report(y_test, y_pred))

# Print calibration report (if probabilities available)
if y_pred_proba is not None:
    print(cefr_calibration_report(y_test, y_pred_proba))
```

### With Custom Labels

```python
labels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
target_names = ['Beginner 1', 'Beginner 2', 'Intermediate 1',
                'Intermediate 2', 'Advanced 1', 'Advanced 2']

print(cefr_classification_report(
    y_test,
    y_pred,
    labels=labels,
    target_names=target_names,
    digits=3
))
```

## References

- **Adjacent Accuracy**: Inspired by off-by-one tolerance in ordered classification
- **Expected Calibration Error**: Naeini et al. (2015) "Obtaining Well Calibrated Probabilities Using Bayesian Binning"
- **Calibration**: Guo et al. (2017) "On Calibration of Modern Neural Networks"

## Related Files

- `src/predict.py` - Implementation of custom metrics
- `USAGE.md` - Usage examples and integration in pipeline
- `requirements.txt` - Dependencies (removed `imbalanced-learn`)
