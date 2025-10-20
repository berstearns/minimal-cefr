# Ordinal Metrics for CEFR Classification

## Why Ordinal Metrics Matter

CEFR levels are **ordered**: A1 < A2 < B1 < B2 < C1 < C2

Standard classification metrics treat all errors equally:
- Predicting C2 when true label is A1 = **same penalty** as predicting A2 when true is A1
- This ignores the ordinal nature of proficiency levels!

## Available Ordinal Metrics

### 1. Ordinal Accuracy (Adjacent Accuracy)
**Flag:** `--metric ordinal_accuracy`
**Formula:** Prediction is correct if `|predicted - true| ≤ 1`

**Use case:** Lenient evaluation where ±1 level is acceptable

**Example:**
```
True: B1 (level 2)
Predicted: B2 (level 3)
Distance: 1
Score: 1.0 ✓ (correct within tolerance)

True: B1 (level 2)
Predicted: C1 (level 4)
Distance: 2
Score: 0.0 ✗ (outside tolerance)
```

### 2. Ordinal Distance (Linear Penalty)
**Flag:** `--metric ordinal_distance_linear`

**Formula:** `score = 1 - (distance / max_distance)`
- max_distance = 5 (A1 to C2)

**Use case:** Proportional penalty - each level of error reduces score equally

**Score table:**
| True | Predicted | Distance | Score |
|------|-----------|----------|-------|
| A1 | A1 | 0 | 1.00 |
| A1 | A2 | 1 | 0.80 |
| A1 | B1 | 2 | 0.60 |
| A1 | B2 | 3 | 0.40 |
| A1 | C1 | 4 | 0.20 |
| A1 | C2 | 5 | 0.00 |

**Visual:**
```
Perfect  →→→→→→  Worst
1.0  0.8  0.6  0.4  0.2  0.0
 │    │    │    │    │    │
 0    1    2    3    4    5  levels away
```

### 3. Ordinal Distance (Quadratic Penalty) **RECOMMENDED**
**Flag:** `--metric ordinal_distance_quadratic`

**Formula:** `score = 1 - (distance² / max_distance²)`

**Use case:** Heavily penalize far predictions while being lenient on adjacent errors

**Score table:**
| True | Predicted | Distance | Distance² | Score |
|------|-----------|----------|-----------|-------|
| A1 | A1 | 0 | 0 | 1.00 |
| A1 | A2 | 1 | 1 | 0.96 |
| A1 | B1 | 2 | 4 | 0.84 |
| A1 | B2 | 3 | 9 | 0.64 |
| A1 | C1 | 4 | 16 | 0.36 |
| A1 | C2 | 5 | 25 | 0.00 |

**Visual:**
```
Perfect  →→→→→→  Worst
1.00  0.96  0.84  0.64  0.36  0.00
 │     │     │     │     │     │
 0     1     2     3     4     5  levels away

Notice: 1 level off = 0.96 (minor penalty)
        5 levels off = 0.00 (severe penalty)
```

### 4. Ordinal MSE Score
**Flag:** `--metric ordinal_mse`

**Formula:** `score = 1 - (MSE / max_MSE)`
- Treats labels as numerical: A1=0, A2=1, B1=2, B2=3, C1=4, C2=5
- MSE = mean((y_true - y_pred)²)
- max_MSE = 25 (worst case: predicting 0 when true is 5)

**Use case:** Same as quadratic penalty, but using MSE formulation (more intuitive for regression users)

**Score table (identical to quadratic):**
| True | Predicted | Numerical Diff | MSE | Score |
|------|-----------|----------------|-----|-------|
| A1 (0) | A1 (0) | 0 | 0 | 1.00 |
| A1 (0) | A2 (1) | 1 | 1 | 0.96 |
| A1 (0) | B1 (2) | 2 | 4 | 0.84 |
| A1 (0) | B2 (3) | 3 | 9 | 0.64 |
| A1 (0) | C1 (4) | 4 | 16 | 0.36 |
| A1 (0) | C2 (5) | 5 | 25 | 0.00 |

**Note:** This is mathematically equivalent to `ordinal_distance_quadratic`. The difference is:
- `ordinal_distance_quadratic`: score = 1 - (distance² / 5²) = 1 - (distance² / 25)
- `ordinal_mse`: score = 1 - (MSE / 25) where MSE = distance²

Both produce identical scores!

## Comparison of All Metrics

### Scenario: Model makes various prediction errors on A1 samples

| Prediction | Standard Accuracy | Ordinal Accuracy | Linear Distance | Quadratic Distance | MSE Score |
|------------|-------------------|------------------|-----------------|-------------------|-----------|
| A1 (✓) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| A2 (±1) | 0.00 | 1.00 | 0.80 | 0.96 | 0.96 |
| B1 (±2) | 0.00 | 0.00 | 0.60 | 0.84 | 0.84 |
| B2 (±3) | 0.00 | 0.00 | 0.40 | 0.64 | 0.64 |
| C1 (±4) | 0.00 | 0.00 | 0.20 | 0.36 | 0.36 |
| C2 (±5) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Average if model predicts [A1, A2, B1, B2, C1, C2] for 6 A1 samples:**
- Standard Accuracy: 1/6 = **0.167**
- Ordinal Accuracy: 2/6 = **0.333** (A1 and A2 count)
- Linear Distance: (1.0 + 0.8 + 0.6 + 0.4 + 0.2 + 0.0) / 6 = **0.500**
- Quadratic Distance: (1.0 + 0.96 + 0.84 + 0.64 + 0.36 + 0.0) / 6 = **0.633**
- MSE Score: (1.0 + 0.96 + 0.84 + 0.64 + 0.36 + 0.0) / 6 = **0.633** (same as quadratic)

## When to Use Each Metric

### Ordinal Accuracy
✅ Use when:
- Adjacent level errors are acceptable
- Pass/fail evaluation (±1 is OK)
- Quick screening of models

❌ Avoid when:
- Need to differentiate between 1-level and 2-level errors
- Want to heavily penalize far predictions

### Ordinal Distance (Linear)
✅ Use when:
- Want proportional penalty for each level of error
- All distances matter equally (1→2 same importance as 4→5)

❌ Avoid when:
- Far errors should be penalized much more heavily

### Ordinal Distance (Quadratic) **RECOMMENDED**
✅ Use when:
- Far errors are much worse than near errors
- Natural for CEFR: A1→C2 confusion is catastrophic
- Want to optimize for both accuracy AND ordinal correctness

✅ **Best default for CEFR classification**

## Hyperparameter Optimization Impact

When optimizing with different metrics:

### With `--metric accuracy`
- Treats all errors equally
- May optimize to avoid worst-performing classes
- Ignores proficiency ordering

### With `--metric ordinal_accuracy`
- Optimizes to get within ±1 level
- May be too lenient for some applications
- Good for "close enough" scenarios

### With `--metric ordinal_distance_quadratic`
- Balances perfect predictions with near-misses
- Heavily penalizes catastrophic errors (A1↔C2)
- **Recommended for most CEFR applications**

## Example: Model Comparison

**Model A:** Predicts [A2, B1, B1, B2, C1, C1] for true labels [A1, A2, B1, B2, C1, C2]

**Model B:** Predicts [A1, A2, B2, B2, C2, C2] for true labels [A1, A2, B1, B2, C1, C2]

| Metric | Model A | Model B | Winner |
|--------|---------|---------|--------|
| Accuracy | 3/6 = 0.500 | 4/6 = 0.667 | B |
| Ordinal Accuracy | 6/6 = 1.000 | 5/6 = 0.833 | A |
| Linear Distance | 0.867 | 0.867 | Tie |
| Quadratic Distance | 0.947 | 0.933 | A |

**Analysis:**
- Model A: Consistently off by ±1 level
- Model B: Mostly correct but one catastrophic error (C1→C2 off by 1)
- Standard accuracy prefers Model B
- Ordinal metrics prefer Model A (more consistent, no far errors)

## Recommendation

For CEFR classification:
```bash
--metric ordinal_distance_quadratic \
--weight-strategy inverse \
--weight-alpha 0.7
```

This combination:
1. Uses quadratic penalty for ordinal nature
2. Balances class sizes with inverse weighting
3. Moderate calibration (0.7) to not over-weight small classes
