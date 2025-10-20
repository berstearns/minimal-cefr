# Ordinal MSE Metric - Quick Reference

## What is it?

`ordinal_mse` treats CEFR labels as numerical indices and computes a score based on Mean Squared Error.

**Encoding:** A1=0, A2=1, B1=2, B2=3, C1=4, C2=5

**Formula:** `score = 1 - (MSE / max_MSE)`
- MSE = mean((y_true - y_pred)²)
- max_MSE = 25 (worst case: 0↔5)

## Usage

```bash
python -m src.train_classifiers_with_ho_multifeat \
    -e data/experiments/zero-shot-2 \
    --features-base-dir data/experiments/zero-shot-2/features \
    --metric ordinal_mse \
    --top-k 5
```

## Score Examples

| True Label | Predicted | Numerical Diff | Squared Error | Score |
|------------|-----------|----------------|---------------|-------|
| A1 (0) | A1 (0) | 0 | 0 | 1.00 ✓ Perfect |
| A1 (0) | A2 (1) | 1 | 1 | 0.96 ≈ Good |
| A1 (0) | B1 (2) | 2 | 4 | 0.84 ≈ OK |
| A1 (0) | B2 (3) | 3 | 9 | 0.64 ≈ Poor |
| A1 (0) | C1 (4) | 4 | 16 | 0.36 ✗ Bad |
| A1 (0) | C2 (5) | 5 | 25 | 0.00 ✗✗ Terrible |

## Relationship to Other Metrics

### Identical to `ordinal_distance_quadratic`

Both metrics produce **exactly the same scores**:

```python
# ordinal_distance_quadratic
score = 1 - (distance² / max_distance²)
score = 1 - (distance² / 5²)
score = 1 - (distance² / 25)

# ordinal_mse
MSE = distance²  # for single prediction
score = 1 - (MSE / max_MSE)
score = 1 - (distance² / 25)

# They're the same!
```

### Why have both?

**Use `ordinal_distance_quadratic` if:**
- You think in terms of "distance-based penalties"
- You want the mathematical interpretation of quadratic penalty

**Use `ordinal_mse` if:**
- You're familiar with regression metrics (MSE, RMSE)
- You want to interpret the model as doing "ordinal regression"
- You prefer thinking numerically (A1=0, C2=5)

## Comparison with Other Ordinal Metrics

### Example: Predicting A2 (1) when true is A1 (0)

| Metric | Calculation | Score | Interpretation |
|--------|-------------|-------|----------------|
| `ordinal_accuracy` | distance ≤ 1 | 1.00 | Perfect (within tolerance) |
| `ordinal_distance_linear` | 1 - 1/5 | 0.80 | Good (proportional) |
| `ordinal_distance_quadratic` | 1 - 1²/5² | 0.96 | Very good (slight penalty) |
| `ordinal_mse` | 1 - 1/25 | 0.96 | Very good (same as quadratic) |

### Example: Predicting C2 (5) when true is A1 (0)

| Metric | Calculation | Score | Interpretation |
|--------|-------------|-------|----------------|
| `ordinal_accuracy` | distance ≤ 1 | 0.00 | Wrong (outside tolerance) |
| `ordinal_distance_linear` | 1 - 5/5 | 0.00 | Terrible (max distance) |
| `ordinal_distance_quadratic` | 1 - 5²/5² | 0.00 | Terrible (max penalty) |
| `ordinal_mse` | 1 - 25/25 | 0.00 | Terrible (max MSE) |

## When to Use

✅ **Use `ordinal_mse` when:**
- You want to heavily penalize far predictions (A1↔C2)
- You're comfortable with regression-style metrics
- You want to treat CEFR as a numerical scale

❌ **Don't use when:**
- You want to treat ±1 level errors as acceptable → use `ordinal_accuracy`
- You want proportional (not quadratic) penalties → use `ordinal_distance_linear`

## Practical Example

### Dataset with mixed errors

Predictions for 4 samples:
1. True: A1 (0), Pred: A1 (0) → SE=0
2. True: B1 (2), Pred: B2 (3) → SE=1
3. True: C1 (4), Pred: B2 (3) → SE=1
4. True: A2 (1), Pred: C2 (5) → SE=16

**MSE = (0 + 1 + 1 + 16) / 4 = 4.5**

**Score = 1 - (4.5 / 25) = 1 - 0.18 = 0.82**

**Interpretation:**
- Model is 82% of the way to perfect (vs 0% for worst possible)
- The one catastrophic error (A2→C2) significantly hurts the score
- Two minor errors (±1 level) have minimal impact

## Recommendation

For CEFR classification, `ordinal_mse` (or equivalently `ordinal_distance_quadratic`) is **recommended** because:

1. **Penalizes catastrophic errors:** A1↔C2 confusion gets score of 0.0
2. **Lenient on adjacent errors:** B1↔B2 confusion gets score of 0.96
3. **Matches ordinal nature:** Respects that CEFR is ordered
4. **Interpretable:** MSE formulation is familiar to many ML practitioners

**Alternative names this could have been called:**
- `1_minus_normalized_mse`
- `ordinal_regression_score`
- `numerical_ordinal_score`
- `squared_distance_penalty`

All would be mathematically equivalent to the current implementation!
