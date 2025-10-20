# Two-Stage Hyperparameter Optimization with Metric & Weighting Options

## Overview

`train_classifiers_with_ho_multifeat.py` performs two-stage optimization across multiple feature configurations with flexible optimization goals and class weighting strategies.

## Key Features

### 🎯 Optimization Metrics (Default: F1 Macro)

#### Standard Classification Metrics
- `f1_macro` - Macro-averaged F1 score **(DEFAULT)** - Best for imbalanced classes
- `f1_weighted` - Weighted F1 score
- `f1_micro` - Micro-averaged F1 score
- `accuracy` - Standard accuracy
- `precision_macro` - Macro-averaged precision
- `recall_macro` - Macro-averaged recall

#### Ordinal Metrics (CEFR-Aware)
- `ordinal_accuracy` - Adjacent accuracy (allows ±1 level error)
  - Example: Predicting B1 when true is B2 counts as correct
- `ordinal_distance_linear` - Linear distance penalty
  - Score = 1 - (distance / max_distance)
  - 1 level off = 0.8, 2 levels = 0.6, 3 levels = 0.4, etc.
- `ordinal_distance_quadratic` - Quadratic distance penalty **(RECOMMENDED for ordinal)**
  - Score = 1 - (distance² / max_distance²)
  - 1 level off = 0.96, 2 levels = 0.84, 3 levels = 0.64
  - **Penalizes far predictions more heavily**
- `ordinal_mse` - MSE-based score treating labels as numerical indices
  - Treats A1=0, A2=1, ..., C2=5 as numbers
  - Score = 1 - (MSE / max_MSE), where max_MSE = 25
  - Identical to `ordinal_distance_quadratic` but more intuitive for regression-minded users

### ⚖️ Sample Weighting Strategies
- `equal` **(DEFAULT)** - All samples have equal weight
- `inverse` - Weight inversely proportional to class frequency (favors small classes)
- `inverse_sqrt` - Weight inversely proportional to sqrt of class frequency (moderate favoring)

### 🎚️ Weight Calibration (`--weight-alpha`)
- `0.0` = Equal weights (ignore strategy)
- `0.5` = Moderate class balancing
- `1.0` = Full inverse weighting **(DEFAULT)**

Formula: `weight = (1 - alpha) * 1.0 + alpha * inverse_weight`

## Usage Examples

### Basic: F1 Macro (Default)
```bash
python -m src.train_classifiers_with_ho_multifeat \
    -e data/experiments/zero-shot-2 \
    --features-base-dir data/experiments/zero-shot-2/features \
    --top-k 5 \
    --stage2-trials 100
```

### Optimize for Accuracy
```bash
python -m src.train_classifiers_with_ho_multifeat \
    -e data/experiments/zero-shot-2 \
    --features-base-dir data/experiments/zero-shot-2/features \
    --metric accuracy \
    --top-k 5
```

### Favor Small Classes (Inverse Weighting)
```bash
python -m src.train_classifiers_with_ho_multifeat \
    -e data/experiments/zero-shot-2 \
    --features-base-dir data/experiments/zero-shot-2/features \
    --metric f1_macro \
    --weight-strategy inverse \
    --weight-alpha 1.0 \
    --top-k 5
```

### Slight Favoring of Small Classes (50% Inverse)
```bash
python -m src.train_classifiers_with_ho_multifeat \
    -e data/experiments/zero-shot-2 \
    --features-base-dir data/experiments/zero-shot-2/features \
    --metric f1_macro \
    --weight-strategy inverse \
    --weight-alpha 0.5 \
    --top-k 5
```

### Moderate Favoring (Inverse Sqrt)
```bash
python -m src.train_classifiers_with_ho_multifeat \
    -e data/experiments/zero-shot-2 \
    --features-base-dir data/experiments/zero-shot-2/features \
    --metric f1_macro \
    --weight-strategy inverse_sqrt \
    --weight-alpha 0.7 \
    --top-k 5
```

### Ordinal Distance (Quadratic Penalty)
```bash
python -m src.train_classifiers_with_ho_multifeat \
    -e data/experiments/zero-shot-2 \
    --features-base-dir data/experiments/zero-shot-2/features \
    --metric ordinal_distance_quadratic \
    --weight-strategy inverse \
    --weight-alpha 0.7 \
    --top-k 5
```

### Ordinal Accuracy (Adjacent Levels OK)
```bash
python -m src.train_classifiers_with_ho_multifeat \
    -e data/experiments/zero-shot-2 \
    --features-base-dir data/experiments/zero-shot-2/features \
    --metric ordinal_accuracy \
    --top-k 5
```

### Ordinal MSE (Regression-style)
```bash
python -m src.train_classifiers_with_ho_multifeat \
    -e data/experiments/zero-shot-2 \
    --features-base-dir data/experiments/zero-shot-2/features \
    --metric ordinal_mse \
    --weight-strategy inverse \
    --weight_alpha 0.7 \
    --top-k 5
```

### With Early Discarding
```bash
python -m src.train_classifiers_with_ho_multifeat \
    -e data/experiments/zero-shot-2 \
    --features-base-dir data/experiments/zero-shot-2/features \
    --metric ordinal_distance_quadratic \
    --weight-strategy inverse \
    --weight-alpha 0.5 \
    --early-discard-threshold 0.70 \
    --early-discard-percentile 25 \
    --top-k 3 \
    --stage2-trials 150
```

## Complete Flag Reference

### Metric & Weighting
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--metric` | choice | `f1_macro` | Optimization metric |
| `--weight-strategy` | choice | `equal` | Sample weighting strategy |
| `--weight-alpha` | float | `1.0` | Weight calibration (0-1) |

### Feature Selection
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--features-base-dir` | str | **required** | Base features directory |
| `--feature-pattern` | str | `*_tfidf*` | Glob pattern for feature dirs |

### Early Discarding
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--early-discard-threshold` | float | None | Absolute metric threshold |
| `--early-discard-percentile` | float | None | Relative percentile threshold |

### Stage 2
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--top-k` | int | `5` | Top configs to optimize |
| `--stage2-trials` | int | `50` | Trials per config |

### Classifier
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--classifier` | choice | `xgboost` | xgboost or logistic |
| `--xgb-use-gpu` | flag | False | Use GPU for XGBoost |

## Outputs

All outputs saved to `{experiment_dir}/ho_multifeat_results/`:

1. **`stage1_screening_results.json`**
   - All feature configs with metric scores
   - Pass/fail screening status

2. **`stage2_optimization_results.json`**
   - Top-K configs with best hyperparameters
   - Improvement metrics

3. **`stage2_detailed_results.json`**
   - Full Optuna trial history for each config
   - All trials with params and scores

4. **`optimization_summary.md`**
   - Human-readable report
   - Best configuration summary
   - Ranking tables

## Example Output

```
================================================================================
STAGE 1: Feature Configuration Screening
================================================================================
Evaluating 24 feature configurations...
Classifier: xgboost
Metric: f1_macro
Weight strategy: inverse (alpha=0.5)
Validation split: 0.2
Early discard threshold: f1_macro < 0.3500

[1/24] Evaluating 84cbc90c_tfidf_grouped/norm-EFCAMDAT-train... f1_macro: 0.4823
[2/24] Evaluating 252cd532_tfidf/norm-EFCAMDAT-train... f1_macro: 0.4621
...

Top 10 feature configurations:
  1. [✓] 84cbc90c_tfidf_grouped   f1_macro=0.4823 ( 5000 features)
  2. [✓] 252cd532_tfidf           f1_macro=0.4621 ( 5000 features)
  ...
  9. [✗] abc12345_tfidf           f1_macro=0.3301 ( 500 features) - f1_macro < threshold

Passed screening: 18
Discarded: 6

================================================================================
STAGE 2: Deep Hyperparameter Optimization
================================================================================
Optimizing top 5 feature configurations
Metric: f1_macro
Weight strategy: inverse (alpha=0.5)

[1/5] Optimizing: 84cbc90c_tfidf_grouped
  Stage 1 f1_macro: 0.4823
  Running 100 trials...
  ✓ Best f1_macro: 0.5412 (+0.0589 improvement)

...

STAGE 2 COMPLETE - Final Rankings:
Rank   Feature Config            Stage1     Stage2     Improve
--------------------------------------------------------------------------------
1      84cbc90c_tfidf_grouped    0.4823     0.5412     +0.0589
2      252cd532_tfidf            0.4621     0.5234     +0.0613
```

## Recommendations

### For CEFR Classification (Ordinal Nature) - **RECOMMENDED**
```bash
--metric ordinal_distance_quadratic \
--weight-strategy inverse \
--weight-alpha 0.7
```
**Why:** Heavily penalizes confusing A1 with C2, while treating B1/B2 confusion as less severe.

### For Imbalanced CEFR Data (Standard)
```bash
--metric f1_macro \
--weight-strategy inverse \
--weight-alpha 0.7
```

### For Adjacent Accuracy (Lenient)
```bash
--metric ordinal_accuracy \
--weight-strategy inverse_sqrt \
--weight-alpha 0.5
```
**Why:** Useful if you consider ±1 level acceptable (e.g., B1 vs B2 is OK)

### For Balanced Data
```bash
--metric accuracy \
--weight-strategy equal
```

### For Precision-Critical Applications
```bash
--metric precision_macro \
--weight-strategy inverse_sqrt \
--weight-alpha 0.5
```

### For Recall-Critical Applications
```bash
--metric recall_macro \
--weight-strategy inverse \
--weight-alpha 1.0
```

## Metric Comparison Table

| Metric | A1→A2 Error | A1→B1 Error | A1→C2 Error | Best For |
|--------|-------------|-------------|-------------|----------|
| `accuracy` | 0.0 | 0.0 | 0.0 | Equal weight all errors |
| `f1_macro` | varies | varies | varies | Class-balanced performance |
| `ordinal_accuracy` | 1.0 ✓ | 0.0 ✗ | 0.0 ✗ | Lenient (±1 OK) |
| `ordinal_distance_linear` | 0.8 | 0.6 | 0.0 | Proportional penalty |
| `ordinal_distance_quadratic` | 0.96 | 0.84 | 0.0 | **Heavy penalty for far errors** |
| `ordinal_mse` | 0.96 | 0.84 | 0.0 | **Same as quadratic (MSE formulation)** |

**Example scores for predicting A2 when true label is A1:**
- Distance = 1 level
- `accuracy` = 0.0 (wrong)
- `ordinal_accuracy` = 1.0 (within ±1)
- `ordinal_distance_linear` = 1 - 1/5 = 0.8
- `ordinal_distance_quadratic` = 1 - 1²/5² = 0.96
- `ordinal_mse` = 1 - 1/25 = 0.96 (MSE=1, max_MSE=25)
