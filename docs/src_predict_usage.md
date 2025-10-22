# `src.predict` Usage Guide

Generate predictions using trained CEFR classification models.

## Table of Contents
- [Overview](#overview)
- [Two Operating Modes](#two-operating-modes)
- [Quick Start Examples](#quick-start-examples)
- [Common Usage Patterns](#common-usage-patterns)
- [Configuration Flags Reference](#configuration-flags-reference)
- [Expected File Structure](#expected-file-structure)
- [Common Pitfalls](#common-pitfalls)
- [Output Structure](#output-structure)

---

## Overview

The `src.predict` module makes predictions on test data using trained classifiers. It supports:

- **Features mode** (default): Use pre-extracted TF-IDF features
- **Text mode** (legacy): Preprocess raw text with TF-IDF pipeline
- **Single model** prediction or **batch processing** of multiple models
- **Multiple prediction strategies**: argmax, rounded average
- **Automatic evaluation** with ground truth labels

**⚠️ IMPORTANT**: For most workflows, use `src.pipeline --steps 4` instead of calling `src.predict` directly. The pipeline automatically handles model-feature matching.

---

## Two Operating Modes

### 1. Features Mode (Default & Recommended)

Uses pre-extracted TF-IDF features from `src.extract_features`.

```bash
python -m src.predict \
    -e data/experiments/90-10 \
    -m CELVA-SP__90-10__train_xgboost_063988a0_tfidf \
    -d data/experiments/90-10/features/063988a0_tfidf/CELVA-SP__90-10__test \
    --cefr-column cefr_level
```

**When to use**: When features are already extracted (after running pipeline steps 1-2).

### 2. Text Mode (Legacy)

Preprocesses raw text using TF-IDF pipeline from scratch.

```bash
python -m src.predict \
    -e data/experiments/90-10 \
    -m CELVA-SP__90-10__train_xgboost_063988a0_tfidf \
    --preprocess-text \
    -t data/experiments/90-10/ml-test-data/CELVA-SP__90-10__test.csv \
    --text-column text \
    --cefr-column cefr_level
```

**When to use**: Rarely. Only for quick testing or when features aren't pre-extracted.

---

## Quick Start Examples

### Example 1: Single Model, Single Dataset (Features Mode)

```bash
python -m src.predict \
    -e data/experiments/90-10 \
    -m KUPA-KEYS__90-10__train_xgboost_063988a0_tfidf \
    -d data/experiments/90-10/features/063988a0_tfidf/KUPA-KEYS__90-10__test \
    --cefr-column cefr_level
```

**Output**: Creates `data/experiments/90-10/results/KUPA-KEYS__90-10__train_xgboost_063988a0_tfidf/KUPA-KEYS__90-10__test/`

### Example 2: Single Model, All Datasets (Batch Features)

```bash
python -m src.predict \
    -e data/experiments/90-10 \
    -m CELVA-SP__90-10__train_logistic_252cd532_tfidf \
    --batch-features-dir data/experiments/90-10/features/252cd532_tfidf \
    --cefr-column cefr_level
```

Predicts on all feature directories within `252cd532_tfidf/` (e.g., `*__test/`, `*__train/`).

### Example 3: All Models, All Datasets (Full Batch)

```bash
python -m src.predict \
    -e data/experiments/90-10 \
    --batch-models-dir data/experiments/90-10/feature-models/classifiers \
    --batch-features-dir data/experiments/90-10/features \
    --cefr-column cefr_level
```

**This is what `pipeline --steps 4` does internally.**

Automatically matches each model with its corresponding TF-IDF features based on hash in model name.

---

## Common Usage Patterns

### Pattern 1: Using the Pipeline (Recommended)

**Instead of calling `src.predict` directly**, use the pipeline:

```bash
python -m src.pipeline \
    -e data/experiments/90-10 \
    --steps 4 \
    --cefr-column cefr_level
```

**Why?** Pipeline handles:
- Model-to-feature matching automatically
- Error handling and progress reporting
- Consistent configuration

### Pattern 2: Predict on New Test Set

If you've trained models and want to evaluate on a new test set:

```bash
# Step 1: Extract features for new test set with all TF-IDF models
python -m src.extract_features \
    -e data/experiments/90-10 \
    --batch-tfidf-dir data/experiments/90-10/feature-models \
    -t data/new-test/test.csv \
    --cefr-column cefr_level

# Step 2: Predict with all models
python -m src.predict \
    -e data/experiments/90-10 \
    --batch-models-dir data/experiments/90-10/feature-models/classifiers \
    --batch-features-dir data/experiments/90-10/features \
    --cefr-column cefr_level
```

### Pattern 3: Evaluate Single Model on Training Data

Check overfitting by evaluating on training set:

```bash
python -m src.predict \
    -e data/experiments/90-10 \
    -m CELVA-SP__90-10__train_xgboost_063988a0_tfidf \
    -d data/experiments/90-10/features/063988a0_tfidf/CELVA-SP__90-10__train \
    --cefr-column cefr_level
```

### Pattern 4: Predict Without Ground Truth Labels

For real inference (no evaluation):

```bash
python -m src.predict \
    -e data/experiments/90-10 \
    -m MODEL_NAME \
    -f unlabeled_features.csv \
    --no-save-results  # Skip evaluation
```

**Output**: Only prediction JSONs, no evaluation report.

---

## Configuration Flags Reference

### Essential Flags

| Flag | Description | Required? |
|------|-------------|-----------|
| `-e, --experiment-dir` | Experiment directory path | ✅ Yes |
| `-m, --classifier-model` | Model name to use | Yes (unless `--batch-models-dir`) |
| `--cefr-column` | Column name for CEFR labels in CSV | ⚠️ Almost always needed |

### Input Selection (Features Mode)

| Flag | Use Case |
|------|----------|
| `-d, --feature-dir` | Single feature directory |
| `-f, --features-file` | Single features CSV file |
| `--batch-features-dir` | All feature directories |
| `--batch-models-dir` | All classifier models (requires `--batch-features-dir`) |

### Labels Specification

| Flag | Description |
|------|-------------|
| `--labels-csv` | CSV file with labels (single file) |
| `--labels-csv-dir` | Directory containing multiple label CSVs (batch mode) |
| `--labels-file` | Plain text file with one label per line |

**Default behavior**: Script looks for labels CSV matching feature directory name in `ml-test-data/` then `ml-training-data/`.

### Output Control

| Flag | Effect |
|------|--------|
| `--no-save-results` | Skip saving any results |
| `--no-save-csv` | Skip CSV predictions (keep JSON) |
| `--no-save-json` | Skip JSON predictions (keep CSV) |
| `-q, --quiet` | Suppress verbose output |

### ⚠️ Dangerous Flags

| Flag | ⚠️ Warning |
|------|-----------|
| `-o, --output-dir` | **DO NOT USE for predictions!** This overrides `models_dir` (where classifiers are loaded from), NOT where results are saved. Results always go to `<experiment-dir>/results/`. Only use `-o` when **training** models. |

---

## Expected File Structure

### Input Structure (Features Mode)

```
data/experiments/90-10/
├── feature-models/
│   └── classifiers/
│       └── CELVA-SP__90-10__train_xgboost_063988a0_tfidf/
│           ├── classifier.pkl
│           ├── label_encoder.pkl
│           ├── xgb_label_mapping.pkl  # (for XGBoost)
│           └── config.json
├── features/
│   └── 063988a0_tfidf/
│       └── CELVA-SP__90-10__test/
│           ├── features_dense.csv      # Input features
│           └── feature_names.csv
└── ml-test-data/
    └── CELVA-SP__90-10__test.csv      # Ground truth labels
```

### Input Structure (Text Mode with --preprocess-text)

```
data/experiments/90-10/
├── feature-models/
│   ├── classifiers/
│   │   └── MODEL_NAME/
│   │       └── classifier.pkl
│   └── 063988a0_tfidf/
│       └── tfidf_model.pkl             # TF-IDF model
└── ml-test-data/
    └── test.csv                        # Raw text + labels
```

---

## Common Pitfalls

### ❌ Pitfall 1: Using `-o` Flag for Predictions

**Wrong:**
```bash
python -m src.predict \
    -e data/experiments/90-10 \
    -o data/experiments/90-10/results \  # ❌ DON'T DO THIS
    --batch-models-dir ...
```

**Error:** `Classifier directory not found: data/experiments/90-10/results/classifiers/MODEL_NAME`

**Why?** The `-o` flag sets `models_dir` (where classifiers are loaded FROM), not where results are saved TO.

**Right:**
```bash
python -m src.predict \
    -e data/experiments/90-10 \  # ✅ No -o flag needed
    --batch-models-dir data/experiments/90-10/feature-models/classifiers \
    --batch-features-dir data/experiments/90-10/features \
    --cefr-column cefr_level
```

### ❌ Pitfall 2: Wrong CEFR Column Name

**Error:** `Column 'cefr_label' not found in CSV`

**Solution:** Check your CSV headers:
```bash
head -1 data/experiments/90-10/ml-test-data/test.csv
```

If it shows `cefr_level`, use `--cefr-column cefr_level`.

### ❌ Pitfall 3: Model-Feature Hash Mismatch

**Wrong:**
```bash
python -m src.predict \
    -m MODEL_xgboost_063988a0_tfidf \      # Model uses 063988a0
    -d features/252cd532_tfidf/test \      # ❌ Features use 252cd532
```

**Error:** Model trained on different features, predictions will be garbage.

**Solution:** Match TF-IDF hashes:
```bash
# Extract hash from model name
MODEL=CELVA-SP__90-10__train_xgboost_063988a0_tfidf
HASH=$(echo $MODEL | grep -oP '\w{8}(?=_tfidf)')  # 063988a0

# Use matching features
python -m src.predict \
    -m $MODEL \
    -d features/${HASH}_tfidf/test_set \
    --cefr-column cefr_level
```

Or just use batch mode:
```bash
python -m src.predict \
    -e data/experiments/90-10 \
    --batch-models-dir feature-models/classifiers \
    --batch-features-dir features \
    --cefr-column cefr_level
```

### ❌ Pitfall 4: Missing Ground Truth Labels

**Warning:** `No ground truth CSV found for DATASET_NAME`

**Cause:** Script can't find CSV matching feature directory name.

**Solution:** Ensure naming consistency:
```
features/063988a0_tfidf/CELVA-SP__90-10__test/  ← Feature dir name
ml-test-data/CELVA-SP__90-10__test.csv          ← Must match exactly
```

Or specify explicitly:
```bash
python -m src.predict \
    -d features/063988a0_tfidf/my-test \
    --labels-csv data/ground-truth/my-test.csv \
    --cefr-column cefr_level
```

---

## Output Structure

### Generated Files

For each model-dataset combination, the script creates:

```
results/
└── CELVA-SP__90-10__train_xgboost_063988a0_tfidf/
    └── CELVA-SP__90-10__test/
        ├── evaluation_report.md          # Human-readable metrics
        ├── soft_predictions.json         # Raw probabilities
        ├── argmax_predictions.json       # Argmax predictions
        └── rounded_avg_predictions.json  # Rounded average predictions
```

### Evaluation Report Contents

The `evaluation_report.md` includes:

- **Dataset Info**: Samples, classes present
- **Model Info**: Classifier type, TF-IDF config
- **Strategy 1 (Argmax)**: Accuracy, adjacent accuracy, F1 scores, confusion matrix
- **Strategy 2 (Rounded Avg)**: Same metrics for regression-style rounding

### Prediction JSON Format

```json
{
  "sample_id_1": {
    "A1": 0.05,
    "A2": 0.10,
    "B1": 0.60,
    "B2": 0.20,
    "C1": 0.04,
    "C2": 0.01
  },
  "sample_id_2": { ... }
}
```

---

## Advanced Examples

### Example A: Cross-Dataset Evaluation

Evaluate CELVA-trained model on KUPA test set:

```bash
python -m src.predict \
    -e data/experiments/90-10 \
    -m CELVA-SP__90-10__train_xgboost_063988a0_tfidf \
    -d data/experiments/90-10/features/063988a0_tfidf/KUPA-KEYS__90-10__test \
    --cefr-column cefr_level
```

### Example B: Ensemble Prediction

Evaluate multiple models on same dataset to compare:

```bash
for model in CELVA*_logistic_*_tfidf CELVA*_xgboost_*_tfidf; do
    python -m src.predict \
        -e data/experiments/90-10 \
        -m $model \
        -d features/063988a0_tfidf/CELVA-SP__90-10__test \
        --cefr-column cefr_level \
        -q
done
```

### Example C: Quick Model Test (Text Mode)

Test a model without pre-extracting features:

```bash
python -m src.predict \
    -e data/experiments/90-10 \
    -m CELVA-SP__90-10__train_logistic_063988a0_tfidf \
    --preprocess-text \
    -t data/quick-test/samples.csv \
    --text-column text \
    --cefr-column cefr_level \
    -q
```

**Note:** This is slower and only works if the TF-IDF model exists.

---

## Integration with Other Modules

### Typical Workflow

```bash
# 1. Train TF-IDF and extract features
python -m src.pipeline -e data/experiments/90-10 --steps 1 2 --cefr-column cefr_level

# 2. Train classifiers
python -m src.pipeline -e data/experiments/90-10 --steps 3 --cefr-column cefr_level

# 3. Make predictions (THIS MODULE)
python -m src.pipeline -e data/experiments/90-10 --steps 4 --cefr-column cefr_level

# 4. Generate reports
python -m src.report -e data/experiments/90-10 --rank accuracy --summary-report results_summary.md
```

### Or All at Once

```bash
python -m src.pipeline \
    -e data/experiments/90-10 \
    --max-features-list 1000 5000 10000 \
    --classifiers logistic xgboost \
    --cefr-column cefr_level \
    --summarize
```

---

## Troubleshooting

### Check Model-Feature Compatibility

```bash
# Get model's TF-IDF hash
model_config=$(cat feature-models/classifiers/MODEL_NAME/config.json)
echo $model_config | grep tfidf_hash

# Verify matching features exist
ls features/HASH_tfidf/
```

### Verify Labels Column Name

```bash
head -1 ml-test-data/test.csv | tr ',' '\n' | nl
```

### Test Single Sample

```bash
# Create minimal test file
echo "writing_id,cefr_level,text" > test_single.csv
echo "test1,B1,This is a test sentence." >> test_single.csv

# Run prediction
python -m src.predict \
    -e data/experiments/90-10 \
    -m MODEL_NAME \
    --preprocess-text \
    -t test_single.csv \
    --cefr-column cefr_level \
    -v
```

---

## See Also

- **[`src.pipeline`](./pipeline_usage.md)**: Recommended wrapper for end-to-end workflows
- **[`src.train_classifiers`](./train_classifiers_usage.md)**: Train models before prediction
- **[`src.extract_features`](./extract_features_usage.md)**: Extract features before prediction
- **[`src.report`](./report_usage.md)**: Analyze prediction results
