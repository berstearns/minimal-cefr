# Report Functionality Guide

Complete guide to using `src.report` for analyzing and ranking model results across experiments.

## Overview

The report tool (`src.report`) scans experiment results, extracts performance metrics, and generates comprehensive summaries and rankings. It aggregates results from multiple models, datasets, and prediction strategies to help you identify the best-performing configurations.

## Table of Contents

- [Input Structure](#input-structure)
- [Prediction Files Used](#prediction-files-used)
- [File Formats](#file-formats)
- [Usage Examples](#usage-examples)
- [Output Types](#output-types)
- [Understanding the Results](#understanding-the-results)

---

## Input Structure

The report tool expects results to be organized in the following structure:

```
experiment-dir/
├── results/
│   ├── model-1_xgboost_abc123_tfidf/
│   │   ├── test-set-1/
│   │   │   ├── evaluation_report.md       # ← Parsed by report tool
│   │   │   ├── argmax_predictions.json
│   │   │   ├── rounded_avg_predictions.json
│   │   │   └── soft_predictions.json
│   │   └── test-set-2/
│   │       ├── evaluation_report.md       # ← Parsed by report tool
│   │       └── ...
│   └── model-2_logistic_def456_tfidf/
│       └── ...
└── feature-models/
    └── classifiers/
        ├── model-1_xgboost_abc123_tfidf/
        │   └── config.json                # ← Model configuration
        └── model-2_logistic_def456_tfidf/
            └── config.json
```

### Required Files

**For each model/dataset combination:**
- `results/{model_name}/{dataset_name}/evaluation_report.md`

**For each model:**
- `feature-models/classifiers/{model_name}/config.json`

---

## Prediction Files Used

### Primary File: `evaluation_report.md`

The report tool **primarily parses** `evaluation_report.md` files, which contain pre-computed metrics for both prediction strategies. It does **not** directly read the JSON prediction files.

### Location Pattern

```
results/{model_name}/{dataset_name}/evaluation_report.md
```

Where:
- `{model_name}`: e.g., `train-data_xgboost_c2b5a010_tfidf`
- `{dataset_name}`: e.g., `norm-KUPA-KEYS`, `norm-CELVA-SP`, `test-data`

### Configuration File

```
feature-models/classifiers/{model_name}/config.json
```

Contains model metadata:
```json
{
  "tfidf_hash": "c2b5a010",
  "tfidf_max_features": 5000,
  "tfidf_readable_name": "5k-features",
  "classifier_type": "xgboost"
}
```

---

## File Formats

### 1. Evaluation Report Format (`evaluation_report.md`)

The report tool extracts metrics from two sections in each evaluation report:

#### Section 1: Argmax Predictions

```markdown
## Strategy 1: Argmax Predictions

### CEFR Classification Report

```
    precision    recall  f1-score   support

B1       0.10      0.38      0.16       109
B2       0.60      0.54      0.57       570
C1       0.35      0.07      0.12       312
C2       0.00      0.00      0.00        15

macro avg       0.26      0.25      0.21      1006
weighted avg       0.46      0.37      0.38      1006

accuracy      0.37      1006          ← Extracted
adjacent accuracy      0.83      1006  ← Extracted
```
```

**Extracted Metrics:**
- `accuracy`: Overall prediction accuracy
- `adjacent_accuracy`: CEFR-specific metric (predictions within ±1 level)
- `macro_f1`: Macro-averaged F1-score (from "macro avg" row, 3rd column)
- `weighted_f1`: Weighted F1-score (from "weighted avg" row, 3rd column)

#### Section 2: Rounded Average Predictions

```markdown
## Strategy 2: Rounded Average Predictions

### CEFR Classification Report

```
    precision    recall  f1-score   support

B1       0.11      0.50      0.19       109
B2       0.62      0.54      0.58       570
...

accuracy      0.38      1006          ← Extracted
adjacent accuracy      0.84      1006  ← Extracted
```
```

Same metrics extracted for this strategy.

#### Dataset Metadata

Extracted from report header:
```markdown
**Samples**: 1006                       ← Extracted as n_samples
**Classes in test set**: B1, B2, C1, C2  ← Extracted as classes_in_test
```

### 2. Model Configuration Format (`config.json`)

```json
{
  "tfidf_hash": "c2b5a010",
  "tfidf_max_features": 5000,
  "tfidf_readable_name": "5k-features",
  "classifier_type": "xgboost",
  "timestamp": "2024-10-18T01:15:32"
}
```

**Fields Used by Report Tool:**
- `tfidf_hash`: Unique identifier for TF-IDF configuration
- `tfidf_max_features`: Number of TF-IDF features (e.g., 1000, 5000)
- `tfidf_readable_name`: Human-readable name for TF-IDF config
- `classifier_type`: Classifier algorithm (xgboost, logistic, etc.)

### 3. Prediction Files (Referenced but not parsed by report tool)

#### `argmax_predictions.json`

Standard argmax prediction strategy (highest probability class):

```json
[
  {
    "sample_id": 0,
    "predicted_label": "B1",
    "true_label": "C1",
    "confidence": 0.3506
  },
  {
    "sample_id": 1,
    "predicted_label": "B2",
    "true_label": "B1",
    "confidence": 0.4258
  }
]
```

#### `rounded_avg_predictions.json`

Regression-style prediction (expected class index, rounded):

```json
[
  {
    "sample_id": 0,
    "predicted_label": "B2",
    "true_label": "C1",
    "expected_index": 2.15
  }
]
```

#### `soft_predictions.json`

Full probability distributions:

```json
[
  {
    "sample_id": 0,
    "probabilities": {
      "A1": 0.0449,
      "A2": 0.2783,
      "B1": 0.3506,
      "B2": 0.2468,
      "C1": 0.0792
    }
  }
]
```

**Note:** These JSON files are created by `src.predict` but are **not directly read** by `src.report`. The report tool only parses the pre-computed metrics in `evaluation_report.md`.

---

## Usage Examples

### Basic Usage

#### 1. Quick Overview

```bash
python -m src.report -e data/experiments/zero-shot
```

**Output:**
```
Quick Overview:
  Total evaluations: 120
  Unique models: 30
  Datasets: 4

Use --rank or --summary-report to see detailed results.
```

#### 2. Rank by Accuracy

```bash
python -m src.report -e data/experiments/zero-shot --rank accuracy
```

**Output:**
```
================================================================================
RANKING BY: ACCURACY (Grouped by Dataset)
================================================================================

📊 Dataset: norm-CELVA-SP
--------------------------------------------------------------------------------
Rank   Model                           Strategy      Accuracy   TF-IDF              Classifier
------ ------------------------------- ------------ ---------- ------------------- ------------
1      train-data_xgboost_c2b5a010...  argmax       0.6234     5k-features         xgboost
2      train-data_logistic_c2b5a010... argmax       0.6102     5k-features         logistic
3      train-data_xgboost_abc123...    argmax       0.5987     10k-features        xgboost
...
```

#### 3. Rank Specific Dataset

```bash
python -m src.report -e data/experiments/zero-shot \
    --rank accuracy \
    --dataset norm-KUPA-KEYS \
    --top 10
```

Shows top 10 models for a specific dataset (flat ranking, not grouped).

#### 4. Compare Prediction Strategies

```bash
python -m src.report -e data/experiments/zero-shot \
    --rank accuracy \
    --strategy argmax
```

Only shows results for argmax strategy.

#### 5. Rank by Adjacent Accuracy

```bash
python -m src.report -e data/experiments/zero-shot \
    --rank adjacent_accuracy \
    --strategy rounded_avg
```

Ranks by CEFR-specific adjacent accuracy metric for rounded_avg strategy.

### Advanced Usage

#### 6. Generate Comprehensive Summary Report

```bash
python -m src.report -e data/experiments/zero-shot \
    --summary-report results_summary.md
```

Creates detailed markdown report including:
- Top 10 models by accuracy
- Top 10 models by adjacent accuracy
- Top 10 models by macro F1
- Top 10 models by weighted F1
- Performance breakdown by dataset
- TF-IDF configuration comparison

#### 7. Rank by F1 Scores

```bash
# Macro F1
python -m src.report -e data/experiments/zero-shot \
    --rank macro_f1 \
    --top 20

# Weighted F1
python -m src.report -e data/experiments/zero-shot \
    --rank weighted_f1 \
    --top 20
```

#### 8. Minimal Output (No Config Details)

```bash
python -m src.report -e data/experiments/zero-shot \
    --rank accuracy \
    --no-config
```

Hides TF-IDF and classifier configuration columns for cleaner output.

#### 9. Flat Ranking (No Grouping)

```bash
python -m src.report -e data/experiments/zero-shot \
    --rank accuracy \
    --no-group
```

Shows all results in a single ranked list (instead of grouped by dataset).

#### 10. Verbose Mode

```bash
python -m src.report -e data/experiments/zero-shot \
    --rank accuracy \
    -v
```

Shows progress information while scanning directories.

---

## Output Types

### 1. Grouped Ranking (Default)

When `--rank` is used without `--dataset` or `--no-group`:

```
📊 Dataset: norm-CELVA-SP
--------------------------------------------------------------------------------
Rank   Model                           Strategy      Accuracy   ...
1      model-a                         argmax        0.6234     ...
2      model-b                         argmax        0.6102     ...

📊 Dataset: norm-KUPA-KEYS
--------------------------------------------------------------------------------
Rank   Model                           Strategy      Accuracy   ...
1      model-c                         argmax        0.5523     ...
2      model-d                         argmax        0.5401     ...
```

### 2. Flat Ranking

When `--dataset` filter is used or `--no-group` is specified:

```
================================================================================
RANKING BY: ACCURACY
================================================================================
Rank   Model                Dataset          Strategy    Accuracy   ...
1      model-a              norm-CELVA-SP    argmax      0.6234     ...
2      model-b              norm-CELVA-SP    argmax      0.6102     ...
3      model-c              norm-KUPA-KEYS   argmax      0.5523     ...
```

### 3. Summary Report (Markdown)

Generated with `--summary-report`:

```markdown
# Experiment Results Summary: zero-shot

## Overview

- **Total Models:** 30
- **Datasets Evaluated:** 4
- **Prediction Strategies:** 2
- **Total Evaluations:** 240

## Top 10 Models by Accuracy

### Strategy: Argmax

| Rank | Model | Dataset | Accuracy | TF-IDF Config | Classifier |
|------|-------|---------|----------|---------------|------------|
| 1    | `train-data_xgboost_c2b5a010_tfidf` | norm-CELVA-SP | 0.6234 | 5k-features | xgboost |
...

## Performance by Dataset

### norm-CELVA-SP

- **Best Accuracy:** 0.6234 (`train-data_xgboost_c2b5a010_tfidf`)
- **Best Adjacent Accuracy:** 0.8912 (`train-data_logistic_abc123_tfidf`)
- **Models Evaluated:** 30

## TF-IDF Configuration Analysis

| TF-IDF Config | Max Features | Avg Accuracy | Min Accuracy | Max Accuracy | Evaluations |
|---------------|--------------|--------------|--------------|--------------|-------------|
| 5k-features   | 5000         | 0.5834       | 0.4521       | 0.6234       | 120         |
| 10k-features  | 10000        | 0.5723       | 0.4401       | 0.6102       | 120         |
```

---

## Understanding the Results

### Metrics Explained

**accuracy**
- Standard classification accuracy
- Percentage of exact matches between predicted and true labels
- Range: 0.0 (0%) to 1.0 (100%)

**adjacent_accuracy**
- CEFR-specific ordinal metric
- Predictions within ±1 CEFR level count as correct
- Example: Predicting B2 when true label is B1 or C1 counts as correct
- Higher values than standard accuracy due to ordinal nature of CEFR

**macro_f1**
- Average F1-score across all classes (unweighted)
- Treats all classes equally, regardless of support
- Good for balanced datasets or when minority classes are important

**weighted_f1**
- Average F1-score weighted by class support
- Better reflects performance on imbalanced datasets
- Emphasizes performance on majority classes

### Prediction Strategies

**argmax** (Strategy 1)
- Standard classification approach
- Select class with highest probability
- Formula: `argmax(P(class | input))`
- Best for: Clear decision boundaries

**rounded_avg** (Strategy 2)
- Regression-style approach for ordinal data
- Calculate expected class index from probabilities
- Round to nearest integer and map to class
- Formula: `round(Σ(class_index × P(class)))`
- Best for: Ordinal data like CEFR levels

### Model Name Format

Model names follow the pattern:
```
{train-set}_{classifier}_{tfidf-hash}[_{feature-type}]
```

Examples:
- `train-data_xgboost_c2b5a010_tfidf`
  - Trained on: `train-data`
  - Classifier: `xgboost`
  - TF-IDF config hash: `c2b5a010`
  - Feature type: `tfidf`

- `norm-EFCAMDAT-train_logistic_abc123_tfidf`
  - Trained on: `norm-EFCAMDAT-train`
  - Classifier: `logistic`
  - TF-IDF config hash: `abc123`
  - Feature type: `tfidf`

### TF-IDF Hash

The TF-IDF hash (e.g., `c2b5a010`) uniquely identifies a TF-IDF configuration:
- `max_features`: Number of vocabulary terms
- `ngram_range`: N-gram range (e.g., 1-2 for unigrams + bigrams)
- `min_df`, `max_df`: Document frequency thresholds
- Other TF-IDF parameters

Models with the same hash use identical TF-IDF vectorizers.

### Reading the Rankings

**When grouped by dataset:**
- Each dataset shows independent rankings
- Rank 1 in each group is best for that specific dataset
- Useful for comparing: "What works best for CELVA-SP vs KUPA-KEYS?"

**When flat ranking:**
- All results ranked globally
- Best overall model appears first
- Useful for comparing: "What's the single best model across all data?"

**When filtering by strategy:**
- Compare argmax vs rounded_avg independently
- Some models may perform better with one strategy vs the other
- Adjacent accuracy often higher with rounded_avg for ordinal tasks

---

## Common Workflows

### Workflow 1: Find Best Model for Production

```bash
# 1. Generate full summary
python -m src.report -e experiments/my-exp \
    --summary-report summary.md

# 2. Rank by accuracy for target dataset
python -m src.report -e experiments/my-exp \
    --rank accuracy \
    --dataset production-test-set \
    --top 5

# 3. Check adjacent accuracy for top candidates
python -m src.report -e experiments/my-exp \
    --rank adjacent_accuracy \
    --dataset production-test-set \
    --top 5
```

### Workflow 2: Compare TF-IDF Configurations

```bash
# Generate summary with TF-IDF analysis
python -m src.report -e experiments/my-exp \
    --summary-report tfidf_comparison.md

# Review "TF-IDF Configuration Analysis" section in output
```

### Workflow 3: Compare Prediction Strategies

```bash
# Argmax results
python -m src.report -e experiments/my-exp \
    --rank accuracy \
    --strategy argmax \
    --top 10

# Rounded average results
python -m src.report -e experiments/my-exp \
    --rank accuracy \
    --strategy rounded_avg \
    --top 10
```

### Workflow 4: Quick Model Selection

```bash
# Rank all models by adjacent accuracy (CEFR-specific metric)
python -m src.report -e experiments/my-exp \
    --rank adjacent_accuracy \
    --no-group \
    --top 20
```

---

## Troubleshooting

### No results found

**Cause:** Missing `evaluation_report.md` files

**Solution:**
```bash
# Run predictions first
python -m src.predict -e experiments/my-exp \
    --classifier-model your-model \
    --features-file path/to/features.csv
```

### Missing configuration data

**Cause:** Missing `config.json` in `feature-models/classifiers/{model_name}/`

**Solution:**
- Config files are created automatically during training
- Re-train the model or manually create config.json with required fields

### Metrics showing "N/A"

**Cause:** Parsing failed or metrics not present in evaluation report

**Solution:**
- Check that `evaluation_report.md` has the expected format
- Ensure it contains both "Strategy 1" and "Strategy 2" sections
- Verify the classification reports use the expected format

---

## Configuration Files Reference

### Experiment Directory Structure

```
experiments/my-experiment/
├── ml-training-data/
│   └── train.csv
├── ml-test-data/
│   ├── test-set-1.csv
│   └── test-set-2.csv
├── feature-models/
│   ├── tfidf/
│   │   └── hash_c2b5a010/
│   │       └── tfidf_model.pkl
│   └── classifiers/
│       ├── train_xgboost_c2b5a010_tfidf/
│       │   ├── model.pkl
│       │   └── config.json              ← Used by report tool
│       └── train_logistic_c2b5a010_tfidf/
│           ├── model.pkl
│           └── config.json
├── features/
│   └── c2b5a010_tfidf/
│       ├── train/
│       └── test-set-1/
├── results/                              ← Scanned by report tool
│   ├── train_xgboost_c2b5a010_tfidf/
│   │   ├── test-set-1/
│   │   │   ├── evaluation_report.md     ← Primary data source
│   │   │   ├── argmax_predictions.json
│   │   │   ├── rounded_avg_predictions.json
│   │   │   └── soft_predictions.json
│   │   └── test-set-2/
│   │       └── [same structure]
│   └── train_logistic_c2b5a010_tfidf/
│       └── [same structure]
└── config.yaml
```

### Required Fields in `config.json`

```json
{
  "tfidf_hash": "c2b5a010",           # Required
  "tfidf_max_features": 5000,         # Optional (shown in tables)
  "tfidf_readable_name": "5k-feats",  # Optional (shown in tables)
  "classifier_type": "xgboost"        # Optional (shown in tables)
}
```

---

## Manual Predictions Mode

The report tool also supports **manual predictions mode** for analyzing results from external models (LLMs, etc.).

See **[REPORT_MANUAL_PREDICTIONS.md](REPORT_MANUAL_PREDICTIONS.md)** for:
- Using the report tool with prediction JSON files
- Auto-detection of structure type
- Fuzzy dataset name matching
- Computing metrics on-the-fly

**Quick Example:**
```bash
python -m src.report \
    -e data/experiments/prompting \
    --labels-dir data/labels \
    --rank accuracy
```

---

## See Also

- **[REPORT_MANUAL_PREDICTIONS.md](REPORT_MANUAL_PREDICTIONS.md)** - Manual predictions mode guide
- **[USAGE.md](USAGE.md)** - Running the prediction pipeline
- **[CEFR_METRICS.md](CEFR_METRICS.md)** - Understanding CEFR metrics
- **[SIMPLE_TRAINING_GUIDE.md](SIMPLE_TRAINING_GUIDE.md)** - Training workflow
- **[MANUAL_PREDICTIONS_GUIDE.md](MANUAL_PREDICTIONS_GUIDE.md)** - Adding custom predictions

---

Last updated: 2025-10-21
