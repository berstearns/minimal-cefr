# Manual Predictions Guide

This guide explains how to manually add predictions for a model to an experiment folder, ensuring compliance with the expected folder structure.

## Overview

This is useful when you have predictions from external models (e.g., language models like GPT, Claude, or other systems) that you want to evaluate using the existing pipeline infrastructure.

## Experiment Folder Structure

An experiment folder (e.g., `data/experiments/zero-shot-2`) has the following structure:

```
data/experiments/zero-shot-2/
├── ml-training-data/
│   └── norm-EFCAMDAT-train.csv          # Training data (text + CEFR labels)
├── ml-test-data/
│   ├── norm-CELVA-SP.csv                # Test dataset 1
│   ├── norm-EFCAMDAT-test.csv           # Test dataset 2
│   └── norm-KUPA-KEYS.csv               # Test dataset 3
├── feature-models/                       # Created by pipeline
│   ├── {hash}_tfidf/
│   │   ├── tfidf_model.pkl
│   │   └── config.json
│   └── ...
├── features/                             # Created by pipeline
│   ├── {hash}_tfidf/
│   │   ├── train-data/
│   │   │   ├── features_dense.csv
│   │   │   └── feature_names.csv
│   │   ├── norm-CELVA-SP/
│   │   └── ...
│   └── ...
└── results/                              # Where predictions go
    ├── {model_name}/
    │   ├── {test-set-1}/
    │   │   ├── argmax_predictions.json
    │   │   ├── soft_predictions.json
    │   │   ├── rounded_avg_predictions.json
    │   │   └── evaluation_report.md
    │   └── {test-set-2}/
    │       └── ...
    └── ...
```

## Required Test Data Format

Each test dataset CSV must contain these columns:

```csv
writing_id,l1,cefr_level,text
0,French,B1,"Text content here..."
1,Spanish,A2,"More text content..."
```

**Required columns:**
- `writing_id`: Unique identifier for each sample
- `cefr_level`: Ground truth CEFR level (A1, A2, B1, B2, C1, C2)
- `text`: The text to classify
- `l1`: (optional) First language of the writer

The column names can be configured, but these are the defaults used by the pipeline.

## Adding Manual Predictions

### Step 1: Create Model Directory

Create a directory under `results/` with a descriptive name for your model:

```bash
mkdir -p data/experiments/zero-shot-2/results/my-model-name
```

**Naming convention:**
- Use descriptive names like `gpt4-zero-shot`, `claude-3.5-fewshot`, `llama-3.1-70b`
- For traditional ML models: `{train-data}_{classifier}_{hash}_{features}`
  - Example: `norm-EFCAMDAT-train_logistic_005ebc16_tfidf`

### Step 2: Create Test Set Subdirectories

For each test set you want to evaluate, create a subdirectory:

```bash
mkdir -p data/experiments/zero-shot-2/results/my-model-name/norm-CELVA-SP
mkdir -p data/experiments/zero-shot-2/results/my-model-name/norm-EFCAMDAT-test
```

The directory name must **exactly match** the test set filename (without `.csv`).

### Step 3: Create Prediction Files

You need to create up to 3 prediction JSON files in each test set directory:

#### 3a. `argmax_predictions.json` (Required)

Contains the final predicted label for each sample:

```json
[
  {
    "sample_id": 0,
    "predicted_label": "B2",
    "true_label": "B1",
    "confidence": 0.5591052386869284
  },
  {
    "sample_id": 1,
    "predicted_label": "B2",
    "true_label": "A1",
    "confidence": 0.6084546508709694
  }
]
```

**Fields:**
- `sample_id`: Integer index matching the row number in the test CSV (0-indexed)
- `predicted_label`: Your model's prediction (A1, A2, B1, B2, C1, or C2)
- `true_label`: Ground truth from the test CSV
- `confidence`: (Optional) Confidence score between 0 and 1

#### 3b. `soft_predictions.json` (Optional)

Contains probability distributions over all classes:

```json
[
  {
    "sample_id": 0,
    "probabilities": {
      "A1": 0.000028,
      "A2": 0.000409,
      "B1": 0.014735,
      "B2": 0.559105,
      "C1": 0.425722
    },
    "true_label": "B1"
  }
]
```

**Fields:**
- `sample_id`: Integer index (0-indexed)
- `probabilities`: Dictionary with all 6 CEFR levels and their probabilities
  - Probabilities should sum to ~1.0
  - Include all levels, even if probability is very low
- `true_label`: Ground truth label

#### 3c. `rounded_avg_predictions.json` (Optional)

Alternative prediction strategy based on expected value:

```json
[
  {
    "sample_id": 0,
    "predicted_label": "B2",
    "expected_value": 3.85,
    "true_label": "B1",
    "confidence": 0.559105
  }
]
```

This is typically generated from soft predictions by:
1. Computing expected CEFR level (A1=0, A2=1, B1=2, B2=3, C1=4, C2=5)
2. Rounding to nearest integer
3. Converting back to CEFR label

**If you only have hard predictions** (just the predicted class), you only need `argmax_predictions.json`.

### Step 4: Verify File Structure

Your structure should look like:

```
data/experiments/zero-shot-2/results/my-model-name/
├── norm-CELVA-SP/
│   ├── argmax_predictions.json
│   ├── soft_predictions.json          # Optional
│   └── rounded_avg_predictions.json   # Optional
├── norm-EFCAMDAT-test/
│   └── argmax_predictions.json
└── norm-KUPA-KEYS/
    └── argmax_predictions.json
```

### Step 5: Generate Evaluation Reports

Run the report generation to create `evaluation_report.md` files:

```bash
python -m src_cq2.report --experiment-dir data/experiments/zero-shot-2
```

This will:
1. Read all prediction files in `results/`
2. Compare with ground truth from `ml-test-data/`
3. Generate `evaluation_report.md` in each subdirectory
4. Create `results_summary.md` ranking all models

## Important Notes

### Sample ID Alignment

**Critical**: The `sample_id` in your predictions must match the row index in the test CSV file (0-indexed, excluding header).

Example:
```csv
writing_id,l1,cefr_level,text
0,French,B1,"First text..."      # sample_id: 0
1,Spanish,A2,"Second text..."    # sample_id: 1
2,German,B2,"Third text..."      # sample_id: 2
```

### Valid CEFR Labels

Only these labels are valid:
- `A1`, `A2`, `B1`, `B2`, `C1`, `C2`

Case-sensitive! Use uppercase.

### Probability Requirements

If providing `soft_predictions.json`:
- Include all 6 CEFR levels in each `probabilities` dict
- Probabilities should sum to approximately 1.0
- Use small values (e.g., 1e-6) for very unlikely classes, not 0

### Missing Predictions

- All samples in the test set must have predictions
- Sample IDs should be sequential from 0 to N-1
- No gaps allowed

## Example: Adding GPT-4 Predictions

```bash
# 1. Create model directory
mkdir -p data/experiments/zero-shot-2/results/gpt4-zero-shot

# 2. Create test set directory
mkdir -p data/experiments/zero-shot-2/results/gpt4-zero-shot/norm-CELVA-SP

# 3. Generate predictions (your code)
python my_gpt4_predictor.py \
  --test-file data/experiments/zero-shot-2/ml-test-data/norm-CELVA-SP.csv \
  --output data/experiments/zero-shot-2/results/gpt4-zero-shot/norm-CELVA-SP/argmax_predictions.json

# 4. Generate evaluation report
python -m src_cq2.report --experiment-dir data/experiments/zero-shot-2
```

## Validation Checklist

Before running the report generator, verify:

- [ ] Model directory name is descriptive and unique
- [ ] Test set subdirectories match test CSV filenames exactly (without `.csv`)
- [ ] At minimum, `argmax_predictions.json` exists for each test set
- [ ] Sample IDs are 0-indexed and sequential
- [ ] All predicted labels are valid CEFR levels (A1-C2)
- [ ] Number of predictions matches number of samples in test CSV
- [ ] JSON files are valid (no syntax errors)
- [ ] If using soft predictions, probabilities sum to ~1.0

## Troubleshooting

### "No such file or directory" errors
- Check that test set directory names match CSV filenames exactly
- Ensure you're using the correct experiment directory path

### "Sample ID mismatch" errors
- Verify sample IDs are 0-indexed and sequential
- Check that prediction count matches test set size

### "Invalid CEFR label" errors
- Ensure labels are uppercase (B1, not b1)
- Check for typos in labels
- Valid labels: A1, A2, B1, B2, C1, C2

### Probabilities don't sum to 1.0
- Acceptable range is usually 0.99-1.01
- Use normalization if needed: `probs = {k: v/sum(probs.values()) for k, v in probs.items()}`

## See Also

- `LANGUAGE_MODEL_CLASSIFIER_GUIDE.md` - Using language models with the pipeline
- `LM_CLASSIFIER_QUICK_START.md` - Quick start for LM classifiers
- `STRUCTURE_VALIDATION.md` - Validating experiment structure
- `CEFR_METRICS.md` - Understanding CEFR evaluation metrics
