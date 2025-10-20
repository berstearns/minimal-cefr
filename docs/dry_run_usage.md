# Dry Run Mode - Usage Guide

The dry run mode allows you to preview the directory structure and files that will be created by the CEFR classification pipeline without actually running the computation-heavy training and prediction steps.

## What is Dry Run Mode?

Dry run mode creates **dummy empty files** in the same directory structure that the real pipeline would create. This helps you:

- **Preview output structure** before running expensive computations
- **Verify paths and configurations** are correct
- **Estimate disk space** requirements
- **Understand the pipeline** output organization
- **Test configurations** without waiting for actual training

## Quick Start

### Basic Usage

```bash
# Preview what the pipeline would create
python -m src.pipeline \
  -e data/experiments/zero-shot \
  --cefr-column cefr_level \
  --dry-run
```

### With Multiple Classifiers

```bash
# Preview outputs for multiple classifiers
python -m src.pipeline \
  -e data/experiments/zero-shot \
  --classifiers xgboost logistic randomforest \
  --cefr-column cefr_level \
  --dry-run
```

### With Multiple TF-IDF Configurations

```bash
# Preview outputs for different feature sizes
python -m src.pipeline \
  -e data/experiments/zero-shot \
  --max-features-list 1000 5000 10000 \
  --classifier xgboost \
  --cefr-column cefr_level \
  --dry-run
```

### Preview Specific Steps Only

```bash
# Preview only feature extraction and training steps
python -m src.pipeline \
  -e data/experiments/zero-shot \
  --steps 2 3 \
  --cefr-column cefr_level \
  --dry-run
```

## What Gets Created?

The dry run creates the following structure (with empty/dummy files):

```
experiment_dir/
├── feature-models/
│   ├── {hash}_tfidf/              # TF-IDF models (Step 1)
│   │   ├── tfidf_model.pkl        # Dummy pickle file
│   │   └── config.json            # Dummy config with metadata
│   └── classifiers/               # Trained classifiers (Step 3)
│       └── {model_name}/
│           ├── classifier.pkl     # Dummy classifier
│           ├── label_encoder.pkl  # Dummy encoder
│           └── config.json        # Dummy config
├── features/                      # Extracted features (Step 2)
│   └── {hash}_tfidf/
│       └── {dataset_name}/
│           ├── features_dense.csv     # Empty CSV
│           ├── feature_names.csv      # Empty CSV
│           └── config.json            # Dummy config
└── results/                       # Predictions (Step 4)
    └── {model_name}/
        └── {dataset_name}/
            ├── soft_predictions.json          # Dummy JSON
            ├── argmax_predictions.json        # Dummy JSON
            ├── rounded_avg_predictions.json   # Dummy JSON
            └── evaluation_report.md           # Dummy markdown
```

## Dummy File Contents

### JSON Files
All `.json` files contain minimal metadata:
```json
{
  "_dry_run": true,
  "_description": "Description of what this file would contain",
  "_note": "This is a dry run dummy file. Run the actual pipeline to generate real data."
}
```

### Markdown Files
All `.md` files contain:
```markdown
# Dry Run Output

<description>

This is a dummy file created by dry run mode.
```

### Other Files
- `.pkl` files: empty
- `.csv` files: empty
- `.txt` files: empty

## Examples

### Example 1: Simple Preview

```bash
python -m src.pipeline \
  -e dummy-experiment \
  --cefr-column cefr_level \
  --dry-run
```

Output:
```
======================================================================
DRY RUN MODE - CEFR CLASSIFICATION PIPELINE
======================================================================
Experiment: dummy-experiment
TF-IDF configs: 1
Classifiers: ['xgboost']
Steps to run: [1, 2, 3, 4]
======================================================================

Creating dummy files to preview output structure...

======================================================================
DRY RUN - STEP 1: Train TF-IDF Vectorizer(s)
======================================================================

Config 1/1:
  Hash: e2752d18
  max_features: 5000
  Directory: dummy-experiment/feature-models/e2752d18_tfidf
  Created: dummy-experiment/feature-models/e2752d18_tfidf/tfidf_model.pkl
  Created: dummy-experiment/feature-models/e2752d18_tfidf/config.json

...
```

### Example 2: Grid Search Preview

Preview a large grid search without running it:

```bash
python -m src.pipeline \
  -e experiments/grid-search \
  --max-features-list 1000 5000 10000 20000 \
  --classifiers xgboost logistic randomforest svm \
  --cefr-column cefr_level \
  --dry-run
```

This would show you exactly what files would be created for 4 × 4 = 16 model combinations.

### Example 3: Specific Steps Preview

Preview only the prediction step:

```bash
python -m src.pipeline \
  -e experiments/test \
  --steps predict \
  --cefr-column cefr_level \
  --dry-run
```

## Standalone Dry Run Tool

You can also run the dry run tool standalone:

```bash
# Run standalone dry run
python -m src.dry_run \
  -e data/experiments/zero-shot \
  --cefr-column cefr_level

# With specific configuration
python -m src.dry_run \
  -e data/experiments/zero-shot \
  --max-features-list 5000 10000 \
  --classifiers xgboost logistic \
  --cefr-column cefr_level
```

## After Dry Run

Once you've verified the structure looks correct:

1. **Clean up dummy files** (optional):
   ```bash
   # Remove dummy directories if desired
   rm -rf experiment_dir/feature-models
   rm -rf experiment_dir/features
   rm -rf experiment_dir/results
   ```

2. **Run actual pipeline** (remove `--dry-run` flag):
   ```bash
   python -m src.pipeline \
     -e data/experiments/zero-shot \
     --cefr-column cefr_level
   # Now runs for real!
   ```

## Use Cases

### 1. Verify Configuration Before Long Run
```bash
# First: dry run to check
python -m src.pipeline -e exp --dry-run

# Verify structure looks good
ls -R exp/

# Then: run for real
python -m src.pipeline -e exp
```

### 2. Estimate Disk Space
```bash
# Dry run creates the directory structure
python -m src.pipeline -e exp --dry-run

# Count files to estimate
find exp -type f | wc -l

# Estimate based on file counts
# Each .pkl file ≈ 100MB, each .csv ≈ 50MB, etc.
```

### 3. Test Complex Configurations
```bash
# Test a complex multi-classifier, multi-config setup
python -m src.pipeline \
  -e exp \
  --max-features-list 1000 5000 10000 \
  --classifiers xgboost logistic randomforest svm multinomialnb \
  --cefr-column cefr_level \
  --dry-run

# Outputs: "Creating 15 models (3 TF-IDF × 5 classifiers)"
# Shows exact directory structure for all combinations
```

### 4. Debug Path Issues
```bash
# Dry run with custom paths
python -m src.pipeline \
  -e experiments/custom \
  -o /models/shared \
  --dry-run

# Verify paths are correct before running
```

## Tips

- **Always dry run first** for complex configurations
- **Use `-q` flag** for less verbose output: `--dry-run -q`
- **Combine with `--steps`** to preview specific pipeline stages
- **Check file counts** to estimate total outputs
- **Inspect dummy JSON files** to see what metadata will be saved

## Differences from Real Run

| Aspect | Dry Run | Real Run |
|--------|---------|----------|
| File creation | Empty/dummy files | Real data files |
| Execution time | Seconds | Hours |
| Disk space | Minimal (<1MB) | GBs |
| Computation | None | Full training |
| Models | No actual models | Trained models |
| Predictions | No predictions | Real predictions |

## Troubleshooting

### "Experiment directory not found"
The dry run still requires the input data directories to exist:
- `experiment_dir/ml-training-data/` (with .csv files)
- `experiment_dir/ml-test-data/` (with .csv files)

### "No CSV files found"
Make sure your experiment directory contains:
```
experiment_dir/
├── ml-training-data/
│   └── *.csv  # At least one CSV file
└── ml-test-data/
    └── *.csv  # At least one CSV file
```

### Cleanup
To remove all dummy files created by dry run:
```bash
# Remove feature models
rm -rf experiment_dir/feature-models

# Remove features
rm -rf experiment_dir/features

# Remove results
rm -rf experiment_dir/results
```

Or selectively remove only dry run files:
```bash
# Remove files marked as dry run
find experiment_dir -name "*.json" -exec grep -l '"_dry_run": true' {} \; | xargs rm
```

## Related Commands

- `python -m src.pipeline --help` - Full pipeline options
- `python -m src.pipeline --list-steps` - List all pipeline steps
- `python -m src.dry_run --help` - Standalone dry run help

## Summary

The dry run mode is a powerful tool for:
- ✅ Previewing pipeline outputs
- ✅ Verifying configurations
- ✅ Estimating resource requirements
- ✅ Learning the pipeline structure
- ✅ Testing without computation

Always dry run first for complex experiments!
