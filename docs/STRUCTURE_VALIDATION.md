# Experiment Structure Documentation and Validation

## Overview

This document describes the experiment folder structure standardization implemented across the CEFR classification pipeline.

## What Was Done

### 1. Created Central Structure Definition (`src/experiment_structure.py`)

A new module that:
- **Documents the canonical experiment folder structure** used across all scripts
- **Provides validation functions** to check if an experiment directory is properly structured
- **Can be used standalone** to validate, create, or display the structure

### 2. Updated `src/predict.py`

The predict script now:
- ✅ **Has structure documentation in docstring** showing exactly what folders it expects
- ✅ **Imports validation function** from `src.experiment_structure`
- ✅ **Validates structure** before running predictions
- ✅ **Shows clear error messages** if structure is invalid

### 3. Created Validation Utility (`utils/validate_structure_docs.py`)

A utility script that:
- **Checks all scripts** in `src/` for structure documentation
- **Verifies validation imports and calls** are present in main scripts
- **Reports which scripts** need to be updated

## Canonical Experiment Structure

```
experiment_dir/
├── ml-training-data/           # Training data CSVs
│   └── *.csv                   # Training datasets
├── ml-test-data/               # Test data CSVs
│   └── *.csv                   # Test datasets
├── features-training-data/     # Training features (optional)
│   └── *.csv                   # Training feature CSVs
├── feature-models/             # TF-IDF models and classifiers
│   ├── {config_hash}_tfidf/    # TF-IDF model directories
│   │   ├── config.json         # TF-IDF configuration
│   │   └── tfidf_model.pkl     # Trained TF-IDF vectorizer
│   └── classifiers/            # Trained classifier models
│       └── {model_name}/       # Classifier directory
│           ├── classifier.pkl  # Trained classifier
│           ├── config.json     # Classifier configuration
│           └── label_encoder.pkl  # Label encoder
├── features/                   # Extracted TF-IDF features
│   └── {config_hash}_tfidf/    # Feature directory per TF-IDF config
│       └── {dataset_name}/     # Dataset subdirectories
│           ├── config.json     # Feature extraction config
│           ├── feature_names.csv  # Feature names
│           └── features_dense.csv # Dense feature matrix
└── results/                    # Prediction results
    └── {model_name}/           # Results per model
        └── {dataset_name}/     # Results per dataset
            ├── soft_predictions.json
            ├── argmax_predictions.json
            ├── rounded_avg_predictions.json
            └── evaluation_report.md
```

## Using the Structure Validation

### Standalone Validation

```bash
# Validate an experiment directory
python -m src.experiment_structure validate data/experiments/zero-shot-2

# Create a new experiment structure
python -m src.experiment_structure create data/experiments/my-experiment

# Show documentation
python -m src.experiment_structure show
```

### In Python Scripts

```python
from src.experiment_structure import validate_experiment_structure

# In main() function:
required_dirs = ['ml-test-data', 'feature-models', 'features']
is_valid, errors = validate_experiment_structure(
    experiment_dir,
    required_dirs=required_dirs,
    verbose=True
)

if not is_valid:
    print("Experiment structure validation failed!")
    for error in errors:
        print(f"  {error}")
    raise SystemExit(1)
```

### Check All Scripts

```bash
# Validate that all scripts have proper documentation
python utils/validate_structure_docs.py
```

## Current Status

### ✅ Completed Scripts
- `src/predict.py` - Fully documented and validated

### ⚠️ Scripts Needing Updates

The following scripts need to be updated with:
1. Structure documentation in their module docstrings
2. Import statement: `from src.experiment_structure import validate_experiment_structure`
3. Validation call in their `main()` function

- `src/train_tfidf.py`
- `src/train_tfidf_groupby.py`
- `src/extract_features.py`
- `src/train_classifiers.py`
- `src/train_classifiers_with_ho.py`
- `src/train_classifiers_with_ho_multifeat.py`
- `src/extract_perplexity_features.py`
- `src/pipeline.py`
- `src/report.py`
- `src/mock_pytorch_lm.py`

### Library Scripts (Documentation Only)
- `src/config.py` - Needs structure docs in docstring (no validation needed)

## How to Update Remaining Scripts

### Step 1: Add Structure Documentation to Docstring

Copy the structure documentation from `src/predict.py` and add it to the module docstring. Customize the comments to indicate which directories the script reads from and writes to.

Example for `train_classifiers.py`:

```python
"""
Step 3: Train Classifiers on Extracted Features

Expected Experiment Folder Structure:
=====================================

experiment_dir/
├── features/                   # Extracted TF-IDF features (input)
│   └── {config_hash}_tfidf/
│       └── {dataset_name}/
│           └── features_dense.csv
└── feature-models/             # Trained classifiers (output)
    └── classifiers/
        └── {model_name}/
            ├── classifier.pkl
            ├── config.json
            └── label_encoder.pkl

For detailed structure documentation, see: src/experiment_structure.py
"""
```

### Step 2: Add Import

```python
from src.experiment_structure import validate_experiment_structure
```

### Step 3: Add Validation in main()

```python
def main():
    # ... parse args and load config ...

    # Validate experiment structure
    required_dirs = ['features', 'feature-models']  # Adjust based on script needs
    is_valid, errors = validate_experiment_structure(
        config.experiment_config.experiment_dir,
        required_dirs=required_dirs,
        verbose=config.output_config.verbose
    )
    if not is_valid:
        print("\n✗ Experiment structure validation failed!")
        for error in errors:
            print(f"  {error}")
        print("\nSee docstring or src/experiment_structure.py for expected structure.")
        raise SystemExit(1)

    # ... rest of main() ...
```

## Benefits

1. **Consistency**: All scripts use the same structure definition
2. **Early Error Detection**: Structure validation happens before long-running operations
3. **Clear Documentation**: Users can see exactly what folder structure is expected
4. **Easy Validation**: Can validate structure without running the full pipeline
5. **Centralized Maintenance**: Update structure definition in one place

## Next Steps

1. Run `python utils/validate_structure_docs.py` to see current status
2. Update remaining scripts following the template above
3. Test each script with validation enabled
4. Run validation utility again to confirm all scripts pass

## Reference

- **Structure Definition**: `src/experiment_structure.py`
- **Example Implementation**: `src/predict.py`
- **Validation Utility**: `utils/validate_structure_docs.py`
- **This Document**: `STRUCTURE_VALIDATION.md`
