# Minimal Fake Example and Pipeline Test Script

## Overview

`minimal-fake-example-and-pipeline.sh` is a comprehensive test script for validating the CEFR classification pipeline. It creates synthetic data and runs the entire pipeline from data generation through prediction.

## Features

### 🎯 Core Capabilities

- **Synthetic Data Generation**: Creates minimal fake CEFR-labeled text data for testing
- **Full Pipeline Execution**: Runs all steps: TF-IDF training, feature extraction, classifier training, and prediction
- **Timestamped Experiments**: Automatically adds timestamps to experiment folders to prevent overwrites
- **Extensive Validation**: Validates all configuration parameters before execution

### 🏗️ Architecture

- **Centralized Configuration**: Global `Config` associative array (dictionary-like) stores all settings
- **Granular Argument Parsing**: Command-line arguments with full validation
- **Modular Pipeline Steps**: Each step can be skipped independently for debugging
- **Comprehensive Error Handling**: Extensive checks with clear error messages

## Quick Start

### Basic Usage

```bash
# Run with all defaults
./utils/minimal-fake-example-and-pipeline.sh

# View help
./utils/minimal-fake-example-and-pipeline.sh --help
```

### Common Scenarios

```bash
# Test with more data
./utils/minimal-fake-example-and-pipeline.sh \
    --num-train 100 \
    --num-test 50

# Test different classifier
./utils/minimal-fake-example-and-pipeline.sh \
    --classifier xgboost

# Quick test with cleanup
./utils/minimal-fake-example-and-pipeline.sh \
    --no-timestamp \
    --cleanup \
    --num-train 20 \
    --num-test 10

# Debug specific step
./utils/minimal-fake-example-and-pipeline.sh \
    --skip-data-gen \
    --skip-tfidf \
    --skip-features
```

## Configuration

### Default Values

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PYTHON_BIN` | `~/.pyenv/versions/3.10.18/bin/python3` | Python interpreter path |
| `EXPERIMENT_NAME` | `test-pipeline` | Name of the experiment |
| `EXPERIMENT_BASE_DIR` | `./test-experiments` | Base directory for experiments |
| `USE_TIMESTAMP` | `true` | Add timestamp to experiment folder |
| `NUM_TRAIN_SAMPLES` | `50` | Number of training samples |
| `NUM_TEST_SAMPLES` | `20` | Number of test samples |
| `CEFR_LEVELS` | `A1,A2,B1,B2,C1` | CEFR levels to generate |
| `TFIDF_MAX_FEATURES` | `100` | TF-IDF max features |
| `TFIDF_NGRAM_MIN` | `1` | N-gram minimum |
| `TFIDF_NGRAM_MAX` | `2` | N-gram maximum |
| `TFIDF_MIN_DF` | `1` | Minimum document frequency |
| `TFIDF_MAX_DF` | `0.95` | Maximum document frequency |
| `CLASSIFIER_TYPE` | `logistic` | Classifier type |

### Command-Line Arguments

#### Experiment Configuration
- `--experiment-name NAME`: Set experiment name
- `--experiment-dir DIR`: Set base directory for experiments
- `--timestamp` / `--no-timestamp`: Control timestamp in folder name

#### Data Generation
- `--num-train N`: Number of training samples
- `--num-test N`: Number of test samples
- `--cefr-levels LEVELS`: Comma-separated CEFR levels (e.g., `A1,A2,B1,B2,C1,C2`)

#### TF-IDF Configuration
- `--max-features N`: TF-IDF maximum features
- `--ngram-min N`: N-gram minimum (default: 1)
- `--ngram-max N`: N-gram maximum (default: 2)
- `--min-df N`: Minimum document frequency
- `--max-df FLOAT`: Maximum document frequency (0.0-1.0)

#### Classifier Configuration
- `--classifier TYPE`: Classifier type
  - Valid types: `multinomialnb`, `logistic`, `randomforest`, `svm`, `xgboost`

#### Pipeline Control
- `--skip-data-gen`: Skip data generation
- `--skip-tfidf`: Skip TF-IDF training
- `--skip-features`: Skip feature extraction
- `--skip-classifier`: Skip classifier training
- `--skip-prediction`: Skip prediction

#### Python Configuration
- `--python PATH`: Path to Python binary

#### Output Control
- `--verbose` / `--quiet`: Control verbosity
- `--cleanup`: Remove experiment directory after success
- `--no-strict`: Disable strict mode (allow existing directories)
- `--skip-python-check`: Skip Python version check
- `--skip-dep-check`: Skip dependency check

### Environment Variables

All options can be set via environment variables:

```bash
export EXPERIMENT_NAME="my-test"
export NUM_TRAIN_SAMPLES=100
export TFIDF_MAX_FEATURES=500
export CLASSIFIER_TYPE="xgboost"
export PYTHON_BIN="/usr/bin/python3"

./utils/minimal-fake-example-and-pipeline.sh
```

## Examples

### Example 1: Basic Test

```bash
./utils/minimal-fake-example-and-pipeline.sh
```

This will:
1. Create experiment directory: `./test-experiments/test-pipeline-20251019_123456/`
2. Generate 50 training samples and 20 test samples
3. Train TF-IDF model with 100 features
4. Extract features for training and test data
5. Train logistic regression classifier
6. Run predictions on test data

### Example 2: Large Test with XGBoost

```bash
./utils/minimal-fake-example-and-pipeline.sh \
    --experiment-name large-xgboost-test \
    --num-train 200 \
    --num-test 80 \
    --classifier xgboost \
    --max-features 500 \
    --ngram-max 3
```

### Example 3: Debug Specific Step

```bash
# Skip data generation and TF-IDF training to debug classifier
./utils/minimal-fake-example-and-pipeline.sh \
    --skip-data-gen \
    --skip-tfidf \
    --skip-features
```

### Example 4: CI/CD Integration

```bash
# Quick test for CI/CD pipeline
./utils/minimal-fake-example-and-pipeline.sh \
    --no-timestamp \
    --cleanup \
    --num-train 30 \
    --num-test 10 \
    --quiet
```

## Validation

The script performs extensive validation:

### Python Environment
- ✓ Python binary exists and is executable
- ✓ Python version >= 3.7
- ✓ Required packages installed (`pandas`, `numpy`, `sklearn`, `yaml`)

### Configuration Parameters
- ✓ All numeric values are positive integers or valid floats
- ✓ Boolean values are `true` or `false`
- ✓ Classifier type is valid
- ✓ CEFR levels format is correct
- ✓ N-gram range is valid (min <= max)
- ✓ Directories are writable

## Directory Structure

When executed, the script creates this structure:

```
test-experiments/
└── test-pipeline-20251019_123456/
    ├── config.yaml                    # Pipeline configuration
    ├── ml-training-data/
    │   ├── train-data.csv            # Generated training data
    │   └── generate_data.py          # Data generation script
    ├── ml-test-data/
    │   └── test-data.csv             # Generated test data
    ├── features-training-data/       # (unused in this script)
    ├── features/
    │   └── {hash}_tfidf/
    │       ├── train-data/
    │       │   └── features_dense.csv
    │       └── test-data/
    │           └── features_dense.csv
    ├── feature-models/
    │   ├── {hash}_tfidf/
    │   │   ├── tfidf_vectorizer.pkl
    │   │   └── config.json
    │   └── classifiers/
    │       └── train-data_{classifier}_{hash}_tfidf/
    │           ├── classifier.pkl
    │           ├── label_encoder.pkl
    │           └── config.json
    └── results/
        └── {classifier_name}/
            └── test-data/
                ├── argmax_predictions.json
                ├── soft_predictions.json
                ├── rounded_avg_predictions.json
                └── evaluation_report.md
```

## Exit Status

- `0`: Success
- `1`: Configuration error
- `2`: Validation error
- `3`: Pipeline execution error

## Troubleshooting

### Python Binary Not Found

```bash
./utils/minimal-fake-example-and-pipeline.sh \
    --python /usr/bin/python3
```

Or set environment variable:

```bash
export PYTHON_BIN="/usr/bin/python3"
```

### Directory Already Exists

By default, the script uses timestamps to prevent overwrites. If you get this error:

```bash
# Use different experiment name
./utils/minimal-fake-example-and-pipeline.sh \
    --experiment-name my-unique-test

# Or disable strict mode
./utils/minimal-fake-example-and-pipeline.sh \
    --no-strict
```

### Missing Dependencies

```bash
# Install required packages
~/.pyenv/versions/3.10.18/bin/python3 -m pip install pandas numpy scikit-learn pyyaml

# Or skip dependency check (not recommended)
./utils/minimal-fake-example-and-pipeline.sh \
    --skip-dep-check
```

## Advanced Usage

### Custom Python Environment

```bash
# Use specific Python version
PYTHON_BIN="/usr/local/bin/python3.11" \
    ./utils/minimal-fake-example-and-pipeline.sh

# Use virtual environment
source venv/bin/activate
./utils/minimal-fake-example-and-pipeline.sh \
    --python $(which python3)
```

### Integration with Existing Experiments

```bash
# Use existing experiment directory structure
./utils/minimal-fake-example-and-pipeline.sh \
    --experiment-dir ./data/experiments \
    --experiment-name pipeline-validation \
    --no-timestamp
```

### Batch Testing Multiple Configurations

```bash
#!/bin/bash

# Test multiple classifiers
for classifier in logistic xgboost randomforest; do
    ./utils/minimal-fake-example-and-pipeline.sh \
        --experiment-name "batch-test-${classifier}" \
        --classifier "$classifier" \
        --cleanup
done
```

## Notes

- **Timestamp Format**: `YYYYMMDD_HHMMSS` (e.g., `20251019_153045`)
- **Default Python**: `~/.pyenv/versions/3.10.18/bin/python3`
- **Synthetic Data**: Generated text uses level-specific vocabularies to simulate CEFR progression
- **Config Hash**: TF-IDF configuration generates a unique hash to prevent overwrites

## See Also

- Main pipeline documentation: `docs/PIPELINE.md`
- Configuration system: `src_cq/config.py`
- Training scripts: `src_cq/train_*.py`
- Prediction script: `src_cq/predict.py`
