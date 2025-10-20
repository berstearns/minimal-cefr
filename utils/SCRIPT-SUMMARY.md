# Minimal Fake Example and Pipeline Test Script - Summary

## What Was Created

### Main Script
**File**: `utils/minimal-fake-example-and-pipeline.sh` (33KB, ~1000 lines)

A comprehensive bash script for testing the CEFR classification pipeline with the following features:

#### Key Features

1. **Hyper-Granular Argument Parsing**
   - 30+ command-line options
   - Full environment variable support
   - Extensive validation for all parameters
   - Clear error messages with suggestions

2. **Centralized Configuration Structure**
   - Global `Config` associative array (bash dictionary)
   - Similar to Python's dict structure
   - All settings in one place
   - Easy to extend and modify

3. **Automatic Timestamping**
   - Default format: `YYYYMMDD_HHMMSS` (e.g., `20251019_153045`)
   - Prevents accidental overwrites
   - Can be disabled with `--no-timestamp`
   - Configurable experiment names

4. **Pipeline Steps**
   - ✓ Synthetic data generation
   - ✓ Pipeline configuration creation
   - ✓ TF-IDF model training
   - ✓ Feature extraction (train + test)
   - ✓ Classifier training
   - ✓ Prediction and evaluation
   - ✓ Optional cleanup

5. **Comprehensive Validation**
   - Python binary existence and executability
   - Python version check (>= 3.7)
   - Required package installation check
   - Numeric parameter validation
   - Boolean parameter validation
   - Classifier type validation
   - CEFR levels format validation
   - Directory writability checks

6. **Supported Classifiers**
   - `multinomialnb` - Multinomial Naive Bayes
   - `logistic` - Logistic Regression (default)
   - `randomforest` - Random Forest
   - `svm` - Support Vector Machine
   - `xgboost` - XGBoost

7. **Flexible Configuration**
   - Command-line arguments
   - Environment variables
   - Combination of both
   - Skip individual pipeline steps
   - Control output verbosity

### Documentation Files

#### README-pipeline-test.md (9.6KB)
Complete reference documentation covering:
- Overview and features
- Quick start guide
- All configuration options
- Default values table
- Directory structure
- Troubleshooting guide
- Advanced usage examples
- Exit status codes

#### USAGE-EXAMPLES.md (11KB)
Practical examples including:
- Quick smoke tests
- Testing different classifiers
- TF-IDF configuration experiments
- Data size testing
- CEFR level customization
- Debugging workflows
- CI/CD integration
- Batch testing
- Performance benchmarking
- Regression testing
- Stress testing

#### SCRIPT-SUMMARY.md (this file)
High-level overview of the script system.

## Quick Start

```bash
# Make script executable (if needed)
chmod +x utils/minimal-fake-example-and-pipeline.sh

# Run basic test
./utils/minimal-fake-example-and-pipeline.sh

# View all options
./utils/minimal-fake-example-and-pipeline.sh --help

# Quick test with cleanup
./utils/minimal-fake-example-and-pipeline.sh --cleanup --num-train 20
```

## Architecture

### Global Configuration Structure

```bash
declare -A Config=(
    # Python
    [PYTHON_BIN]="/path/to/python3"

    # Experiment
    [EXPERIMENT_NAME]="test-pipeline"
    [EXPERIMENT_DIR]="/path/to/experiment"
    [USE_TIMESTAMP]="true"

    # Data
    [NUM_TRAIN_SAMPLES]="50"
    [NUM_TEST_SAMPLES]="20"
    [CEFR_LEVELS]="A1,A2,B1,B2,C1"

    # TF-IDF
    [TFIDF_MAX_FEATURES]="100"
    [TFIDF_NGRAM_MIN]="1"
    [TFIDF_NGRAM_MAX]="2"

    # Classifier
    [CLASSIFIER_TYPE]="logistic"

    # Control
    [SKIP_*]="false"
    [VERBOSE]="true"
    # ... and more
)
```

### Execution Flow

```
1. Parse Arguments
   ↓
2. Display Configuration
   ↓
3. Validate Configuration
   ├─ Python binary exists?
   ├─ Python version >= 3.7?
   ├─ Dependencies installed?
   ├─ Parameters valid?
   └─ Directories writable?
   ↓
4. Setup Experiment Directory
   ├─ Create timestamped folder
   └─ Create subdirectories
   ↓
5. Generate Synthetic Data
   ├─ Create generation script
   ├─ Generate training data
   └─ Generate test data
   ↓
6. Create Pipeline Config
   └─ Generate config.yaml
   ↓
7. Run Pipeline Steps
   ├─ Train TF-IDF model
   ├─ Extract features
   ├─ Train classifier
   └─ Run predictions
   ↓
8. Optional Cleanup
   ↓
9. Report Results
```

### Generated Directory Structure

```
test-experiments/
└── test-pipeline-20251019_123456/
    ├── config.yaml                    # Auto-generated pipeline config
    │
    ├── ml-training-data/
    │   ├── train-data.csv            # Synthetic training data
    │   └── generate_data.py          # Data generation script
    │
    ├── ml-test-data/
    │   └── test-data.csv             # Synthetic test data
    │
    ├── features-training-data/       # (empty, used by other scripts)
    │
    ├── features/
    │   └── {hash}_tfidf/
    │       ├── train-data/
    │       │   └── features_dense.csv
    │       └── test-data/
    │           └── features_dense.csv
    │
    ├── feature-models/
    │   ├── {hash}_tfidf/
    │   │   ├── tfidf_vectorizer.pkl
    │   │   └── config.json
    │   └── classifiers/
    │       └── train-data_{classifier}_{hash}_tfidf/
    │           ├── classifier.pkl
    │           ├── label_encoder.pkl
    │           └── config.json
    │
    └── results/
        └── {classifier_name}/
            └── test-data/
                ├── argmax_predictions.json
                ├── soft_predictions.json
                ├── rounded_avg_predictions.json
                └── evaluation_report.md
```

## Command-Line Options Reference

### Experiment Configuration
```bash
--experiment-name NAME      # Experiment name (default: test-pipeline)
--experiment-dir DIR        # Base directory (default: ./test-experiments)
--timestamp / --no-timestamp # Add/skip timestamp (default: true)
```

### Data Generation
```bash
--num-train N              # Training samples (default: 50)
--num-test N               # Test samples (default: 20)
--cefr-levels LEVELS       # CEFR levels (default: A1,A2,B1,B2,C1)
```

### TF-IDF Configuration
```bash
--max-features N           # Max features (default: 100)
--ngram-min N              # N-gram min (default: 1)
--ngram-max N              # N-gram max (default: 2)
--min-df N                 # Min doc frequency (default: 1)
--max-df FLOAT             # Max doc frequency (default: 0.95)
```

### Classifier Configuration
```bash
--classifier TYPE          # Type: multinomialnb, logistic, randomforest, svm, xgboost
```

### Pipeline Control
```bash
--skip-data-gen           # Skip data generation
--skip-tfidf              # Skip TF-IDF training
--skip-features           # Skip feature extraction
--skip-classifier         # Skip classifier training
--skip-prediction         # Skip prediction
```

### Python Configuration
```bash
--python PATH             # Python binary path
```

### Output Control
```bash
-v, --verbose             # Enable verbose output
-q, --quiet               # Disable verbose output
--cleanup                 # Cleanup on success
--no-strict               # Disable strict mode
--skip-python-check       # Skip Python version check
--skip-dep-check          # Skip dependency check
```

## Environment Variables

All options can be set via environment variables by converting to uppercase and replacing dashes with underscores:

```bash
export EXPERIMENT_NAME="my-test"
export NUM_TRAIN_SAMPLES=100
export TFIDF_MAX_FEATURES=500
export CLASSIFIER_TYPE="xgboost"
export PYTHON_BIN="/usr/bin/python3"
```

## Validation Features

The script validates:

1. **Python Binary**
   - File exists
   - Is executable
   - Version >= 3.7 (if not skipped)

2. **Dependencies**
   - pandas, numpy, sklearn, yaml installed (if not skipped)

3. **Numeric Parameters**
   - Positive integers where required
   - Valid ranges (e.g., 0.0-1.0 for max_df)
   - Logical constraints (ngram_min <= ngram_max)

4. **Boolean Parameters**
   - Must be "true" or "false"

5. **Classifier Type**
   - Must be one of: multinomialnb, logistic, randomforest, svm, xgboost

6. **CEFR Levels**
   - Must match format: A1,A2,B1,B2,C1,C2
   - Each level must be [A-C][1-2]

7. **Directories**
   - Must be writable (or creatable)

## Exit Codes

- `0` - Success
- `1` - Configuration error
- `2` - Validation error
- `3` - Pipeline execution error

## Default Python Binary

```bash
~/.pyenv/versions/3.10.18/bin/python3
```

Can be overridden with:
- `--python PATH` argument
- `PYTHON_BIN` environment variable

## Common Use Cases

### 1. Quick Validation
```bash
./utils/minimal-fake-example-and-pipeline.sh --cleanup --num-train 20
```

### 2. Test Specific Classifier
```bash
./utils/minimal-fake-example-and-pipeline.sh --classifier xgboost
```

### 3. CI/CD Integration
```bash
./utils/minimal-fake-example-and-pipeline.sh --no-timestamp --cleanup --quiet
```

### 4. Debug Specific Step
```bash
./utils/minimal-fake-example-and-pipeline.sh --skip-data-gen --skip-tfidf
```

### 5. Batch Testing
```bash
for clf in logistic xgboost randomforest; do
    ./utils/minimal-fake-example-and-pipeline.sh --classifier $clf --cleanup
done
```

## Color Output

The script uses colored output for better readability:

- 🔵 **Blue** - INFO messages
- 🟢 **Green** - SUCCESS messages
- 🟡 **Yellow** - WARNING messages
- 🔴 **Red** - ERROR messages
- 🔷 **Cyan** - Step headers
- 🟣 **Magenta** - Substeps

Colors are automatically disabled if output is not a TTY (e.g., piped to file).

## Integration with Existing Pipeline

The script integrates with the existing CEFR classification pipeline by:

1. Using the same `src_cq/` modules
2. Generating compatible `config.yaml` files
3. Creating standard experiment directory structure
4. Using `PYTHONPATH=.` for module imports
5. Following the same naming conventions

## Testing

The script has been tested with:
- ✓ Default configuration
- ✓ Custom arguments
- ✓ Environment variables
- ✓ Skip flags
- ✓ All classifiers
- ✓ Various data sizes
- ✓ Different Python versions

## Future Enhancements

Potential improvements:
- [ ] Support for grouped TF-IDF
- [ ] Support for perplexity features
- [ ] Multi-feature combination testing
- [ ] Parallel execution of multiple configurations
- [ ] JSON/YAML output for results aggregation
- [ ] Integration with existing ablation scripts
- [ ] Support for custom feature types
- [ ] Automated performance comparison reports

## Files Created

1. **utils/minimal-fake-example-and-pipeline.sh** (33KB)
   - Main executable script

2. **utils/README-pipeline-test.md** (9.6KB)
   - Complete reference documentation

3. **utils/USAGE-EXAMPLES.md** (11KB)
   - Practical usage examples

4. **utils/SCRIPT-SUMMARY.md** (this file)
   - High-level overview

Total: ~65KB of documentation and code

## Related Files

- `src_cq/config.py` - Configuration system
- `src_cq/train_tfidf.py` - TF-IDF training
- `src_cq/extract_features.py` - Feature extraction
- `src_cq/train_classifiers.py` - Classifier training
- `src_cq/predict.py` - Prediction script

## Support

For issues or questions:
- See `README-pipeline-test.md` for troubleshooting
- See `USAGE-EXAMPLES.md` for examples
- Run with `--help` for all options
- Use `--verbose` to see detailed output
