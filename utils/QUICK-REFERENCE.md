# Pipeline Test Script - Quick Reference Card

## One-Liner Examples

```bash
# Basic test (50 train, 20 test, logistic classifier)
./utils/minimal-fake-example-and-pipeline.sh

# Quick test with cleanup
./utils/minimal-fake-example-and-pipeline.sh --cleanup --num-train 20

# Test XGBoost classifier
./utils/minimal-fake-example-and-pipeline.sh --classifier xgboost

# Silent mode for CI/CD
./utils/minimal-fake-example-and-pipeline.sh --quiet --cleanup --no-timestamp
```

## Essential Options

| Short | Long | Default | Description |
|-------|------|---------|-------------|
| `-h` | `--help` | - | Show help |
| `-v` | `--verbose` | `true` | Verbose output |
| `-q` | `--quiet` | `false` | Silent mode |
| | `--experiment-name` | `test-pipeline` | Experiment name |
| | `--num-train` | `50` | Training samples |
| | `--num-test` | `20` | Test samples |
| | `--classifier` | `logistic` | Classifier type |
| | `--max-features` | `100` | TF-IDF features |
| | `--cleanup` | `false` | Remove after success |
| | `--python` | `~/.pyenv/.../python3` | Python binary |

## Classifiers

```bash
--classifier multinomialnb    # Naive Bayes (fast)
--classifier logistic         # Logistic Regression (default, fast)
--classifier randomforest     # Random Forest (medium)
--classifier svm              # Support Vector Machine (slow)
--classifier xgboost          # XGBoost (slow, accurate)
```

## Common Workflows

### Test All Classifiers
```bash
for clf in multinomialnb logistic randomforest svm xgboost; do
  ./utils/minimal-fake-example-and-pipeline.sh --classifier $clf --cleanup
done
```

### Test Different Sizes
```bash
for size in 50 100 200 500; do
  ./utils/minimal-fake-example-and-pipeline.sh --num-train $size --cleanup
done
```

### Debug Specific Step
```bash
# Only data generation
./utils/minimal-fake-example-and-pipeline.sh --skip-tfidf --skip-features --skip-classifier --skip-prediction

# Only TF-IDF training
./utils/minimal-fake-example-and-pipeline.sh --skip-data-gen --skip-features --skip-classifier --skip-prediction

# Only prediction
./utils/minimal-fake-example-and-pipeline.sh --skip-data-gen --skip-tfidf --skip-features --skip-classifier
```

## Environment Variables

```bash
export EXPERIMENT_NAME="my-test"
export NUM_TRAIN_SAMPLES=100
export CLASSIFIER_TYPE="xgboost"
export PYTHON_BIN="/usr/bin/python3"

./utils/minimal-fake-example-and-pipeline.sh
```

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Configuration error |
| `2` | Validation error |
| `3` | Pipeline execution error |

## Output Location

```
test-experiments/
└── {name}-{timestamp}/
    ├── config.yaml
    ├── ml-training-data/train-data.csv
    ├── ml-test-data/test-data.csv
    ├── features/{hash}_tfidf/
    ├── feature-models/
    └── results/
```

## Troubleshooting

```bash
# Python not found?
./utils/minimal-fake-example-and-pipeline.sh --python $(which python3)

# Directory exists?
./utils/minimal-fake-example-and-pipeline.sh --no-timestamp --no-strict

# Missing dependencies?
~/.pyenv/versions/3.10.18/bin/python3 -m pip install pandas numpy scikit-learn pyyaml

# Need debug output?
./utils/minimal-fake-example-and-pipeline.sh --verbose

# Skip validation?
./utils/minimal-fake-example-and-pipeline.sh --skip-python-check --skip-dep-check
```

## CI/CD Template

```bash
#!/bin/bash
set -e

./utils/minimal-fake-example-and-pipeline.sh \
    --experiment-name ci-test \
    --no-timestamp \
    --num-train 30 \
    --num-test 15 \
    --cleanup \
    --quiet

echo "Pipeline test passed!"
```

## Performance Estimates

| Samples | Time | Disk |
|---------|------|------|
| 20 | ~15s | ~1MB |
| 50 | ~25s | ~2MB |
| 100 | ~45s | ~4MB |
| 500 | ~3m | ~15MB |
| 1000 | ~7m | ~30MB |

## Files Created

- `utils/minimal-fake-example-and-pipeline.sh` - Main script (33KB)
- `utils/README-pipeline-test.md` - Full documentation (9.6KB)
- `utils/USAGE-EXAMPLES.md` - Examples (11KB)
- `utils/SCRIPT-SUMMARY.md` - Overview (8KB)
- `utils/QUICK-REFERENCE.md` - This file (3KB)

## Documentation

| File | Purpose |
|------|---------|
| `--help` | Built-in help |
| `QUICK-REFERENCE.md` | This file - quick lookup |
| `USAGE-EXAMPLES.md` | Practical examples |
| `README-pipeline-test.md` | Complete reference |
| `SCRIPT-SUMMARY.md` | Architecture overview |
