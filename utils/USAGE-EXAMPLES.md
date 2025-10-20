# Pipeline Test Script - Usage Examples

## Quick Reference

```bash
# Basic test - run everything with defaults
./utils/minimal-fake-example-and-pipeline.sh

# Show all options
./utils/minimal-fake-example-and-pipeline.sh --help

# Quiet mode with cleanup
./utils/minimal-fake-example-and-pipeline.sh --quiet --cleanup
```

## Common Use Cases

### 1. Quick Smoke Test

Fast validation that the pipeline works end-to-end:

```bash
./utils/minimal-fake-example-and-pipeline.sh \
    --num-train 20 \
    --num-test 10 \
    --cleanup
```

**What it does:**
- Creates 20 training samples, 10 test samples
- Runs full pipeline
- Cleans up temporary files on success
- Takes ~30 seconds

### 2. Test Different Classifiers

Compare different classifier types:

```bash
# Logistic Regression (default, fastest)
./utils/minimal-fake-example-and-pipeline.sh \
    --experiment-name test-logistic \
    --classifier logistic

# XGBoost (slower, often better performance)
./utils/minimal-fake-example-and-pipeline.sh \
    --experiment-name test-xgboost \
    --classifier xgboost

# Random Forest
./utils/minimal-fake-example-and-pipeline.sh \
    --experiment-name test-rf \
    --classifier randomforest

# Support Vector Machine
./utils/minimal-fake-example-and-pipeline.sh \
    --experiment-name test-svm \
    --classifier svm

# Multinomial Naive Bayes
./utils/minimal-fake-example-and-pipeline.sh \
    --experiment-name test-nb \
    --classifier multinomialnb
```

### 3. Test TF-IDF Configurations

Experiment with different TF-IDF settings:

```bash
# Small feature set
./utils/minimal-fake-example-and-pipeline.sh \
    --experiment-name small-features \
    --max-features 50 \
    --ngram-max 1

# Large feature set with trigrams
./utils/minimal-fake-example-and-pipeline.sh \
    --experiment-name large-features \
    --max-features 1000 \
    --ngram-max 3

# Balanced configuration
./utils/minimal-fake-example-and-pipeline.sh \
    --experiment-name balanced \
    --max-features 500 \
    --ngram-min 1 \
    --ngram-max 2 \
    --min-df 2 \
    --max-df 0.90
```

### 4. Test with More Data

Simulate realistic data sizes:

```bash
# Medium dataset
./utils/minimal-fake-example-and-pipeline.sh \
    --experiment-name medium-test \
    --num-train 500 \
    --num-test 200

# Large dataset
./utils/minimal-fake-example-and-pipeline.sh \
    --experiment-name large-test \
    --num-train 2000 \
    --num-test 500 \
    --max-features 2000
```

### 5. Custom CEFR Levels

Test with specific CEFR level subsets:

```bash
# Only beginner levels
./utils/minimal-fake-example-and-pipeline.sh \
    --experiment-name beginners-only \
    --cefr-levels A1,A2,B1

# Only advanced levels
./utils/minimal-fake-example-and-pipeline.sh \
    --experiment-name advanced-only \
    --cefr-levels B2,C1,C2

# All 6 levels
./utils/minimal-fake-example-and-pipeline.sh \
    --experiment-name all-levels \
    --cefr-levels A1,A2,B1,B2,C1,C2
```

### 6. Debugging Specific Steps

Skip certain steps to debug others:

```bash
# Only generate data (skip pipeline)
./utils/minimal-fake-example-and-pipeline.sh \
    --skip-tfidf \
    --skip-features \
    --skip-classifier \
    --skip-prediction

# Only train TF-IDF (assumes data exists)
./utils/minimal-fake-example-and-pipeline.sh \
    --skip-data-gen \
    --skip-features \
    --skip-classifier \
    --skip-prediction

# Only run prediction (assumes everything else exists)
./utils/minimal-fake-example-and-pipeline.sh \
    --skip-data-gen \
    --skip-tfidf \
    --skip-features \
    --skip-classifier
```

### 7. CI/CD Integration

Automated testing in continuous integration:

```bash
#!/bin/bash
# ci-test.sh

set -e

# Run pipeline test with strict validation
./utils/minimal-fake-example-and-pipeline.sh \
    --experiment-name ci-test \
    --no-timestamp \
    --num-train 30 \
    --num-test 15 \
    --cleanup \
    --quiet

echo "Pipeline test passed!"
```

### 8. Using Different Python Versions

Test with specific Python installations:

```bash
# System Python
./utils/minimal-fake-example-and-pipeline.sh \
    --python /usr/bin/python3

# Pyenv version
./utils/minimal-fake-example-and-pipeline.sh \
    --python ~/.pyenv/versions/3.11.7/bin/python3

# Virtual environment
source myenv/bin/activate
./utils/minimal-fake-example-and-pipeline.sh \
    --python $(which python3)

# Conda environment
conda activate myenv
./utils/minimal-fake-example-and-pipeline.sh \
    --python $(which python3)
```

### 9. Custom Experiment Location

Use specific directories for experiments:

```bash
# Use existing experiment structure
./utils/minimal-fake-example-and-pipeline.sh \
    --experiment-dir ./data/experiments \
    --experiment-name validation-test

# Use temporary directory
./utils/minimal-fake-example-and-pipeline.sh \
    --experiment-dir /tmp/pipeline-tests \
    --cleanup
```

### 10. Batch Testing

Run multiple configurations:

```bash
#!/bin/bash
# batch-test.sh

classifiers=(logistic xgboost randomforest)
feature_sizes=(100 500 1000)

for clf in "${classifiers[@]}"; do
    for features in "${feature_sizes[@]}"; do
        echo "Testing: $clf with $features features"
        ./utils/minimal-fake-example-and-pipeline.sh \
            --experiment-name "batch-${clf}-${features}" \
            --classifier "$clf" \
            --max-features "$features" \
            --num-train 100 \
            --num-test 50 \
            --quiet
    done
done

echo "All tests complete!"
```

## Environment Variable Configuration

Set defaults via environment variables:

```bash
# Create a configuration file
cat > pipeline-test.env << 'EOF'
export EXPERIMENT_NAME="my-default-test"
export NUM_TRAIN_SAMPLES=100
export NUM_TEST_SAMPLES=40
export TFIDF_MAX_FEATURES=500
export CLASSIFIER_TYPE="xgboost"
export PYTHON_BIN="~/.pyenv/versions/3.10.18/bin/python3"
EOF

# Use it
source pipeline-test.env
./utils/minimal-fake-example-and-pipeline.sh
```

## Advanced Scenarios

### Performance Benchmarking

```bash
#!/bin/bash
# benchmark.sh

echo "Benchmarking pipeline performance..."

for size in 50 100 200 500 1000; do
    echo "Testing with $size training samples..."

    start=$(date +%s)

    ./utils/minimal-fake-example-and-pipeline.sh \
        --experiment-name "benchmark-${size}" \
        --num-train "$size" \
        --num-test $((size / 5)) \
        --quiet \
        --cleanup

    end=$(date +%s)
    duration=$((end - start))

    echo "  Completed in ${duration}s"
done
```

### Regression Testing

```bash
#!/bin/bash
# regression-test.sh

# Test that pipeline still works after code changes

echo "Running regression tests..."

# Test 1: Basic functionality
./utils/minimal-fake-example-and-pipeline.sh \
    --experiment-name regression-basic \
    --cleanup --quiet || exit 1

# Test 2: All classifiers
for clf in multinomialnb logistic randomforest svm xgboost; do
    ./utils/minimal-fake-example-and-pipeline.sh \
        --experiment-name "regression-${clf}" \
        --classifier "$clf" \
        --cleanup --quiet || exit 1
done

# Test 3: Different data sizes
for size in 20 100 500; do
    ./utils/minimal-fake-example-and-pipeline.sh \
        --experiment-name "regression-size-${size}" \
        --num-train "$size" \
        --cleanup --quiet || exit 1
done

echo "All regression tests passed!"
```

### Stress Testing

```bash
#!/bin/bash
# stress-test.sh

# Test pipeline with large configurations

./utils/minimal-fake-example-and-pipeline.sh \
    --experiment-name stress-test \
    --num-train 5000 \
    --num-test 1000 \
    --max-features 5000 \
    --ngram-max 3 \
    --classifier xgboost
```

## Troubleshooting Examples

### Fix Python Path Issues

```bash
# Find your Python installation
which python3

# Use it explicitly
./utils/minimal-fake-example-and-pipeline.sh \
    --python $(which python3)
```

### Fix Permission Issues

```bash
# Ensure script is executable
chmod +x ./utils/minimal-fake-example-and-pipeline.sh

# Use different experiment directory
./utils/minimal-fake-example-and-pipeline.sh \
    --experiment-dir ~/my-experiments
```

### Skip Validation for Quick Tests

```bash
# Skip all validation checks (use with caution!)
./utils/minimal-fake-example-and-pipeline.sh \
    --skip-python-check \
    --skip-dep-check \
    --no-strict
```

## Output Examples

### Successful Run

```
╔════════════════════════════════════════════════════════════════════╗
║  Minimal CEFR Classification Pipeline Test                        ║
║  Testing pipeline integrity with synthetic data                   ║
╚════════════════════════════════════════════════════════════════════╝

==> Configuration Summary
...

==> Validating Configuration
[SUCCESS] Configuration validation passed ✓

==> Setting Up Experiment Directory
[SUCCESS] Experiment directory created: ./test-experiments/test-pipeline-20251019_123456

==> Generating Synthetic CEFR Data
[SUCCESS] Synthetic data generation complete

==> Creating Pipeline Configuration File
[SUCCESS] Configuration file created: ./test-experiments/test-pipeline-20251019_123456/config.yaml

==> Training TF-IDF Model
[SUCCESS] TF-IDF training complete

==> Extracting Features
[SUCCESS] Feature extraction complete

==> Training Classifier
[SUCCESS] Classifier training complete

==> Running Predictions
[SUCCESS] Predictions complete

==> Pipeline Test Complete
[SUCCESS] All steps completed successfully! ✓

[INFO] Experiment directory: ./test-experiments/test-pipeline-20251019_123456
[INFO] Configuration file: ./test-experiments/test-pipeline-20251019_123456/config.yaml
[INFO] Results directory: ./test-experiments/test-pipeline-20251019_123456/results
```

### Configuration Error

```
[ERROR] Invalid classifier type: invalid-classifier. Must be one of: multinomialnb logistic randomforest svm xgboost
[ERROR] Configuration validation failed with 1 error(s)
```

### Experiment Directory Exists

```
[ERROR] Experiment directory already exists: ./test-experiments/test-pipeline
[ERROR] Use --no-timestamp and different --experiment-name, or remove the directory
```

## Tips and Best Practices

1. **Always use timestamps for experiments** (default behavior) to avoid overwrites
2. **Use cleanup flag** for CI/CD to save disk space
3. **Start small** (20-50 samples) for quick validation
4. **Use quiet mode** in automated scripts
5. **Set Python path explicitly** in production environments
6. **Keep temp files** (default) for debugging
7. **Use descriptive experiment names** for easy identification
8. **Test with multiple classifiers** to find best model
9. **Validate configuration** before long-running tests
10. **Use environment variables** for consistent settings across runs
