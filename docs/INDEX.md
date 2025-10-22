# CEFR Classification Pipeline - Documentation Index

Complete documentation for the CEFR text classification pipeline.

## 📚 Table of Contents

### Getting Started

- **[README.md](README.md)** - Project overview and pipeline architecture
- **[QUICK_START.md](QUICK_START.md)** - Quick start for TF-IDF classifiers
- **[QUICK_START_STRUCTURE.md](QUICK_START_STRUCTURE.md)** - Directory structure reference
- **[setup_from_zero.md](setup_from_zero.md)** - Setting up from scratch

### Complete Guides

- **[USAGE.md](USAGE.md)** - Comprehensive usage guide for all pipeline components
- **[SIMPLE_TRAINING_GUIDE.md](SIMPLE_TRAINING_GUIDE.md)** - Step-by-step training walkthrough
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - High-level project summary

### Feature-Specific Guides

#### TF-IDF Classifiers
- **[QUICK_START.md](QUICK_START.md)** - TF-IDF classifier quick start
- **[TRAIN_CLASSIFIERS_REFACTOR.md](TRAIN_CLASSIFIERS_REFACTOR.md)** - Classifier training details
- **[XGBOOST_USAGE.md](XGBOOST_USAGE.md)** - XGBoost-specific usage

#### Language Model Classifiers (NEW!)
- **[LANGUAGE_MODEL_CLASSIFIER_GUIDE.md](LANGUAGE_MODEL_CLASSIFIER_GUIDE.md)** - Complete guide for BERT, GPT-2, RoBERTa training
- **[LM_CLASSIFIER_QUICK_START.md](LM_CLASSIFIER_QUICK_START.md)** - Quick reference for LM classifiers

#### Hyperparameter Optimization
- **[ho_multifeat_usage.md](ho_multifeat_usage.md)** - Multi-feature hyperparameter optimization

### Advanced Topics

- **[dry_run_usage.md](dry_run_usage.md)** - Dry run mode for testing
- **[train-test-split.md](train-test-split.md)** - Data splitting strategies
- **[STRUCTURE_VALIDATION.md](STRUCTURE_VALIDATION.md)** - Directory structure validation
- **[MANUAL_PREDICTIONS_GUIDE.md](MANUAL_PREDICTIONS_GUIDE.md)** - Adding manual predictions to experiments

### Metrics and Evaluation

- **[CEFR_METRICS.md](CEFR_METRICS.md)** - CEFR-specific evaluation metrics
- **[REPORT_GUIDE.md](REPORT_GUIDE.md)** - Complete guide to analyzing and ranking model results
- **[REPORT_MANUAL_PREDICTIONS.md](REPORT_MANUAL_PREDICTIONS.md)** - Using report tool with manual/LLM predictions
- **[ordinal_metrics_explained.md](ordinal_metrics_explained.md)** - Ordinal regression metrics
- **[ordinal_mse_summary.md](ordinal_mse_summary.md)** - MSE for ordinal data

### Testing and Quality

- **[PIPELINE_TEST_SUMMARY.md](PIPELINE_TEST_SUMMARY.md)** - Pipeline testing summary
- **[code_quality.md](code_quality.md)** - Code quality standards
- **[code_quality_final.md](code_quality_final.md)** - Final code quality report

### Change Log

- **[CHANGES.md](CHANGES.md)** - Project changelog

---

## 🚀 Quick Navigation by Task

### I want to train a classifier...

#### Using TF-IDF Features
→ Start here: [QUICK_START.md](QUICK_START.md)
→ Full guide: [USAGE.md](USAGE.md)

#### Using Language Models (BERT, GPT-2)
→ Quick start: [LM_CLASSIFIER_QUICK_START.md](LM_CLASSIFIER_QUICK_START.md)
→ Full guide: [LANGUAGE_MODEL_CLASSIFIER_GUIDE.md](LANGUAGE_MODEL_CLASSIFIER_GUIDE.md)

### I want to run the complete pipeline...
→ [SIMPLE_TRAINING_GUIDE.md](SIMPLE_TRAINING_GUIDE.md)

### I want to optimize hyperparameters...
→ [ho_multifeat_usage.md](ho_multifeat_usage.md)

### I want to understand the metrics...
→ [CEFR_METRICS.md](CEFR_METRICS.md)

### I want to analyze and rank my results...
→ [REPORT_GUIDE.md](REPORT_GUIDE.md)

### I want to set up from scratch...
→ [setup_from_zero.md](setup_from_zero.md)

### I want to add external model predictions...
→ [MANUAL_PREDICTIONS_GUIDE.md](MANUAL_PREDICTIONS_GUIDE.md)

---

## 📖 Documentation by Pipeline Component

### Pipeline Scripts

| Script | Documentation | Description |
|--------|--------------|-------------|
| `src.pipeline` | [USAGE.md](USAGE.md) | Main pipeline orchestrator |
| `src.train_tfidf` | [USAGE.md](USAGE.md) | TF-IDF vectorizer training |
| `src.extract_features` | [USAGE.md](USAGE.md) | Feature extraction |
| `src.train_classifiers` | [QUICK_START.md](QUICK_START.md) | Classifier training (TF-IDF) |
| `src.train_lm_classifiers` | [LANGUAGE_MODEL_CLASSIFIER_GUIDE.md](LANGUAGE_MODEL_CLASSIFIER_GUIDE.md) | Classifier training (LM) |
| `src.predict` | [USAGE.md](USAGE.md) | Prediction and evaluation |
| `src.report` | [REPORT_GUIDE.md](REPORT_GUIDE.md) | Results analysis and model ranking |
| `src.train_classifiers_with_ho` | [ho_multifeat_usage.md](ho_multifeat_usage.md) | Hyperparameter optimization |

---

## 🔍 Key Concepts

### File Formats

All scripts expect CSV files with specific columns:

**Training/Test Data:**
```csv
text,cefr_level
"The student writes fluently.",C1
"Basic sentence structure.",A2
```

**Required columns:**
- `text` (configurable via `--text-column`)
- `cefr_level` or `label` (configurable via `--target-column`)

### Directory Structure

```
experiment-dir/
├── ml-training-data/          # Training CSV files
├── ml-test-data/              # Test CSV files
├── feature-models/            # Trained models
│   ├── tfidf/                 # TF-IDF models
│   └── classifiers/           # Classifier models
├── features/                  # Extracted features
└── results/                   # Predictions and evaluations
```

See [QUICK_START_STRUCTURE.md](QUICK_START_STRUCTURE.md) for details.

### CEFR Levels

The Common European Framework of Reference (CEFR) has 6 levels:
- **A1** - Beginner
- **A2** - Elementary
- **B1** - Intermediate
- **B2** - Upper Intermediate
- **C1** - Advanced
- **C2** - Proficient

See [CEFR_METRICS.md](CEFR_METRICS.md) for evaluation metrics.

---

## 💡 Common Workflows

### Workflow 1: TF-IDF Classifier Training

```bash
# 1. Train TF-IDF
python -m src.train_tfidf -e experiments/my-exp

# 2. Extract features
python -m src.extract_features -e experiments/my-exp

# 3. Train classifier
python -m src.train_classifiers -e experiments/my-exp --classifier xgboost

# 4. Predict
python -m src.predict -e experiments/my-exp
```

→ See [SIMPLE_TRAINING_GUIDE.md](SIMPLE_TRAINING_GUIDE.md)

### Workflow 2: Language Model Classifier Training

```bash
# 1. Train BERT classifier
python -m src.train_lm_classifiers train \
    -i data/train.csv \
    --text-column text \
    --target-column cefr_level \
    --model-name bert-base-uncased \
    -o experiments/bert-cefr

# 2. Predict
python -m src.train_lm_classifiers predict \
    --checkpoint experiments/bert-cefr/checkpoints/best \
    -i data/test.csv \
    --text-column text \
    -o predictions/results.json
```

→ See [LM_CLASSIFIER_QUICK_START.md](LM_CLASSIFIER_QUICK_START.md)

### Workflow 3: Complete Pipeline with Multiple Configs

```bash
python -m src.pipeline \
    -e experiments/my-exp \
    --max-features-list 1000 5000 10000 \
    --classifiers xgboost logistic \
    --cefr-column cefr_level
```

→ See [USAGE.md](USAGE.md)

---

## 🎯 Feature Comparison

| Feature | TF-IDF Classifiers | Language Model Classifiers |
|---------|-------------------|---------------------------|
| **Training Speed** | ⚡⚡⚡ Fast | 🐢 Slower |
| **Memory Usage** | 💾 Low | 💾💾 Higher |
| **Accuracy** | ✓ Good | ✓✓ Better |
| **Interpretability** | ✓✓ High | ✓ Lower |
| **Transfer Learning** | ✗ No | ✓ Yes |
| **GPU Required** | ✗ No | ✓ Recommended |
| **Best For** | Quick experiments, baselines | Maximum accuracy, production |

---

## 📞 Support

- **Issues**: Check [PIPELINE_TEST_SUMMARY.md](PIPELINE_TEST_SUMMARY.md)
- **Code Quality**: See [code_quality.md](code_quality.md)
- **Changes**: Review [CHANGES.md](CHANGES.md)

---

## 🔄 Recent Updates

- ✨ **NEW**: Language Model Classifier training (`train_lm_classifiers.py`)
  - BERT, GPT-2, RoBERTa support
  - Frozen base model training
  - Class weight balancing
  - Early stopping
  - See [LANGUAGE_MODEL_CLASSIFIER_GUIDE.md](LANGUAGE_MODEL_CLASSIFIER_GUIDE.md)

- **Pipeline**: `--add-test-set` flag for adding new test sets to experiments
- **Metrics**: Comprehensive CEFR ordinal metrics
- **Quality**: Full code quality pipeline with Black, Flake8, Mypy

---

Last updated: 2025-10-21
