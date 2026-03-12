# Step 3: TF-IDF Feature Experiments

This step runs the TF-IDF feature pipeline for both zero-shot and 90-10
scenarios. TF-IDF serves as one of the four feature configurations evaluated
in the paper (Feature Config 4).

## Scenario 1: Zero-Shot TF-IDF

### Pre-requisites

The zero-shot experiment directory should already be set up per
[`data-preparation.md`](data-preparation.md):

```
data/experiments/zero-shot/
├── features-training-data/
│   └── norm-EFCAMDAT-remainder.csv
├── ml-training-data/
│   └── norm-EFCAMDAT-train.csv
└── ml-test-data/
    ├── norm-EFCAMDAT-test.csv
    ├── norm-CELVA-SP.csv
    └── norm-KUPA-KEYS.csv
```

### Run Full Pipeline (LR + XGBoost)

```bash
# Logistic Regression
python -m src.pipeline \
    -e data/experiments/zero-shot \
    --classifier logistic \
    --cefr-column cefr_level \
    --max-features 5000 \
    --summarize

# XGBoost
python -m src.pipeline \
    -e data/experiments/zero-shot \
    --classifier xgboost \
    --cefr-column cefr_level \
    --max-features 5000 \
    --summarize
```

Or train both in a single run:

```bash
python -m src.pipeline \
    -e data/experiments/zero-shot \
    --classifiers logistic xgboost \
    --cefr-column cefr_level \
    --max-features 5000 \
    --summarize
```

### What Happens

1. **Step 1** -- Fits TF-IDF on `features-training-data/norm-EFCAMDAT-remainder.csv`
2. **Step 2** -- Extracts TF-IDF features for all CSVs in `ml-training-data/`
   and `ml-test-data/`
3. **Step 3** -- Trains LR and XGBoost on `norm-EFCAMDAT-train` features
4. **Step 4** -- Predicts on `norm-EFCAMDAT-test`, `norm-CELVA-SP`, `norm-KUPA-KEYS`
5. **Summary** -- Generates `results_summary.md` with ranked results

### Generated Outputs

```
data/experiments/zero-shot/
├── feature-models/
│   ├── tfidf/
│   │   └── <hash>_tfidf/
│   │       ├── tfidf_model.pkl
│   │       └── config.json
│   └── classifiers/
│       ├── norm-EFCAMDAT-train_logistic_<hash>/
│       │   ├── classifier.pkl
│       │   ├── label_encoder.pkl
│       │   └── config.json
│       └── norm-EFCAMDAT-train_xgboost_<hash>/
│           └── ...
├── features/
│   └── <hash>_tfidf/
│       ├── norm-EFCAMDAT-train/
│       ├── norm-EFCAMDAT-test/
│       ├── norm-CELVA-SP/
│       └── norm-KUPA-KEYS/
├── results/
│   ├── norm-EFCAMDAT-train_logistic_<hash>/
│   │   ├── norm-EFCAMDAT-test/
│   │   │   ├── evaluation_report.md
│   │   │   ├── argmax_predictions.json
│   │   │   └── soft_predictions.json
│   │   ├── norm-CELVA-SP/
│   │   └── norm-KUPA-KEYS/
│   └── norm-EFCAMDAT-train_xgboost_<hash>/
│       └── ...
└── results_summary.md
```

## Scenario 2: 90-10 TF-IDF

### Pre-requisites

The 90-10 experiment directory should be set up per
[`data-preparation.md`](data-preparation.md).

### Run Full Pipeline

```bash
# Both classifiers
python -m src.pipeline \
    -e data/experiments/90-10 \
    --classifiers logistic xgboost \
    --cefr-column cefr_level \
    --max-features 5000 \
    --summarize
```

### What Happens

1. TF-IDF is fit on `norm-EFCAMDAT-remainder.csv` (same vocabulary as zero-shot)
2. Features extracted for 90% train splits and 10% test splits
3. LR and XGBoost trained on `CELVA-SP__90-10__train` and `KUPA-KEYS__90-10__train`
4. Predictions on `CELVA-SP__90-10__test` and `KUPA-KEYS__90-10__test`

## Running Individual Steps

If you need to re-run only part of the pipeline:

```bash
# Step 1 only: Train TF-IDF
python -m src.pipeline \
    -e data/experiments/zero-shot \
    --steps 1 \
    --cefr-column cefr_level

# Steps 2-4: Extract features, train classifiers, predict
python -m src.pipeline \
    -e data/experiments/zero-shot \
    --steps 2 3 4 \
    --classifier xgboost \
    --cefr-column cefr_level
```

## Adding a New Test Set to Existing Experiment

If you later want to evaluate on an additional test set:

```bash
python -m src.pipeline \
    -e data/experiments/zero-shot \
    --add-test-set /path/to/new-test-set.csv \
    --cefr-column cefr_level
```

This will:
1. Copy the CSV into `ml-test-data/`
2. Extract features using all existing TF-IDF models
3. Predict using all trained classifiers
4. Generate evaluation reports

## TF-IDF Hyperparameter Notes

The paper uses default TF-IDF settings. For ablation studies:

```bash
# Vary vocabulary size
python -m src.pipeline \
    -e data/experiments/zero-shot \
    --max-features-list 1000 5000 10000 20000 \
    --classifiers logistic xgboost \
    --cefr-column cefr_level

# Vary n-gram range
python -m src.pipeline \
    -e data/experiments/zero-shot \
    --ngram-min 1 --ngram-max 3 \
    --classifier xgboost \
    --cefr-column cefr_level
```

Each TF-IDF configuration gets a unique hashed directory, so experiments
never overwrite each other.
