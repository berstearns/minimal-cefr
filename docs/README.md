# Minimal CEFR Classification Pipeline

A minimal scikit-learn compliant pipeline for CEFR (Common European Framework of Reference for Languages) text classification using pretrained TF-IDF and ML models with `make_pipeline`.

## Project Structure

```
minimal-cefr/
├── src/                          # Core pipeline modules
│   ├── config.py                # Centralized configuration system
│   ├── train_tfidf.py           # Step 1: Train TF-IDF vectorizer
│   ├── extract_features.py      # Step 2: Extract test features
│   ├── train_classifiers.py     # Step 3: Train ML classifiers
│   ├── predict.py               # Step 4: Make predictions
│   └── pipeline.py              # Main pipeline orchestrator
├── data/experiments/            # Experiment directories
│   ├── zero-shot/
│   └── 90-10/
├── pipeline.py                  # Wrapper script for main pipeline
├── train_tfidf.py               # Wrapper script for step 1
├── extract_features.py          # Wrapper script for step 2
├── train_classifiers.py         # Wrapper script for step 3
├── predict.py                   # Wrapper script for step 4
├── create_sample_data.py        # Generate sample CEFR data
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── USAGE.md                     # Comprehensive usage guide
```

## Features

- ✅ Scikit-learn API compliant with `make_pipeline` support
- ✅ Pretrained model reusability (TF-IDF + Classifiers)
- ✅ Multiple experiment configurations
- ✅ Multiple classifier support (Naive Bayes, Logistic Regression, SVM, Random Forest)
- ✅ Imbalanced dataset evaluation with `classification_report_imbalanced`
- ✅ Modular step-by-step pipeline

## Pipeline Overview

```
Step 1: Train TF-IDF Vectorizer
  └─ Input: features-training-data/*.csv
  └─ Output: models/tfidf/tfidf_model.pkl

Step 2: Extract Test Features
  └─ Input: ml-test-data/*.csv
  └─ Output: features/<test_name>/{features_dense.csv, feature_names.csv, config.json}

Step 3: Train ML Classifiers
  └─ Input: ml-training-data/*.csv (each file = separate model)
  └─ Output: models/classifiers/<model_name>/classifier.pkl

Step 4: Predict with make_pipeline
  └─ Uses: make_pipeline(PretrainedTfidf, PretrainedClassifier)
  └─ Output: results/<model_name>/{predictions.csv, results.json}
  └─ Evaluation: classification_report_imbalanced
```

## Directory Structure

```
data/experiments/
├── zero-shot/
│   ├── features-training-data/    # For TF-IDF training
│   │   └── norm-efcamdat-remainder.csv
│   ├── ml-training-data/          # For classifier training (1 file = 1 model)
│   │   └── norm-efcamdat-train.csv
│   ├── ml-test-data/              # Test sets
│   │   ├── norm-celva-sp.csv
│   │   ├── norm-efcamdat-test.csv
│   │   └── norm-kupa-keys.csv
│   ├── models/                    # Generated models
│   ├── features/                  # Generated test features
│   └── results/                   # Generated predictions
└── 90-10/
    └── (same structure)
```

## Data Format

All CSV files must contain:
- `text`: The text content to classify
- `label`: The CEFR level (A1, A2, B1, B2, C1, C2)

Example:
```csv
text,label
"Hello. My name is John.",A1
"I believe learning languages is important.",B1
"The proliferation of digital technologies has transformed communication.",C1
```

## Installation

```bash
pip install pandas scikit-learn imbalanced-learn
```

## Usage

### Quick Start - Run Full Pipeline

Run the complete pipeline for a single experiment:
```bash
python run_pipeline.py data/experiments/zero-shot
```

Run with a specific classifier:
```bash
python run_pipeline.py data/experiments/zero-shot logistic
```

Run all experiments:
```bash
python run_pipeline.py all
python run_pipeline.py all randomforest
```

### Step-by-Step Execution

#### Step 1: Train TF-IDF Vectorizer
```bash
python step1_train_tfidf.py data/experiments/zero-shot
```

#### Step 2: Extract Test Features
```bash
# Extract features for all test sets
python step2_extract_test_features.py data/experiments/zero-shot

# Extract features for specific test set
python step2_extract_test_features.py data/experiments/zero-shot norm-celva-sp.csv
```

#### Step 3: Train Classifiers
```bash
# Train Multinomial Naive Bayes (default)
python step3_train_classifiers.py data/experiments/zero-shot

# Train with specific classifier
python step3_train_classifiers.py data/experiments/zero-shot logistic

# Train specific file only
python step3_train_classifiers.py data/experiments/zero-shot multinomialnb norm-efcamdat-train.csv
```

#### Step 4: Predict with Pipeline
```bash
# Predict all combinations (all classifiers × all test sets)
python step4_predict_with_pipeline.py data/experiments/zero-shot

# Predict specific combination
python step4_predict_with_pipeline.py data/experiments/zero-shot norm-efcamdat-train_multinomialnb norm-celva-sp.csv
```

## Available Classifiers

- `multinomialnb` - Multinomial Naive Bayes (default)
- `logistic` - Logistic Regression
- `randomforest` - Random Forest
- `svm` - Linear SVM

## Pipeline Implementation Details

### make_pipeline with Pretrained Models

The pipeline uses wrapper classes to make pretrained models compatible with sklearn's `make_pipeline`:

```python
from sklearn.pipeline import make_pipeline

# Create pipeline with pretrained models
pipeline = make_pipeline(
    PretrainedTfidfWrapper(tfidf_model_path),
    PretrainedClassifierWrapper(classifier_path)
)

# Use like any sklearn pipeline
y_pred = pipeline.predict(X_test)
```

The wrappers implement `fit()`, `transform()`, and `predict()` methods for sklearn compatibility, while using the pretrained models internally.

## Evaluation Metrics

The pipeline outputs:
1. **classification_report_imbalanced** - Handles imbalanced CEFR datasets
2. **Standard classification report** - Precision, recall, F1-score per class
3. **Confusion matrix** - Visualization of prediction errors
4. **Overall accuracy**

Example output:
```
CLASSIFICATION REPORT (imbalanced-learn)
                   pre       rec       spe        f1       geo       iba       sup

         A1       0.85      0.90      0.98      0.87      0.94      0.87        10
         A2       0.80      0.75      0.96      0.77      0.85      0.70        10
         B1       0.75      0.80      0.95      0.77      0.87      0.74        10
         ...
```

## Output Files

After running the pipeline, you'll find:

```
models/
├── tfidf/
│   ├── tfidf_model.pkl
│   └── config.json
└── classifiers/
    └── norm-efcamdat-train_multinomialnb/
        ├── classifier.pkl
        └── config.json

features/
└── norm-celva-sp/
    ├── features_dense.csv
    ├── feature_names.csv
    └── config.json

results/
└── norm-efcamdat-train_multinomialnb/
    ├── norm-celva-sp_predictions.csv
    └── norm-celva-sp_results.json
```

## Create Sample Data

To test the pipeline with sample data:
```bash
python create_sample_data.py
```

This creates sample CEFR-labeled texts in the experiment directories.

## Multiple Training Files

The pipeline supports training multiple independent models from different training files. Each file in `ml-training-data/` creates a separate model:

```
ml-training-data/
├── dataset1.csv  → models/classifiers/dataset1_multinomialnb/
├── dataset2.csv  → models/classifiers/dataset2_multinomialnb/
└── dataset3.csv  → models/classifiers/dataset3_multinomialnb/
```

## Extending the Pipeline

### Add New Classifier

Edit `step3_train_classifiers.py`:
```python
CLASSIFIERS = {
    'multinomialnb': MultinomialNB,
    'logistic': lambda: LogisticRegression(...),
    'yournewclassifier': lambda: YourClassifier(...)
}
```

### Customize TF-IDF Parameters

Edit `step1_train_tfidf.py`:
```python
tfidf = TfidfVectorizer(
    max_features=10000,      # Increase vocabulary
    ngram_range=(1, 3),      # Add trigrams
    min_df=5,                # Different min frequency
    # ... your parameters
)
```

## Troubleshooting

**Issue**: `FileNotFoundError: TF-IDF model not found`
- **Solution**: Run step 1 first: `python step1_train_tfidf.py <experiment_dir>`

**Issue**: Empty data files
- **Solution**: Populate CSV files with data or run `python create_sample_data.py` for testing

**Issue**: `'text' or 'label' column not found`
- **Solution**: Ensure all CSV files have `text` and `label` columns

## License

MIT
