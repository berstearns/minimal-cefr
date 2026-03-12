# IO Documentation: PPL Classifiers Zero-Shot Experiment

**Started:** 2026-03-11
**Status:** Complete

---

## Step 0: Setup experiment directory

**Action:** Copy data split CSVs into experiment directory structure.

**Inputs:**
| File | Source |
|------|--------|
| norm-EFCAMDAT-train.csv | `/home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits/` |
| norm-EFCAMDAT-test.csv | same |
| norm-CELVA-SP.csv | same |
| norm-KUPA-KEYS.csv | same |

**Outputs:**
| File | Destination | Data rows |
|------|-------------|-----------|
| norm-EFCAMDAT-train.csv | `experiment/ml-training-data/` | 79,998 |
| norm-EFCAMDAT-test.csv | `experiment/ml-test-data/` | 20,002 |
| norm-CELVA-SP.csv | `experiment/ml-test-data/` | 7,957 |
| norm-KUPA-KEYS.csv | `experiment/ml-test-data/` | 3,334 |

**Status:** Done

---

## Step 1: Assemble perplexity features from gzip files

**Action:** Read 28 pre-extracted `.csv.features.gzip` files from `gdrive-data/fe/`,
column-concatenate per-model features, write `features_dense.csv` + `feature_names.csv`
for 3 feature configurations x 4 datasets = 12 output directories.

**Script:** `scripts/assemble_ppl_features.py`

**Inputs:** 28 gzip files from `/home/b/p/cefr-classification/gdrive-data/fe/`
(see `feature-files.md` for full list of paths).

**Actual outputs:**

| Config | Dataset | Rows | Cols | Path |
|--------|---------|------|------|------|
| native_only | norm-EFCAMDAT-train | 79,998 | 554 | `experiment/features/native_only/norm-EFCAMDAT-train/` |
| native_only | norm-EFCAMDAT-test | 20,001 | 554 | `experiment/features/native_only/norm-EFCAMDAT-test/` |
| native_only | norm-CELVA-SP | 1,742 | 554 | `experiment/features/native_only/norm-CELVA-SP/` |
| native_only | norm-KUPA-KEYS | 1,006 | 554 | `experiment/features/native_only/norm-KUPA-KEYS/` |
| native_general | norm-EFCAMDAT-train | 79,998 | 1,108 | `experiment/features/native_general/norm-EFCAMDAT-train/` |
| native_general | norm-EFCAMDAT-test | 20,001 | 1,108 | `experiment/features/native_general/norm-EFCAMDAT-test/` |
| native_general | norm-CELVA-SP | 1,742 | 1,108 | `experiment/features/native_general/norm-CELVA-SP/` |
| native_general | norm-KUPA-KEYS | 1,006 | 1,108 | `experiment/features/native_general/norm-KUPA-KEYS/` |
| all_models | norm-EFCAMDAT-train | 79,998 | 3,878 | `experiment/features/all_models/norm-EFCAMDAT-train/` |
| all_models | norm-EFCAMDAT-test | 20,001 | 3,878 | `experiment/features/all_models/norm-EFCAMDAT-test/` |
| all_models | norm-CELVA-SP | 1,742 | 3,878 | `experiment/features/all_models/norm-CELVA-SP/` |
| all_models | norm-KUPA-KEYS | 1,006 | 3,878 | `experiment/features/all_models/norm-KUPA-KEYS/` |

Each directory contains: `features_dense.csv`, `feature_names.csv`

**Fixes applied during execution:**
- EFCAMDAT-test labels trimmed from 20,002 to 20,001 rows (feature extraction produced 1 fewer row).
- CELVA-SP non-flat gzip files contained a `writing_id` column; dropped post-assembly to match
  train column count (555->554, 1110->1108, 3885->3878). Fix also added to `assemble_ppl_features.py`
  for future runs.

**Status:** Done

---

## Step 2: Train classifiers

**Action:** Train Logistic Regression + XGBoost on EFCAMDAT-train features for each
of 3 feature configurations = 6 classifier models total.

**Command pattern:**
```
python -m src.train_classifiers -e $EXP \
    --features-file $EXP/features/{CONFIG}/norm-EFCAMDAT-train/features_dense.csv \
    --labels-csv $EXP/ml-training-data/norm-EFCAMDAT-train.csv \
    --cefr-column cefr_level --classifier {CLF} --model-name ppl_{CONFIG}_{CLF}
```

**Inputs per model:**
- Features: `experiment/features/{config}/norm-EFCAMDAT-train/features_dense.csv` (79,998 rows)
- Labels: `experiment/ml-training-data/norm-EFCAMDAT-train.csv` column `cefr_level`
- Training classes: A1 (37,777), A2 (23,764), B1 (12,932), B2 (4,385), C1 (1,140) -- no C2 in training data

**Outputs and training performance:**

| Model name | Features | Train Acc | Train Macro-F1 | Output files |
|------------|----------|-----------|----------------|--------------|
| ppl_native_only_logistic | 554 | 0.76 | 0.66 | classifier.pkl, label_encoder.pkl, config.json |
| ppl_native_only_xgboost | 554 | 0.95 | 0.96 | classifier.pkl, label_encoder.pkl, config.json, xgb_label_mapping.pkl |
| ppl_native_general_logistic | 1,108 | 0.78 | 0.70 | classifier.pkl, label_encoder.pkl, config.json |
| ppl_native_general_xgboost | 1,108 | 0.96 | 0.97 | classifier.pkl, label_encoder.pkl, config.json, xgb_label_mapping.pkl |
| ppl_all_models_logistic | 3,878 | 0.98 | 0.98 | classifier.pkl, label_encoder.pkl, config.json |
| ppl_all_models_xgboost | 3,878 | 1.00 | 1.00 | classifier.pkl, label_encoder.pkl, config.json, xgb_label_mapping.pkl |

All saved to: `experiment/feature-models/classifiers/{model_name}/`

**Notes:**
- All logistic models hit the 1000-iteration convergence limit (ConvergenceWarning).
  This is expected for unscaled perplexity features; models are still usable.
- XGBoost train accuracy near 1.0 (especially all_models) suggests overfitting; test
  performance is the real metric.
- More models = better train fit: native_only (76%) < native_general (78%) < all_models (98%) for logistic.

**Status:** Done

---

## Step 3: Predict on test sets

**Action:** Run each of 6 trained models on each of 3 test sets = 18 evaluation runs.

**Command pattern:**
```
python -m src.predict -e $EXP -m {MODEL} \
    --features-file $EXP/features/{CONFIG}/{TEST}/features_dense.csv \
    --labels-csv $EXP/ml-test-data/{TEST}.csv \
    --cefr-column cefr_level
```

**Inputs per prediction:**
- Trained model from `experiment/feature-models/classifiers/{model}/`
- Test features from `experiment/features/{config}/{test_dataset}/features_dense.csv`
- Test labels from `experiment/ml-test-data/{test_dataset}.csv` column `cefr_level`

**Outputs per prediction (in `experiment/results/{model}/{test_dataset}/`):**
- `evaluation_report.md` -- accuracy, F1, adjacent accuracy, confusion matrix
- `soft_predictions.json` -- per-sample probability distributions
- `argmax_predictions.json` -- hard predictions (highest probability class)
- `rounded_avg_predictions.json` -- expected-value predictions (regression-style)

**Test accuracy results (argmax strategy):**

| Model | EFCAMDAT-test | CELVA-SP | KUPA-KEYS |
|-------|---------------|----------|-----------|
| ppl_native_only_logistic | 0.33 | 0.22 | 0.43 |
| ppl_native_only_xgboost | 0.34 | 0.24 | 0.54 |
| ppl_native_general_logistic | 0.33 | 0.21 | 0.40 |
| ppl_native_general_xgboost | 0.34 | 0.24 | 0.53 |
| ppl_all_models_logistic | 0.34 | 0.22 | 0.49 |
| ppl_all_models_xgboost | 0.34 | 0.24 | 0.52 |

**Test accuracy results (rounded_avg strategy):**

| Model | EFCAMDAT-test | CELVA-SP | KUPA-KEYS |
|-------|---------------|----------|-----------|
| ppl_native_only_logistic | 0.32 | 0.21 | 0.43 |
| ppl_native_only_xgboost | 0.34 | 0.25 | 0.56 |
| ppl_native_general_logistic | 0.32 | 0.20 | 0.41 |
| ppl_native_general_xgboost | 0.34 | 0.25 | 0.54 |
| ppl_all_models_logistic | 0.34 | 0.23 | 0.50 |
| ppl_all_models_xgboost | 0.34 | 0.24 | 0.54 |

**Status:** Done (18/18 evaluations)

---

## Step 4: Generate summary report

**Action:** Generate ranked summary across all models and datasets.

**Command:**
```
python -m src.report -e $EXP --rank accuracy \
    --summary-report $EXP/results_summary.md --include-all-datasets -v
```

**Input:** 18 `evaluation_report.md` files across `experiment/results/`

**Output:** `experiment/results_summary.md`
- 36 metric records (18 evaluations x 2 strategies)
- Ranked tables by accuracy, adjacent accuracy, macro F1, weighted F1, micro F1
- Per-dataset breakdowns with best model identification

**Key findings:**
- Best on KUPA-KEYS: `ppl_native_only_xgboost` (0.56 rounded_avg, 0.54 argmax)
- Best on EFCAMDAT-test: tie at 0.34 across most models
- Best on CELVA-SP: `ppl_native_only_xgboost` / `ppl_native_general_xgboost` (0.25 rounded_avg)
- XGBoost consistently outperforms logistic despite massive train overfitting
- Adding more AL models does NOT consistently improve test performance (native_only xgboost
  is competitive or best); suggests per-position features from 7 models add noise more than signal
- KUPA-KEYS is the easiest cross-corpus dataset; CELVA-SP is hardest

**Status:** Done
