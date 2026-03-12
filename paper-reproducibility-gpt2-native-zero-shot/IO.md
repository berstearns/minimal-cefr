# Input/Output Documentation for Each Step

## Design Decisions

### Row Limits for CPU Runs

Full perplexity extraction on CPU is extremely slow (GPT-2 processes each text token-by-token).
For a quick end-to-end pipeline validation, the following limits are applied via `--limit`:

| Dataset | Full rows | Limited to | Reason |
|---|---|---|---|
| norm-EFCAMDAT-train | 80,000 | 2,000 | Training set; 80k texts would take ~40h on CPU |
| norm-EFCAMDAT-test | 20,000 | 500 | In-domain test; 20k texts would take ~10h on CPU |
| norm-CELVA-SP | 1,742 | None (full) | Small enough for CPU in ~45 min |
| norm-KUPA-KEYS | 1,006 | None (full) | Small enough for CPU in ~25 min |

To reproduce full paper results: set all limits to `None` in `scripts/run_experiment.py`.

### Label Trimming

When `--limit` is used, the extracted features have fewer rows than the original labels CSV.
The script automatically trims labels to match by taking the first N rows (preserving order).
Trimmed labels are saved to `trimmed-labels/` to avoid modifying source data.

### CEFR Column

The data uses `cefr_level` (not the default `cefr_label` expected by `src/`).
All commands explicitly pass `--cefr-column cefr_level`.

### Feature Naming

Feature directory is `gpt2_native` to distinguish from fine-tuned GPT-2 variants.
Model names follow pattern: `norm-EFCAMDAT-train_{classifier}_gpt2native`.

### Python Environment

Uses pyenv Python 3.10.18 (`~/.pyenv/versions/3.10.18/bin/python3`).
The `sys.executable` in `run_experiment.py` ensures child processes use the same interpreter.

### Pipeline vs Manual Mode

`src/pipeline.py` only automates TF-IDF experiments. For perplexity features, we use
`src/train_classifiers.py` and `src/predict.py` directly with `--features-file` and
`--labels-csv` flags (manual/external features mode).

---

## Step 0: Setup

**Input:**
- Source CSVs from `/home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits/`
  - `norm-EFCAMDAT-train.csv` (80k rows, schema: writing_id,l1,cefr_level,text)
  - `norm-EFCAMDAT-test.csv` (20k rows, same schema)
  - `norm-CELVA-SP.csv` (1,742 rows, same schema)
  - `norm-KUPA-KEYS.csv` (1,006 rows, same schema)

**Output:**
- `experiment/ml-training-data/norm-EFCAMDAT-train.csv`
- `experiment/ml-test-data/norm-EFCAMDAT-test.csv`
- `experiment/ml-test-data/norm-CELVA-SP.csv`
- `experiment/ml-test-data/norm-KUPA-KEYS.csv`

---

## Step 1: Extract GPT-2 Perplexity

**Input:** Each source CSV + pre-trained GPT-2 model (auto-downloaded from HuggingFace Hub)

**Command:** `python -m src.extract_perplexity_features -i <csv> --text-column text -m gpt2 -d cpu --aggregate-only -f csv -o <output>`

**Output:** `perplexity-raw/{dataset_name}.csv`

Output CSV schema:
```
text,model,mean_perplexity,median_perplexity,std_perplexity,min_perplexity,max_perplexity,mean_entropy,std_entropy,total_tokens
"The student wrote...",gpt2,45.2,32.1,28.4,1.2,312.5,5.5,3.2,42
```

**Key detail:** `--aggregate-only` outputs summary statistics per text, not per-token details.

---

## Step 2: Convert to features_dense.csv

**Input:** `perplexity-raw/{dataset_name}.csv` (from Step 1)

**Processing:** Drop `text` and `model` columns, keep only numeric columns:
`mean_perplexity, median_perplexity, std_perplexity, min_perplexity, max_perplexity, mean_entropy, std_entropy, total_tokens`

**Output:**
- `experiment/features/gpt2_native/{dataset_name}/features_dense.csv` (N rows x 8 numeric columns)
- `experiment/features/gpt2_native/{dataset_name}/feature_names.csv` (8 feature names)

features_dense.csv schema (no text, no model -- pure numeric):
```
mean_perplexity,median_perplexity,std_perplexity,min_perplexity,max_perplexity,mean_entropy,std_entropy,total_tokens
45.2,32.1,28.4,1.2,312.5,5.5,3.2,42
```

---

## Step 3: Train Classifiers

**Input:**
- `experiment/features/gpt2_native/norm-EFCAMDAT-train/features_dense.csv` (training features)
- `experiment/ml-training-data/norm-EFCAMDAT-train.csv` (labels, column: cefr_level)
  - If --limit was used in Step 1, labels are trimmed to match feature count

**Command:** `python -m src.train_classifiers -e experiment/ --features-file <features> --labels-csv <labels> --cefr-column cefr_level --classifier {logistic|xgboost} --model-name <name>`

**Output:** `experiment/feature-models/classifiers/{model_name}/`
- `classifier.pkl` -- trained sklearn/xgboost model
- `label_encoder.pkl` -- maps CEFR levels to integers (A1=0, A2=1, B1=2, B2=3, C1=4, C2=5)
- `config.json` -- training metadata (n_samples, n_features, classes, etc.)
- `xgb_label_mapping.pkl` -- (XGBoost only) contiguous label remapping

Model names: `norm-EFCAMDAT-train_logistic_gpt2native`, `norm-EFCAMDAT-train_xgboost_gpt2native`

---

## Step 4: Predict

**Input:**
- Trained classifier from `experiment/feature-models/classifiers/{model_name}/`
- Test features from `experiment/features/gpt2_native/{test_name}/features_dense.csv`
- Test labels from `experiment/ml-test-data/{test_name}.csv` (for evaluation)

**Command:** `python -m src.predict -e experiment/ -m <model_name> --features-file <features> --labels-csv <labels> --cefr-column cefr_level`

**Output:** `experiment/results/{model_name}/{test_name}/`
- `evaluation_report.md` -- full classification report with accuracy, F1, confusion matrix
- `argmax_predictions.json` -- predicted labels via argmax strategy
- `soft_predictions.json` -- full probability distributions per sample
- `rounded_avg_predictions.json` -- predictions via expected-value rounding

evaluation_report.md contains:
- Strategy 1 (Argmax): precision/recall/F1 per class, macro/weighted averages, accuracy, adjacent accuracy, confusion matrix, calibration report
- Strategy 2 (Rounded Average): same metrics using regression-style prediction

---

## Step 5: Report

**Input:** All `evaluation_report.md` files in `experiment/results/`

**Command:** `python -m src.report -e experiment/ --rank accuracy --summary-report experiment/results_summary.md -v`

**Output:** `experiment/results_summary.md`
- Ranked table of all (model, dataset, strategy) combinations
- Sorted by accuracy (or other chosen metric)
- Shows: rank, model name, dataset, strategy, accuracy, adjacent_accuracy, macro_f1, weighted_f1

---

## Directory Structure After Complete Run

```
paper-reproducibility-gpt2-native-zero-shot/
├── plan/
│   ├── overview.md
│   └── steps.md
├── scripts/
│   └── run_experiment.py
├── perplexity-raw/                          # Step 1 output
│   ├── norm-EFCAMDAT-train.csv
│   ├── norm-EFCAMDAT-test.csv
│   ├── norm-CELVA-SP.csv
│   └── norm-KUPA-KEYS.csv
├── trimmed-labels/                          # Created if --limit used
│   ├── norm-EFCAMDAT-train.csv
│   └── norm-EFCAMDAT-test.csv
├── experiment/
│   ├── ml-training-data/
│   │   └── norm-EFCAMDAT-train.csv
│   ├── ml-test-data/
│   │   ├── norm-EFCAMDAT-test.csv
│   │   ├── norm-CELVA-SP.csv
│   │   └── norm-KUPA-KEYS.csv
│   ├── features/                            # Step 2 output
│   │   └── gpt2_native/
│   │       ├── norm-EFCAMDAT-train/
│   │       │   ├── features_dense.csv
│   │       │   └── feature_names.csv
│   │       ├── norm-EFCAMDAT-test/
│   │       ├── norm-CELVA-SP/
│   │       └── norm-KUPA-KEYS/
│   ├── feature-models/                      # Step 3 output
│   │   └── classifiers/
│   │       ├── norm-EFCAMDAT-train_logistic_gpt2native/
│   │       │   ├── classifier.pkl
│   │       │   ├── label_encoder.pkl
│   │       │   └── config.json
│   │       └── norm-EFCAMDAT-train_xgboost_gpt2native/
│   │           ├── classifier.pkl
│   │           ├── label_encoder.pkl
│   │           ├── xgb_label_mapping.pkl
│   │           └── config.json
│   ├── results/                             # Step 4 output
│   │   ├── norm-EFCAMDAT-train_logistic_gpt2native/
│   │   │   ├── norm-EFCAMDAT-test/
│   │   │   │   ├── evaluation_report.md
│   │   │   │   ├── argmax_predictions.json
│   │   │   │   ├── soft_predictions.json
│   │   │   │   └── rounded_avg_predictions.json
│   │   │   ├── norm-CELVA-SP/
│   │   │   └── norm-KUPA-KEYS/
│   │   └── norm-EFCAMDAT-train_xgboost_gpt2native/
│   │       ├── norm-EFCAMDAT-test/
│   │       ├── norm-CELVA-SP/
│   │       └── norm-KUPA-KEYS/
│   └── results_summary.md                   # Step 5 output
├── IO.md
└── commands.md
```
