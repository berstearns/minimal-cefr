# Environment Setup

## 1. Python Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

## 2. Core Dependencies

```bash
pip install pandas scikit-learn numpy pyyaml tqdm
```

## 3. Optional Dependencies

```bash
# XGBoost classifier (required for paper)
pip install xgboost

# Perplexity feature extraction (required for paper)
pip install torch transformers

# For inter-rater agreement metrics (AC2)
pip install irrCAC

# For ordinal regression (optional baseline)
pip install mord
```

## 4. Verify Installation

```bash
cd /home/b/p/cefr-classification/minimal-cefr
python -c "from src.config import GlobalConfig; print('OK')"
python -c "import xgboost; print('XGBoost OK')"
python -c "import torch; import transformers; print('Torch+Transformers OK')"
```

## 5. Data Location

All source data lives at:

```
/home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits/
```

Key files:

| File | Description | Rows |
|---|---|---|
| `norm-EFCAMDAT-train.csv` | Statistical model training set | 80k |
| `norm-EFCAMDAT-test.csv` | In-domain test set | 20k |
| `norm-EFCAMDAT-remainder.csv` | AL pre-training corpus | ~623k |
| `norm-CELVA-SP.csv` | External test (French L1, ESP writing) | 1,742 |
| `norm-KUPA-KEYS.csv` | External test (42 L1s, keystroke-logged) | 1,006 |
| `norm_andrew100k_remainder_{A1..C1}_texts.txt` | Level-split texts for AL training | varies |

All CSVs use schema: `writing_id, l1, cefr_level, text`

## 6. Experiment Directory Layout

The pipeline expects this structure per experiment:

```
experiment-dir/
├── features-training-data/   # Data for fitting TF-IDF vocabulary
│   └── *.csv
├── ml-training-data/         # Data for training classifiers
│   └── *.csv
├── ml-test-data/             # Test sets for evaluation
│   └── *.csv
├── feature-models/           # (generated) TF-IDF models + classifiers
├── features/                 # (generated) extracted feature matrices
└── results/                  # (generated) predictions + evaluation reports
```

## 7. Working Directory

All commands assume you run from the repo root:

```bash
cd /home/b/p/cefr-classification/minimal-cefr
```
