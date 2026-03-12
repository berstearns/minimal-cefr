# Tangible Commands

All commands run from repo root: `cd /home/b/p/cefr-classification/minimal-cefr`

## Prerequisites

- Python 3.10 via pyenv: `~/.pyenv/versions/3.10.18/bin/python3` (aliased as `py`)
- Required packages: pandas, scikit-learn, xgboost, torch, transformers
- GPT-2 model is auto-downloaded from HuggingFace Hub on first run (~550MB)
- CPU only (no GPU required)

## One-command run (recommended)

```bash
~/.pyenv/versions/3.10.18/bin/python3 paper-reproducibility-gpt2-native-zero-shot/scripts/run_experiment.py
```

This orchestrates all steps automatically with error handling.

## Manual step-by-step commands

### Step 0: Setup data

```bash
DATA=/home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits
EXP=paper-reproducibility-gpt2-native-zero-shot/experiment

cp $DATA/norm-EFCAMDAT-train.csv $EXP/ml-training-data/
cp $DATA/norm-EFCAMDAT-test.csv  $EXP/ml-test-data/
cp $DATA/norm-CELVA-SP.csv       $EXP/ml-test-data/
cp $DATA/norm-KUPA-KEYS.csv      $EXP/ml-test-data/
```

### Step 1: Extract GPT-2 perplexity (CPU, with limits)

```bash
RAWDIR=paper-reproducibility-gpt2-native-zero-shot/perplexity-raw
mkdir -p $RAWDIR

python -m src.extract_perplexity_features \
    -i $DATA/norm-EFCAMDAT-train.csv --text-column text \
    -m gpt2 -d cpu --aggregate-only -f csv \
    --limit 2000 \
    -o $RAWDIR/norm-EFCAMDAT-train.csv

python -m src.extract_perplexity_features \
    -i $DATA/norm-EFCAMDAT-test.csv --text-column text \
    -m gpt2 -d cpu --aggregate-only -f csv \
    --limit 500 \
    -o $RAWDIR/norm-EFCAMDAT-test.csv

python -m src.extract_perplexity_features \
    -i $DATA/norm-CELVA-SP.csv --text-column text \
    -m gpt2 -d cpu --aggregate-only -f csv \
    -o $RAWDIR/norm-CELVA-SP.csv

python -m src.extract_perplexity_features \
    -i $DATA/norm-KUPA-KEYS.csv --text-column text \
    -m gpt2 -d cpu --aggregate-only -f csv \
    -o $RAWDIR/norm-KUPA-KEYS.csv
```

### Step 2: Convert to features_dense.csv

```python
# Python snippet (or run from run_experiment.py)
import pandas as pd
for name in ["norm-EFCAMDAT-train", "norm-EFCAMDAT-test", "norm-CELVA-SP", "norm-KUPA-KEYS"]:
    df = pd.read_csv(f"paper-reproducibility-gpt2-native-zero-shot/perplexity-raw/{name}.csv")
    df_num = df.drop(columns=["text", "model"], errors="ignore")
    out_dir = f"paper-reproducibility-gpt2-native-zero-shot/experiment/features/gpt2_native/{name}"
    import os; os.makedirs(out_dir, exist_ok=True)
    df_num.to_csv(f"{out_dir}/features_dense.csv", index=False)
```

### Step 3: Train classifiers

```bash
EXP=paper-reproducibility-gpt2-native-zero-shot/experiment

# Logistic Regression
python -m src.train_classifiers \
    -e $EXP \
    --features-file $EXP/features/gpt2_native/norm-EFCAMDAT-train/features_dense.csv \
    --labels-csv $EXP/ml-training-data/norm-EFCAMDAT-train.csv \
    --cefr-column cefr_level \
    --classifier logistic \
    --model-name norm-EFCAMDAT-train_logistic_gpt2native

# XGBoost
python -m src.train_classifiers \
    -e $EXP \
    --features-file $EXP/features/gpt2_native/norm-EFCAMDAT-train/features_dense.csv \
    --labels-csv $EXP/ml-training-data/norm-EFCAMDAT-train.csv \
    --cefr-column cefr_level \
    --classifier xgboost \
    --model-name norm-EFCAMDAT-train_xgboost_gpt2native
```

### Step 4: Predict on test sets

```bash
EXP=paper-reproducibility-gpt2-native-zero-shot/experiment

for MODEL in norm-EFCAMDAT-train_logistic_gpt2native norm-EFCAMDAT-train_xgboost_gpt2native; do
    for TEST in norm-EFCAMDAT-test norm-CELVA-SP norm-KUPA-KEYS; do
        python -m src.predict \
            -e $EXP \
            -m $MODEL \
            --features-file $EXP/features/gpt2_native/$TEST/features_dense.csv \
            --labels-csv $EXP/ml-test-data/$TEST.csv \
            --cefr-column cefr_level
    done
done
```

### Step 5: Generate report

```bash
python -m src.report \
    -e paper-reproducibility-gpt2-native-zero-shot/experiment \
    --rank accuracy \
    --summary-report paper-reproducibility-gpt2-native-zero-shot/experiment/results_summary.md \
    --include-all-datasets \
    -v
```

## Remove limits for full paper reproduction

In `scripts/run_experiment.py`, set all values in `LIMITS` to `None`:

```python
LIMITS = {
    "norm-EFCAMDAT-train": None,   # was 2000
    "norm-EFCAMDAT-test":  None,   # was 500
    "norm-CELVA-SP":       None,
    "norm-KUPA-KEYS":      None,
}
```

Warning: Full extraction on CPU takes many hours (80k + 20k + 1.7k + 1k texts through GPT-2).
