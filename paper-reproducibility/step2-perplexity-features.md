# Step 2: Extracting Perplexity Features

This step uses the trained Artificial Learner models (from Step 1) and the
pre-trained GPT-2 to extract perplexity features for all datasets.

## Overview

For each text, we compute aggregate perplexity statistics from 7 models:

1. **Native GPT-2** (`gpt2`) -- native speaker reference
2. **General AL** (`models/al-general`) -- full learner trajectory
3. **A1 AL** through **C1 AL** (`models/al-{A1..C1}`) -- level-specific

The aggregate features (mean, median, std perplexity, etc.) become the input
features for the statistical classifiers.

## Feature Extraction Using `src/extract_perplexity_features.py`

### Extract features for each model on each dataset

The script processes one model at a time. Run for each (model, dataset) pair.

#### Native GPT-2 perplexity

```bash
DATA=/home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits

# On EFCAMDAT train
python -m src.extract_perplexity_features \
    -i $DATA/norm-EFCAMDAT-train.csv \
    --text-column text \
    -m gpt2 \
    -d cuda \
    --aggregate-only \
    -f csv \
    -o perplexity-features/native/efcamdat-train.csv

# On EFCAMDAT test
python -m src.extract_perplexity_features \
    -i $DATA/norm-EFCAMDAT-test.csv \
    --text-column text \
    -m gpt2 \
    -d cuda \
    --aggregate-only \
    -f csv \
    -o perplexity-features/native/efcamdat-test.csv

# On CELVA-SP
python -m src.extract_perplexity_features \
    -i $DATA/norm-CELVA-SP.csv \
    --text-column text \
    -m gpt2 \
    -d cuda \
    --aggregate-only \
    -f csv \
    -o perplexity-features/native/celva-sp.csv

# On KUPA-KEYS
python -m src.extract_perplexity_features \
    -i $DATA/norm-KUPA-KEYS.csv \
    --text-column text \
    -m gpt2 \
    -d cuda \
    --aggregate-only \
    -f csv \
    -o perplexity-features/native/kupa-keys.csv
```

#### General AL perplexity

```bash
for DATASET in norm-EFCAMDAT-train norm-EFCAMDAT-test norm-CELVA-SP norm-KUPA-KEYS; do
    python -m src.extract_perplexity_features \
        -i $DATA/${DATASET}.csv \
        --text-column text \
        -m models/al-general \
        -d cuda \
        --aggregate-only \
        -f csv \
        -o perplexity-features/al-general/${DATASET}.csv
done
```

#### Level-specific AL perplexities

```bash
for LEVEL in A1 A2 B1 B2 C1; do
    for DATASET in norm-EFCAMDAT-train norm-EFCAMDAT-test norm-CELVA-SP norm-KUPA-KEYS; do
        python -m src.extract_perplexity_features \
            -i $DATA/${DATASET}.csv \
            --text-column text \
            -m models/al-${LEVEL} \
            -d cuda \
            --aggregate-only \
            -f csv \
            -o perplexity-features/al-${LEVEL}/${DATASET}.csv
    done
done
```

## Assembling Feature Matrices

After extraction, you need to merge the per-model aggregate CSVs into combined
feature matrices that the classifiers can consume. Each row is one text; columns
are the aggregate perplexity statistics from each model.

### Feature Configuration 1: Native only

Columns: `native_mean_perplexity, native_median_perplexity, native_std_perplexity, ...`

### Feature Configuration 2: Native + General AL

Columns: native features + `general_mean_perplexity, general_median_perplexity, ...`

### Feature Configuration 3: All models

Columns: native + general + A1 + A2 + B1 + B2 + C1 features

### Assembly script (example)

```python
import pandas as pd

def assemble_features(dataset_name, model_names, output_path):
    """Merge per-model perplexity CSVs into a single feature matrix."""
    frames = []
    for model in model_names:
        df = pd.read_csv(f"perplexity-features/{model}/{dataset_name}.csv")
        # Prefix columns with model name
        agg_cols = [c for c in df.columns if c not in ('text', 'model')]
        df = df[agg_cols].rename(columns={c: f"{model}_{c}" for c in agg_cols})
        frames.append(df)
    merged = pd.concat(frames, axis=1)
    merged.to_csv(output_path, index=False)
    return merged

# Config 1: native only
assemble_features("norm-EFCAMDAT-train", ["native"], "features/native-only/train.csv")

# Config 2: native + general
assemble_features("norm-EFCAMDAT-train", ["native", "al-general"], "features/native-general/train.csv")

# Config 3: all models
all_models = ["native", "al-general", "al-A1", "al-A2", "al-B1", "al-B2", "al-C1"]
assemble_features("norm-EFCAMDAT-train", all_models, "features/all-models/train.csv")
```

## Output Structure

```
perplexity-features/
├── native/
│   ├── efcamdat-train.csv
│   ├── efcamdat-test.csv
│   ├── celva-sp.csv
│   └── kupa-keys.csv
├── al-general/
│   ├── ...
├── al-A1/
│   ├── ...
├── al-A2/
├── al-B1/
├── al-B2/
└── al-C1/

features/              # Assembled feature matrices
├── native-only/
├── native-general/
└── all-models/
```

## Notes

- Use `--aggregate-only` to save only summary statistics (mean, median, std,
  min, max perplexity + entropy), not per-token details. Per-token output for
  80k texts would be enormous.
- Use `-d cuda` for GPU acceleration. CPU extraction of 80k texts is very slow.
- The `--limit N` flag is useful for quick sanity checks during development.
