# Commands: Full Pipeline

All commands run from repo root:

```bash
cd /home/b/p/cefr-classification/minimal-cefr
```

## Variables

```bash
GDRIVE=/home/b/p/cefr-classification/gdrive-data/fe
DATA=/home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits
EXP=paper-reproducibility-ppl-classifiers-zero-shot/experiment
```

---

## Step 0: Create experiment directory structure

```bash
mkdir -p $EXP/{ml-training-data,ml-test-data,features}

cp $DATA/norm-EFCAMDAT-train.csv $EXP/ml-training-data/
cp $DATA/norm-EFCAMDAT-test.csv  $EXP/ml-test-data/
cp $DATA/norm-CELVA-SP.csv       $EXP/ml-test-data/
cp $DATA/norm-KUPA-KEYS.csv      $EXP/ml-test-data/
```

---

## Step 1: Assemble feature matrices from pre-extracted gzip files

This step reads the `.csv.features.gzip` files, column-concatenates per-model
features, and writes `features_dense.csv` files into the experiment directory.

No GPU, no model loading -- pure pandas operations.

```python
#!/usr/bin/env python3
"""assemble_ppl_features.py -- Run from repo root."""
import os
import pandas as pd

GDRIVE = "/home/b/p/cefr-classification/gdrive-data/fe"
EXP = "paper-reproducibility-ppl-classifiers-zero-shot/experiment"

# Model files per dataset (non-flat preferred, flat fallback)
# Format: { model_name: { dataset: filepath } }

MODELS = ["gpt2", "AL-all-gpt2", "AL-a1-gpt2", "AL-a2-gpt2",
           "AL-b1-gpt2", "AL-b2-gpt2", "AL-c1-gpt2"]

# Mapping: (dataset_pipeline_name, model) -> gzip file path
FILES = {
    # --- EFCAMDAT-train (andrew100ktrain) ---
    ("norm-EFCAMDAT-train", "gpt2"):
        f"{GDRIVE}/andrew100ktrain_df-gpt2_avg+ppp+ppos-fe_2025-04-23-09-31.csv.features.gzip",
    ("norm-EFCAMDAT-train", "AL-all-gpt2"):
        f"{GDRIVE}/andrew100ktrain_df-AL-all-gpt2_avg+ppp+ppos-fe_2025-04-23-13-31.csv.features.gzip",
    ("norm-EFCAMDAT-train", "AL-a1-gpt2"):
        f"{GDRIVE}/andrew100ktrain_df-AL-a1-gpt2_avg+ppp+ppos-fe_2025-04-23-09-31.csv.features.gzip",
    ("norm-EFCAMDAT-train", "AL-a2-gpt2"):
        f"{GDRIVE}/andrew100ktrain_df-AL-a2-gpt2_avg+ppp+ppos-fe_2025-04-23-09-31.csv.features.gzip",
    ("norm-EFCAMDAT-train", "AL-b1-gpt2"):
        f"{GDRIVE}/andrew100ktrain_df-AL-b1-gpt2_avg+ppp+ppos-fe_2025-04-23-09-31.csv.features.gzip",
    ("norm-EFCAMDAT-train", "AL-b2-gpt2"):
        f"{GDRIVE}/andrew100ktrain_df-AL-b2-gpt2_avg+ppp+ppos-fe_2025-04-23-09-31.csv.features.gzip",
    ("norm-EFCAMDAT-train", "AL-c1-gpt2"):
        f"{GDRIVE}/andrew100ktrain_df-AL-c1-gpt2_avg+ppp+ppos-fe_2025-04-23-09-31.csv.features.gzip",

    # --- EFCAMDAT-test (andrew100ktest) ---
    ("norm-EFCAMDAT-test", "gpt2"):
        f"{GDRIVE}/andrew100ktest_df-gpt2_avg+ppp+ppos-fe-flat_2025-04-24-00-56.csv.features.gzip",
    ("norm-EFCAMDAT-test", "AL-all-gpt2"):
        f"{GDRIVE}/andrew100ktest_df-AL-all-gpt2_avg+ppp+ppos-fe-flat_2025-04-24-00-56.csv.features.gzip",
    ("norm-EFCAMDAT-test", "AL-a1-gpt2"):
        f"{GDRIVE}/andrew100ktest_df-AL-a1-gpt2_avg+ppp+ppos-fe-flat_2025-04-24-00-56.csv.features.gzip",
    ("norm-EFCAMDAT-test", "AL-a2-gpt2"):
        f"{GDRIVE}/andrew100ktest_df-AL-a2-gpt2_avg+ppp+ppos-fe-flat_2025-04-24-00-56.csv.features.gzip",
    ("norm-EFCAMDAT-test", "AL-b1-gpt2"):
        f"{GDRIVE}/andrew100ktest_df-AL-b1-gpt2_avg+ppp+ppos-fe-flat_2025-04-24-00-56.csv.features.gzip",
    ("norm-EFCAMDAT-test", "AL-b2-gpt2"):
        f"{GDRIVE}/andrew100ktest_df-AL-b2-gpt2_avg+ppp+ppos-fe-flat_2025-04-24-00-56.csv.features.gzip",
    ("norm-EFCAMDAT-test", "AL-c1-gpt2"):
        f"{GDRIVE}/andrew100ktest_df-AL-c1-gpt2_avg+ppp+ppos-fe-flat_2025-04-24-00-56.csv.features.gzip",

    # --- CELVA-SP ---
    ("norm-CELVA-SP", "gpt2"):
        f"{GDRIVE}/celva-gpt2_avg+ppp+ppos-fe_2025-04-23-08-31.csv.features.gzip",
    ("norm-CELVA-SP", "AL-all-gpt2"):
        f"{GDRIVE}/celva-AL-all-gpt2_avg+ppp+ppos-fe_2025-04-23-08-31.csv.features.gzip",
    ("norm-CELVA-SP", "AL-a1-gpt2"):
        f"{GDRIVE}/celva-AL-a1-gpt2_avg+ppp+ppos-fe_2025-04-23-08-31.csv.features.gzip",
    ("norm-CELVA-SP", "AL-a2-gpt2"):
        f"{GDRIVE}/celva-AL-a2-gpt2_avg+ppp+ppos-fe_2025-04-23-08-31.csv.features.gzip",
    ("norm-CELVA-SP", "AL-b1-gpt2"):
        f"{GDRIVE}/celva-AL-b1-gpt2_avg+ppp+ppos-fe_2025-04-23-08-31.csv.features.gzip",
    ("norm-CELVA-SP", "AL-b2-gpt2"):
        f"{GDRIVE}/celva-AL-b2-gpt2_avg+ppp+ppos-fe_2025-04-23-08-31.csv.features.gzip",
    ("norm-CELVA-SP", "AL-c1-gpt2"):
        f"{GDRIVE}/celva-AL-c1-gpt2_avg+ppp+ppos-fe_2025-04-23-08-31.csv.features.gzip",

    # --- KUPA-KEYS ---
    ("norm-KUPA-KEYS", "gpt2"):
        f"{GDRIVE}/KUPA-KEYS-gpt2_avg+ppp+ppos-fe_2025-04-23-06-04.csv.features.gzip",
    ("norm-KUPA-KEYS", "AL-all-gpt2"):
        f"{GDRIVE}/KUPA-KEYS-AL-all-gpt2_avg+ppp+ppos-fe_2025-04-23-06-04.csv.features.gzip",
    ("norm-KUPA-KEYS", "AL-a1-gpt2"):
        f"{GDRIVE}/KUPA-KEYS-AL-a1-gpt2_avg+ppp+ppos-fe_2025-04-23-06-04.csv.features.gzip",
    ("norm-KUPA-KEYS", "AL-a2-gpt2"):
        f"{GDRIVE}/KUPA-KEYS-AL-a2-gpt2_avg+ppp+ppos-fe_2025-04-23-06-04.csv.features.gzip",
    ("norm-KUPA-KEYS", "AL-b1-gpt2"):
        f"{GDRIVE}/KUPA-KEYS-AL-b1-gpt2_avg+ppp+ppos-fe_2025-04-23-06-04.csv.features.gzip",
    ("norm-KUPA-KEYS", "AL-b2-gpt2"):
        f"{GDRIVE}/KUPA-KEYS-AL-b2-gpt2_avg+ppp+ppos-fe_2025-04-23-06-04.csv.features.gzip",
    ("norm-KUPA-KEYS", "AL-c1-gpt2"):
        f"{GDRIVE}/KUPA-KEYS-AL-c1-gpt2_avg+ppp+ppos-fe_2025-04-23-06-04.csv.features.gzip",
}

DATASETS = ["norm-EFCAMDAT-train", "norm-EFCAMDAT-test", "norm-CELVA-SP", "norm-KUPA-KEYS"]

# Feature configurations to assemble
CONFIGS = {
    "native_only":    ["gpt2"],
    "native_general": ["gpt2", "AL-all-gpt2"],
    "all_models":     MODELS,
}


def load_features(path):
    """Load gzip features, drop index column if present."""
    df = pd.read_csv(path, compression="gzip")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def assemble(config_name, model_list):
    """Assemble features for one configuration across all datasets."""
    for dataset in DATASETS:
        frames = []
        for model in model_list:
            key = (dataset, model)
            path = FILES.get(key)
            if path is None or not os.path.exists(path):
                print(f"  MISSING: {key} -> {path}")
                continue
            df = load_features(path)
            # Prefix columns with model name to avoid collisions
            df.columns = [f"{model}__{c}" for c in df.columns]
            frames.append(df)

        if not frames:
            print(f"  SKIP {config_name}/{dataset}: no feature files found")
            continue

        merged = pd.concat(frames, axis=1)
        out_dir = f"{EXP}/features/{config_name}/{dataset}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/features_dense.csv"
        merged.to_csv(out_path, index=False)

        # Save feature names
        pd.DataFrame({"feature_name": merged.columns}).to_csv(
            f"{out_dir}/feature_names.csv", index=False
        )
        print(f"  {config_name}/{dataset}: {merged.shape} -> {out_path}")


if __name__ == "__main__":
    for config_name, model_list in CONFIGS.items():
        print(f"\n=== {config_name} ===")
        assemble(config_name, model_list)
    print("\nDone.")
```

Save as `paper-reproducibility-ppl-classifiers-zero-shot/assemble_ppl_features.py`
and run:

```bash
python paper-reproducibility-ppl-classifiers-zero-shot/assemble_ppl_features.py
```

---

## Step 2: Train classifiers

For each feature configuration, train LR and XGBoost on EFCAMDAT-train.

```bash
for CONFIG in native_only native_general all_models; do
    for CLF in logistic xgboost; do
        python -m src.train_classifiers \
            -e $EXP \
            --features-file $EXP/features/${CONFIG}/norm-EFCAMDAT-train/features_dense.csv \
            --labels-csv $EXP/ml-training-data/norm-EFCAMDAT-train.csv \
            --cefr-column cefr_level \
            --classifier $CLF \
            --model-name ppl_${CONFIG}_${CLF}
    done
done
```

This produces 6 trained models under `$EXP/feature-models/classifiers/`:
- `ppl_native_only_logistic`
- `ppl_native_only_xgboost`
- `ppl_native_general_logistic`
- `ppl_native_general_xgboost`
- `ppl_all_models_logistic`
- `ppl_all_models_xgboost`

---

## Step 3: Predict on all test sets

```bash
for CONFIG in native_only native_general all_models; do
    for CLF in logistic xgboost; do
        MODEL=ppl_${CONFIG}_${CLF}
        for TEST in norm-EFCAMDAT-test norm-CELVA-SP norm-KUPA-KEYS; do
            python -m src.predict \
                -e $EXP \
                -m $MODEL \
                --features-file $EXP/features/${CONFIG}/${TEST}/features_dense.csv \
                --labels-csv $EXP/ml-test-data/${TEST}.csv \
                --cefr-column cefr_level
        done
    done
done
```

This produces 18 evaluation reports (6 models x 3 test sets) under
`$EXP/results/{model_name}/{test_dataset}/evaluation_report.md`.

---

## Step 4: Generate summary report

```bash
python -m src.report \
    -e $EXP \
    --rank accuracy \
    --summary-report $EXP/results_summary.md \
    --include-all-datasets \
    -v
```

Additional views:

```bash
# Ranked by macro F1
python -m src.report -e $EXP --rank macro_f1 --top 20

# Per-dataset breakdown
python -m src.report -e $EXP --rank adjacent_accuracy --dataset norm-CELVA-SP
python -m src.report -e $EXP --rank adjacent_accuracy --dataset norm-KUPA-KEYS
python -m src.report -e $EXP --rank accuracy --dataset norm-EFCAMDAT-test
```

---

## One-liner: Full Pipeline

```bash
cd /home/b/p/cefr-classification/minimal-cefr && \
GDRIVE=/home/b/p/cefr-classification/gdrive-data/fe && \
DATA=/home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits && \
EXP=paper-reproducibility-ppl-classifiers-zero-shot/experiment && \
mkdir -p $EXP/{ml-training-data,ml-test-data,features} && \
cp $DATA/norm-EFCAMDAT-train.csv $EXP/ml-training-data/ && \
cp $DATA/norm-EFCAMDAT-test.csv $EXP/ml-test-data/ && \
cp $DATA/norm-CELVA-SP.csv $EXP/ml-test-data/ && \
cp $DATA/norm-KUPA-KEYS.csv $EXP/ml-test-data/ && \
python paper-reproducibility-ppl-classifiers-zero-shot/assemble_ppl_features.py && \
for CONFIG in native_only native_general all_models; do \
    for CLF in logistic xgboost; do \
        python -m src.train_classifiers -e $EXP \
            --features-file $EXP/features/${CONFIG}/norm-EFCAMDAT-train/features_dense.csv \
            --labels-csv $EXP/ml-training-data/norm-EFCAMDAT-train.csv \
            --cefr-column cefr_level --classifier $CLF \
            --model-name ppl_${CONFIG}_${CLF}; \
    done; \
done && \
for CONFIG in native_only native_general all_models; do \
    for CLF in logistic xgboost; do \
        MODEL=ppl_${CONFIG}_${CLF}; \
        for TEST in norm-EFCAMDAT-test norm-CELVA-SP norm-KUPA-KEYS; do \
            python -m src.predict -e $EXP -m $MODEL \
                --features-file $EXP/features/${CONFIG}/${TEST}/features_dense.csv \
                --labels-csv $EXP/ml-test-data/${TEST}.csv \
                --cefr-column cefr_level; \
        done; \
    done; \
done && \
python -m src.report -e $EXP --rank accuracy \
    --summary-report $EXP/results_summary.md --include-all-datasets -v
```

---

## Expected Output Structure

```
paper-reproducibility-ppl-classifiers-zero-shot/
├── README.md
├── feature-files.md
├── commands.md
├── assemble_ppl_features.py
└── experiment/
    ├── ml-training-data/
    │   └── norm-EFCAMDAT-train.csv
    ├── ml-test-data/
    │   ├── norm-EFCAMDAT-test.csv
    │   ├── norm-CELVA-SP.csv
    │   └── norm-KUPA-KEYS.csv
    ├── features/
    │   ├── native_only/
    │   │   ├── norm-EFCAMDAT-train/features_dense.csv   (554 cols)
    │   │   ├── norm-EFCAMDAT-test/features_dense.csv
    │   │   ├── norm-CELVA-SP/features_dense.csv
    │   │   └── norm-KUPA-KEYS/features_dense.csv
    │   ├── native_general/                              (~1108 cols)
    │   │   └── ...
    │   └── all_models/                                  (~3878 cols)
    │       └── ...
    ├── feature-models/
    │   └── classifiers/
    │       ├── ppl_native_only_logistic/
    │       ├── ppl_native_only_xgboost/
    │       ├── ppl_native_general_logistic/
    │       ├── ppl_native_general_xgboost/
    │       ├── ppl_all_models_logistic/
    │       └── ppl_all_models_xgboost/
    ├── results/
    │   ├── ppl_native_only_logistic/
    │   │   ├── norm-EFCAMDAT-test/evaluation_report.md
    │   │   ├── norm-CELVA-SP/evaluation_report.md
    │   │   └── norm-KUPA-KEYS/evaluation_report.md
    │   ├── ppl_native_only_xgboost/
    │   │   └── ...
    │   └── ... (6 models x 3 test sets = 18 reports)
    └── results_summary.md
```

---

## Verification Checklist

- [ ] `assemble_ppl_features.py` runs without MISSING warnings
- [ ] Row counts match: training features rows == labels rows (both ~80k)
- [ ] 6 classifier models trained (3 configs x 2 classifiers)
- [ ] 18 evaluation reports generated (6 models x 3 test sets)
- [ ] `results_summary.md` contains ranked table with accuracy, F1, adjacent accuracy
- [ ] All-models XGBoost on CELVA-SP achieves highest adjacent accuracy (expected best)
