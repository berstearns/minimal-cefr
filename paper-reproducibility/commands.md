# Complete Reproduction Commands

All commands in execution order. Run from the repo root:

```bash
cd /home/b/p/cefr-classification/minimal-cefr
```

Data source:

```bash
DATA=/home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits
```

---

## Phase 0: Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas scikit-learn numpy pyyaml tqdm xgboost
pip install torch transformers   # for perplexity features & AL training
```

---

## Phase 1: Data Preparation

### 1.1 Zero-Shot Experiment Directory

```bash
EXP=data/experiments/zero-shot

mkdir -p $EXP/{features-training-data,ml-training-data,ml-test-data}

cp $DATA/norm-EFCAMDAT-remainder.csv $EXP/features-training-data/
cp $DATA/norm-EFCAMDAT-train.csv     $EXP/ml-training-data/
cp $DATA/norm-EFCAMDAT-test.csv      $EXP/ml-test-data/
cp $DATA/norm-CELVA-SP.csv           $EXP/ml-test-data/
cp $DATA/norm-KUPA-KEYS.csv          $EXP/ml-test-data/
```

### 1.2 90-10 Experiment Directory

```bash
EXP=data/experiments/90-10

mkdir -p $EXP/{features-training-data,ml-training-data,ml-test-data}

# Same TF-IDF vocabulary as zero-shot
cp $DATA/norm-EFCAMDAT-remainder.csv $EXP/features-training-data/

# Create stratified 90/10 splits
python -m utils.ds_split \
    -i $DATA/norm-CELVA-SP.csv \
    -o $EXP/ \
    --stratify-column cefr_level \
    --train-name CELVA-SP__90-10__train.csv \
    --test-name CELVA-SP__90-10__test.csv \
    --test-size 0.1 \
    --random-state 42

python -m utils.ds_split \
    -i $DATA/norm-KUPA-KEYS.csv \
    -o $EXP/ \
    --stratify-column cefr_level \
    --train-name KUPA-KEYS__90-10__train.csv \
    --test-name KUPA-KEYS__90-10__test.csv \
    --test-size 0.1 \
    --random-state 42

# Move splits to correct subdirs
mv $EXP/CELVA-SP__90-10__train.csv  $EXP/ml-training-data/
mv $EXP/KUPA-KEYS__90-10__train.csv $EXP/ml-training-data/
mv $EXP/CELVA-SP__90-10__test.csv   $EXP/ml-test-data/
mv $EXP/KUPA-KEYS__90-10__test.csv  $EXP/ml-test-data/
```

---

## Phase 2: Train Artificial Learner Models

> These commands use HuggingFace's `run_clm.py` (or equivalent). Adjust paths
> to your training script. GPU required.

### 2.1 General AL (all EFCAMDAT remainder)

```bash
python run_clm.py \
    --model_name_or_path gpt2 \
    --train_file $DATA/norm-EFCAMDAT-remainder.csv \
    --text_column text \
    --output_dir models/al-general \
    --do_train \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --save_strategy epoch \
    --fp16
```

### 2.2 Level-Specific ALs

```bash
for LEVEL in A1 A2 B1 B2 C1; do
    python run_clm.py \
        --model_name_or_path gpt2 \
        --train_file $DATA/norm_andrew100k_remainder_${LEVEL}_texts.txt \
        --output_dir models/al-${LEVEL} \
        --do_train \
        --num_train_epochs 3 \
        --per_device_train_batch_size 8 \
        --save_strategy epoch \
        --fp16
done
```

---

## Phase 3: Extract Perplexity Features

### 3.1 Native GPT-2 perplexity on all datasets

```bash
mkdir -p perplexity-features/{native,al-general,al-A1,al-A2,al-B1,al-B2,al-C1}

for DATASET in norm-EFCAMDAT-train norm-EFCAMDAT-test norm-CELVA-SP norm-KUPA-KEYS; do
    python -m src.extract_perplexity_features \
        -i $DATA/${DATASET}.csv \
        --text-column text \
        -m gpt2 \
        -d cuda \
        --aggregate-only \
        -f csv \
        -o perplexity-features/native/${DATASET}.csv
done
```

### 3.2 General AL perplexity

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

### 3.3 Level-specific AL perplexities

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

### 3.4 Extract perplexity features for 90-10 splits

```bash
EXP=data/experiments/90-10

for MODEL_NAME in gpt2 models/al-general models/al-A1 models/al-A2 models/al-B1 models/al-B2 models/al-C1; do
    # Derive output dir name from model path
    MNAME=$(basename $MODEL_NAME)
    [ "$MNAME" = "gpt2" ] && MNAME="native"

    for SPLIT in CELVA-SP__90-10__train CELVA-SP__90-10__test KUPA-KEYS__90-10__train KUPA-KEYS__90-10__test; do
        python -m src.extract_perplexity_features \
            -i $EXP/ml-training-data/${SPLIT}.csv 2>/dev/null || \
        python -m src.extract_perplexity_features \
            -i $EXP/ml-test-data/${SPLIT}.csv \
            --text-column text \
            -m $MODEL_NAME \
            -d cuda \
            --aggregate-only \
            -f csv \
            -o perplexity-features/${MNAME}/${SPLIT}.csv
    done
done
```

### 3.5 Assemble feature matrices

Merge per-model CSVs into combined feature matrices for each configuration.
Create a small helper script or run inline:

```python
# assemble_features.py
import pandas as pd
import os

DATA_DIR = os.environ.get("DATA",
    "/home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits")

CONFIGS = {
    "native-only":     ["native"],
    "native-general":  ["native", "al-general"],
    "all-models":      ["native", "al-general", "al-A1", "al-A2", "al-B1", "al-B2", "al-C1"],
}

DATASETS = [
    "norm-EFCAMDAT-train", "norm-EFCAMDAT-test",
    "norm-CELVA-SP", "norm-KUPA-KEYS",
]

for config_name, models in CONFIGS.items():
    os.makedirs(f"features/{config_name}", exist_ok=True)
    for dataset in DATASETS:
        frames = []
        for model in models:
            path = f"perplexity-features/{model}/{dataset}.csv"
            if not os.path.exists(path):
                print(f"  SKIP {path}")
                continue
            df = pd.read_csv(path)
            agg_cols = [c for c in df.columns if c not in ("text", "model")]
            df = df[agg_cols].rename(columns={c: f"{model}_{c}" for c in agg_cols})
            frames.append(df)
        if frames:
            merged = pd.concat(frames, axis=1)
            out = f"features/{config_name}/{dataset}.csv"
            merged.to_csv(out, index=False)
            print(f"  Wrote {out} ({merged.shape})")
```

```bash
python assemble_features.py
```

---

## Phase 4: TF-IDF Experiments

### 4.1 Zero-Shot: LR + XGBoost with TF-IDF

```bash
python -m src.pipeline \
    -e data/experiments/zero-shot \
    --classifiers logistic xgboost \
    --cefr-column cefr_level \
    --max-features 5000 \
    --summarize
```

### 4.2 90-10: LR + XGBoost with TF-IDF

```bash
python -m src.pipeline \
    -e data/experiments/90-10 \
    --classifiers logistic xgboost \
    --cefr-column cefr_level \
    --max-features 5000 \
    --summarize
```

---

## Phase 5: Perplexity-Feature Experiments

> For perplexity features, the assembled CSVs from Phase 3.5 need to be placed
> into experiment directories and classifiers trained on them. This uses the
> same pipeline but with pre-extracted features.

### 5.1 Zero-Shot with perplexity features

For each feature config (native-only, native-general, all-models), create
an experiment directory with the assembled feature CSVs:

```bash
for CONFIG in native-only native-general all-models; do
    EXP=data/experiments/zero-shot-ppx-${CONFIG}
    mkdir -p $EXP/{features-training-data,ml-training-data,ml-test-data}

    # Training data labels (needed for classifier fitting)
    cp $DATA/norm-EFCAMDAT-train.csv $EXP/ml-training-data/
    # Test set labels
    cp $DATA/norm-EFCAMDAT-test.csv $EXP/ml-test-data/
    cp $DATA/norm-CELVA-SP.csv      $EXP/ml-test-data/
    cp $DATA/norm-KUPA-KEYS.csv     $EXP/ml-test-data/

    # Copy assembled perplexity feature CSVs as training features
    # (The pipeline can use pre-extracted features via feature dirs)
done
```

Then train classifiers on each feature set using the pipeline's training step
(or a custom script that loads the assembled feature CSVs and fits LR/XGBoost).

### 5.2 90-10 with perplexity features

Same approach but with 90/10 splits of CELVA-SP and KUPA-KEYS.

---

## Phase 6: Generate Reports

### 6.1 Zero-Shot summary

```bash
python -m src.report \
    -e data/experiments/zero-shot \
    --rank accuracy \
    --summary-report data/experiments/zero-shot/results_summary.md \
    -v

python -m src.report \
    -e data/experiments/zero-shot \
    --rank macro_f1 \
    --top 20

python -m src.report \
    -e data/experiments/zero-shot \
    --rank adjacent_accuracy \
    --dataset norm-CELVA-SP

python -m src.report \
    -e data/experiments/zero-shot \
    --rank adjacent_accuracy \
    --dataset norm-KUPA-KEYS
```

### 6.2 90-10 summary

```bash
python -m src.report \
    -e data/experiments/90-10 \
    --rank accuracy \
    --summary-report data/experiments/90-10/results_summary.md \
    -v

python -m src.report \
    -e data/experiments/90-10 \
    --rank macro_f1 \
    --top 20
```

### 6.3 Per-dataset detailed results

```bash
# View a specific evaluation report
cat data/experiments/zero-shot/results/*/norm-CELVA-SP/evaluation_report.md
cat data/experiments/zero-shot/results/*/norm-KUPA-KEYS/evaluation_report.md
cat data/experiments/zero-shot/results/*/norm-EFCAMDAT-test/evaluation_report.md
```

---

## Phase 7: Baselines (Optional)

### 7.1 Fine-tuned BERT

```bash
python -m src.train_lm_classifiers train \
    --model bert-base-uncased \
    --train-csv $DATA/norm-EFCAMDAT-train.csv \
    --val-csv $DATA/norm-EFCAMDAT-test.csv \
    --text-column text \
    --label-column cefr_level \
    --output-dir models/bert-finetuned \
    --epochs 5 \
    --batch-size 16 \
    --device cuda

# Predict on test sets
for TESTSET in norm-EFCAMDAT-test norm-CELVA-SP norm-KUPA-KEYS; do
    python -m src.train_lm_classifiers predict \
        --model-dir models/bert-finetuned \
        --input-csv $DATA/${TESTSET}.csv \
        --text-column text \
        --label-column cefr_level \
        --output-dir results/bert/${TESTSET} \
        --device cuda
done
```

### 7.2 Zero-Shot LLM Baselines

Zero-shot LLM predictions are handled via the prompting experiment. Results
are placed in `data/experiments/prompting/results/` and analyzed with:

```bash
python -m src.report \
    -e data/experiments/prompting \
    --mode manual \
    --labels-dir data/experiments/zero-shot/ml-test-data \
    --rank accuracy \
    -v
```

---

## Quick Reference: Key Flags

| Flag | Purpose | Example |
|---|---|---|
| `-e` | Experiment directory | `-e data/experiments/zero-shot` |
| `--classifier` | Single classifier | `--classifier xgboost` |
| `--classifiers` | Multiple classifiers | `--classifiers logistic xgboost` |
| `--cefr-column` | CEFR label column name | `--cefr-column cefr_level` |
| `--max-features` | TF-IDF vocabulary size | `--max-features 5000` |
| `--steps` | Run specific pipeline steps | `--steps 1 2` |
| `--summarize` | Generate summary after pipeline | `--summarize` |
| `--add-test-set` | Add new test set to experiment | `--add-test-set /path/to.csv` |
| `--rank` | Ranking metric for report | `--rank macro_f1` |
| `--dataset` | Filter report by dataset | `--dataset norm-CELVA-SP` |
| `-v` | Verbose output (report) | `-v` |
| `-q` | Quiet output (pipeline) | `-q` |
| `-d cuda` | GPU for perplexity extraction | `-d cuda` |
| `--aggregate-only` | Skip per-token features | `--aggregate-only` |

---

## Verification Checklist

After all experiments complete, verify:

- [ ] Zero-shot TF-IDF results exist for LR and XGBoost on all 3 test sets
- [ ] 90-10 TF-IDF results exist for LR and XGBoost on CELVA-SP and KUPA-KEYS
- [ ] Perplexity features extracted for all 7 models on all datasets
- [ ] Feature matrices assembled for 3 perplexity configs
- [ ] Perplexity-based classifiers trained and evaluated
- [ ] `results_summary.md` generated for each experiment
- [ ] Metrics match paper tables (accuracy, F1, adjacent accuracy)
