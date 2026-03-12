# Data Preparation

## Source Data

All normalized CSVs are at:
```
DATA=/home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits
```

## Experiment 1: Zero-Shot

The zero-shot experiment trains on EFCAMDAT and evaluates on all three test
sets (EFCAMDAT-test, CELVA-SP, KUPA-KEYS) without any target-domain exposure.

### Directory Setup

```bash
EXP=data/experiments/zero-shot
DATA=/home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits

mkdir -p $EXP/{features-training-data,ml-training-data,ml-test-data}

# TF-IDF is fit on the remainder (same corpus used to train ALs)
cp $DATA/norm-EFCAMDAT-remainder.csv $EXP/features-training-data/

# Statistical classifiers train on the 80k EFCAMDAT split
cp $DATA/norm-EFCAMDAT-train.csv $EXP/ml-training-data/

# Three test sets
cp $DATA/norm-EFCAMDAT-test.csv $EXP/ml-test-data/
cp $DATA/norm-CELVA-SP.csv      $EXP/ml-test-data/
cp $DATA/norm-KUPA-KEYS.csv     $EXP/ml-test-data/
```

### Rationale

- **features-training-data**: The TF-IDF vectorizer vocabulary is learned from
  the remainder corpus (~623k texts). This mirrors the AL pre-training data and
  ensures the feature space covers the full proficiency range.
- **ml-training-data**: The 80k EFCAMDAT training split is used to fit
  classifier weights. This is separate from the AL pre-training data.
- **ml-test-data**: All three evaluation sets are placed here. The pipeline
  evaluates each model on every test set automatically.

## Experiment 2: 90-10 Split

The 90-10 experiment uses EFCAMDAT-trained perplexity features but adapts
classifiers to each external corpus by training on 90% and testing on 10%.

### Create 90-10 Splits

```bash
DATA=/home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits

# CELVA-SP 90/10
python -m utils.ds_split \
    -i $DATA/norm-CELVA-SP.csv \
    -o data/experiments/90-10/ \
    --stratify-column cefr_level \
    --train-name CELVA-SP__90-10__train.csv \
    --test-name CELVA-SP__90-10__test.csv \
    --test-size 0.1 \
    --random-state 42

# KUPA-KEYS 90/10
python -m utils.ds_split \
    -i $DATA/norm-KUPA-KEYS.csv \
    -o data/experiments/90-10/ \
    --stratify-column cefr_level \
    --train-name KUPA-KEYS__90-10__train.csv \
    --test-name KUPA-KEYS__90-10__test.csv \
    --test-size 0.1 \
    --random-state 42
```

### Directory Setup

```bash
EXP=data/experiments/90-10
DATA=/home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits

mkdir -p $EXP/{features-training-data,ml-training-data,ml-test-data}

# TF-IDF vocabulary from EFCAMDAT remainder (same as zero-shot)
cp $DATA/norm-EFCAMDAT-remainder.csv $EXP/features-training-data/

# 90% train splits go to ml-training-data
mv $EXP/CELVA-SP__90-10__train.csv  $EXP/ml-training-data/
mv $EXP/KUPA-KEYS__90-10__train.csv $EXP/ml-training-data/

# 10% test splits go to ml-test-data
mv $EXP/CELVA-SP__90-10__test.csv  $EXP/ml-test-data/
mv $EXP/KUPA-KEYS__90-10__test.csv $EXP/ml-test-data/
```

### Rationale

- TF-IDF features come from the same vocabulary as zero-shot (EFCAMDAT
  remainder), so the feature representation is identical.
- Classifiers are trained on 90% of each external corpus to test whether
  domain-specific decision boundaries improve over zero-shot transfer.
- This isolates the contribution of in-domain classifier training while keeping
  feature representations constant.

## CSV Schema Reference

All data CSVs follow the same schema:

```
writing_id,l1,cefr_level,text
115499,German,B1,"The student wrote an essay about..."
```

The pipeline must be told to use `cefr_level` (not the default `cefr_label`)
via the `--cefr-column cefr_level` flag.
