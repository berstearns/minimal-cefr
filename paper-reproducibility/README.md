# Reproducing: Artificial Learner CEFR Classification (LREC 2026)

This directory contains step-by-step instructions to reproduce all experiments
from the paper using the `minimal-cefr` pipeline.

## Paper Summary

The paper presents a statistical approach to CEFR classification using
LLM-derived perplexity features combined with interpretable ML models (Logistic
Regression and XGBoost). "Artificial Learners" (GPT-2 models fine-tuned on
proficiency-stratified learner text) generate perplexity-based feature
representations that are then consumed by traditional classifiers.

## Mapping Paper to Code

| Paper Concept | Repo Component |
|---|---|
| TF-IDF feature extraction | `src/train_tfidf.py` + `src/extract_features.py` |
| Perplexity feature extraction | `src/extract_perplexity_features.py` |
| LR / XGBoost training | `src/train_classifiers.py` (classifier types: `logistic`, `xgboost`) |
| Full pipeline orchestration | `src/pipeline.py` |
| Evaluation & ranking | `src/predict.py` + `src/report.py` |
| Data splitting utility | `utils/ds_split.py` |

## Datasets

| Dataset | Role | Samples | Path |
|---|---|---|---|
| EFCAMDAT remainder | AL pre-training (held-out) | ~623k | `norm-EFCAMDAT-remainder.csv` |
| EFCAMDAT train | Statistical model training | 80k | `norm-EFCAMDAT-train.csv` |
| EFCAMDAT test | In-domain evaluation | 20k | `norm-EFCAMDAT-test.csv` |
| CELVA-SP | Cross-corpus evaluation | 1,742 | `norm-CELVA-SP.csv` |
| KUPA-KEYS | Cross-corpus evaluation | 1,006 | `norm-KUPA-KEYS.csv` |

All source CSVs share the schema: `writing_id, l1, cefr_level, text`

## Experimental Scenarios

1. **Zero-Shot** (Scenario 1): Train on EFCAMDAT only, test on all three test
   sets without any target-domain exposure.
2. **90-10 Split** (Scenario 2): Use EFCAMDAT-trained perplexity features but
   train classifiers on 90% of each external corpus; test on remaining 10%.

## Feature Configurations (per scenario)

Each scenario trains LR and XGBoost on four feature sets:

1. Native perplexities only (pre-trained GPT-2)
2. Native + General Artificial Learner perplexities
3. All model perplexities (native + general AL + 5 level-specific ALs)
4. TF-IDF features

## Documentation Files

| File | Purpose |
|---|---|
| [`setup.md`](setup.md) | Environment, dependencies, directory layout |
| [`data-preparation.md`](data-preparation.md) | Preparing data splits and symlinks |
| [`step1-artificial-learners.md`](step1-artificial-learners.md) | Training the GPT-2 Artificial Learner models |
| [`step2-perplexity-features.md`](step2-perplexity-features.md) | Extracting perplexity features from ALs |
| [`step3-tfidf-experiments.md`](step3-tfidf-experiments.md) | TF-IDF feature pipeline (zero-shot & 90-10) |
| [`step4-evaluation.md`](step4-evaluation.md) | Generating reports, ranking, and paper tables |
| [`commands.md`](commands.md) | All concrete commands in execution order |
