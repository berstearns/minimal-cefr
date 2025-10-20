#!/bin/bash
# Ablation Study: max_features × max_df (COMPACT VERSION)
# Grid: 3 max_features (100, 1000, 5000) × 3 max_df (0.8, 0.9, 0.95) = 9 configs
# Uses --max-features-list to run all max_features values in one command

# ═══════════════════════════════════════════════════════════════════════════
# MAX_DF = 0.8 (Remove terms appearing in >80% of documents)
# ═══════════════════════════════════════════════════════════════════════════

echo "Running: max_features=[100, 1000, 5000], max_df=0.8"
python -m src.pipeline \
    -e data/experiments/zero-shot \
    --max-features-list 100 1000 5000 \
    --ngram-min 1 \
    --ngram-max 2 \
    --max-df 0.8 \
    --classifier xgboost \
    --cefr-column cefr_level

# ═══════════════════════════════════════════════════════════════════════════
# MAX_DF = 0.9 (Remove terms appearing in >90% of documents)
# ═══════════════════════════════════════════════════════════════════════════

echo "Running: max_features=[100, 1000, 5000], max_df=0.9"
python -m src.pipeline \
    -e data/experiments/zero-shot \
    --max-features-list 100 1000 5000 \
    --ngram-min 1 \
    --ngram-max 2 \
    --max-df 0.9 \
    --classifier xgboost \
    --cefr-column cefr_level

# ═══════════════════════════════════════════════════════════════════════════
# MAX_DF = 0.95 (Default - Remove terms appearing in >95% of documents)
# ═══════════════════════════════════════════════════════════════════════════

echo "Running: max_features=[100, 1000, 5000], max_df=0.95"
python -m src.pipeline \
    -e data/experiments/zero-shot \
    --max-features-list 100 1000 5000 \
    --ngram-min 1 \
    --ngram-max 2 \
    --max-df 0.95 \
    --classifier xgboost \
    --cefr-column cefr_level

echo "═══════════════════════════════════════════════════════════════════════════"
echo "Ablation study complete! 9 TF-IDF configurations trained."
echo "Results organized by unique hash directories - no overwrites!"
echo "═══════════════════════════════════════════════════════════════════════════"

