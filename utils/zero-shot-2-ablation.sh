#!/bin/bash
# ============================================================================
# EXTENSIVE ABLATION STUDY: TF-IDF vs Grouped TF-IDF
# ============================================================================
# Explores combinations of:
# - Feature types: standard TF-IDF vs grouped TF-IDF (concatenated)
# - max_features: 500, 1000, 2000, 5000, 10000
# - n-grams: (1,1), (1,2), (1,3), (2,2), (2,3)
# - min_df: 1, 2, 5, 10
# - max_df: 0.8, 0.9, 0.95, 1.0
# - classifiers: xgboost, logistic
# ============================================================================

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================
PYTHON_BIN="/home/b/.pyenv/versions/3.10.18/bin/python3"  # Change to your Python path (e.g., "python3", "/usr/bin/python3", "py")
EXPERIMENT_DIR="data/experiments/zero-shot-2"
CEFR_COLUMN="cefr_level"

# Skip to specific part (set to 1 to skip parts 1-7, useful for resuming)
# START_FROM_PART=${START_FROM_PART:-1}  # Default: start from part 1
START_FROM_PART=1

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          EXTENSIVE ABLATION STUDY: TF-IDF CONFIGURATIONS            ║${NC}"
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo ""
echo "Experiment Directory: $EXPERIMENT_DIR"
echo "CEFR Column: $CEFR_COLUMN"
echo ""

# ============================================================================
# PART 1: STANDARD TF-IDF - MAX FEATURES ABLATION
# ============================================================================
if [ $START_FROM_PART -le 1 ]; then
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}PART 1: Standard TF-IDF - Max Features Ablation${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

MAX_FEATURES_LIST=(500 1000 2000 5000 10000)

for max_feat in "${MAX_FEATURES_LIST[@]}"; do
    echo -e "${YELLOW}[1.1] Training TF-IDF: max_features=${max_feat}, ngram=(1,2), min_df=2, max_df=0.95${NC}"

    $PYTHON_BIN -m src.train_tfidf \
        -e $EXPERIMENT_DIR \
        --max-features $max_feat \
        --ngram-min 1 \
        --ngram-max 2 \
        --min-df 2 \
        --max-df 0.95

    echo -e "${GREEN}✓ TF-IDF trained (max_features=${max_feat})${NC}\n"
done
fi

# ============================================================================
# PART 2: STANDARD TF-IDF - N-GRAM ABLATION
# ============================================================================
if [ $START_FROM_PART -le 2 ]; then
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}PART 2: Standard TF-IDF - N-gram Ablation${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

NGRAM_CONFIGS=(
    "1 1"  # unigrams only
    "1 2"  # unigrams + bigrams
    "1 3"  # unigrams + bigrams + trigrams
    "2 2"  # bigrams only
    "2 3"  # bigrams + trigrams
)

for ngram in "${NGRAM_CONFIGS[@]}"; do
    read -r ngram_min ngram_max <<< "$ngram"
    echo -e "${YELLOW}[2.1] Training TF-IDF: max_features=5000, ngram=(${ngram_min},${ngram_max}), min_df=2, max_df=0.95${NC}"

    $PYTHON_BIN -m src.train_tfidf \
        -e $EXPERIMENT_DIR \
        --max-features 5000 \
        --ngram-min $ngram_min \
        --ngram-max $ngram_max \
        --min-df 2 \
        --max-df 0.95

    echo -e "${GREEN}✓ TF-IDF trained (ngram=${ngram_min}-${ngram_max})${NC}\n"
done
fi

# ============================================================================
# PART 3: STANDARD TF-IDF - MIN_DF ABLATION
# ============================================================================
if [ $START_FROM_PART -le 3 ]; then
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}PART 3: Standard TF-IDF - Min DF Ablation${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

MIN_DF_LIST=(1 2 5 10)

for min_df in "${MIN_DF_LIST[@]}"; do
    echo -e "${YELLOW}[3.1] Training TF-IDF: max_features=5000, ngram=(1,2), min_df=${min_df}, max_df=0.95${NC}"

    $PYTHON_BIN -m src.train_tfidf \
        -e $EXPERIMENT_DIR \
        --max-features 5000 \
        --ngram-min 1 \
        --ngram-max 2 \
        --min-df $min_df \
        --max-df 0.95

    echo -e "${GREEN}✓ TF-IDF trained (min_df=${min_df})${NC}\n"
done
fi

# ============================================================================
# PART 4: STANDARD TF-IDF - MAX_DF ABLATION
# ============================================================================
if [ $START_FROM_PART -le 4 ]; then
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}PART 4: Standard TF-IDF - Max DF Ablation${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

MAX_DF_LIST=(0.8 0.9 0.95 1.0)

for max_df in "${MAX_DF_LIST[@]}"; do
    echo -e "${YELLOW}[4.1] Training TF-IDF: max_features=5000, ngram=(1,2), min_df=2, max_df=${max_df}${NC}"

    $PYTHON_BIN -m src.train_tfidf \
        -e $EXPERIMENT_DIR \
        --max-features 5000 \
        --ngram-min 1 \
        --ngram-max 2 \
        --min-df 2 \
        --max-df $max_df

    echo -e "${GREEN}✓ TF-IDF trained (max_df=${max_df})${NC}\n"
done
fi

# ============================================================================
# PART 5: GROUPED TF-IDF (CONCATENATED) - MAX FEATURES ABLATION
# ============================================================================
if [ $START_FROM_PART -le 5 ]; then
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}PART 5: Grouped TF-IDF - Max Features Per Group Ablation${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

GROUPED_MAX_FEATURES=(100 200 500 1000 2000)

for max_feat in "${GROUPED_MAX_FEATURES[@]}"; do
    echo -e "${YELLOW}[5.1] Training Grouped TF-IDF: max_features=${max_feat} per group (×5 groups = $((max_feat * 5)) total)${NC}"

    $PYTHON_BIN -m src.train_tfidf_groupby \
        -e $EXPERIMENT_DIR \
        --group-by $CEFR_COLUMN \
        --max-features $max_feat \
        --ngram-min 1 \
        --ngram-max 2 \
        --min-df 2 \
        --max-df 0.95

    echo -e "${GREEN}✓ Grouped TF-IDF trained (${max_feat} features per group, $((max_feat * 5)) total)${NC}\n"
done
fi

# ============================================================================
# PART 6: GROUPED TF-IDF - N-GRAM ABLATION
# ============================================================================
if [ $START_FROM_PART -le 6 ]; then
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}PART 6: Grouped TF-IDF - N-gram Ablation${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

for ngram in "${NGRAM_CONFIGS[@]}"; do
    read -r ngram_min ngram_max <<< "$ngram"
    echo -e "${YELLOW}[6.1] Training Grouped TF-IDF: max_features=1000, ngram=(${ngram_min},${ngram_max})${NC}"

    $PYTHON_BIN -m src.train_tfidf_groupby \
        -e $EXPERIMENT_DIR \
        --group-by $CEFR_COLUMN \
        --max-features 1000 \
        --ngram-min $ngram_min \
        --ngram-max $ngram_max \
        --min-df 2 \
        --max-df 0.95

    echo -e "${GREEN}✓ Grouped TF-IDF trained (ngram=${ngram_min}-${ngram_max})${NC}\n"
done
fi

# ============================================================================
# PART 7: EXTRACT FEATURES FOR ALL MODELS
# ============================================================================
if [ $START_FROM_PART -le 7 ]; then
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}PART 7: Extracting Features for All Models${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo -e "${YELLOW}[7.1] Finding all trained TF-IDF models...${NC}"

# Find all TF-IDF model directories
TFIDF_MODELS=$(find $EXPERIMENT_DIR/feature-models -type d -name "*_tfidf*" -maxdepth 1 2>/dev/null)

if [ -z "$TFIDF_MODELS" ]; then
    echo -e "${RED}✗ No TF-IDF models found!${NC}"
    exit 1
fi

echo "Found $(echo "$TFIDF_MODELS" | wc -l) TF-IDF models"
echo ""

# Extract features for each model
for model_dir in $TFIDF_MODELS; do
    model_name=$(basename $model_dir)
    echo -e "${YELLOW}[7.2] Extracting features: ${model_name}${NC}"

    $PYTHON_BIN -m src.extract_features \
        -e $EXPERIMENT_DIR \
        -p $model_dir \
        -s both \
        -q

    echo -e "${GREEN}✓ Features extracted for ${model_name}${NC}\n"
done
fi

# ============================================================================
# PART 8: TRAIN CLASSIFIERS ON ALL FEATURE SETS
# ============================================================================
if [ $START_FROM_PART -le 8 ]; then
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}PART 8: Training Classifiers on All Feature Sets${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Find all feature directories
FEATURE_DIRS=$(find $EXPERIMENT_DIR/features -mindepth 1 -maxdepth 1 -type d 2>/dev/null)

if [ -z "$FEATURE_DIRS" ]; then
    echo -e "${RED}✗ No feature directories found!${NC}"
    exit 1
fi

echo "Found $(echo "$FEATURE_DIRS" | wc -l) feature sets"
echo ""

CLASSIFIERS="xgboost logistic"

for feature_dir in $FEATURE_DIRS; do
    feature_name=$(basename $feature_dir)
    echo -e "${YELLOW}[8.1] Training classifiers on: ${feature_name}${NC}"

    for clf in $CLASSIFIERS; do
        echo -e "${YELLOW}  [8.1.${clf}] Training ${clf}...${NC}"
        $PYTHON_BIN -m src.train_classifiers \
            -e $EXPERIMENT_DIR \
            --batch-features-dir $feature_dir \
            --classifier $clf \
            --cefr-column $CEFR_COLUMN \
            -q
    done

    echo -e "${GREEN}✓ Classifiers trained on ${feature_name}${NC}\n"
done
fi

# ============================================================================
# PART 9: GENERATE PREDICTIONS
# ============================================================================
if [ $START_FROM_PART -le 9 ]; then
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}PART 9: Generating Predictions for All Models${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo -e "${YELLOW}[9.1] Running predictions on all classifiers...${NC}"

$PYTHON_BIN -m src.predict \
    -e $EXPERIMENT_DIR \
    --cefr-column $CEFR_COLUMN

echo -e "${GREEN}✓ Predictions generated for all models${NC}\n"
fi

# ============================================================================
# PART 10: GENERATE ANALYSIS REPORTS
# ============================================================================
if [ $START_FROM_PART -le 10 ]; then
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}PART 10: Generating Analysis Reports${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

REPORT_FILE="${EXPERIMENT_DIR}/ablation_study_results_$(date +%Y%m%d_%H%M%S).md"

echo -e "${YELLOW}[10.1] Generating comprehensive summary report...${NC}"

$PYTHON_BIN -m src.report \
    -e $EXPERIMENT_DIR \
    --summary-report $REPORT_FILE

echo -e "${GREEN}✓ Summary report generated: ${REPORT_FILE}${NC}\n"

echo -e "${YELLOW}[10.2] Top 20 models by adjacent accuracy...${NC}"

$PYTHON_BIN -m src.report \
    -e $EXPERIMENT_DIR \
    --rank adjacent_accuracy \
    --top 20 \
    > "${EXPERIMENT_DIR}/top_20_adjacent_accuracy.txt"

echo -e "${GREEN}✓ Top 20 ranking saved${NC}\n"

echo -e "${YELLOW}[10.3] Top 20 models by accuracy...${NC}"

$PYTHON_BIN -m src.report \
    -e $EXPERIMENT_DIR \
    --rank accuracy \
    --top 20 \
    > "${EXPERIMENT_DIR}/top_20_accuracy.txt"

echo -e "${GREEN}✓ Top 20 ranking saved${NC}\n"
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    ABLATION STUDY COMPLETE!                          ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Summary:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Count models
TFIDF_COUNT=$(find $EXPERIMENT_DIR/feature-models -maxdepth 1 -name "*_tfidf" -type d 2>/dev/null | wc -l)
GROUPED_COUNT=$(find $EXPERIMENT_DIR/feature-models -maxdepth 1 -name "*_tfidf_grouped" -type d 2>/dev/null | wc -l)
CLASSIFIER_COUNT=$(find $EXPERIMENT_DIR/feature-models/classifiers -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)

echo "📊 Standard TF-IDF models: $TFIDF_COUNT"
echo "📊 Grouped TF-IDF models: $GROUPED_COUNT"
echo "📊 Trained classifiers: $CLASSIFIER_COUNT"
echo ""
echo "📁 Results directory: $EXPERIMENT_DIR"
echo "📄 Summary report: $REPORT_FILE"
echo "📄 Top 20 (adjacent accuracy): ${EXPERIMENT_DIR}/top_20_adjacent_accuracy.txt"
echo "📄 Top 20 (accuracy): ${EXPERIMENT_DIR}/top_20_accuracy.txt"
echo ""
echo -e "${YELLOW}View results:${NC}"
echo "  cat $REPORT_FILE"
echo "  cat ${EXPERIMENT_DIR}/top_20_adjacent_accuracy.txt"
echo ""
echo -e "${YELLOW}Interactive ranking:${NC}"
echo "  $PYTHON_BIN -m src.report -e $EXPERIMENT_DIR --rank adjacent_accuracy --top 10"
echo ""
echo -e "${GREEN}✓ All experiments completed successfully!${NC}"

