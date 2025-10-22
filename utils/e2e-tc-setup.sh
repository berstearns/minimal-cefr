#!/usr/bin/env bash

################################################################################
# Language Model Text Classifier Experiment Setup Script
#
# Purpose: Set up folder structure for training BERT and GPT-2 text classifiers
#
# Features:
#   - Creates organized experiment directory structure
#   - Supports both BERT and GPT-2 classifier setups
#   - Prepares data directories for train/validation/test splits
#   - Generates configuration templates for quick start
#   - Option to create multiple model experiment folders at once
################################################################################

set -euo pipefail

################################################################################
# CONFIGURATION
################################################################################

declare -A Config=(
    # Default paths
    [EXPERIMENT_NAME]="${EXPERIMENT_NAME:-lm-classifier-experiment}"
    [BASE_EXPERIMENTS_DIR]="${BASE_EXPERIMENTS_DIR:-experiments}"
    [DATA_DIR]="${DATA_DIR:-data}"

    # Model types to set up
    [SETUP_BERT]="${SETUP_BERT:-true}"
    [SETUP_GPT2]="${SETUP_GPT2:-true}"

    # BERT configuration
    [BERT_MODEL_NAME]="${BERT_MODEL_NAME:-bert-base-uncased}"
    [BERT_OUTPUT_NAME]="${BERT_OUTPUT_NAME:-bert-classifier}"

    # GPT-2 configuration
    [GPT2_MODEL_NAME]="${GPT2_MODEL_NAME:-gpt2}"
    [GPT2_OUTPUT_NAME]="${GPT2_OUTPUT_NAME:-gpt2-classifier}"

    # Data configuration
    [CREATE_DATA_STRUCTURE]="${CREATE_DATA_STRUCTURE:-true}"
    [TEXT_COLUMN]="${TEXT_COLUMN:-text}"
    [TARGET_COLUMN]="${TARGET_COLUMN:-cefr_level}"

    # Training configuration defaults
    [NUM_EPOCHS]="${NUM_EPOCHS:-3}"
    [BATCH_SIZE]="${BATCH_SIZE:-16}"
    [LEARNING_RATE]="${LEARNING_RATE:-2e-5}"
    [FREEZE_BASE_MODEL]="${FREEZE_BASE_MODEL:-true}"

    # Output control
    [VERBOSE]="${VERBOSE:-true}"
    [CREATE_CONFIGS]="${CREATE_CONFIGS:-true}"
)

################################################################################
# COLOR OUTPUT
################################################################################

if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    MAGENTA='\033[0;35m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' BLUE='' MAGENTA='' CYAN='' BOLD='' NC=''
fi

################################################################################
# LOGGING FUNCTIONS
################################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_step() {
    echo -e "\n${CYAN}${BOLD}==>${NC} ${BOLD}$*${NC}\n"
}

log_substep() {
    echo -e "  ${MAGENTA}→${NC} $*"
}

################################################################################
# HELP DOCUMENTATION
################################################################################

show_help() {
    cat << 'EOF'
Language Model Text Classifier Experiment Setup Script

USAGE:
    ./e2e-tc-setup.sh [OPTIONS]

DESCRIPTION:
    Sets up folder structure for training BERT and/or GPT-2 text classifiers
    for end-to-end text classification experiments. Creates organized directory
    structure with data folders, model output directories, and config templates.

OPTIONS:
    -h, --help                  Show this help message
    -v, --verbose              Enable verbose output (default: true)
    -q, --quiet                Disable verbose output

    EXPERIMENT CONFIGURATION:
    --experiment-name NAME      Name of the experiment (default: lm-classifier-experiment)
    --base-dir DIR             Base directory for experiments (default: experiments)
    --data-dir DIR             Data directory name (default: data)

    MODEL SELECTION:
    --bert-only                Set up only BERT classifier
    --gpt2-only                Set up only GPT-2 classifier
    --both                     Set up both BERT and GPT-2 (default)

    BERT CONFIGURATION:
    --bert-model NAME          BERT model name (default: bert-base-uncased)
    --bert-output NAME         BERT output directory name (default: bert-classifier)

    GPT-2 CONFIGURATION:
    --gpt2-model NAME          GPT-2 model name (default: gpt2)
    --gpt2-output NAME         GPT-2 output directory name (default: gpt2-classifier)

    DATA CONFIGURATION:
    --text-column NAME         Text column name (default: text)
    --target-column NAME       Target/label column name (default: cefr_level)
    --no-data-structure        Don't create data directory structure

    TRAINING DEFAULTS (for config templates):
    --epochs N                 Number of epochs (default: 3)
    --batch-size N            Batch size (default: 16)
    --learning-rate LR        Learning rate (default: 2e-5)
    --no-freeze               Don't freeze base model (fine-tune entire model)

    OUTPUT CONTROL:
    --no-configs              Don't create config file templates

ENVIRONMENT VARIABLES:
    All options can be set via environment variables. Examples:
        EXPERIMENT_NAME=my-classifier-exp
        SETUP_BERT=true
        SETUP_GPT2=false
        TEXT_COLUMN=essay
        TARGET_COLUMN=level

EXAMPLES:
    # Basic setup with defaults (both BERT and GPT-2)
    ./e2e-tc-setup.sh

    # Setup custom experiment name
    ./e2e-tc-setup.sh --experiment-name cefr-classifiers

    # Setup only BERT classifier
    ./e2e-tc-setup.sh --bert-only --experiment-name bert-only-exp

    # Setup only GPT-2 with custom model
    ./e2e-tc-setup.sh --gpt2-only --gpt2-model gpt2-medium

    # Setup both with custom column names
    ./e2e-tc-setup.sh --text-column essay --target-column cefr_label

    # Advanced: Custom everything
    ./e2e-tc-setup.sh \
        --experiment-name advanced-classifiers \
        --bert-model bert-large-uncased \
        --gpt2-model gpt2-large \
        --epochs 10 \
        --batch-size 32 \
        --no-freeze

OUTPUT STRUCTURE:
    experiments/
    └── <experiment-name>/
        ├── bert-classifier/          # BERT experiment folder
        │   ├── checkpoints/
        │   ├── logs/
        │   ├── predictions/
        │   └── config.yaml           # Config template
        ├── gpt2-classifier/          # GPT-2 experiment folder
        │   ├── checkpoints/
        │   ├── logs/
        │   ├── predictions/
        │   └── config.yaml
        └── data/
            ├── train.csv             # Placeholder for training data
            ├── validation.csv        # Placeholder for validation data
            ├── test.csv              # Placeholder for test data
            └── README.md             # Data format instructions

NEXT STEPS:
    After running this script:

    1. Add your data files to: experiments/<experiment-name>/data/
       - Required: train.csv with columns: <text-column>, <target-column>
       - Optional: validation.csv, test.csv

    2. Train BERT classifier:
       python -m src.train_lm_classifiers train \
           -c experiments/<experiment-name>/bert-classifier/config.yaml

    3. Train GPT-2 classifier:
       python -m src.train_lm_classifiers train \
           -c experiments/<experiment-name>/gpt2-classifier/config.yaml

    4. Make predictions:
       python -m src.train_lm_classifiers predict \
           --checkpoint experiments/<experiment-name>/bert-classifier/checkpoints/best \
           -i experiments/<experiment-name>/data/test.csv \
           --text-column <text-column> \
           -o experiments/<experiment-name>/bert-classifier/predictions/results.json

EXIT STATUS:
    0   Success
    1   Error during execution

EOF
}

################################################################################
# ARGUMENT PARSING
################################################################################

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--verbose)
                Config[VERBOSE]="true"
                shift
                ;;
            -q|--quiet)
                Config[VERBOSE]="false"
                shift
                ;;
            --experiment-name)
                Config[EXPERIMENT_NAME]="$2"
                shift 2
                ;;
            --base-dir)
                Config[BASE_EXPERIMENTS_DIR]="$2"
                shift 2
                ;;
            --data-dir)
                Config[DATA_DIR]="$2"
                shift 2
                ;;
            --bert-only)
                Config[SETUP_BERT]="true"
                Config[SETUP_GPT2]="false"
                shift
                ;;
            --gpt2-only)
                Config[SETUP_BERT]="false"
                Config[SETUP_GPT2]="true"
                shift
                ;;
            --both)
                Config[SETUP_BERT]="true"
                Config[SETUP_GPT2]="true"
                shift
                ;;
            --bert-model)
                Config[BERT_MODEL_NAME]="$2"
                shift 2
                ;;
            --bert-output)
                Config[BERT_OUTPUT_NAME]="$2"
                shift 2
                ;;
            --gpt2-model)
                Config[GPT2_MODEL_NAME]="$2"
                shift 2
                ;;
            --gpt2-output)
                Config[GPT2_OUTPUT_NAME]="$2"
                shift 2
                ;;
            --text-column)
                Config[TEXT_COLUMN]="$2"
                shift 2
                ;;
            --target-column)
                Config[TARGET_COLUMN]="$2"
                shift 2
                ;;
            --no-data-structure)
                Config[CREATE_DATA_STRUCTURE]="false"
                shift
                ;;
            --epochs)
                Config[NUM_EPOCHS]="$2"
                shift 2
                ;;
            --batch-size)
                Config[BATCH_SIZE]="$2"
                shift 2
                ;;
            --learning-rate)
                Config[LEARNING_RATE]="$2"
                shift 2
                ;;
            --no-freeze)
                Config[FREEZE_BASE_MODEL]="false"
                shift
                ;;
            --no-configs)
                Config[CREATE_CONFIGS]="false"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

################################################################################
# DIRECTORY STRUCTURE CREATION
################################################################################

create_experiment_structure() {
    log_step "Creating Experiment Directory Structure"

    local base_dir="${Config[BASE_EXPERIMENTS_DIR]}"
    local exp_name="${Config[EXPERIMENT_NAME]}"
    local exp_dir="$base_dir/$exp_name"

    if [[ -d "$exp_dir" ]]; then
        log_warning "Experiment directory already exists: $exp_dir"
        log_info "Will create missing subdirectories only"
    fi

    log_substep "Creating base: $exp_dir"
    mkdir -p "$exp_dir"

    # Store experiment directory in config for later use
    Config[EXPERIMENT_DIR]="$exp_dir"

    log_success "Experiment directory ready: $exp_dir"
}

create_data_structure() {
    if [[ "${Config[CREATE_DATA_STRUCTURE]}" != "true" ]]; then
        log_info "Skipping data directory creation (--no-data-structure)"
        return 0
    fi

    log_step "Creating Data Directory Structure"

    local exp_dir="${Config[EXPERIMENT_DIR]}"
    local data_dir="$exp_dir/${Config[DATA_DIR]}"

    log_substep "Creating: $data_dir"
    mkdir -p "$data_dir"

    # Create placeholder CSV files with headers
    local text_col="${Config[TEXT_COLUMN]}"
    local target_col="${Config[TARGET_COLUMN]}"

    log_substep "Creating placeholder data files"

    # Create train.csv placeholder
    echo "${text_col},${target_col}" > "$data_dir/train.csv"
    echo "# Add your training data here (text and labels)" >> "$data_dir/train.csv"

    # Create validation.csv placeholder
    echo "${text_col},${target_col}" > "$data_dir/validation.csv"
    echo "# Add your validation data here (optional)" >> "$data_dir/validation.csv"

    # Create test.csv placeholder
    echo "${text_col}" > "$data_dir/test.csv"
    echo "# Add your test data here (text only or with labels)" >> "$data_dir/test.csv"

    # Create data README
    cat > "$data_dir/README.md" << EOF
# Data Directory

This directory should contain your training, validation, and test data.

## Required Files

### train.csv
Training data with text and labels.

**Format:**
\`\`\`csv
${text_col},${target_col}
"The student demonstrates advanced proficiency.",C1
"Simple sentences with basic vocabulary.",A2
"Complex grammatical structures.",B2
\`\`\`

**Columns:**
- \`${text_col}\`: Text to classify
- \`${target_col}\`: CEFR level (A1, A2, B1, B2, C1, C2) or other labels

## Optional Files

### validation.csv
Validation data for early stopping and model selection.

**Format:** Same as train.csv

### test.csv
Test data for final evaluation.

**Format:** Can include labels for evaluation or just text for prediction
\`\`\`csv
${text_col},${target_col}
"Test sample 1",B1
"Test sample 2",C1
\`\`\`

Or without labels:
\`\`\`csv
${text_col}
"Test sample 1"
"Test sample 2"
\`\`\`

## Supported Label Formats

- **CEFR levels:** A1, A2, B1, B2, C1, C2
- **Numeric labels:** 0, 1, 2, 3, 4, 5
- **Any string labels:** The script will automatically encode them

## Example Dataset

For CEFR classification with 1000 training samples:

\`\`\`bash
# Assuming you have your data
cp /path/to/your/training_data.csv $data_dir/train.csv
cp /path/to/your/validation_data.csv $data_dir/validation.csv
cp /path/to/your/test_data.csv $data_dir/test.csv
\`\`\`

## Next Steps

1. Replace placeholder CSV files with your actual data
2. Ensure column names match: \`${text_col}\` and \`${target_col}\`
3. Run the training commands shown in the main experiment README
EOF

    log_success "Data directory structure created: $data_dir"
}

create_model_structure() {
    local model_type="$1"
    local model_name="$2"
    local output_name="$3"

    log_step "Creating $model_type Classifier Structure"

    local exp_dir="${Config[EXPERIMENT_DIR]}"
    local model_dir="$exp_dir/$output_name"

    log_substep "Creating: $model_dir"
    mkdir -p "$model_dir"

    # Create subdirectories
    local subdirs=(
        "checkpoints"
        "logs"
        "predictions"
    )

    for subdir in "${subdirs[@]}"; do
        log_substep "Creating: $model_dir/$subdir"
        mkdir -p "$model_dir/$subdir"
    done

    # Create config file if requested
    if [[ "${Config[CREATE_CONFIGS]}" == "true" ]]; then
        create_config_file "$model_type" "$model_name" "$model_dir"
    fi

    # Create model-specific README
    create_model_readme "$model_type" "$model_name" "$output_name" "$model_dir"

    log_success "$model_type classifier structure created: $model_dir"
}

create_config_file() {
    local model_type="$1"
    local model_name="$2"
    local model_dir="$3"

    local config_file="$model_dir/config.yaml"

    log_substep "Creating config template: $config_file"

    local exp_dir="${Config[EXPERIMENT_DIR]}"
    local data_dir="${Config[DATA_DIR]}"

    cat > "$config_file" << EOF
# ${model_type} Text Classifier Configuration
# Auto-generated by e2e-tc-setup.sh

# Model configuration
model_name: "${model_name}"
num_epochs: ${Config[NUM_EPOCHS]}
batch_size: ${Config[BATCH_SIZE]}
learning_rate: ${Config[LEARNING_RATE]}
freeze_base_model: ${Config[FREEZE_BASE_MODEL]}

# Data configuration
input_file: "${exp_dir}/${data_dir}/train.csv"
validation_file: "${exp_dir}/${data_dir}/validation.csv"
text_column: "${Config[TEXT_COLUMN]}"
target_column: "${Config[TARGET_COLUMN]}"

# Training options
max_length: 128
warmup_steps: 0
weight_decay: 0.01
early_stopping_patience: 3

# Device configuration
device: cpu  # Change to 'cuda' for GPU training
fp16: false  # Set to true for mixed precision on GPU

# Output options
save_steps: 500
eval_steps: 100
logging_steps: 50
save_total_limit: 3

# Class balancing (for imbalanced datasets)
use_class_weights: false

# TensorBoard logging
tensorboard: true

# Random seed for reproducibility
seed: 42
EOF

    log_success "Config template created: $config_file"
}

create_model_readme() {
    local model_type="$1"
    local model_name="$2"
    local output_name="$3"
    local model_dir="$4"

    local readme_file="$model_dir/README.md"
    local exp_name="${Config[EXPERIMENT_NAME]}"
    local data_dir="${Config[DATA_DIR]}"

    log_substep "Creating README: $readme_file"

    cat > "$readme_file" << EOF
# ${model_type} Text Classifier

Experiment: **${exp_name}**
Model: **${model_name}**

## Quick Start

### 1. Prepare Data

Add your data files to \`../${data_dir}/\`:
- \`train.csv\` - Required
- \`validation.csv\` - Optional (recommended)
- \`test.csv\` - Optional

### 2. Train Model

\`\`\`bash
# Using config file (recommended)
python -m src.train_lm_classifiers train -c config.yaml

# Or with command-line arguments
python -m src.train_lm_classifiers train \\
    -i ../${data_dir}/train.csv \\
    --validation-file ../${data_dir}/validation.csv \\
    --text-column ${Config[TEXT_COLUMN]} \\
    --target-column ${Config[TARGET_COLUMN]} \\
    --model-name ${model_name} \\
    --num-epochs ${Config[NUM_EPOCHS]} \\
    --batch-size ${Config[BATCH_SIZE]} \\
    --learning-rate ${Config[LEARNING_RATE]} \\
    --freeze-base-model \\
    -o .
\`\`\`

### 3. Make Predictions

\`\`\`bash
# After training completes, use the best checkpoint
python -m src.train_lm_classifiers predict \\
    --checkpoint checkpoints/best \\
    -i ../${data_dir}/test.csv \\
    --text-column ${Config[TEXT_COLUMN]} \\
    --output-probabilities \\
    -o predictions/test_results.json
\`\`\`

### 4. Evaluate Results

\`\`\`bash
# With ground truth labels for evaluation
python -m src.train_lm_classifiers predict \\
    --checkpoint checkpoints/best \\
    -i ../${data_dir}/test.csv \\
    --text-column ${Config[TEXT_COLUMN]} \\
    --target-column ${Config[TARGET_COLUMN]} \\
    -o predictions/test_eval.json

# Check evaluation metrics
cat predictions/evaluation_metrics.json
\`\`\`

## Directory Structure

\`\`\`
${output_name}/
├── checkpoints/           # Model checkpoints saved during training
│   ├── checkpoint-100/
│   ├── checkpoint-200/
│   └── best/             # ← Use this for predictions
├── logs/                 # Training logs
│   ├── training_log.txt
│   └── tensorboard/      # TensorBoard logs (if enabled)
├── predictions/          # Prediction outputs
│   ├── test_results.json
│   └── evaluation_metrics.json
├── config.yaml           # Training configuration
└── README.md            # This file
\`\`\`

## Configuration

Edit \`config.yaml\` to customize:

- **Model:** Change \`model_name\` to try different models:
  - BERT: \`bert-base-uncased\`, \`bert-large-uncased\`
  - GPT-2: \`gpt2\`, \`gpt2-medium\`, \`gpt2-large\`
  - RoBERTa: \`roberta-base\`, \`roberta-large\`
  - DistilBERT: \`distilbert-base-uncased\`

- **Training:** Adjust \`num_epochs\`, \`batch_size\`, \`learning_rate\`

- **GPU:** Set \`device: cuda\` and \`fp16: true\` for faster training

- **Early Stopping:** Requires \`validation_file\` in config

## Common Tasks

### Train on GPU

\`\`\`bash
# Edit config.yaml:
device: cuda
fp16: true
batch_size: 32

# Then train
python -m src.train_lm_classifiers train -c config.yaml
\`\`\`

### Handle Imbalanced Data

\`\`\`bash
# Edit config.yaml:
use_class_weights: true

# Then train
python -m src.train_lm_classifiers train -c config.yaml
\`\`\`

### Monitor Training with TensorBoard

\`\`\`bash
# TensorBoard is enabled by default in config.yaml
# Start TensorBoard server:
tensorboard --logdir logs/tensorboard

# Open in browser: http://localhost:6006
\`\`\`

## Troubleshooting

### Out of Memory

Reduce batch size in \`config.yaml\`:
\`\`\`yaml
batch_size: 8
\`\`\`

Or freeze the base model:
\`\`\`yaml
freeze_base_model: true
\`\`\`

### Slow Training

Use GPU with mixed precision:
\`\`\`yaml
device: cuda
fp16: true
\`\`\`

Or use a smaller model:
\`\`\`yaml
model_name: distilbert-base-uncased
\`\`\`

### Poor Accuracy

Add validation set for early stopping:
\`\`\`yaml
validation_file: ../${data_dir}/validation.csv
early_stopping_patience: 3
\`\`\`

Increase training epochs:
\`\`\`yaml
num_epochs: 10
\`\`\`

Use class weights for imbalanced data:
\`\`\`yaml
use_class_weights: true
\`\`\`

## Next Steps

1. **Compare Models:** Train both BERT and GPT-2, compare results
2. **Hyperparameter Tuning:** Experiment with different learning rates
3. **Ensemble:** Combine predictions from multiple models
4. **Production:** Export best checkpoint for deployment

## Documentation

- [Language Model Classifier Quick Start](../../../docs/LM_CLASSIFIER_QUICK_START.md)
- [Complete Guide](../../../docs/LANGUAGE_MODEL_CLASSIFIER_GUIDE.md)
EOF

    log_success "README created: $readme_file"
}

create_main_readme() {
    log_step "Creating Main Experiment README"

    local exp_dir="${Config[EXPERIMENT_DIR]}"
    local readme_file="$exp_dir/README.md"
    local exp_name="${Config[EXPERIMENT_NAME]}"

    log_substep "Creating: $readme_file"

    local models_list=""
    if [[ "${Config[SETUP_BERT]}" == "true" ]]; then
        models_list="${models_list}\n- **BERT:** \`${Config[BERT_OUTPUT_NAME]}/\` - ${Config[BERT_MODEL_NAME]}"
    fi
    if [[ "${Config[SETUP_GPT2]}" == "true" ]]; then
        models_list="${models_list}\n- **GPT-2:** \`${Config[GPT2_OUTPUT_NAME]}/\` - ${Config[GPT2_MODEL_NAME]}"
    fi

    cat > "$readme_file" << EOF
# ${exp_name}

Language Model Text Classification Experiment

## Experiment Structure

\`\`\`
${exp_name}/
├── data/                     # Training, validation, and test data
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
EOF

    if [[ "${Config[SETUP_BERT]}" == "true" ]]; then
        cat >> "$readme_file" << EOF
├── ${Config[BERT_OUTPUT_NAME]}/         # BERT classifier
│   ├── checkpoints/
│   ├── logs/
│   ├── predictions/
│   ├── config.yaml
│   └── README.md
EOF
    fi

    if [[ "${Config[SETUP_GPT2]}" == "true" ]]; then
        cat >> "$readme_file" << EOF
├── ${Config[GPT2_OUTPUT_NAME]}/         # GPT-2 classifier
│   ├── checkpoints/
│   ├── logs/
│   ├── predictions/
│   ├── config.yaml
│   └── README.md
EOF
    fi

    cat >> "$readme_file" << EOF
└── README.md                # This file
\`\`\`

## Models

This experiment includes:
${models_list}

## Quick Start

### 1. Add Your Data

Place your CSV files in \`data/\`:

\`\`\`bash
# Copy your training data
cp /path/to/your/train.csv data/train.csv

# Copy validation data (optional but recommended)
cp /path/to/your/val.csv data/validation.csv

# Copy test data
cp /path/to/your/test.csv data/test.csv
\`\`\`

**Data format:**
- Columns: \`${Config[TEXT_COLUMN]}\`, \`${Config[TARGET_COLUMN]}\`
- Labels: A1, A2, B1, B2, C1, C2 (or any classification labels)

### 2. Train Models

EOF

    if [[ "${Config[SETUP_BERT]}" == "true" ]]; then
        cat >> "$readme_file" << EOF
**Train BERT:**

\`\`\`bash
cd ${Config[BERT_OUTPUT_NAME]}
python -m src.train_lm_classifiers train -c config.yaml
cd ..
\`\`\`

EOF
    fi

    if [[ "${Config[SETUP_GPT2]}" == "true" ]]; then
        cat >> "$readme_file" << EOF
**Train GPT-2:**

\`\`\`bash
cd ${Config[GPT2_OUTPUT_NAME]}
python -m src.train_lm_classifiers train -c config.yaml
cd ..
\`\`\`

EOF
    fi

    cat >> "$readme_file" << EOF
### 3. Compare Results

After training both models, compare their predictions:

\`\`\`bash
# View BERT results
cat ${Config[BERT_OUTPUT_NAME]}/predictions/evaluation_metrics.json

# View GPT-2 results
cat ${Config[GPT2_OUTPUT_NAME]}/predictions/evaluation_metrics.json
\`\`\`

## Training Tips

### Use GPU for Faster Training

Edit the \`config.yaml\` files:

\`\`\`yaml
device: cuda
fp16: true
batch_size: 32
\`\`\`

### Enable Early Stopping

Make sure you have a validation file:

\`\`\`yaml
validation_file: ../data/validation.csv
early_stopping_patience: 3
\`\`\`

### Handle Class Imbalance

\`\`\`yaml
use_class_weights: true
\`\`\`

## Monitor Training

### TensorBoard

\`\`\`bash
# For BERT
tensorboard --logdir ${Config[BERT_OUTPUT_NAME]}/logs/tensorboard

# For GPT-2
tensorboard --logdir ${Config[GPT2_OUTPUT_NAME]}/logs/tensorboard
\`\`\`

### Training Logs

\`\`\`bash
# View BERT training progress
tail -f ${Config[BERT_OUTPUT_NAME]}/logs/training_log.txt

# View GPT-2 training progress
tail -f ${Config[GPT2_OUTPUT_NAME]}/logs/training_log.txt
\`\`\`

## Model Comparison

| Model | Pros | Cons |
|-------|------|------|
| BERT | Better for understanding context | Slower training |
| GPT-2 | Faster, good for generation tasks | May need more data |

## Next Steps

1. **Data Preparation:** Ensure your CSV files are properly formatted
2. **Train Models:** Run both BERT and GPT-2 training
3. **Evaluate:** Compare performance on test set
4. **Tune:** Adjust hyperparameters in config.yaml files
5. **Deploy:** Use best checkpoint for production

## Documentation

- See individual model READMEs in each model directory
- Full documentation: [docs/LANGUAGE_MODEL_CLASSIFIER_GUIDE.md](../../docs/LANGUAGE_MODEL_CLASSIFIER_GUIDE.md)
- Quick reference: [docs/LM_CLASSIFIER_QUICK_START.md](../../docs/LM_CLASSIFIER_QUICK_START.md)

## Experiment Details

- **Created:** $(date)
- **Text Column:** ${Config[TEXT_COLUMN]}
- **Target Column:** ${Config[TARGET_COLUMN]}
- **Training Epochs:** ${Config[NUM_EPOCHS]}
- **Batch Size:** ${Config[BATCH_SIZE]}
- **Learning Rate:** ${Config[LEARNING_RATE]}
- **Freeze Base Model:** ${Config[FREEZE_BASE_MODEL]}
EOF

    log_success "Main README created: $readme_file"
}

################################################################################
# DISPLAY SUMMARY
################################################################################

display_summary() {
    local exp_dir="${Config[EXPERIMENT_DIR]}"

    echo ""
    log_step "Setup Complete!"
    echo ""

    echo -e "${BOLD}Experiment Directory:${NC} $exp_dir"
    echo ""

    echo -e "${BOLD}Models Created:${NC}"
    if [[ "${Config[SETUP_BERT]}" == "true" ]]; then
        echo -e "  ${GREEN}✓${NC} BERT (${Config[BERT_MODEL_NAME]}) → ${Config[BERT_OUTPUT_NAME]}/"
    fi
    if [[ "${Config[SETUP_GPT2]}" == "true" ]]; then
        echo -e "  ${GREEN}✓${NC} GPT-2 (${Config[GPT2_MODEL_NAME]}) → ${Config[GPT2_OUTPUT_NAME]}/"
    fi
    echo ""

    echo -e "${BOLD}Data Directory:${NC}"
    if [[ "${Config[CREATE_DATA_STRUCTURE]}" == "true" ]]; then
        echo -e "  ${GREEN}✓${NC} ${Config[DATA_DIR]}/"
        echo -e "    - train.csv (placeholder created)"
        echo -e "    - validation.csv (placeholder created)"
        echo -e "    - test.csv (placeholder created)"
    else
        echo -e "  ${YELLOW}⊘${NC} Not created (--no-data-structure)"
    fi
    echo ""

    echo -e "${BOLD}Next Steps:${NC}"
    echo ""
    echo -e "  ${CYAN}1.${NC} Add your data:"
    echo -e "     cd $exp_dir/${Config[DATA_DIR]}"
    echo -e "     # Replace placeholder CSVs with your actual data"
    echo ""

    if [[ "${Config[SETUP_BERT]}" == "true" ]]; then
        echo -e "  ${CYAN}2.${NC} Train BERT classifier:"
        echo -e "     cd $exp_dir/${Config[BERT_OUTPUT_NAME]}"
        echo -e "     python -m src.train_lm_classifiers train -c config.yaml"
        echo ""
    fi

    if [[ "${Config[SETUP_GPT2]}" == "true" ]]; then
        local step_num="2"
        if [[ "${Config[SETUP_BERT]}" == "true" ]]; then
            step_num="3"
        fi
        echo -e "  ${CYAN}${step_num}.${NC} Train GPT-2 classifier:"
        echo -e "     cd $exp_dir/${Config[GPT2_OUTPUT_NAME]}"
        echo -e "     python -m src.train_lm_classifiers train -c config.yaml"
        echo ""
    fi

    echo -e "${BOLD}Documentation:${NC}"
    echo -e "  - Main README: $exp_dir/README.md"
    echo -e "  - Data guide: $exp_dir/${Config[DATA_DIR]}/README.md"
    if [[ "${Config[SETUP_BERT]}" == "true" ]]; then
        echo -e "  - BERT guide: $exp_dir/${Config[BERT_OUTPUT_NAME]}/README.md"
    fi
    if [[ "${Config[SETUP_GPT2]}" == "true" ]]; then
        echo -e "  - GPT-2 guide: $exp_dir/${Config[GPT2_OUTPUT_NAME]}/README.md"
    fi
    echo ""
}

################################################################################
# MAIN EXECUTION
################################################################################

main() {
    echo -e "${BOLD}${CYAN}"
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║  Language Model Text Classifier Experiment Setup                  ║"
    echo "║  BERT & GPT-2 Training Structure Generator                        ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    # Parse arguments
    parse_arguments "$@"

    # Create structures
    create_experiment_structure

    if [[ "${Config[CREATE_DATA_STRUCTURE]}" == "true" ]]; then
        create_data_structure
    fi

    if [[ "${Config[SETUP_BERT]}" == "true" ]]; then
        create_model_structure \
            "BERT" \
            "${Config[BERT_MODEL_NAME]}" \
            "${Config[BERT_OUTPUT_NAME]}"
    fi

    if [[ "${Config[SETUP_GPT2]}" == "true" ]]; then
        create_model_structure \
            "GPT-2" \
            "${Config[GPT2_MODEL_NAME]}" \
            "${Config[GPT2_OUTPUT_NAME]}"
    fi

    create_main_readme

    # Display summary
    display_summary

    log_success "All done! Your experiment structure is ready."
    echo ""
}

# Execute main
main "$@"
