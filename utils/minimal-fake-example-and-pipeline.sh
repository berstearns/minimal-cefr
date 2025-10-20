#!/usr/bin/env bash

################################################################################
# Minimal Fake Example and Pipeline Test Script
#
# Purpose: Test that the current CEFR classification pipeline is working with
#          minimal synthetic data and comprehensive configuration validation.
#
# Features:
#   - Hyper-granular argument parsing with extensive validation
#   - Centralized Config data structure (dict-like)
#   - Automatic timestamped experiment folders to prevent overwrites
#   - Full pipeline test from data creation to prediction
################################################################################

set -euo pipefail

################################################################################
# GLOBAL CONFIGURATION STRUCTURE (Dict-like)
################################################################################

declare -A Config=(
    # Python interpreter
    [PYTHON_BIN]="${PYTHON_BIN:-$HOME/.pyenv/versions/3.10.18/bin/python3}"

    # Default paths
    [BASE_DIR]="${BASE_DIR:-$(pwd)}"
    [TIMESTAMP]="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

    # Experiment configuration
    [EXPERIMENT_NAME]="${EXPERIMENT_NAME:-test-pipeline}"
    [EXPERIMENT_BASE_DIR]="${EXPERIMENT_BASE_DIR:-./test-experiments}"
    [USE_TIMESTAMP]="${USE_TIMESTAMP:-true}"

    # Data generation parameters
    [NUM_TRAIN_SAMPLES]="${NUM_TRAIN_SAMPLES:-50}"
    [NUM_TEST_SAMPLES]="${NUM_TEST_SAMPLES:-20}"
    [CEFR_LEVELS]="${CEFR_LEVELS:-A1,A2,B1,B2,C1}"

    # TF-IDF configuration
    [TFIDF_MAX_FEATURES]="${TFIDF_MAX_FEATURES:-100}"
    [TFIDF_NGRAM_MIN]="${TFIDF_NGRAM_MIN:-1}"
    [TFIDF_NGRAM_MAX]="${TFIDF_NGRAM_MAX:-2}"
    [TFIDF_MIN_DF]="${TFIDF_MIN_DF:-1}"
    [TFIDF_MAX_DF]="${TFIDF_MAX_DF:-0.95}"

    # Classifier configuration
    [CLASSIFIER_TYPE]="${CLASSIFIER_TYPE:-logistic}"

    # Pipeline steps control
    [SKIP_DATA_GENERATION]="${SKIP_DATA_GENERATION:-false}"
    [SKIP_TFIDF_TRAINING]="${SKIP_TFIDF_TRAINING:-false}"
    [SKIP_FEATURE_EXTRACTION]="${SKIP_FEATURE_EXTRACTION:-false}"
    [SKIP_CLASSIFIER_TRAINING]="${SKIP_CLASSIFIER_TRAINING:-false}"
    [SKIP_PREDICTION]="${SKIP_PREDICTION:-false}"

    # Output control
    [VERBOSE]="${VERBOSE:-true}"
    [CLEANUP_ON_SUCCESS]="${CLEANUP_ON_SUCCESS:-false}"
    [KEEP_TEMP_FILES]="${KEEP_TEMP_FILES:-true}"

    # Validation flags
    [STRICT_MODE]="${STRICT_MODE:-true}"
    [CHECK_PYTHON_VERSION]="${CHECK_PYTHON_VERSION:-true}"
    [CHECK_DEPENDENCIES]="${CHECK_DEPENDENCIES:-true}"
)

################################################################################
# COLOR OUTPUT UTILITIES
################################################################################

if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    MAGENTA='\033[0;35m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    MAGENTA=''
    CYAN=''
    BOLD=''
    NC=''
fi

################################################################################
# LOGGING FUNCTIONS
################################################################################

# Write status to JSONL file
write_status() {
    local step="$1"
    local status="$2"  # success, error, warning, info, started
    local message="${3:-}"
    local timestamp=$(date -Iseconds 2>/dev/null || date +%Y-%m-%dT%H:%M:%S%z)

    # Only write if status file is configured
    if [[ -n "${Config[STATUS_FILE]:-}" ]]; then
        # Escape special characters in message for JSON
        local escaped_message=$(echo "$message" | sed 's/"/\\"/g' | sed "s/'/\\'/g")

        local json_line="{\"timestamp\":\"$timestamp\",\"step\":\"$step\",\"status\":\"$status\",\"message\":\"$escaped_message\"}"

        echo "$json_line" >> "${Config[STATUS_FILE]}"
    fi
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    local message="$*"
    echo -e "${GREEN}[SUCCESS]${NC} $message"
}

log_warning() {
    local message="$*"
    echo -e "${YELLOW}[WARNING]${NC} $message"
}

log_error() {
    local message="$*"
    echo -e "${RED}[ERROR]${NC} $message" >&2
}

log_step() {
    echo -e "\n${CYAN}${BOLD}==>${NC} ${BOLD}$*${NC}\n"
}

log_substep() {
    echo -e "  ${MAGENTA}→${NC} $*"
}

log_config() {
    if [[ "${Config[VERBOSE]}" == "true" ]]; then
        echo -e "${CYAN}[CONFIG]${NC} $1=${BOLD}$2${NC}"
    fi
}

################################################################################
# VALIDATION FUNCTIONS
################################################################################

# Validate that a value is a positive integer
validate_positive_int() {
    local value="$1"
    local name="$2"

    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        log_error "$name must be a positive integer, got: $value"
        return 1
    fi

    if [[ "$value" -le 0 ]]; then
        log_error "$name must be greater than 0, got: $value"
        return 1
    fi

    return 0
}

# Validate that a value is a number between 0 and 1
validate_float_0_1() {
    local value="$1"
    local name="$2"

    if ! [[ "$value" =~ ^[0-9]*\.?[0-9]+$ ]]; then
        log_error "$name must be a number, got: $value"
        return 1
    fi

    if (( $(echo "$value < 0" | bc -l) )) || (( $(echo "$value > 1" | bc -l) )); then
        log_error "$name must be between 0 and 1, got: $value"
        return 1
    fi

    return 0
}

# Validate boolean value
validate_boolean() {
    local value="$1"
    local name="$2"

    if [[ "$value" != "true" && "$value" != "false" ]]; then
        log_error "$name must be 'true' or 'false', got: $value"
        return 1
    fi

    return 0
}

# Validate classifier type
validate_classifier_type() {
    local value="$1"
    local valid_types=("multinomialnb" "logistic" "randomforest" "svm" "xgboost")

    for type in "${valid_types[@]}"; do
        if [[ "$value" == "$type" ]]; then
            return 0
        fi
    done

    log_error "Invalid classifier type: $value. Must be one of: ${valid_types[*]}"
    return 1
}

# Validate CEFR levels format
validate_cefr_levels() {
    local levels="$1"

    if [[ ! "$levels" =~ ^[A-C][1-2](,[A-C][1-2])*$ ]]; then
        log_error "Invalid CEFR levels format: $levels. Expected format: A1,A2,B1,B2,C1,C2"
        return 1
    fi

    return 0
}

# Check if Python binary exists and is executable
check_python_binary() {
    local python_bin="${Config[PYTHON_BIN]}"

    if [[ ! -f "$python_bin" ]]; then
        log_error "Python binary not found: $python_bin"
        return 1
    fi

    if [[ ! -x "$python_bin" ]]; then
        log_error "Python binary is not executable: $python_bin"
        return 1
    fi

    return 0
}

# Validate Python version
validate_python_version() {
    local python_bin="${Config[PYTHON_BIN]}"
    local version

    version=$("$python_bin" --version 2>&1 | awk '{print $2}')
    local major minor patch
    IFS='.' read -r major minor patch <<< "$version"

    if [[ "$major" -lt 3 ]] || [[ "$major" -eq 3 && "$minor" -lt 7 ]]; then
        log_error "Python version must be 3.7 or higher, found: $version"
        return 1
    fi

    log_info "Python version: $version ✓"
    return 0
}

# Check Python dependencies
check_python_dependencies() {
    local python_bin="${Config[PYTHON_BIN]}"
    local required_packages=("pandas" "numpy" "sklearn" "yaml")
    local missing=()

    log_info "Checking Python dependencies..."

    for package in "${required_packages[@]}"; do
        if ! "$python_bin" -c "import $package" 2>/dev/null; then
            missing+=("$package")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing required Python packages: ${missing[*]}"
        log_error "Install with: ${python_bin} -m pip install ${missing[*]}"
        return 1
    fi

    log_success "All required Python packages are installed ✓"
    return 0
}

# Validate directory is writable
validate_writable_directory() {
    local dir="$1"
    local name="$2"

    if [[ -e "$dir" && ! -d "$dir" ]]; then
        log_error "$name exists but is not a directory: $dir"
        return 1
    fi

    if [[ -d "$dir" && ! -w "$dir" ]]; then
        log_error "$name is not writable: $dir"
        return 1
    fi

    return 0
}

################################################################################
# CONFIGURATION VALIDATION
################################################################################

validate_config() {
    log_step "Validating Configuration"

    local errors=0

    # Validate Python binary
    if ! check_python_binary; then
        ((errors++))
    fi

    # Validate Python version
    if [[ "${Config[CHECK_PYTHON_VERSION]}" == "true" ]]; then
        if ! validate_python_version; then
            ((errors++))
        fi
    fi

    # Check dependencies
    if [[ "${Config[CHECK_DEPENDENCIES]}" == "true" ]]; then
        if ! check_python_dependencies; then
            ((errors++))
        fi
    fi

    # Validate numeric parameters
    if ! validate_positive_int "${Config[NUM_TRAIN_SAMPLES]}" "NUM_TRAIN_SAMPLES"; then
        ((errors++))
    fi

    if ! validate_positive_int "${Config[NUM_TEST_SAMPLES]}" "NUM_TEST_SAMPLES"; then
        ((errors++))
    fi

    if ! validate_positive_int "${Config[TFIDF_MAX_FEATURES]}" "TFIDF_MAX_FEATURES"; then
        ((errors++))
    fi

    if ! validate_positive_int "${Config[TFIDF_NGRAM_MIN]}" "TFIDF_NGRAM_MIN"; then
        ((errors++))
    fi

    if ! validate_positive_int "${Config[TFIDF_NGRAM_MAX]}" "TFIDF_NGRAM_MAX"; then
        ((errors++))
    fi

    if ! validate_positive_int "${Config[TFIDF_MIN_DF]}" "TFIDF_MIN_DF"; then
        ((errors++))
    fi

    if ! validate_float_0_1 "${Config[TFIDF_MAX_DF]}" "TFIDF_MAX_DF"; then
        ((errors++))
    fi

    # Validate ngram range
    if [[ "${Config[TFIDF_NGRAM_MIN]}" -gt "${Config[TFIDF_NGRAM_MAX]}" ]]; then
        log_error "TFIDF_NGRAM_MIN must be <= TFIDF_NGRAM_MAX"
        ((errors++))
    fi

    # Validate boolean parameters
    for key in USE_TIMESTAMP SKIP_DATA_GENERATION SKIP_TFIDF_TRAINING \
               SKIP_FEATURE_EXTRACTION SKIP_CLASSIFIER_TRAINING SKIP_PREDICTION \
               VERBOSE CLEANUP_ON_SUCCESS KEEP_TEMP_FILES STRICT_MODE \
               CHECK_PYTHON_VERSION CHECK_DEPENDENCIES; do
        if ! validate_boolean "${Config[$key]}" "$key"; then
            ((errors++))
        fi
    done

    # Validate classifier type
    if ! validate_classifier_type "${Config[CLASSIFIER_TYPE]}"; then
        ((errors++))
    fi

    # Validate CEFR levels
    if ! validate_cefr_levels "${Config[CEFR_LEVELS]}"; then
        ((errors++))
    fi

    # Validate directories
    if ! validate_writable_directory "${Config[EXPERIMENT_BASE_DIR]}" "EXPERIMENT_BASE_DIR"; then
        ((errors++))
    fi

    if [[ $errors -gt 0 ]]; then
        log_error "Configuration validation failed with $errors error(s)"
        return 1
    fi

    log_success "Configuration validation passed ✓"
    return 0
}

################################################################################
# HELP DOCUMENTATION
################################################################################

show_help() {
    cat << 'EOF'
Minimal Fake Example and Pipeline Test Script

USAGE:
    ./minimal-fake-example-and-pipeline.sh [OPTIONS]

DESCRIPTION:
    Creates minimal synthetic CEFR classification data and runs the full
    pipeline to verify everything is working correctly. Supports extensive
    configuration via environment variables or command-line arguments.

OPTIONS:
    -h, --help                  Show this help message
    -v, --verbose              Enable verbose output (default: true)
    -q, --quiet                Disable verbose output

    EXPERIMENT CONFIGURATION:
    --experiment-name NAME      Name of the experiment (default: test-pipeline)
    --experiment-dir DIR        Base directory for experiments (default: ./test-experiments)
    --no-timestamp             Don't add timestamp to experiment folder
    --timestamp                Add timestamp to experiment folder (default)

    DATA GENERATION:
    --num-train N              Number of training samples (default: 50)
    --num-test N               Number of test samples (default: 20)
    --cefr-levels LEVELS       Comma-separated CEFR levels (default: A1,A2,B1,B2,C1)

    TF-IDF CONFIGURATION:
    --max-features N           TF-IDF max features (default: 100)
    --ngram-min N              N-gram minimum (default: 1)
    --ngram-max N              N-gram maximum (default: 2)
    --min-df N                 TF-IDF min document frequency (default: 1)
    --max-df FLOAT             TF-IDF max document frequency (default: 0.95)

    CLASSIFIER CONFIGURATION:
    --classifier TYPE          Classifier type: multinomialnb, logistic, randomforest,
                              svm, xgboost (default: logistic)

    PIPELINE CONTROL:
    --skip-data-gen           Skip data generation step
    --skip-tfidf              Skip TF-IDF training step
    --skip-features           Skip feature extraction step
    --skip-classifier         Skip classifier training step
    --skip-prediction         Skip prediction step

    PYTHON CONFIGURATION:
    --python PATH             Path to Python binary
                             (default: $HOME/.pyenv/versions/3.10.18/bin/python3)

    OUTPUT CONTROL:
    --cleanup                 Cleanup temporary files on success
    --no-keep-temp           Don't keep temporary files
    --no-strict              Disable strict mode
    --skip-python-check      Skip Python version check
    --skip-dep-check         Skip dependency check

ENVIRONMENT VARIABLES:
    All options can be set via environment variables by converting the
    option name to uppercase and replacing dashes with underscores.

    Examples:
        EXPERIMENT_NAME=my-test
        NUM_TRAIN_SAMPLES=100
        TFIDF_MAX_FEATURES=500
        CLASSIFIER_TYPE=xgboost
        PYTHON_BIN=/usr/bin/python3

EXAMPLES:
    # Basic usage with defaults
    ./minimal-fake-example-and-pipeline.sh

    # Custom experiment with more samples
    ./minimal-fake-example-and-pipeline.sh --experiment-name large-test \
        --num-train 200 --num-test 50

    # Use XGBoost classifier with custom TF-IDF
    ./minimal-fake-example-and-pipeline.sh --classifier xgboost \
        --max-features 500 --ngram-max 3

    # Quick test without timestamp and cleanup after
    ./minimal-fake-example-and-pipeline.sh --no-timestamp --cleanup

    # Skip certain steps (useful for debugging)
    ./minimal-fake-example-and-pipeline.sh --skip-data-gen --skip-tfidf

EXIT STATUS:
    0   Success
    1   Configuration error
    2   Validation error
    3   Pipeline execution error

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
            --experiment-dir)
                Config[EXPERIMENT_BASE_DIR]="$2"
                shift 2
                ;;
            --no-timestamp)
                Config[USE_TIMESTAMP]="false"
                shift
                ;;
            --timestamp)
                Config[USE_TIMESTAMP]="true"
                shift
                ;;
            --num-train)
                Config[NUM_TRAIN_SAMPLES]="$2"
                shift 2
                ;;
            --num-test)
                Config[NUM_TEST_SAMPLES]="$2"
                shift 2
                ;;
            --cefr-levels)
                Config[CEFR_LEVELS]="$2"
                shift 2
                ;;
            --max-features)
                Config[TFIDF_MAX_FEATURES]="$2"
                shift 2
                ;;
            --ngram-min)
                Config[TFIDF_NGRAM_MIN]="$2"
                shift 2
                ;;
            --ngram-max)
                Config[TFIDF_NGRAM_MAX]="$2"
                shift 2
                ;;
            --min-df)
                Config[TFIDF_MIN_DF]="$2"
                shift 2
                ;;
            --max-df)
                Config[TFIDF_MAX_DF]="$2"
                shift 2
                ;;
            --classifier)
                Config[CLASSIFIER_TYPE]="$2"
                shift 2
                ;;
            --skip-data-gen)
                Config[SKIP_DATA_GENERATION]="true"
                shift
                ;;
            --skip-tfidf)
                Config[SKIP_TFIDF_TRAINING]="true"
                shift
                ;;
            --skip-features)
                Config[SKIP_FEATURE_EXTRACTION]="true"
                shift
                ;;
            --skip-classifier)
                Config[SKIP_CLASSIFIER_TRAINING]="true"
                shift
                ;;
            --skip-prediction)
                Config[SKIP_PREDICTION]="true"
                shift
                ;;
            --python)
                Config[PYTHON_BIN]="$2"
                shift 2
                ;;
            --cleanup)
                Config[CLEANUP_ON_SUCCESS]="true"
                shift
                ;;
            --no-keep-temp)
                Config[KEEP_TEMP_FILES]="false"
                shift
                ;;
            --no-strict)
                Config[STRICT_MODE]="false"
                shift
                ;;
            --skip-python-check)
                Config[CHECK_PYTHON_VERSION]="false"
                shift
                ;;
            --skip-dep-check)
                Config[CHECK_DEPENDENCIES]="false"
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
# CONFIGURATION DISPLAY
################################################################################

display_config() {
    if [[ "${Config[VERBOSE]}" != "true" ]]; then
        return
    fi

    log_step "Configuration Summary"

    echo -e "${BOLD}Python:${NC}"
    log_config "  PYTHON_BIN" "${Config[PYTHON_BIN]}"

    echo -e "\n${BOLD}Experiment:${NC}"
    log_config "  EXPERIMENT_NAME" "${Config[EXPERIMENT_NAME]}"
    log_config "  EXPERIMENT_BASE_DIR" "${Config[EXPERIMENT_BASE_DIR]}"
    log_config "  USE_TIMESTAMP" "${Config[USE_TIMESTAMP]}"

    echo -e "\n${BOLD}Data Generation:${NC}"
    log_config "  NUM_TRAIN_SAMPLES" "${Config[NUM_TRAIN_SAMPLES]}"
    log_config "  NUM_TEST_SAMPLES" "${Config[NUM_TEST_SAMPLES]}"
    log_config "  CEFR_LEVELS" "${Config[CEFR_LEVELS]}"

    echo -e "\n${BOLD}TF-IDF Configuration:${NC}"
    log_config "  TFIDF_MAX_FEATURES" "${Config[TFIDF_MAX_FEATURES]}"
    log_config "  TFIDF_NGRAM_RANGE" "${Config[TFIDF_NGRAM_MIN]},${Config[TFIDF_NGRAM_MAX]}"
    log_config "  TFIDF_MIN_DF" "${Config[TFIDF_MIN_DF]}"
    log_config "  TFIDF_MAX_DF" "${Config[TFIDF_MAX_DF]}"

    echo -e "\n${BOLD}Classifier:${NC}"
    log_config "  CLASSIFIER_TYPE" "${Config[CLASSIFIER_TYPE]}"

    echo ""
}

################################################################################
# EXPERIMENT DIRECTORY SETUP
################################################################################

setup_experiment_directory() {
    log_step "Setting Up Experiment Directory"
    write_status "setup_experiment_directory" "started" "Setting up experiment directory"

    # Construct experiment directory name
    if [[ "${Config[USE_TIMESTAMP]}" == "true" ]]; then
        Config[EXPERIMENT_DIR]="${Config[EXPERIMENT_BASE_DIR]}/${Config[EXPERIMENT_NAME]}-${Config[TIMESTAMP]}"
    else
        Config[EXPERIMENT_DIR]="${Config[EXPERIMENT_BASE_DIR]}/${Config[EXPERIMENT_NAME]}"
    fi

    local exp_dir="${Config[EXPERIMENT_DIR]}"

    # Check if directory already exists
    if [[ -d "$exp_dir" ]]; then
        if [[ "${Config[STRICT_MODE]}" == "true" ]]; then
            log_error "Experiment directory already exists: $exp_dir"
            log_error "Use --no-timestamp and different --experiment-name, or remove the directory"
            write_status "setup_experiment_directory" "error" "Directory already exists: $exp_dir"
            return 1
        else
            log_warning "Experiment directory already exists: $exp_dir"
            log_warning "Continuing in non-strict mode..."
            write_status "setup_experiment_directory" "warning" "Directory exists, continuing in non-strict mode"
        fi
    fi

    # Create directory structure
    log_substep "Creating: $exp_dir"
    mkdir -p "$exp_dir"

    # Set status file path now that experiment dir exists
    Config[STATUS_FILE]="$exp_dir/__status.jsonl"

    # Create subdirectories matching expected structure
    local subdirs=(
        "ml-training-data"
        "ml-test-data"
        "features-training-data"
        "features"
        "feature-models"
        "results"
    )

    for subdir in "${subdirs[@]}"; do
        log_substep "Creating: $exp_dir/$subdir"
        mkdir -p "$exp_dir/$subdir"
    done

    Config[ML_TRAIN_DIR]="$exp_dir/ml-training-data"
    Config[ML_TEST_DIR]="$exp_dir/ml-test-data"
    Config[FEATURES_TRAIN_DIR]="$exp_dir/features-training-data"
    Config[FEATURES_OUTPUT_DIR]="$exp_dir/features"
    Config[MODELS_DIR]="$exp_dir/feature-models"
    Config[RESULTS_DIR]="$exp_dir/results"

    log_success "Experiment directory created: $exp_dir"
    write_status "setup_experiment_directory" "success" "Experiment directory created successfully"
    return 0
}

################################################################################
# GENERATE SYNTHETIC DATA
################################################################################

generate_synthetic_data() {
    if [[ "${Config[SKIP_DATA_GENERATION]}" == "true" ]]; then
        log_warning "Skipping data generation (--skip-data-gen)"
        write_status "generate_synthetic_data" "skipped" "Data generation skipped by user"
        return 0
    fi

    log_step "Generating Synthetic CEFR Data"
    write_status "generate_synthetic_data" "started" "Starting synthetic data generation"

    local python_bin="${Config[PYTHON_BIN]}"
    local ml_train_dir="${Config[ML_TRAIN_DIR]}"
    local ml_test_dir="${Config[ML_TEST_DIR]}"
    local num_train="${Config[NUM_TRAIN_SAMPLES]}"
    local num_test="${Config[NUM_TEST_SAMPLES]}"
    local cefr_levels="${Config[CEFR_LEVELS]}"

    # Create synthetic data generation script in experiment root (not in data dir)
    local gen_script="${Config[EXPERIMENT_DIR]}/generate_data.py"

    log_substep "Creating data generation script: $gen_script"

    cat > "$gen_script" << 'PYTHON_EOF'
import pandas as pd
import sys
import random
from pathlib import Path

def generate_fake_text(cefr_level, length=50):
    """Generate fake text with level-specific vocabulary."""
    vocabularies = {
        'A1': ['hello', 'good', 'morning', 'cat', 'dog', 'I', 'you', 'am', 'is', 'nice', 'day', 'book', 'red', 'blue'],
        'A2': ['hello', 'yesterday', 'tomorrow', 'friend', 'family', 'school', 'work', 'like', 'want', 'need', 'think', 'know'],
        'B1': ['however', 'although', 'because', 'therefore', 'consequently', 'furthermore', 'meanwhile', 'previously', 'subsequently'],
        'B2': ['nevertheless', 'notwithstanding', 'albeit', 'whereas', 'moreover', 'accordingly', 'henceforth', 'predominantly'],
        'C1': ['quintessential', 'ubiquitous', 'paradigm', 'methodology', 'juxtaposition', 'phenomenological', 'epistemological'],
        'C2': ['exacerbate', 'obfuscate', 'ameliorate', 'parsimonious', 'recalcitrant', 'perspicacious', 'insouciant']
    }

    vocab = vocabularies.get(cefr_level, vocabularies['B1'])
    words = [random.choice(vocab) for _ in range(length)]
    return ' '.join(words)

def generate_dataset(num_samples, cefr_levels, output_file):
    """Generate synthetic CEFR dataset."""
    data = []
    levels = cefr_levels.split(',')

    for i in range(num_samples):
        level = levels[i % len(levels)]
        text = generate_fake_text(level, length=random.randint(30, 100))
        data.append({
            'text': text,
            'cefr_label': level,
            'label': level  # Backward compatibility
        })

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} samples -> {output_file}")
    print(f"CEFR level distribution:\n{df['cefr_label'].value_counts().sort_index()}")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python generate_data.py <num_samples> <cefr_levels> <output_file>")
        sys.exit(1)

    num_samples = int(sys.argv[1])
    cefr_levels = sys.argv[2]
    output_file = sys.argv[3]

    generate_dataset(num_samples, cefr_levels, output_file)
PYTHON_EOF

    # Generate training data for ml-training-data (used by classifier training)
    local train_file="$ml_train_dir/train-data.csv"
    log_substep "Generating training data: $train_file"
    if ! "$python_bin" "$gen_script" "$num_train" "$cefr_levels" "$train_file"; then
        log_error "Failed to generate training data"
        write_status "generate_synthetic_data" "error" "Failed to generate training data"
        return 1
    fi

    # Copy training data to features-training-data (used by TF-IDF training)
    local features_train_file="${Config[FEATURES_TRAIN_DIR]}/train-data.csv"
    log_substep "Copying to features-training-data: $features_train_file"
    cp "$train_file" "$features_train_file"

    # Generate test data
    local test_file="$ml_test_dir/test-data.csv"
    log_substep "Generating test data: $test_file"
    if ! "$python_bin" "$gen_script" "$num_test" "$cefr_levels" "$test_file"; then
        log_error "Failed to generate test data"
        write_status "generate_synthetic_data" "error" "Failed to generate test data"
        return 1
    fi

    Config[TRAIN_CSV]="$train_file"
    Config[TEST_CSV]="$test_file"

    log_success "Synthetic data generation complete"
    write_status "generate_synthetic_data" "success" "Generated $num_train training and $num_test test samples"
    return 0
}

################################################################################
# CREATE PIPELINE CONFIG FILE
################################################################################

create_pipeline_config() {
    log_step "Creating Pipeline Configuration File"
    write_status "create_pipeline_config" "started" "Creating pipeline configuration file"

    local config_file="${Config[EXPERIMENT_DIR]}/config.yaml"
    local exp_dir="${Config[EXPERIMENT_DIR]}"

    log_substep "Creating: $config_file"

    cat > "$config_file" << EOF
# Auto-generated pipeline configuration
# Timestamp: ${Config[TIMESTAMP]}
# Experiment: ${Config[EXPERIMENT_NAME]}

experiment_config:
  experiment_dir: "$exp_dir"

tfidf_config:
  max_features: ${Config[TFIDF_MAX_FEATURES]}
  ngram_range: [${Config[TFIDF_NGRAM_MIN]}, ${Config[TFIDF_NGRAM_MAX]}]
  min_df: ${Config[TFIDF_MIN_DF]}
  max_df: ${Config[TFIDF_MAX_DF]}
  sublinear_tf: true

classifier_config:
  classifier_type: "${Config[CLASSIFIER_TYPE]}"
  random_state: 42
  logistic_max_iter: 1000
  logistic_class_weight: "balanced"

data_config:
  text_column: "text"
  label_column: "label"
  cefr_column: "cefr_label"
  min_text_length: 0

output_config:
  save_config: true
  save_models: true
  save_features: true
  save_results: true
  verbose: true
  save_csv: true
  save_json: true
EOF

    Config[CONFIG_FILE]="$config_file"

    log_success "Configuration file created: $config_file"
    write_status "create_pipeline_config" "success" "Configuration file created successfully"
    return 0
}

################################################################################
# PIPELINE EXECUTION STEPS
################################################################################

run_tfidf_training() {
    if [[ "${Config[SKIP_TFIDF_TRAINING]}" == "true" ]]; then
        log_warning "Skipping TF-IDF training (--skip-tfidf)"
        write_status "run_tfidf_training" "skipped" "TF-IDF training skipped by user"
        return 0
    fi

    log_step "Training TF-IDF Model"
    write_status "run_tfidf_training" "started" "Starting TF-IDF model training"

    local python_bin="${Config[PYTHON_BIN]}"
    local config_file="${Config[CONFIG_FILE]}"

    log_substep "Training from features-training-data/"

    # Note: train_tfidf.py reads from features-training-data/ automatically
    if ! PYTHONPATH=. "$python_bin" src/train_tfidf.py \
        --config-file "$config_file"; then
        log_error "TF-IDF training failed"
        write_status "run_tfidf_training" "error" "TF-IDF training failed"
        return 1
    fi

    log_success "TF-IDF training complete"
    write_status "run_tfidf_training" "success" "TF-IDF model trained successfully"
    return 0
}

run_feature_extraction() {
    if [[ "${Config[SKIP_FEATURE_EXTRACTION]}" == "true" ]]; then
        log_warning "Skipping feature extraction (--skip-features)"
        write_status "run_feature_extraction" "skipped" "Feature extraction skipped by user"
        return 0
    fi

    log_step "Extracting Features"
    write_status "run_feature_extraction" "started" "Starting feature extraction"

    local python_bin="${Config[PYTHON_BIN]}"
    local config_file="${Config[CONFIG_FILE]}"
    local models_dir="${Config[MODELS_DIR]}"

    # Find the TF-IDF model directory that was just created
    local tfidf_model_dir
    tfidf_model_dir=$(find "$models_dir" -maxdepth 1 -type d -name "*_tfidf" | head -n1)

    if [[ -z "$tfidf_model_dir" || ! -d "$tfidf_model_dir" ]]; then
        log_error "TF-IDF model directory not found in: $models_dir"
        write_status "run_feature_extraction" "error" "TF-IDF model not found"
        return 1
    fi

    log_substep "Using TF-IDF model from: $tfidf_model_dir"
    log_substep "Extracting features from both training and test data"

    # Note: extract_features.py uses --data-source to read from ml-training-data and ml-test-data
    # We explicitly point it to the pretrained TF-IDF model
    if ! PYTHONPATH=. "$python_bin" src/extract_features.py \
        --config-file "$config_file" \
        --pretrained-dir "$tfidf_model_dir" \
        --data-source both; then
        log_error "Feature extraction failed"
        write_status "run_feature_extraction" "error" "Feature extraction failed"
        return 1
    fi

    log_success "Feature extraction complete"
    write_status "run_feature_extraction" "success" "Features extracted for training and test data"
    return 0
}

run_classifier_training() {
    if [[ "${Config[SKIP_CLASSIFIER_TRAINING]}" == "true" ]]; then
        log_warning "Skipping classifier training (--skip-classifier)"
        write_status "run_classifier_training" "skipped" "Classifier training skipped by user"
        return 0
    fi

    log_step "Training Classifier"
    write_status "run_classifier_training" "started" "Starting ${Config[CLASSIFIER_TYPE]} classifier training"

    local python_bin="${Config[PYTHON_BIN]}"
    local config_file="${Config[CONFIG_FILE]}"
    local train_csv="${Config[TRAIN_CSV]}"
    local features_dir="${Config[FEATURES_OUTPUT_DIR]}"

    # Find the training features directory (should be {hash}_tfidf/train-data/)
    local train_features_dir
    train_features_dir=$(find "$features_dir" -maxdepth 2 -type d -name "train-data" | head -n1)

    if [[ -z "$train_features_dir" || ! -d "$train_features_dir" ]]; then
        log_error "Training features directory not found in: $features_dir"
        write_status "run_classifier_training" "error" "Training features not found"
        return 1
    fi

    log_substep "Training ${Config[CLASSIFIER_TYPE]} classifier"
    log_substep "Using features from: $train_features_dir"
    log_substep "Using labels from: $train_csv"

    # Note: train_classifiers.py uses --feature-dir and --labels-csv
    if ! PYTHONPATH=. "$python_bin" src/train_classifiers.py \
        --config-file "$config_file" \
        --feature-dir "$train_features_dir" \
        --labels-csv "$train_csv"; then
        log_error "Classifier training failed"
        write_status "run_classifier_training" "error" "Classifier training failed"
        return 1
    fi

    log_success "Classifier training complete"
    write_status "run_classifier_training" "success" "${Config[CLASSIFIER_TYPE]} classifier trained successfully"
    return 0
}

run_prediction() {
    if [[ "${Config[SKIP_PREDICTION]}" == "true" ]]; then
        log_warning "Skipping prediction (--skip-prediction)"
        write_status "run_prediction" "skipped" "Prediction skipped by user"
        return 0
    fi

    log_step "Running Predictions"
    write_status "run_prediction" "started" "Starting prediction on test data"

    local python_bin="${Config[PYTHON_BIN]}"
    local config_file="${Config[CONFIG_FILE]}"
    local models_dir="${Config[MODELS_DIR]}/classifiers"
    local features_dir="${Config[FEATURES_OUTPUT_DIR]}"

    # Find the trained classifier model
    local classifier_model
    classifier_model=$(find "$models_dir" -maxdepth 1 -type d -name "*${Config[CLASSIFIER_TYPE]}*" | head -n1)

    if [[ -z "$classifier_model" ]]; then
        log_error "No classifier model found in: $models_dir"
        write_status "run_prediction" "error" "No classifier model found in $models_dir"
        return 1
    fi

    local model_name
    model_name=$(basename "$classifier_model")

    log_substep "Using classifier: $model_name"

    # Find the test features directory
    local test_features_dir
    test_features_dir=$(find "$features_dir" -maxdepth 1 -type d -name "*tfidf" | head -n1)

    if [[ -z "$test_features_dir" ]]; then
        log_error "No TF-IDF features directory found in: $features_dir"
        write_status "run_prediction" "error" "No TF-IDF features directory found"
        return 1
    fi

    local features_subdir="$test_features_dir/test-data"

    if [[ ! -d "$features_subdir" ]]; then
        log_error "Test features not found: $features_subdir"
        write_status "run_prediction" "error" "Test features not found: $features_subdir"
        return 1
    fi

    log_substep "Using features from: $features_subdir"

    if ! PYTHONPATH=. "$python_bin" src/predict.py \
        --config-file "$config_file" \
        --classifier-model "$model_name" \
        --feature-dir "$features_subdir"; then
        log_error "Prediction failed"
        write_status "run_prediction" "error" "Prediction execution failed"
        return 1
    fi

    log_success "Predictions complete"
    write_status "run_prediction" "success" "Predictions completed successfully"
    return 0
}

################################################################################
# CLEANUP
################################################################################

cleanup_temp_files() {
    if [[ "${Config[CLEANUP_ON_SUCCESS]}" != "true" ]]; then
        return 0
    fi

    log_step "Cleaning Up"

    local exp_dir="${Config[EXPERIMENT_DIR]}"

    log_substep "Removing experiment directory: $exp_dir"
    rm -rf "$exp_dir"

    log_success "Cleanup complete"
    return 0
}

################################################################################
# MAIN EXECUTION
################################################################################

main() {
    echo -e "${BOLD}${CYAN}"
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║  Minimal CEFR Classification Pipeline Test                        ║"
    echo "║  Testing pipeline integrity with synthetic data                   ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    # Parse command-line arguments
    parse_arguments "$@"

    # Display configuration
    display_config

    # Validate configuration
    if ! validate_config; then
        log_error "Configuration validation failed"
        exit 2
    fi

    # Setup experiment directory
    if ! setup_experiment_directory; then
        log_error "Failed to setup experiment directory"
        exit 3
    fi

    # Generate synthetic data
    if ! generate_synthetic_data; then
        log_error "Failed to generate synthetic data"
        exit 3
    fi

    # Create pipeline configuration
    if ! create_pipeline_config; then
        log_error "Failed to create pipeline configuration"
        exit 3
    fi

    # Run pipeline steps
    local pipeline_errors=0

    if ! run_tfidf_training; then
        log_error "TF-IDF training failed"
        ((pipeline_errors++))
    fi

    if ! run_feature_extraction; then
        log_error "Feature extraction failed"
        ((pipeline_errors++))
    fi

    if ! run_classifier_training; then
        log_error "Classifier training failed"
        ((pipeline_errors++))
    fi

    if ! run_prediction; then
        log_error "Prediction failed"
        ((pipeline_errors++))
    fi

    # Check for pipeline errors
    if [[ $pipeline_errors -gt 0 ]]; then
        log_error "Pipeline completed with $pipeline_errors error(s)"
        exit 3
    fi

    # Cleanup if requested
    cleanup_temp_files

    # Final success message
    echo ""
    log_step "Pipeline Test Complete"
    log_success "All steps completed successfully! ✓"
    write_status "pipeline_complete" "success" "All pipeline steps completed successfully"
    echo ""
    log_info "Experiment directory: ${Config[EXPERIMENT_DIR]}"
    log_info "Configuration file: ${Config[CONFIG_FILE]}"
    log_info "Results directory: ${Config[RESULTS_DIR]}"
    log_info "Status log: ${Config[STATUS_FILE]}"
    echo ""

    # Display status summary
    if [[ -f "${Config[STATUS_FILE]}" ]]; then
        echo -e "${CYAN}${BOLD}Status Summary:${NC}"
        local total_steps=$(grep -c '"step"' "${Config[STATUS_FILE]}" 2>/dev/null | tr -d ' \n' || echo "0")
        local success_steps=$(grep -c '"status":"success"' "${Config[STATUS_FILE]}" 2>/dev/null | tr -d ' \n' || echo "0")
        local error_steps=$(grep -c '"status":"error"' "${Config[STATUS_FILE]}" 2>/dev/null | tr -d ' \n' || echo "0")
        local skipped_steps=$(grep -c '"status":"skipped"' "${Config[STATUS_FILE]}" 2>/dev/null | tr -d ' \n' || echo "0")

        echo -e "  Total events: $total_steps"
        echo -e "  ${GREEN}Successes: $success_steps${NC}"
        if [[ "$error_steps" -gt 0 ]]; then
            echo -e "  ${RED}Errors: $error_steps${NC}"
        fi
        if [[ "$skipped_steps" -gt 0 ]]; then
            echo -e "  ${YELLOW}Skipped: $skipped_steps${NC}"
        fi
        echo ""
    fi

    return 0
}

# Execute main function
main "$@"
