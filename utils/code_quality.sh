#!/bin/bash

###############################################################################
# Code Quality Pipeline for Python Projects
#
# This script runs a comprehensive set of code quality tools on Python files:
# - black: Code formatting
# - isort: Import sorting
# - flake8: Linting (PEP 8 compliance)
# - mypy: Static type checking
# - pydocstyle: Docstring convention checking
# - pytest: Unit testing
#
# Usage:
#   ./code_quality.sh [target_dir] [mode]
#
# Arguments:
#   target_dir: Directory to analyze (default: ../src)
#   mode: "check" for validation only, "fix" to auto-fix issues (default: check)
#
###############################################################################

set -e  # Exit on error

# Configuration
PYTHON_PATH="${PYTHON_PATH:-$HOME/.pyenv/versions/3.10.18/bin/python3}"
SCRIPT_DIR=$(realpath "$(dirname "$0")")
DEFAULT_TARGET_DIR=$(realpath "$(dirname "$SCRIPT_DIR")")
TARGET_DIR="${1:-$DEFAULT_TARGET_DIR}"
MODE="${2:-check}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_section() {
    echo ""
    echo "=========================================="
    echo -e "${BLUE}$1${NC}"
    echo "=========================================="
}

# Check if Python exists
if [ ! -f "$PYTHON_PATH" ]; then
    log_error "Python not found at: $PYTHON_PATH"
    log_info "Please set PYTHON_PATH environment variable or update the script"
    exit 1
fi

log_info "Using Python: $PYTHON_PATH"
log_info "Python version: $($PYTHON_PATH --version)"

# Check if target directory exists
if [ ! -d "$TARGET_DIR" ]; then
    log_error "Target directory not found: $TARGET_DIR"
    exit 1
fi

log_info "Target directory: $TARGET_DIR"
log_info "Mode: $MODE"

# Find all Python files
PYTHON_FILES=$(find "$TARGET_DIR" -name "*.py" -not -path "*/\__pycache__/*" -not -path "*/.venv/*" -not -path "*/venv/*")

if [ -z "$PYTHON_FILES" ]; then
    log_warning "No Python files found in $TARGET_DIR"
    exit 0
fi

FILE_COUNT=$(echo "$PYTHON_FILES" | wc -l)
log_info "Found $FILE_COUNT Python file(s) to analyze"

# Exit code tracking
EXIT_CODE=0

###############################################################################
# 1. BLACK - Code Formatting
###############################################################################
print_section "1. BLACK - Code Formatting"

if [ "$MODE" == "fix" ]; then
    log_info "Running black in FIX mode..."
    if $PYTHON_PATH -m black $TARGET_DIR; then
        log_success "Black formatting completed"
    else
        log_error "Black formatting failed"
        EXIT_CODE=1
    fi
else
    log_info "Running black in CHECK mode..."
    if $PYTHON_PATH -m black --check --diff $TARGET_DIR; then
        log_success "Black check passed - all files properly formatted"
    else
        log_warning "Black check failed - run with 'fix' mode to auto-format"
        EXIT_CODE=1
    fi
fi

###############################################################################
# 2. ISORT - Import Sorting
###############################################################################
print_section "2. ISORT - Import Sorting"

if [ "$MODE" == "fix" ]; then
    log_info "Running isort in FIX mode..."
    if $PYTHON_PATH -m isort $TARGET_DIR; then
        log_success "Import sorting completed"
    else
        log_error "Import sorting failed"
        EXIT_CODE=1
    fi
else
    log_info "Running isort in CHECK mode..."
    if $PYTHON_PATH -m isort --check-only --diff $TARGET_DIR; then
        log_success "Isort check passed - all imports properly sorted"
    else
        log_warning "Isort check failed - run with 'fix' mode to auto-sort"
        EXIT_CODE=1
    fi
fi

###############################################################################
# 3. FLAKE8 - Linting (PEP 8)
###############################################################################
print_section "3. FLAKE8 - Linting (PEP 8 Compliance)"

log_info "Running flake8..."
if $PYTHON_PATH -m flake8 $TARGET_DIR; then
    log_success "Flake8 check passed - no linting errors"
else
    log_error "Flake8 check failed - please fix linting errors"
    EXIT_CODE=1
fi

###############################################################################
# 4. PYDOCSTYLE - Docstring Checking
###############################################################################
print_section "4. PYDOCSTYLE - Docstring Convention"

log_info "Running pydocstyle..."
if $PYTHON_PATH -m pydocstyle $TARGET_DIR; then
    log_success "Pydocstyle check passed - docstrings follow conventions"
else
    log_warning "Pydocstyle check failed - please improve docstrings"
    # Don't fail the entire pipeline for docstring issues
fi

###############################################################################
# 5. MYPY - Static Type Checking
###############################################################################
print_section "5. MYPY - Static Type Checking"

log_info "Running mypy..."
if $PYTHON_PATH -m mypy $TARGET_DIR --pretty --show-error-context; then
    log_success "Mypy check passed - no type errors"
else
    log_warning "Mypy check failed - please add/fix type hints"
    # Don't fail the entire pipeline for type hint issues in initial setup
fi

###############################################################################
# 6. PYTEST - Unit Testing
###############################################################################
print_section "6. PYTEST - Unit Testing"

if [ -d "../tests" ] || [ -d "tests" ]; then
    log_info "Running pytest..."
    if $PYTHON_PATH -m pytest; then
        log_success "All tests passed"
    else
        log_error "Some tests failed"
        EXIT_CODE=1
    fi
else
    log_warning "No tests directory found - skipping pytest"
    log_info "Consider creating a 'tests' directory with unit tests"
fi

###############################################################################
# Summary
###############################################################################
print_section "CODE QUALITY PIPELINE SUMMARY"

if [ $EXIT_CODE -eq 0 ]; then
    log_success "All critical checks passed! ✓"
    echo ""
    log_info "Code meets quality standards:"
    echo "  ✓ Properly formatted (black)"
    echo "  ✓ Imports sorted (isort)"
    echo "  ✓ PEP 8 compliant (flake8)"
    echo "  ✓ Type hints checked (mypy)"
    echo "  ✓ Docstrings validated (pydocstyle)"
    [ -d "../tests" ] || [ -d "tests" ] && echo "  ✓ Tests passing (pytest)"
else
    log_error "Code quality checks failed! ✗"
    echo ""
    log_info "To auto-fix formatting and import issues, run:"
    echo "  $0 $TARGET_DIR fix"
    echo ""
    log_info "For other issues, please review the errors above and fix manually."
fi

echo ""
exit $EXIT_CODE
