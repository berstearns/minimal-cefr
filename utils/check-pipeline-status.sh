#!/usr/bin/env bash

################################################################################
# Check Pipeline Status from JSONL Status File
#
# Usage: ./check-pipeline-status.sh [status_file]
#
# This script reads the __status.jsonl file and reports on pipeline execution.
################################################################################

set -euo pipefail

# Colors
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    BOLD=''
    NC=''
fi

# Get status file from argument or find most recent
STATUS_FILE="${1:-}"

if [[ -z "$STATUS_FILE" ]]; then
    # Find most recent __status.jsonl file
    STATUS_FILE=$(find test-experiments -name "__status.jsonl" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

    if [[ -z "$STATUS_FILE" ]]; then
        echo -e "${RED}Error: No status file found${NC}"
        echo "Usage: $0 [path/to/__status.jsonl]"
        exit 1
    fi

    echo -e "${BLUE}Using most recent status file:${NC} $STATUS_FILE"
    echo ""
fi

if [[ ! -f "$STATUS_FILE" ]]; then
    echo -e "${RED}Error: Status file not found: $STATUS_FILE${NC}"
    exit 1
fi

# Display header
echo -e "${CYAN}${BOLD}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}${BOLD}║  Pipeline Status Report                                            ║${NC}"
echo -e "${CYAN}${BOLD}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Parse JSONL and display summary
echo -e "${BOLD}Summary:${NC}"

total_lines=$(wc -l < "$STATUS_FILE" | tr -d ' \n')
success_count=$(grep -c '"status":"success"' "$STATUS_FILE" 2>/dev/null | tr -d ' \n' || echo "0")
error_count=$(grep -c '"status":"error"' "$STATUS_FILE" 2>/dev/null | tr -d ' \n' || echo "0")
warning_count=$(grep -c '"status":"warning"' "$STATUS_FILE" 2>/dev/null | tr -d ' \n' || echo "0")
started_count=$(grep -c '"status":"started"' "$STATUS_FILE" 2>/dev/null | tr -d ' \n' || echo "0")
skipped_count=$(grep -c '"status":"skipped"' "$STATUS_FILE" 2>/dev/null | tr -d ' \n' || echo "0")

echo -e "  Total events: $total_lines"
echo -e "  ${GREEN}✓ Successes: $success_count${NC}"
echo -e "  ${RED}✗ Errors: $error_count${NC}"
echo -e "  ${YELLOW}⚠ Warnings: $warning_count${NC}"
echo -e "  ${BLUE}⟳ Started: $started_count${NC}"
echo -e "  ${YELLOW}⊘ Skipped: $skipped_count${NC}"
echo ""

# Overall status
if [[ $error_count -eq 0 ]] && grep -q '"status":"success"' "$STATUS_FILE" && grep -q '"step":"pipeline_complete"' "$STATUS_FILE"; then
    echo -e "${GREEN}${BOLD}Overall Status: SUCCESS ✓${NC}"
    exit_code=0
elif [[ $error_count -gt 0 ]]; then
    echo -e "${RED}${BOLD}Overall Status: FAILED ✗${NC}"
    exit_code=1
else
    echo -e "${YELLOW}${BOLD}Overall Status: INCOMPLETE ⚠${NC}"
    exit_code=2
fi
echo ""

# Display detailed timeline
echo -e "${BOLD}Timeline:${NC}"
echo ""

# Read JSONL line by line and format output
line_num=0
while IFS= read -r line; do
    ((line_num++))

    # Extract fields using basic parsing (avoiding jq dependency)
    timestamp=$(echo "$line" | sed -n 's/.*"timestamp":"\([^"]*\)".*/\1/p')
    step=$(echo "$line" | sed -n 's/.*"step":"\([^"]*\)".*/\1/p')
    status=$(echo "$line" | sed -n 's/.*"status":"\([^"]*\)".*/\1/p')
    message=$(echo "$line" | sed -n 's/.*"message":"\([^"]*\)".*/\1/p')

    # Format timestamp (just time portion)
    time_only=$(echo "$timestamp" | sed 's/.*T\([0-9:]*\).*/\1/')

    # Color code based on status
    case "$status" in
        success)
            status_icon="${GREEN}✓${NC}"
            status_text="${GREEN}SUCCESS${NC}"
            ;;
        error)
            status_icon="${RED}✗${NC}"
            status_text="${RED}ERROR${NC}"
            ;;
        warning)
            status_icon="${YELLOW}⚠${NC}"
            status_text="${YELLOW}WARNING${NC}"
            ;;
        started)
            status_icon="${BLUE}⟳${NC}"
            status_text="${BLUE}STARTED${NC}"
            ;;
        skipped)
            status_icon="${YELLOW}⊘${NC}"
            status_text="${YELLOW}SKIPPED${NC}"
            ;;
        *)
            status_icon="?"
            status_text="UNKNOWN"
            ;;
    esac

    printf "%3d. [%s] %b %-50s %b\n" "$line_num" "$time_only" "$status_icon" "$step" "$status_text"

    # Show message if not empty and verbose
    if [[ -n "$message" && "$message" != "null" ]]; then
        echo "     ${CYAN}↳${NC} $message"
    fi

done < "$STATUS_FILE"

echo ""

# Check if all critical steps succeeded
echo -e "${BOLD}Critical Steps Check:${NC}"

critical_steps=(
    "setup_experiment_directory"
    "generate_synthetic_data"
    "create_pipeline_config"
    "run_tfidf_training"
    "run_feature_extraction"
    "run_classifier_training"
    "run_prediction"
)

all_passed=true

for step in "${critical_steps[@]}"; do
    if grep -q "\"step\":\"$step\".*\"status\":\"success\"" "$STATUS_FILE"; then
        echo -e "  ${GREEN}✓${NC} $step"
    elif grep -q "\"step\":\"$step\".*\"status\":\"skipped\"" "$STATUS_FILE"; then
        echo -e "  ${YELLOW}⊘${NC} $step (skipped)"
    elif grep -q "\"step\":\"$step\".*\"status\":\"error\"" "$STATUS_FILE"; then
        echo -e "  ${RED}✗${NC} $step (failed)"
        all_passed=false
    else
        echo -e "  ${YELLOW}?${NC} $step (not found)"
        all_passed=false
    fi
done

echo ""

if [[ "$all_passed" == "true" ]]; then
    echo -e "${GREEN}${BOLD}All critical steps passed! ✓${NC}"
else
    echo -e "${RED}${BOLD}Some critical steps failed or missing! ✗${NC}"
fi

echo ""
echo "Status file: $STATUS_FILE"

exit $exit_code
