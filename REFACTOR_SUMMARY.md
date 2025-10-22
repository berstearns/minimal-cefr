# Report Tool Refactor Summary

## Overview

Refactored `src/report.py` to support two modes:
1. **Standard mode**: Original behavior using `evaluation_report.md` files
2. **Manual mode**: New feature for analyzing prediction JSON files from external models

## Changes Made

### 1. New Dependencies

Added imports:
- `pandas` - For CSV reading (already used elsewhere in codebase)
- `numpy` - For numerical operations
- `sklearn.metrics` - For metric computation
- `Tuple` - Type hint

### 2. New Functions

#### `parse_manual_predictions(predictions_json_path: Path)`
- Parses JSON files with format: `{"sample_id": {"A1": 0.0, ..., "C2": 0.0}, ...}`
- Returns tuple of (sample_ids, probabilities)

#### `load_ground_truth_labels(labels_csv_path, id_column, label_column)`
- Loads ground truth labels from CSV
- Converts all IDs to strings for consistent matching
- Returns dict mapping sample_id -> true_label

#### `compute_metrics_from_predictions(sample_ids, probabilities, ground_truth, strategy)`
- Computes metrics on-the-fly from predictions
- Supports both "argmax" and "rounded_avg" strategies
- Returns dict with: accuracy, adjacent_accuracy, macro_f1, weighted_f1

#### `detect_results_structure(results_dir: Path)`
- Auto-detects whether results use standard or manual structure
- Checks for `evaluation_report.md` vs JSON files
- Returns "standard" or "manual"

#### `collect_metrics_from_manual_predictions(...)`
- Collects metrics from manual prediction JSON files
- Performs fuzzy matching between dataset names and CSV files
- Handles both string and numeric sample IDs
- Computes metrics for both prediction strategies

### 3. Modified Functions

#### `collect_all_metrics(...)`
- Added parameters: `mode`, `labels_dir`, `id_column`, `label_column`
- Routes to appropriate collection function based on mode
- Maintains backward compatibility (defaults to auto-detect)

#### `main()`
- Added CLI arguments:
  - `--mode` (auto/standard/manual)
  - `--labels-dir` (path to ground truth CSVs)
  - `--id-column` (CSV column for sample IDs)
  - `--label-column` (CSV column for labels)
- Added examples for manual mode usage

## Key Features

### Auto-Detection

```python
mode = detect_results_structure(results_dir)
# Returns "manual" if JSON files found, "standard" if evaluation_report.md found
```

### Fuzzy Dataset Matching

Tries multiple matching strategies:
1. Exact match: `dataset.csv`, `norm-dataset.csv`, `DATASET.csv`
2. Fuzzy match: Case-insensitive substring matching

Example:
- Dataset dir: `celva` → Matches: `norm-CELVA-SP.csv` ✓
- Dataset dir: `kupa` → Matches: `norm-KUPA-KEYS.csv` ✓

### ID Type Normalization

Converts all sample IDs to strings to handle:
- Numeric IDs from CSV: `0, 1, 2, ...`
- String IDs from JSON: `"0", "1", "2", ...`
- Mixed formats: `R_00RbUqO7jXLDItP`

## Usage Examples

### Before (Standard Mode Only)

```bash
python -m src.report -e data/experiments/zero-shot --rank accuracy
```

### After (Auto-Detect Mode)

```bash
# Same command works for both standard and manual structures
python -m src.report -e data/experiments/zero-shot --rank accuracy

# Manual predictions with labels directory
python -m src.report \
    -e data/experiments/prompting \
    --labels-dir data/labels \
    --rank accuracy
```

### Explicit Mode Selection

```bash
# Force manual mode
python -m src.report \
    -e data/experiments/prompting \
    --mode manual \
    --labels-dir data/labels \
    --rank adjacent_accuracy

# Force standard mode (skip auto-detection)
python -m src.report \
    -e data/experiments/zero-shot \
    --mode standard \
    --rank accuracy
```

## Backward Compatibility

✓ All existing commands work without modification
✓ Standard mode behavior unchanged
✓ Auto-detection prevents breaking changes
✓ New arguments are optional

## Testing

Tested with:
- ✓ Standard mode: `data/experiments/zero-shot` (evaluation_report.md files)
- ✓ Manual mode: `data/experiments/prompting` (JSON prediction files)
- ✓ Auto-detection: Correctly identifies both structures
- ✓ Fuzzy matching: Handles `celva`→`norm-CELVA-SP.csv`, `kupa`→`norm-KUPA-KEYS.csv`
- ✓ ID normalization: Works with both numeric and string IDs
- ✓ Both strategies: Argmax and rounded_avg compute correctly

## Documentation

Created/Updated:
- ✓ `docs/REPORT_MANUAL_PREDICTIONS.md` - Complete manual mode guide
- ✓ `docs/REPORT_GUIDE.md` - Updated with manual mode reference
- ✓ `docs/INDEX.md` - Added new guide to index
- ✓ `src/report.py` - Added usage examples in docstring

## Performance

- Standard mode: No performance impact (same code path)
- Manual mode: Slower than standard (computes metrics vs reading them)
  - Acceptable for typical dataset sizes (1000-2000 samples per dataset)
  - Metrics computed in <1 second per model/dataset combination

## Future Improvements

Potential enhancements:
- [ ] Cache computed metrics to avoid recomputation
- [ ] Support for additional prediction formats (CSV, TSV)
- [ ] Configuration file for dataset→CSV mappings
- [ ] Parallel processing for large experiments
- [ ] Support for additional metrics (MSE, MAE for ordinal)

---

Last updated: 2025-10-21
