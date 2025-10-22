# Report Tool - Manual Predictions Mode

Guide for using `src.report` with manual predictions from language models.

## Overview

The report tool now supports **two modes**:

1. **Standard mode**: Reads pre-computed metrics from `evaluation_report.md` files (original behavior)
2. **Manual mode**: Computes metrics on-the-fly from prediction JSON files (new feature)

The tool **auto-detects** which mode to use based on the directory structure, or you can specify manually.

---

## Manual Predictions Structure

### Expected Directory Layout

```
experiment-dir/
└── results/
    ├── model-1/
    │   ├── dataset-1/
    │   │   └── predictions.json          # Any .json file with predictions
    │   └── dataset-2/
    │       └── predictions.json
    └── model-2/
        └── ...
```

### Prediction JSON Format

Each JSON file should contain probability distributions keyed by sample ID:

```json
{
  "sample_id_1": {
    "A1": 0.0,
    "A2": 0.1,
    "B1": 0.7,
    "B2": 0.2,
    "C1": 0.0,
    "C2": 0.0
  },
  "sample_id_2": {
    "A1": 0.0,
    "A2": 0.0,
    "B1": 0.1,
    "B2": 0.6,
    "C1": 0.3,
    "C2": 0.0
  }
}
```

**Requirements:**
- Keys are sample IDs (strings or numbers, will be converted to strings)
- Values are objects with CEFR level probabilities
- Probabilities should sum to approximately 1.0
- All 6 CEFR levels should be present (A1, A2, B1, B2, C1, C2)

### Ground Truth Labels

You must provide CSV files with ground truth labels in a separate directory:

```
labels-dir/
├── dataset-1.csv
├── dataset-2.csv
└── ...
```

**CSV Format:**
```csv
writing_id,cefr_level,text
sample_id_1,B1,"Text content..."
sample_id_2,B2,"More text..."
```

**Required Columns:**
- Sample ID column (default: `writing_id`, configurable via `--id-column`)
- Label column (default: `cefr_level`, configurable via `--label-column`)

---

## Usage

### Auto-Detection Mode (Recommended)

The tool automatically detects whether to use standard or manual mode:

```bash
python -m src.report \
    -e data/experiments/prompting \
    --labels-dir data/experiments/zero-shot/ml-test-data \
    --rank accuracy \
    -v
```

**Output:**
```
Auto-detected mode: manual
Found 4 model directories
  Processing: gemma:2b/celva (combined_gemma_2b_celva.json)
     Matched celva -> norm-CELVA-SP.csv
  ...
```

### Explicit Manual Mode

Force manual mode even if auto-detection would choose standard:

```bash
python -m src.report \
    -e data/experiments/prompting \
    --mode manual \
    --labels-dir /path/to/ground-truth-csvs \
    --rank accuracy
```

### Custom Column Names

If your CSV uses different column names:

```bash
python -m src.report \
    -e data/experiments/prompting \
    --labels-dir data/labels \
    --id-column "id" \
    --label-column "level" \
    --rank accuracy
```

---

## Dataset Name Matching

The tool uses flexible matching to find the correct ground truth CSV for each dataset:

### 1. Exact Match (First Try)

Tries these patterns in order:
- `{dataset_name}.csv`
- `norm-{dataset_name}.csv`
- `{dataset_name.upper()}.csv`
- `norm-{dataset_name.upper()}.csv`

### 2. Fuzzy Match (Fallback)

If exact match fails, performs case-insensitive substring matching:

**Example:**
- Dataset dir: `celva`
- Matches CSV: `norm-CELVA-SP.csv` ✓

**Example:**
- Dataset dir: `kupa`
- Matches CSV: `norm-KUPA-KEYS.csv` ✓

### Manual Override

To ensure correct matching, name your dataset directories to match CSV files:

```bash
# Option 1: Rename dataset directories
results/
├── gemma:7b/
│   ├── norm-CELVA-SP/      # Matches norm-CELVA-SP.csv exactly
│   └── norm-KUPA-KEYS/     # Matches norm-KUPA-KEYS.csv exactly

# Option 2: Create specific CSV names
labels/
├── celva.csv               # Matches celva/ directory
└── kupa.csv                # Matches kupa/ directory
```

---

## Computed Metrics

The tool computes the following metrics for **both** prediction strategies:

### Strategy 1: Argmax

Standard classification approach - select class with highest probability:

```python
predicted_label = argmax(probabilities)
```

### Strategy 2: Rounded Average

Regression-style approach - calculate expected class index and round:

```python
expected_index = sum(class_index * probability for each class)
predicted_label = cefr_levels[round(expected_index)]
```

### Metrics Calculated

For each strategy:

- **accuracy**: Exact match accuracy
- **adjacent_accuracy**: CEFR-specific ordinal accuracy (±1 level)
- **macro_f1**: Macro-averaged F1 score
- **weighted_f1**: Weighted F1 score

---

## Example Workflows

### Workflow 1: Quick Ranking

Rank all models by accuracy with auto-detection:

```bash
python -m src.report \
    -e data/experiments/prompting \
    --labels-dir data/labels \
    --rank accuracy
```

### Workflow 2: Compare Strategies

Compare argmax vs rounded_avg strategies:

```bash
# Argmax strategy
python -m src.report \
    -e data/experiments/prompting \
    --labels-dir data/labels \
    --rank accuracy \
    --strategy argmax

# Rounded average strategy
python -m src.report \
    -e data/experiments/prompting \
    --labels-dir data/labels \
    --rank accuracy \
    --strategy rounded_avg
```

### Workflow 3: CEFR-Specific Ranking

Rank by adjacent accuracy (most relevant for CEFR):

```bash
python -m src.report \
    -e data/experiments/prompting \
    --labels-dir data/labels \
    --rank adjacent_accuracy \
    --no-group
```

### Workflow 4: Generate Summary Report

Create comprehensive markdown report:

```bash
python -m src.report \
    -e data/experiments/prompting \
    --labels-dir data/labels \
    --summary-report lm_results_summary.md
```

### Workflow 5: Filter Specific Dataset

Rank models for a specific test set:

```bash
python -m src.report \
    -e data/experiments/prompting \
    --labels-dir data/labels \
    --rank accuracy \
    --dataset kupa \
    --top 10
```

---

## Example Output

### Grouped by Dataset (Default)

```
========================================================================================================================
RANKING BY: ACCURACY (Grouped by Dataset)
========================================================================================================================

📊 Dataset: celva
------------------------------------------------------------------------------------------------------------------------
Rank   Model                Strategy     Accuracy   TF-IDF     Classifier
------ -------------------- ------------ ---------- ---------- ------------
1      llama3:8b            rounded_avg  0.3800     N/A        N/A
2      llama3:8b            argmax       0.3794     N/A        N/A
3      mistral:7b           argmax       0.3714     N/A        N/A
...

📊 Dataset: kupa
------------------------------------------------------------------------------------------------------------------------
Rank   Model                Strategy     Accuracy   TF-IDF     Classifier
------ -------------------- ------------ ---------- ---------- ------------
1      gemma:7b             argmax       0.3926     N/A        N/A
2      gemma:7b             rounded_avg  0.3926     N/A        N/A
...
```

### Flat Ranking (`--no-group`)

```
========================================================================================================================
RANKING BY: ACCURACY
========================================================================================================================
Rank   Model           Dataset    Strategy     Accuracy   TF-IDF     Classifier
------ --------------- ---------- ------------ ---------- ---------- ------------
1      gemma:7b        kupa       argmax       0.3926     N/A        N/A
2      gemma:7b        kupa       rounded_avg  0.3926     N/A        N/A
3      llama3:8b       celva      rounded_avg  0.3800     N/A        N/A
4      llama3:8b       celva      argmax       0.3794     N/A        N/A
...
```

---

## Troubleshooting

### Issue: "No ground truth CSV found"

**Cause:** Dataset name doesn't match any CSV file

**Solution:**
```bash
# Check what CSVs are available
ls data/labels/

# Use verbose mode to see matching attempts
python -m src.report -e exp --labels-dir data/labels --rank accuracy -v

# Option 1: Rename dataset directories to match CSVs
mv results/model1/dataset1 results/model1/exact-csv-name

# Option 2: Create CSV with matching name
cp data/labels/norm-DATASET-FULL.csv data/labels/dataset1.csv
```

### Issue: "Column 'writing_id' not found"

**Cause:** CSV uses different column name for IDs

**Solution:**
```bash
# Check CSV columns
head -1 data/labels/dataset.csv

# Specify correct column name
python -m src.report \
    -e exp \
    --labels-dir data/labels \
    --id-column "id" \
    --rank accuracy
```

### Issue: No celva results showing (only kupa)

**Cause:** Sample ID type mismatch (integers vs strings)

**Status:** Fixed in v2 - IDs are now always converted to strings

**Verification:**
```bash
# Should show both datasets now
python -m src.report -e exp --labels-dir labels --rank accuracy
```

### Issue: "ModuleNotFoundError: No module named 'pandas'"

**Cause:** Using system Python instead of project environment

**Solution:**
```bash
# Use the correct Python environment
~/.pyenv/versions/3.10.18/bin/python3 -m src.report ...

# Or activate environment first
pyenv shell 3.10.18
python -m src.report ...
```

---

## Comparison: Manual vs Standard Mode

| Feature | Standard Mode | Manual Mode |
|---------|---------------|-------------|
| **Input** | `evaluation_report.md` | Prediction JSON files |
| **Metrics** | Pre-computed (read from report) | Computed on-the-fly |
| **Ground Truth** | Embedded in report | Separate CSV required |
| **Model Config** | Read from `config.json` | Optional (shows N/A if missing) |
| **Speed** | Fast (just parsing) | Slower (computes metrics) |
| **Flexibility** | Limited to report format | Works with any JSON predictions |
| **Use Case** | Pipeline-generated results | External model predictions |

---

## Real-World Example

### Scenario: Evaluating LLM Predictions

You ran 4 language models (Gemma, Llama, Mistral) on 2 test sets (CELVA, KUPA) and collected predictions in JSON format.

**Directory Structure:**
```
data/experiments/prompting/
└── results/
    ├── gemma:2b/
    │   ├── celva/
    │   │   └── combined_gemma_2b_celva.json
    │   └── kupa/
    │       └── combined_gemma_2b_kupa.json
    ├── gemma:7b/
    │   └── ...
    ├── llama3:8b/
    │   └── ...
    └── mistral:7b/
        └── ...
```

**Ground Truth:**
```
data/experiments/zero-shot/ml-test-data/
├── norm-CELVA-SP.csv
└── norm-KUPA-KEYS.csv
```

**Commands:**

```bash
# 1. Quick overview
python -m src.report \
    -e data/experiments/prompting \
    --labels-dir data/experiments/zero-shot/ml-test-data

# 2. Rank by accuracy
python -m src.report \
    -e data/experiments/prompting \
    --labels-dir data/experiments/zero-shot/ml-test-data \
    --rank accuracy

# 3. Best adjacent accuracy (CEFR-appropriate metric)
python -m src.report \
    -e data/experiments/prompting \
    --labels-dir data/experiments/zero-shot/ml-test-data \
    --rank adjacent_accuracy \
    --strategy argmax

# 4. Generate comprehensive report
python -m src.report \
    -e data/experiments/prompting \
    --labels-dir data/experiments/zero-shot/ml-test-data \
    --summary-report lm_results.md
```

---

## Integration with MANUAL_PREDICTIONS_GUIDE.md

This tool complements the manual predictions workflow:

1. **Add predictions** following [MANUAL_PREDICTIONS_GUIDE.md](MANUAL_PREDICTIONS_GUIDE.md)
2. **Analyze results** using this report tool
3. **Compare models** across different datasets and strategies

The report tool accepts the same JSON format documented in the manual predictions guide.

---

## See Also

- **[REPORT_GUIDE.md](REPORT_GUIDE.md)** - Complete report tool documentation (standard mode)
- **[MANUAL_PREDICTIONS_GUIDE.md](MANUAL_PREDICTIONS_GUIDE.md)** - Adding manual predictions to experiments
- **[CEFR_METRICS.md](CEFR_METRICS.md)** - Understanding CEFR evaluation metrics
- **[USAGE.md](USAGE.md)** - Complete pipeline usage guide

---

Last updated: 2025-10-21
