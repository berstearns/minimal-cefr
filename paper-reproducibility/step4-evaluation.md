# Step 4: Evaluation, Reporting & Paper Tables

## Generating Result Summaries

After running experiments, use the report tool to rank and compare models.

### Basic Summary

```bash
# Zero-shot experiment
python -m src.report \
    -e data/experiments/zero-shot \
    --rank accuracy \
    --summary-report data/experiments/zero-shot/results_summary.md \
    -v

# 90-10 experiment
python -m src.report \
    -e data/experiments/90-10 \
    --rank accuracy \
    --summary-report data/experiments/90-10/results_summary.md \
    -v
```

### Ranking by Different Metrics

```bash
# Rank by macro F1
python -m src.report \
    -e data/experiments/zero-shot \
    --rank macro_f1

# Rank by adjacent accuracy (within 1 CEFR level)
python -m src.report \
    -e data/experiments/zero-shot \
    --rank adjacent_accuracy

# Rank by average micro-F1 across all test sets
python -m src.report \
    -e data/experiments/zero-shot \
    --rank avg_micro_f1
```

### Filter by Dataset or Strategy

```bash
# Only CELVA-SP results
python -m src.report \
    -e data/experiments/zero-shot \
    --rank accuracy \
    --dataset norm-CELVA-SP

# Only KUPA-KEYS results
python -m src.report \
    -e data/experiments/zero-shot \
    --rank accuracy \
    --dataset norm-KUPA-KEYS

# Only argmax predictions (vs rounded average)
python -m src.report \
    -e data/experiments/zero-shot \
    --rank accuracy \
    --strategy argmax
```

## Evaluation Metrics

The pipeline generates `evaluation_report.md` files for each (model, test-set)
combination containing:

| Metric | Description | Paper Column |
|---|---|---|
| Accuracy | Exact CEFR level match | Reported |
| Adjacent Accuracy | Within 1 level of reference | Within1 |
| Macro F1 | Average F1 across all levels | F1 (macro) |
| Weighted F1 | Class-size-weighted F1 | -- |
| Per-class F1 | F1 per CEFR level | -- |
| Confusion Matrix | Full confusion matrix | -- |

### Paper-Specific Metrics

The paper additionally reports:
- **RMSE** -- Root Mean Square Error (treating CEFR as ordinal 1-6)
- **Spearman rho** -- Rank correlation
- **AC2 (Gwet's)** -- Inter-rater agreement coefficient

These are computed from the prediction JSON files. The `argmax_predictions.json`
and `soft_predictions.json` files contain all needed data.

## Reading Individual Evaluation Reports

```bash
# View a specific evaluation report
cat data/experiments/zero-shot/results/norm-EFCAMDAT-train_xgboost_<hash>/norm-CELVA-SP/evaluation_report.md
```

## Comparing Across Experiments

To compare zero-shot vs. 90-10 results, run the report tool on each experiment
directory and compare the output tables.

## Mapping to Paper Tables

### Table: Zero-Shot Results (KUPA-KEYS, CELVA-SP, EFCAMDAT-test)

For each feature configuration (native, native+general, all-models, tfidf) x
classifier (LR, XGBoost):

1. Run the pipeline with that feature configuration
2. Extract metrics from `evaluation_report.md` for each test set
3. Compute RMSE, Within1, Spearman rho, AC2 from predictions

### Table: 90-10 Results

Same structure but using the 90-10 experiment directory.

### Table: Baseline Comparisons

Baseline results (BERT, zero-shot LLMs) are produced separately:
- **Fine-tuned BERT**: Use `src/train_lm_classifiers.py`
- **Zero-shot LLMs**: Use the prompting experiment (`data/experiments/prompting/`)

## Output Files Reference

| File | Contents |
|---|---|
| `evaluation_report.md` | Human-readable metrics per (model, dataset) |
| `argmax_predictions.json` | Predicted labels using argmax strategy |
| `soft_predictions.json` | Full probability distributions |
| `rounded_avg_predictions.json` | Rounded average predictions |
| `results_summary.md` | Ranked comparison across all models |
