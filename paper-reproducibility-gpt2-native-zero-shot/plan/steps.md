# Execution Steps

## Step 1: Copy data into experiment directory
- Copy CSVs from splits/ into experiment/ml-training-data/ and experiment/ml-test-data/

## Step 2: Extract GPT-2 perplexity features
- Run `python -m src.extract_perplexity_features` for each dataset
- Uses `--aggregate-only -f csv` to get summary statistics per text
- Model: `gpt2` (auto-downloaded from HuggingFace)
- Device: cpu
- Output columns: text, model, mean_perplexity, median_perplexity, std_perplexity, min_perplexity, max_perplexity, mean_entropy, std_entropy, total_tokens

## Step 3: Convert to features_dense.csv
- Strip `text` and `model` columns, keep only numeric columns
- Save as features_dense.csv in experiment/features/gpt2_native/{dataset_name}/

## Step 4: Train classifiers
- LR: `python -m src.train_classifiers -e experiment/ --features-file ... --labels-csv ... --classifier logistic --cefr-column cefr_level`
- XGBoost: same with `--classifier xgboost`

## Step 5: Predict on test sets
- For each (classifier, test_set) pair:
  `python -m src.predict -e experiment/ -m model_name --features-file ... --labels-csv ... --cefr-column cefr_level`

## Step 6: Generate report
- `python -m src.report -e experiment/ --rank accuracy -v`
