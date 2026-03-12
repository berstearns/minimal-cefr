# GPT-2 Native Perplexity Zero-Shot (Colab Version)

Google Colab version of the GPT-2 native perplexity zero-shot CEFR classification experiment.

## Quick Start

1. Open `gpt2_native_zero_shot_colab.ipynb` in Google Colab
2. Set runtime to **T4 GPU** (`Runtime > Change runtime type`)
3. Run cells in order, choosing **one** data acquisition option (A or B)

## What It Does

Extracts aggregate perplexity features from pre-trained GPT-2, trains Logistic Regression + XGBoost classifiers, and evaluates on 3 test sets.

| Step | Description | Runtime (T4 GPU) |
|------|-------------|------------------|
| 1+2 | Extract perplexity + convert features | ~15 min |
| 3 | Train LR + XGBoost | ~10 sec |
| 4 | Predict on 3 test sets | ~10 sec |
| 5 | Generate report | ~1 sec |

## Data Acquisition

**Option A - Google Drive:** Upload your CSVs to `MyDrive/phd-experimental-data/data/splits/` before running.

**Option B - wget:** Paste a direct download URL to a .zip containing the CSVs.

See [DATA_SETUP.md](DATA_SETUP.md) for details on required files and format.

## Row Limits

Default limits for quick validation (~20 min on GPU):

| Dataset | Full rows | Default limit |
|---------|-----------|---------------|
| norm-EFCAMDAT-train | 80,000 | 2,000 |
| norm-EFCAMDAT-test | 20,000 | 500 |
| norm-CELVA-SP | 1,742 | None (full) |
| norm-KUPA-KEYS | 1,006 | None (full) |

Set any limit to `None` in the Configuration cell for full paper reproduction.

## Repository Setup

The notebook needs the `minimal-cefr` `src/` package. Options:
- **Public repo:** Edit the `git clone` URL in the notebook
- **Private repo:** Upload `src/` as a zip file (instructions in notebook)

## Differences from Local Version

- GPU auto-detection (`cuda` if available, falls back to `cpu`)
- Data comes from Google Drive or wget instead of local paths
- Experiment outputs go to `/content/experiment/` instead of a subdirectory
- Results displayed inline as rendered Markdown
- Optional download of results as .zip
