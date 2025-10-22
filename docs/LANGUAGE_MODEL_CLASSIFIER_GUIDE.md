# Language Model Classifier Training Guide

Complete guide to training BERT, GPT-2, and other transformer-based classifiers for CEFR text classification.

## Table of Contents

- [Quick Start](#quick-start)
- [Overview](#overview)
- [Installation Requirements](#installation-requirements)
- [Basic Usage](#basic-usage)
- [Input Formats](#input-formats)
- [Supported Models](#supported-models)
- [Training Options](#training-options)
- [Making Predictions](#making-predictions)
- [Advanced Usage](#advanced-usage)
- [Output Structure](#output-structure)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Train BERT Classifier (3 commands)

```bash
# 1. Train model
python -m src.train_lm_classifiers train \
    -i data/train.csv \
    --text-column text \
    --target-column cefr_level \
    --model-name bert-base-uncased \
    -o experiments/bert-cefr

# 2. Make predictions
python -m src.train_lm_classifiers predict \
    --checkpoint experiments/bert-cefr/checkpoints/best \
    -i data/test.csv \
    --text-column text \
    -o predictions/test_results.json

# 3. View results
cat predictions/test_results.json
```

## Overview

The `train_lm_classifiers.py` script provides a streamlined interface for:

1. **Training** pre-trained language models for text classification
2. **Freezing** the base model (training only the classification head)
3. **Making predictions** on new data
4. **Evaluating** model performance with ground truth labels

**Key Features:**
- ✅ Supports BERT, GPT-2, RoBERTa, DistilBERT, and any HuggingFace model
- ✅ Automatic label encoding (handles string labels)
- ✅ Class weight balancing for imbalanced datasets
- ✅ Early stopping with validation
- ✅ Mixed precision training (FP16) for faster training
- ✅ Multi-GPU support
- ✅ TensorBoard logging
- ✅ Comprehensive checkpointing

## Installation Requirements

### Core Dependencies

```bash
pip install transformers torch pandas numpy
```

### Optional Dependencies

```bash
# For YAML config files
pip install pyyaml

# For HuggingFace datasets
pip install datasets

# For GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Verify Installation

```bash
python -c "import transformers; print(transformers.__version__)"
```

## Basic Usage

### Training Subcommand

The `train` subcommand trains a language model classifier:

```bash
python -m src.train_lm_classifiers train \
    -i <input-file> \
    --text-column <column-name> \
    --target-column <label-column> \
    --model-name <model-name> \
    -o <output-dir>
```

### Prediction Subcommand

The `predict` subcommand makes predictions:

```bash
python -m src.train_lm_classifiers predict \
    --checkpoint <checkpoint-dir> \
    -i <input-file> \
    --text-column <column-name> \
    -o <output-file>
```

## Input Formats

### CSV Files

**Training data** (must have text and label columns):

```csv
text,cefr_level
"The student writes fluently.",C1
"Basic sentence structure.",A2
"Complex grammatical patterns.",B2
```

```bash
python -m src.train_lm_classifiers train \
    -i data/train.csv \
    --text-column text \
    --target-column cefr_level \
    --model-name bert-base-uncased \
    -o experiments/bert-classifier
```

**Test data** (labels optional for prediction):

```csv
text
"Evaluate this text."
"Another text sample."
```

```bash
python -m src.train_lm_classifiers predict \
    --checkpoint experiments/bert-classifier/checkpoints/best \
    -i data/test.csv \
    --text-column text \
    -o predictions/results.json
```

### Text Files

Plain text files (one document per line):

```bash
# Training - labels derived from filename or not applicable
python -m src.train_lm_classifiers train \
    -i data/documents.txt \
    --model-name gpt2 \
    -o experiments/gpt2-classifier
```

### HuggingFace Datasets

Load datasets directly from HuggingFace Hub:

```bash
python -m src.train_lm_classifiers train \
    --dataset glue \
    --dataset-config sst2 \
    --dataset-split train \
    --text-column sentence \
    --target-column label \
    --model-name bert-base-uncased \
    -o experiments/bert-sst2
```

## Supported Models

### BERT Models

```bash
# Base BERT
--model-name bert-base-uncased

# Large BERT
--model-name bert-large-uncased

# Multilingual BERT
--model-name bert-base-multilingual-cased
```

### GPT-2 Models

```bash
# Small GPT-2
--model-name gpt2

# Medium GPT-2
--model-name gpt2-medium

# Large GPT-2
--model-name gpt2-large
```

### RoBERTa Models

```bash
# Base RoBERTa
--model-name roberta-base

# Large RoBERTa
--model-name roberta-large
```

### DistilBERT Models

```bash
# DistilBERT (faster, smaller)
--model-name distilbert-base-uncased
```

### Any HuggingFace Model

Any model compatible with `AutoModelForSequenceClassification`:

```bash
--model-name facebook/bart-large
--model-name microsoft/deberta-v3-base
--model-name xlnet-base-cased
```

## Training Options

### Freeze Base Model (Default)

Train only the classification head, keeping the language model frozen:

```bash
python -m src.train_lm_classifiers train \
    -i data/train.csv \
    --text-column text \
    --target-column label \
    --model-name bert-base-uncased \
    --freeze-base-model \
    -o experiments/bert-frozen
```

**Advantages:**
- ⚡ Faster training
- 💾 Lower memory usage
- 🎯 Good for small datasets
- 🔒 Preserves pre-trained knowledge

### Fine-tune Entire Model

Train both the base model and classification head:

```bash
python -m src.train_lm_classifiers train \
    -i data/train.csv \
    --text-column text \
    --target-column label \
    --model-name bert-base-uncased \
    -o experiments/bert-finetuned
```

### With Validation Set

Use a validation set for early stopping:

```bash
python -m src.train_lm_classifiers train \
    -i data/train.csv \
    --validation-file data/val.csv \
    --text-column text \
    --target-column label \
    --model-name bert-base-uncased \
    --early-stopping-patience 3 \
    -o experiments/bert-validated
```

### Custom Hyperparameters

```bash
python -m src.train_lm_classifiers train \
    -i data/train.csv \
    --text-column text \
    --target-column label \
    --model-name bert-base-uncased \
    --num-epochs 10 \
    --batch-size 32 \
    --learning-rate 3e-5 \
    --warmup-steps 500 \
    --weight-decay 0.01 \
    -o experiments/bert-custom
```

### Class Imbalance Handling

Automatically compute class weights:

```bash
python -m src.train_lm_classifiers train \
    -i data/train.csv \
    --text-column text \
    --target-column label \
    --model-name bert-base-uncased \
    --use-class-weights \
    -o experiments/bert-weighted
```

### GPU Training

```bash
# Single GPU with mixed precision
python -m src.train_lm_classifiers train \
    -i data/train.csv \
    --text-column text \
    --target-column label \
    --model-name bert-base-uncased \
    --device cuda \
    --fp16 \
    --batch-size 64 \
    -o experiments/bert-gpu

# Multi-GPU training
python -m src.train_lm_classifiers train \
    -i data/train.csv \
    --text-column text \
    --target-column label \
    --model-name bert-large-uncased \
    --device cuda \
    --n-gpu 2 \
    --batch-size 128 \
    -o experiments/bert-multi-gpu
```

### Configuration Files

#### YAML Config

Create `config.yaml`:

```yaml
lm_config:
  model_name: bert-base-uncased
  num_epochs: 5
  batch_size: 16
  learning_rate: 2e-5
  freeze_base_model: true
  use_class_weights: true

data_config:
  input_file: data/train.csv
  text_column: text
  target_column: cefr_level
  validation_file: data/val.csv

output_config:
  output_dir: experiments/bert-cefr
  verbose: true
  tensorboard: true
```

```bash
python -m src.train_lm_classifiers train -c config.yaml
```

#### JSON Config

Create `config.json`:

```json
{
  "lm_config": {
    "model_name": "bert-base-uncased",
    "num_epochs": 5,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "freeze_base_model": true
  },
  "data_config": {
    "input_file": "data/train.csv",
    "text_column": "text",
    "target_column": "cefr_level"
  },
  "output_config": {
    "output_dir": "experiments/bert-cefr"
  }
}
```

```bash
python -m src.train_lm_classifiers train -c config.json
```

## Making Predictions

### Basic Prediction

```bash
python -m src.train_lm_classifiers predict \
    --checkpoint experiments/bert-classifier/checkpoints/best \
    -i data/test.csv \
    --text-column text \
    -o predictions/test_results.json
```

### With Ground Truth Labels (for Evaluation)

```bash
python -m src.train_lm_classifiers predict \
    --checkpoint experiments/bert-classifier/checkpoints/best \
    -i data/test.csv \
    --text-column text \
    --target-column label \
    -o predictions/test_eval.json
```

This will:
1. Make predictions
2. Compare with ground truth
3. Compute accuracy
4. Mark correct/incorrect predictions
5. Save metrics to `evaluation_metrics.json`

### With Probability Outputs

```bash
python -m src.train_lm_classifiers predict \
    --checkpoint experiments/bert-classifier/checkpoints/best \
    -i data/test.csv \
    --text-column text \
    --output-probabilities \
    -o predictions/test_probs.json
```

Output format:

```json
[
  {
    "index": 0,
    "text": "The student writes fluently.",
    "predicted_label": "C1",
    "predicted_label_id": 4,
    "probabilities": {
      "A1": 0.01,
      "A2": 0.03,
      "B1": 0.08,
      "B2": 0.15,
      "C1": 0.68,
      "C2": 0.05
    }
  }
]
```

### Batch Prediction on GPU

```bash
python -m src.train_lm_classifiers predict \
    --checkpoint experiments/bert-classifier/checkpoints/best \
    -i data/large_test.csv \
    --text-column text \
    --batch-size 128 \
    --device cuda \
    -o predictions/large_results.json
```

## Advanced Usage

### Model Comparison Workflow

Compare BERT vs GPT-2 vs RoBERTa:

```bash
# Train BERT
python -m src.train_lm_classifiers train \
    -i data/train.csv --text-column text --target-column label \
    --model-name bert-base-uncased -o experiments/bert

# Train GPT-2
python -m src.train_lm_classifiers train \
    -i data/train.csv --text-column text --target-column label \
    --model-name gpt2 -o experiments/gpt2

# Train RoBERTa
python -m src.train_lm_classifiers train \
    -i data/train.csv --text-column text --target-column label \
    --model-name roberta-base -o experiments/roberta

# Evaluate all
for model in bert gpt2 roberta; do
    python -m src.train_lm_classifiers predict \
        --checkpoint experiments/${model}/checkpoints/best \
        -i data/test.csv --text-column text --target-column label \
        -o predictions/${model}_results.json
done
```

### Hyperparameter Search

Learning rate ablation:

```bash
for lr in 1e-5 2e-5 3e-5 5e-5; do
    python -m src.train_lm_classifiers train \
        -i data/train.csv --text-column text --target-column label \
        --model-name bert-base-uncased \
        --learning-rate $lr \
        -o experiments/bert_lr_${lr}
done
```

Batch size ablation:

```bash
for bs in 8 16 32 64; do
    python -m src.train_lm_classifiers train \
        -i data/train.csv --text-column text --target-column label \
        --model-name bert-base-uncased \
        --batch-size $bs \
        -o experiments/bert_bs_${bs}
done
```

### TensorBoard Monitoring

Enable TensorBoard during training:

```bash
python -m src.train_lm_classifiers train \
    -i data/train.csv \
    --text-column text \
    --target-column label \
    --model-name bert-base-uncased \
    --tensorboard \
    -o experiments/bert-tb
```

View training progress:

```bash
tensorboard --logdir experiments/bert-tb/logs/tensorboard
```

Open browser to: `http://localhost:6006`

## Output Structure

After training, the output directory contains:

```
experiments/bert-classifier/
├── checkpoints/                    # Model checkpoints
│   ├── checkpoint-100/             # Checkpoint at step 100
│   │   ├── pytorch_model.bin       # Model weights
│   │   ├── config.json             # Model config
│   │   ├── tokenizer_config.json   # Tokenizer config
│   │   ├── vocab.txt               # Vocabulary
│   │   └── training_args.bin       # Training arguments
│   ├── checkpoint-200/
│   │   └── ...
│   └── best/                       # Best checkpoint (loaded for prediction)
│       ├── pytorch_model.bin
│       ├── config.json
│       └── ...
├── logs/
│   ├── training_log.txt            # Complete training log
│   └── tensorboard/                # TensorBoard logs (if enabled)
│       └── events.out.tfevents.*
├── config.json                     # Training configuration
└── model_card.md                   # Model card with details
```

### Model Card Example

```markdown
# Language Model Classifier

## Model: bert-base-uncased

## Training Details

- Number of classes: 6
- Training examples: 5000
- Validation examples: 1000
- Epochs: 5
- Batch size: 16
- Learning rate: 2e-05
- Frozen base model: True

## Label Mapping

- 0: A1
- 1: A2
- 2: B1
- 3: B2
- 4: C1
- 5: C2
```

## Troubleshooting

### Out of Memory (OOM)

**Problem:** `RuntimeError: CUDA out of memory`

**Solutions:**

```bash
# 1. Reduce batch size
--batch-size 8

# 2. Use gradient accumulation
--batch-size 8 --gradient-accumulation-steps 4

# 3. Freeze base model
--freeze-base-model

# 4. Use smaller model
--model-name distilbert-base-uncased

# 5. Reduce sequence length
--max-length 64
```

### Slow Training

**Problem:** Training is too slow

**Solutions:**

```bash
# 1. Use GPU
--device cuda

# 2. Enable mixed precision
--fp16

# 3. Increase batch size
--batch-size 64

# 4. Use smaller model
--model-name distilbert-base-uncased

# 5. Reduce max length
--max-length 64
```

### Poor Performance

**Problem:** Low accuracy on test set

**Solutions:**

```bash
# 1. Add validation set with early stopping
--validation-file data/val.csv --early-stopping-patience 3

# 2. Increase epochs
--num-epochs 10

# 3. Adjust learning rate
--learning-rate 3e-5

# 4. Use class weights for imbalanced data
--use-class-weights

# 5. Try different model
--model-name roberta-base

# 6. Fine-tune entire model (not just classification head)
# Remove --freeze-base-model flag
```

### Label Encoding Issues

**Problem:** `ValueError: labels must be in range`

**Solution:** The script automatically encodes string labels to integers. Check that:

1. Labels are consistent across train/validation/test
2. All labels in test set exist in training set
3. No missing or NaN values in label column

### GPU Not Detected

**Problem:** `torch.cuda.is_available()` returns False

**Solutions:**

```bash
# 1. Install CUDA-compatible PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. Verify CUDA installation
nvidia-smi

# 3. Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"
```

## Best Practices

### 1. Start Small

Test your workflow with a small dataset first:

```bash
# Use first 100 examples
head -101 data/train.csv > data/train_small.csv

python -m src.train_lm_classifiers train \
    -i data/train_small.csv \
    --text-column text \
    --target-column label \
    --model-name distilbert-base-uncased \
    --num-epochs 2 \
    --batch-size 8 \
    -o experiments/test-run
```

### 2. Use Validation for Early Stopping

Always use a validation set:

```bash
python -m src.train_lm_classifiers train \
    -i data/train.csv \
    --validation-file data/val.csv \
    --text-column text \
    --target-column label \
    --model-name bert-base-uncased \
    --early-stopping-patience 3 \
    -o experiments/bert-validated
```

### 3. Save Checkpoints Frequently

For long training runs:

```bash
--save-steps 500 --save-total-limit 5 --eval-steps 100
```

### 4. Monitor with TensorBoard

Enable TensorBoard for real-time monitoring:

```bash
--tensorboard
```

### 5. Reproducibility

Set a random seed:

```bash
--seed 42
```

## Performance Expectations

### Training Time (on GPU)

| Model | Dataset Size | Epochs | GPU | Time |
|-------|-------------|--------|-----|------|
| DistilBERT | 5K | 3 | V100 | ~5 min |
| BERT-base | 5K | 3 | V100 | ~10 min |
| BERT-large | 5K | 3 | V100 | ~20 min |
| GPT-2 | 5K | 3 | V100 | ~8 min |
| RoBERTa-base | 5K | 3 | V100 | ~12 min |

### Memory Requirements

| Model | Batch Size | GPU Memory |
|-------|-----------|------------|
| DistilBERT | 32 | ~4 GB |
| BERT-base | 32 | ~6 GB |
| BERT-large | 16 | ~10 GB |
| GPT-2 | 32 | ~5 GB |
| RoBERTa-base | 32 | ~6 GB |

## Next Steps

- **Pipeline Integration**: Use with the main pipeline in `src.pipeline`
- **Feature Comparison**: Compare with TF-IDF classifiers from `src.train_classifiers`
- **Ensemble Methods**: Combine predictions from multiple models
- **Model Deployment**: Export models for production use
- **Advanced Fine-tuning**: Explore layer-wise learning rates and adapter methods

## Related Documentation

- [Main Pipeline Guide](USAGE.md)
- [TF-IDF Classifier Training](QUICK_START.md)
- [Project Structure](STRUCTURE_VALIDATION.md)
- [Hyperparameter Optimization](ho_multifeat_usage.md)
