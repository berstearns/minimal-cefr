# Language Model Classifier - Quick Start

Fast reference guide for training BERT, GPT-2, and transformer models for CEFR classification.

## 🚀 One-Command Training

```bash
python -m src.train_lm_classifiers train \
    -i data/train.csv \
    --text-column text \
    --target-column cefr_level \
    --model-name bert-base-uncased \
    -o experiments/bert-cefr
```

## 📋 What You Need

### Input CSV Format

```csv
text,cefr_level
"The student demonstrates advanced proficiency.",C1
"Simple sentences with basic vocabulary.",A2
"Complex grammatical structures used correctly.",B2
```

**Required:**
- Text column (default: `text`)
- Label column (default: `label`)

**Supported labels:** A1, A2, B1, B2, C1, C2 (or any string/integer labels)

## 🎯 Common Use Cases

### 1. Train BERT (Recommended)

```bash
python -m src.train_lm_classifiers train \
    -i data/train.csv \
    --text-column text \
    --target-column cefr_level \
    --model-name bert-base-uncased \
    --freeze-base-model \
    -o experiments/bert-cefr
```

### 2. Train GPT-2

```bash
python -m src.train_lm_classifiers train \
    -i data/train.csv \
    --text-column text \
    --target-column cefr_level \
    --model-name gpt2 \
    --freeze-base-model \
    -o experiments/gpt2-cefr
```

### 3. With Validation & Early Stopping

```bash
python -m src.train_lm_classifiers train \
    -i data/train.csv \
    --validation-file data/val.csv \
    --text-column text \
    --target-column cefr_level \
    --model-name bert-base-uncased \
    --early-stopping-patience 3 \
    -o experiments/bert-validated
```

### 4. GPU Training with Mixed Precision

```bash
python -m src.train_lm_classifiers train \
    -i data/train.csv \
    --text-column text \
    --target-column cefr_level \
    --model-name bert-base-uncased \
    --device cuda \
    --fp16 \
    --batch-size 64 \
    -o experiments/bert-gpu
```

### 5. Handle Imbalanced Data

```bash
python -m src.train_lm_classifiers train \
    -i data/train.csv \
    --text-column text \
    --target-column cefr_level \
    --model-name bert-base-uncased \
    --use-class-weights \
    -o experiments/bert-weighted
```

## 🔮 Making Predictions

### Basic Prediction

```bash
python -m src.train_lm_classifiers predict \
    --checkpoint experiments/bert-cefr/checkpoints/best \
    -i data/test.csv \
    --text-column text \
    -o predictions/results.json
```

### With Ground Truth (for Evaluation)

```bash
python -m src.train_lm_classifiers predict \
    --checkpoint experiments/bert-cefr/checkpoints/best \
    -i data/test.csv \
    --text-column text \
    --target-column cefr_level \
    -o predictions/eval_results.json
```

### With Probabilities

```bash
python -m src.train_lm_classifiers predict \
    --checkpoint experiments/bert-cefr/checkpoints/best \
    -i data/test.csv \
    --text-column text \
    --output-probabilities \
    -o predictions/probs.json
```

## 📦 Output Structure

```
experiments/bert-cefr/
├── checkpoints/
│   └── best/                       # ← Use this for predictions
│       ├── pytorch_model.bin
│       └── config.json
├── logs/
│   └── training_log.txt
├── config.json
└── model_card.md
```

## 🎨 Supported Models

| Model | Command | Best For |
|-------|---------|----------|
| BERT | `--model-name bert-base-uncased` | General purpose, balanced |
| DistilBERT | `--model-name distilbert-base-uncased` | Faster, lighter |
| RoBERTa | `--model-name roberta-base` | Higher accuracy |
| GPT-2 | `--model-name gpt2` | Generative context |
| BERT Large | `--model-name bert-large-uncased` | Maximum accuracy |

## ⚙️ Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-epochs` | 3 | Training epochs |
| `--batch-size` | 16 | Batch size |
| `--learning-rate` | 2e-5 | Learning rate |
| `--max-length` | 128 | Max sequence length |
| `--freeze-base-model` | False | Freeze LM weights |
| `--use-class-weights` | False | Balance classes |
| `--device` | cpu | cpu/cuda/mps |
| `--fp16` | False | Mixed precision |

## 🐛 Common Issues

### Out of Memory

```bash
# Reduce batch size
--batch-size 8

# Or freeze base model
--freeze-base-model

# Or use smaller model
--model-name distilbert-base-uncased
```

### Slow Training

```bash
# Use GPU
--device cuda

# Enable mixed precision
--fp16

# Use smaller max length
--max-length 64
```

### Poor Accuracy

```bash
# Add validation with early stopping
--validation-file data/val.csv --early-stopping-patience 3

# Use class weights for imbalanced data
--use-class-weights

# Increase epochs
--num-epochs 10
```

## 📚 Full Documentation

For comprehensive guide, see: [LANGUAGE_MODEL_CLASSIFIER_GUIDE.md](LANGUAGE_MODEL_CLASSIFIER_GUIDE.md)

## 💡 Tips

1. **Start with frozen base model** (`--freeze-base-model`) - faster and uses less memory
2. **Always use validation set** for early stopping
3. **Enable TensorBoard** (`--tensorboard`) to monitor training
4. **Test on small dataset first** to verify your setup
5. **Use GPU with FP16** for production training

## 🔗 Next Steps

After training:

1. **Evaluate predictions**: Compare with TF-IDF baseline
2. **Hyperparameter tuning**: Try different learning rates and batch sizes
3. **Ensemble models**: Combine multiple model predictions
4. **Deploy**: Export best checkpoint for production

---

**Quick command reference:**

```bash
# Train
python -m src.train_lm_classifiers train -i data.csv --text-column text --target-column label --model-name bert-base-uncased -o output/

# Predict
python -m src.train_lm_classifiers predict --checkpoint output/checkpoints/best -i test.csv --text-column text -o results.json

# Help
python -m src.train_lm_classifiers train --help
python -m src.train_lm_classifiers predict --help
```
