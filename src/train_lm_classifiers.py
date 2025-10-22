"""
Language Model Fine-tuning for Text Classification (Classification Head Only)

This module fine-tunes pre-trained language models (BERT, GPT-2) for text classification
by training only the classification head while keeping the language model frozen.

Expected File Structure
-----------------------
INPUT:
For CSV input:
data/
└── train.csv                   # Must have text_column and target_column

For text file input:
data/
└── documents.txt               # Plain text, one document per line

For HuggingFace dataset:
--dataset <name>                # Dataset from HuggingFace Hub

OUTPUT:
output-dir/
├── checkpoints/                # Model checkpoints during training
│   ├── checkpoint-100/
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   ├── tokenizer_config.json
│   │   └── training_args.bin
│   ├── checkpoint-200/
│   │   └── ...
│   └── best/                   # Best checkpoint based on validation metric
│       └── ...
├── logs/
│   ├── training_log.txt        # Training progress logs
│   └── tensorboard/            # TensorBoard logs (optional)
│       └── events.out.tfevents.*
├── predictions/                # Predictions from predict subcommand
│   ├── predictions.json
│   └── evaluation_metrics.json
├── config.json                 # Complete training configuration
└── model_card.md               # Model card with training details

Supported Models
----------------
- bert: BERT-based models (bert-base-uncased, bert-large-uncased, etc.)
- gpt2: GPT-2 models (gpt2, gpt2-medium, gpt2-large, etc.)
- distilbert: DistilBERT models
- roberta: RoBERTa models
- Any HuggingFace model with AutoModelForSequenceClassification support

COMBINATORIC USAGE PATTERNS
============================

The script supports various combinations of:
1. Input formats (CSV, text file, HuggingFace dataset)
2. Model types (BERT, GPT-2, RoBERTa, etc.)
3. Configuration methods (CLI args, JSON/YAML config files)
4. Subcommands (train, predict)

USAGE EXAMPLES - TRAIN SUBCOMMAND
==================================

Basic Training
--------------
1. Train BERT on CSV file:
   $ python -m src.train_lm_classifiers train \\
       -i data/train.csv \\
       --text-column text \\
       --target-column cefr_level \\
       --model-name bert-base-uncased \\
       -o experiments/bert-classifier

2. Train GPT-2 on CSV file:
   $ python -m src.train_lm_classifiers train \\
       -i data/train.csv \\
       --text-column essay \\
       --target-column level \\
       --model-name gpt2 \\
       -o experiments/gpt2-classifier

3. Train on text file (unsupervised - will use file as label):
   $ python -m src.train_lm_classifiers train \\
       -i data/documents.txt \\
       --model-name distilbert-base-uncased \\
       -o experiments/distilbert-classifier

HuggingFace Dataset
-------------------
4. Train on HuggingFace dataset:
   $ python -m src.train_lm_classifiers train \\
       --dataset glue \\
       --dataset-config sst2 \\
       --dataset-split train \\
       --text-column sentence \\
       --target-column label \\
       --model-name bert-base-uncased \\
       -o experiments/bert-sst2

5. Train with validation split:
   $ python -m src.train_lm_classifiers train \\
       -i data/train.csv \\
       --text-column text \\
       --target-column label \\
       --validation-file data/val.csv \\
       --model-name roberta-base \\
       -o experiments/roberta-classifier

Custom Training Configuration
------------------------------
6. Custom hyperparameters:
   $ python -m src.train_lm_classifiers train \\
       -i data/train.csv \\
       --text-column text \\
       --target-column cefr_level \\
       --model-name bert-base-uncased \\
       --num-epochs 10 \\
       --batch-size 16 \\
       --learning-rate 2e-5 \\
       --warmup-steps 500 \\
       --weight-decay 0.01 \\
       -o experiments/bert-custom

7. With validation and early stopping:
   $ python -m src.train_lm_classifiers train \\
       -i data/train.csv \\
       --text-column text \\
       --target-column label \\
       --validation-file data/val.csv \\
       --model-name bert-base-uncased \\
       --early-stopping-patience 3 \\
       --eval-steps 100 \\
       -o experiments/bert-early-stop

8. Multi-class with class weights:
   $ python -m src.train_lm_classifiers train \\
       -i data/train.csv \\
       --text-column text \\
       --target-column cefr_level \\
       --model-name bert-base-uncased \\
       --use-class-weights \\
       --num-labels 6 \\
       -o experiments/bert-weighted

GPU and Performance
-------------------
9. Train on GPU with mixed precision:
   $ python -m src.train_lm_classifiers train \\
       -i data/train.csv \\
       --text-column text \\
       --target-column label \\
       --model-name bert-base-uncased \\
       --device cuda \\
       --fp16 \\
       --batch-size 32 \\
       -o experiments/bert-gpu

10. Multi-GPU training:
    $ python -m src.train_lm_classifiers train \\
        -i data/train.csv \\
        --text-column text \\
        --target-column label \\
        --model-name bert-large-uncased \\
        --device cuda \\
        --n-gpu 2 \\
        --batch-size 64 \\
        -o experiments/bert-multi-gpu

Configuration Files
-------------------
11. Using JSON config file:
    $ python -m src.train_lm_classifiers train \\
        -c config.json \\
        -o experiments/from-config

    config.json:
    {
      "input_file": "data/train.csv",
      "text_column": "text",
      "target_column": "cefr_level",
      "model_name": "bert-base-uncased",
      "num_epochs": 5,
      "batch_size": 16,
      "learning_rate": 2e-5
    }

12. Using YAML config with CLI overrides:
    $ python -m src.train_lm_classifiers train \\
        -c config.yaml \\
        --num-epochs 10 \\
        -o experiments/from-yaml

Advanced Options
----------------
13. Custom tokenizer settings:
    $ python -m src.train_lm_classifiers train \\
        -i data/train.csv \\
        --text-column text \\
        --target-column label \\
        --model-name bert-base-uncased \\
        --max-length 256 \\
        --truncation \\
        --padding max_length \\
        -o experiments/bert-custom-tokenizer

14. With logging and checkpointing:
    $ python -m src.train_lm_classifiers train \\
        -i data/train.csv \\
        --text-column text \\
        --target-column label \\
        --model-name gpt2 \\
        --save-steps 500 \\
        --save-total-limit 3 \\
        --logging-steps 50 \\
        --tensorboard \\
        -o experiments/gpt2-logged

15. Freeze all layers except classification head:
    $ python -m src.train_lm_classifiers train \\
        -i data/train.csv \\
        --text-column text \\
        --target-column label \\
        --model-name bert-base-uncased \\
        --freeze-base-model \\
        -o experiments/bert-frozen

USAGE EXAMPLES - PREDICT SUBCOMMAND
====================================

Basic Prediction
----------------
16. Predict on CSV file:
    $ python -m src.train_lm_classifiers predict \\
        --checkpoint experiments/bert-classifier/checkpoints/best \\
        -i data/test.csv \\
        --text-column text \\
        -o predictions/test_predictions.json

17. Predict with ground truth labels (for evaluation):
    $ python -m src.train_lm_classifiers predict \\
        --checkpoint experiments/bert-classifier/checkpoints/best \\
        -i data/test.csv \\
        --text-column text \\
        --target-column label \\
        -o predictions/test_eval.json

18. Predict on text file:
    $ python -m src.train_lm_classifiers predict \\
        --checkpoint experiments/gpt2-classifier/checkpoints/best \\
        -i data/documents.txt \\
        -o predictions/documents_predictions.json

19. Predict with probabilities:
    $ python -m src.train_lm_classifiers predict \\
        --checkpoint experiments/bert-classifier/checkpoints/best \\
        -i data/test.csv \\
        --text-column text \\
        --output-probabilities \\
        -o predictions/test_probs.json

20. Batch prediction with GPU:
    $ python -m src.train_lm_classifiers predict \\
        --checkpoint experiments/bert-classifier/checkpoints/best \\
        -i data/large_test.csv \\
        --text-column text \\
        --batch-size 64 \\
        --device cuda \\
        -o predictions/large_test_predictions.json

SYSTEMATIC EXPERIMENTATION
===========================

Model Comparison
----------------
21. Compare BERT vs GPT-2:
    # Train BERT
    $ python -m src.train_lm_classifiers train \\
        -i data/train.csv --text-column text --target-column label \\
        --model-name bert-base-uncased -o experiments/bert

    # Train GPT-2
    $ python -m src.train_lm_classifiers train \\
        -i data/train.csv --text-column text --target-column label \\
        --model-name gpt2 -o experiments/gpt2

    # Predict with both
    $ python -m src.train_lm_classifiers predict \\
        --checkpoint experiments/bert/checkpoints/best \\
        -i data/test.csv --text-column text --target-column label \\
        -o predictions/bert_results.json

    $ python -m src.train_lm_classifiers predict \\
        --checkpoint experiments/gpt2/checkpoints/best \\
        -i data/test.csv --text-column text --target-column label \\
        -o predictions/gpt2_results.json

Hyperparameter Search
----------------------
22. Learning rate comparison:
    for lr in 1e-5 2e-5 3e-5 5e-5; do
        python -m src.train_lm_classifiers train \\
            -i data/train.csv --text-column text --target-column label \\
            --model-name bert-base-uncased \\
            --learning-rate $lr \\
            -o experiments/bert_lr_${lr}
    done

23. Batch size ablation:
    for bs in 8 16 32 64; do
        python -m src.train_lm_classifiers train \\
            -i data/train.csv --text-column text --target-column label \\
            --model-name bert-base-uncased \\
            --batch-size $bs \\
            -o experiments/bert_bs_${bs}
    done

TIPS AND BEST PRACTICES
========================

1. **Memory Management**: For large models (bert-large, gpt2-large), use smaller
   batch sizes or gradient accumulation:
   --batch-size 8 --gradient-accumulation-steps 4

2. **Learning Rate**: Typical ranges:
   - BERT: 2e-5 to 5e-5
   - GPT-2: 1e-5 to 3e-5
   - RoBERTa: 1e-5 to 3e-5

3. **Early Stopping**: Use validation file and early stopping for best results:
   --validation-file data/val.csv --early-stopping-patience 3

4. **Checkpointing**: Save regularly to avoid losing progress:
   --save-steps 500 --save-total-limit 5

5. **Freezing Base Model**: For quick experiments, freeze the base model:
   --freeze-base-model (trains only classification head)

6. **Mixed Precision**: Speed up training on modern GPUs:
   --fp16 (requires CUDA-compatible GPU)

7. **Class Imbalance**: For imbalanced datasets:
   --use-class-weights

8. **Reproducibility**: Set random seed:
   --seed 42
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class LMClassifierConfig:
    """Configuration for language model classification training."""

    # Model settings
    model_name: str = "bert-base-uncased"
    num_labels: Optional[int] = None  # Auto-detected from data if None
    freeze_base_model: bool = True  # Only train classification head

    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_steps: int = 0
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1

    # Tokenizer settings
    max_length: int = 128
    truncation: bool = True
    padding: str = "max_length"

    # Training options
    use_class_weights: bool = False
    early_stopping_patience: Optional[int] = None
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 100

    # Device settings
    device: str = "cpu"  # cpu, cuda, mps
    fp16: bool = False  # Mixed precision training
    n_gpu: int = 1

    # Reproducibility
    seed: int = 42

    # Output settings
    tensorboard: bool = False
    verbose: bool = True


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""

    input_file: Optional[str] = None
    text_column: str = "text"
    target_column: str = "label"
    validation_file: Optional[str] = None

    # HuggingFace dataset settings
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    dataset_split: str = "train"
    validation_split: Optional[str] = None


@dataclass
class OutputConfig:
    """Configuration for output and logging."""

    output_dir: str = "experiments/lm-classifier"
    save_config: bool = True
    save_model_card: bool = True
    verbose: bool = True


@dataclass
class GlobalLMConfig:
    """Global configuration combining all settings."""

    lm_config: LMClassifierConfig
    data_config: DataConfig
    output_config: OutputConfig

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            "lm_config": {
                k: v
                for k, v in self.lm_config.__dict__.items()
                if not k.startswith("_")
            },
            "data_config": {
                k: v
                for k, v in self.data_config.__dict__.items()
                if not k.startswith("_")
            },
            "output_config": {
                k: v
                for k, v in self.output_config.__dict__.items()
                if not k.startswith("_")
            },
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "GlobalLMConfig":
        """Create config from dictionary."""
        return cls(
            lm_config=LMClassifierConfig(**config_dict.get("lm_config", {})),
            data_config=DataConfig(**config_dict.get("data_config", {})),
            output_config=OutputConfig(**config_dict.get("output_config", {})),
        )

    @classmethod
    def from_json_file(cls, json_path: str) -> "GlobalLMConfig":
        """Load configuration from JSON file."""
        with open(json_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_yaml_file(cls, yaml_path: str) -> "GlobalLMConfig":
        """Load configuration from YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def save(self, output_path: str):
        """Save configuration to JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class TextClassificationDataset(Dataset):
    """Custom dataset for text classification."""

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer=None,
        max_length: int = 128,
        truncation: bool = True,
        padding: str = "max_length",
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            return_tensors="pt",
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def load_data_from_csv(
    csv_path: str, text_column: str, target_column: Optional[str] = None
) -> Tuple[List[str], Optional[List[str]]]:
    """Load data from CSV file."""
    df = pd.read_csv(csv_path)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in {csv_path}")

    texts = df[text_column].fillna("").astype(str).tolist()

    labels = None
    if target_column:
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in {csv_path}")
        labels = df[target_column].tolist()

    return texts, labels


def load_data_from_txt(txt_path: str) -> Tuple[List[str], None]:
    """Load data from text file (one document per line)."""
    with open(txt_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    return texts, None


def load_data_from_huggingface(
    dataset_name: str,
    dataset_config: Optional[str],
    dataset_split: str,
    text_column: str,
    target_column: Optional[str] = None,
) -> Tuple[List[str], Optional[List]]:
    """Load data from HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library not installed. Install with: pip install datasets"
        )

    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
    else:
        dataset = load_dataset(dataset_name, split=dataset_split)

    if text_column not in dataset.column_names:
        raise ValueError(f"Column '{text_column}' not found in dataset")

    texts = [str(example[text_column]) for example in dataset]

    labels = None
    if target_column:
        if target_column not in dataset.column_names:
            raise ValueError(f"Column '{target_column}' not found in dataset")
        labels = [example[target_column] for example in dataset]

    return texts, labels


def encode_labels(labels: List[Union[str, int]]) -> Tuple[List[int], Dict[str, int]]:
    """Encode string labels to integers."""
    unique_labels = sorted(set(labels))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_labels = [label2id[label] for label in labels]
    return encoded_labels, label2id


def compute_class_weights(labels: List[int], num_labels: int) -> torch.Tensor:
    """Compute class weights for imbalanced datasets."""
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.arange(num_labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float)


def train_lm_classifier(config: GlobalLMConfig) -> str:
    """
    Train a language model classifier.

    Args:
        config: GlobalLMConfig containing all settings

    Returns:
        Path to best checkpoint
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers not installed. Install with: pip install transformers torch"
        )

    lm_config = config.lm_config
    data_config = config.data_config
    output_config = config.output_config

    # Set random seed for reproducibility
    torch.manual_seed(lm_config.seed)
    np.random.seed(lm_config.seed)

    # Create output directories
    output_dir = Path(output_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Setup file logging
    file_handler = logging.FileHandler(logs_dir / "training_log.txt")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    logger.info("=" * 80)
    logger.info("LANGUAGE MODEL CLASSIFIER TRAINING")
    logger.info("=" * 80)
    logger.info(f"Model: {lm_config.model_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {lm_config.device}")
    logger.info("=" * 80)

    # Load data
    logger.info("\nLoading data...")
    if data_config.dataset_name:
        texts, labels = load_data_from_huggingface(
            data_config.dataset_name,
            data_config.dataset_config,
            data_config.dataset_split,
            data_config.text_column,
            data_config.target_column,
        )
        val_texts, val_labels = None, None
        if data_config.validation_split:
            val_texts, val_labels = load_data_from_huggingface(
                data_config.dataset_name,
                data_config.dataset_config,
                data_config.validation_split,
                data_config.text_column,
                data_config.target_column,
            )
    elif data_config.input_file:
        input_path = Path(data_config.input_file)
        if input_path.suffix == ".csv":
            texts, labels = load_data_from_csv(
                str(input_path), data_config.text_column, data_config.target_column
            )
        elif input_path.suffix == ".txt":
            texts, labels = load_data_from_txt(str(input_path))
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")

        val_texts, val_labels = None, None
        if data_config.validation_file:
            val_path = Path(data_config.validation_file)
            if val_path.suffix == ".csv":
                val_texts, val_labels = load_data_from_csv(
                    str(val_path), data_config.text_column, data_config.target_column
                )
            elif val_path.suffix == ".txt":
                val_texts, val_labels = load_data_from_txt(str(val_path))
    else:
        raise ValueError("Must provide either input_file or dataset_name")

    if labels is None:
        raise ValueError("Labels are required for training")

    logger.info(f"Loaded {len(texts)} training examples")
    if val_texts:
        logger.info(f"Loaded {len(val_texts)} validation examples")

    # Encode labels
    logger.info("\nEncoding labels...")
    if isinstance(labels[0], str):
        labels_encoded, label2id = encode_labels(labels)
        id2label = {idx: label for label, idx in label2id.items()}
        if val_labels and isinstance(val_labels[0], str):
            val_labels_encoded = [label2id[label] for label in val_labels]
        else:
            val_labels_encoded = val_labels
    else:
        labels_encoded = labels
        num_labels = len(set(labels))
        label2id = {i: i for i in range(num_labels)}
        id2label = {i: i for i in range(num_labels)}
        val_labels_encoded = val_labels

    num_labels = len(label2id)
    logger.info(f"Number of classes: {num_labels}")
    logger.info(f"Label mapping: {label2id}")

    # Load tokenizer and model
    logger.info(f"\nLoading model: {lm_config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(lm_config.model_name)

    # Add padding token if missing (e.g., for GPT-2)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(
        lm_config.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        lm_config.model_name, config=model_config
    )

    # Freeze base model if requested
    if lm_config.freeze_base_model:
        logger.info(
            "Freezing base model parameters (training only classification head)"
        )
        for name, param in model.named_parameters():
            if "classifier" not in name and "score" not in name:
                param.requires_grad = False

    # Create datasets
    logger.info("\nCreating datasets...")
    train_dataset = TextClassificationDataset(
        texts,
        labels_encoded,
        tokenizer,
        lm_config.max_length,
        lm_config.truncation,
        lm_config.padding,
    )

    eval_dataset = None
    if val_texts:
        eval_dataset = TextClassificationDataset(
            val_texts,
            val_labels_encoded,
            tokenizer,
            lm_config.max_length,
            lm_config.truncation,
            lm_config.padding,
        )

    # Compute class weights if requested
    class_weights = None
    if lm_config.use_class_weights:
        logger.info("Computing class weights for imbalanced data...")
        class_weights = compute_class_weights(labels_encoded, num_labels)
        logger.info(f"Class weights: {class_weights.tolist()}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        num_train_epochs=lm_config.num_epochs,
        per_device_train_batch_size=lm_config.batch_size,
        per_device_eval_batch_size=lm_config.batch_size,
        learning_rate=lm_config.learning_rate,
        warmup_steps=lm_config.warmup_steps,
        weight_decay=lm_config.weight_decay,
        gradient_accumulation_steps=lm_config.gradient_accumulation_steps,
        logging_dir=str(logs_dir / "tensorboard") if lm_config.tensorboard else None,
        logging_steps=lm_config.logging_steps,
        save_steps=lm_config.save_steps,
        save_total_limit=lm_config.save_total_limit,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=lm_config.eval_steps if eval_dataset else None,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="accuracy" if eval_dataset else None,
        fp16=lm_config.fp16,
        seed=lm_config.seed,
        report_to="tensorboard" if lm_config.tensorboard else "none",
    )

    # Custom trainer for class weights
    class WeightedTrainer(Trainer):
        def __init__(self, *args, class_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            if self.class_weights is not None:
                loss_fct = torch.nn.CrossEntropyLoss(
                    weight=self.class_weights.to(logits.device)
                )
                loss = loss_fct(logits, labels)
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)

            return (loss, outputs) if return_outputs else loss

    # Metrics computation
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    # Create trainer
    trainer_cls = WeightedTrainer if class_weights is not None else Trainer
    trainer_kwargs = (
        {"class_weights": class_weights} if class_weights is not None else {}
    )

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics if eval_dataset else None,
        **trainer_kwargs,
    )

    # Train
    logger.info("\nStarting training...")
    logger.info(f"Total epochs: {lm_config.num_epochs}")
    logger.info(f"Batch size: {lm_config.batch_size}")
    logger.info(f"Learning rate: {lm_config.learning_rate}")
    logger.info("=" * 80)

    train_result = trainer.train()

    logger.info("\nTraining completed!")
    logger.info(f"Training loss: {train_result.training_loss:.4f}")

    # Save best model
    best_checkpoint_dir = output_dir / "checkpoints" / "best"
    trainer.save_model(str(best_checkpoint_dir))
    tokenizer.save_pretrained(str(best_checkpoint_dir))

    logger.info(f"\nBest model saved to: {best_checkpoint_dir}")

    # Save configuration
    if output_config.save_config:
        config_path = output_dir / "config.json"
        config.save(str(config_path))
        logger.info(f"Configuration saved to: {config_path}")

    # Save model card
    if output_config.save_model_card:
        model_card_path = output_dir / "model_card.md"
        with open(model_card_path, "w") as f:
            f.write("# Language Model Classifier\n\n")
            f.write(f"## Model: {lm_config.model_name}\n\n")
            f.write("## Training Details\n\n")
            f.write(f"- Number of classes: {num_labels}\n")
            f.write(f"- Training examples: {len(texts)}\n")
            if val_texts:
                f.write(f"- Validation examples: {len(val_texts)}\n")
            f.write(f"- Epochs: {lm_config.num_epochs}\n")
            f.write(f"- Batch size: {lm_config.batch_size}\n")
            f.write(f"- Learning rate: {lm_config.learning_rate}\n")
            f.write(f"- Frozen base model: {lm_config.freeze_base_model}\n")
            f.write("\n## Label Mapping\n\n")
            for label, idx in sorted(label2id.items(), key=lambda x: x[1]):
                f.write(f"- {idx}: {label}\n")
        logger.info(f"Model card saved to: {model_card_path}")

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

    return str(best_checkpoint_dir)


def predict_lm_classifier(
    checkpoint_dir: str,
    input_file: str,
    text_column: str,
    target_column: Optional[str],
    output_file: str,
    batch_size: int = 32,
    device: str = "cpu",
    output_probabilities: bool = False,
) -> str:
    """
    Make predictions using trained classifier.

    Args:
        checkpoint_dir: Path to model checkpoint
        input_file: Path to input file (CSV or TXT)
        text_column: Column name for text
        target_column: Column name for labels (optional, for evaluation)
        output_file: Path to save predictions
        batch_size: Batch size for prediction
        device: Device to use (cpu, cuda)
        output_probabilities: Whether to output class probabilities

    Returns:
        Path to predictions file
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers not installed. Install with: pip install transformers torch"
        )

    logger.info("=" * 80)
    logger.info("MAKING PREDICTIONS")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {checkpoint_dir}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Device: {device}")
    logger.info("=" * 80)

    # Load model and tokenizer
    logger.info("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.to(device)
    model.eval()

    # Load data
    logger.info("Loading data...")
    input_path = Path(input_file)
    if input_path.suffix == ".csv":
        texts, labels = load_data_from_csv(str(input_path), text_column, target_column)
    elif input_path.suffix == ".txt":
        texts, labels = load_data_from_txt(str(input_path))
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    logger.info(f"Loaded {len(texts)} examples")

    # Create dataset
    dataset = TextClassificationDataset(
        texts, None, tokenizer, max_length=128, truncation=True, padding="max_length"
    )

    # Predict
    logger.info("\nMaking predictions...")
    all_predictions = []
    all_probabilities = []

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            logits = outputs.logits

            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            all_predictions.extend(predictions.tolist())

            if output_probabilities:
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probabilities.extend(probabilities.tolist())

    # Get label names
    id2label = model.config.id2label

    # Create results
    results = []
    for i, (text, pred) in enumerate(zip(texts, all_predictions)):
        result = {
            "index": i,
            "text": text,
            "predicted_label": id2label[pred],
            "predicted_label_id": pred,
        }

        if labels:
            result["true_label"] = labels[i]
            result["correct"] = str(labels[i]) == str(id2label[pred])

        if output_probabilities:
            result["probabilities"] = {
                id2label[j]: float(prob) for j, prob in enumerate(all_probabilities[i])
            }

        results.append(result)

    # Save predictions
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nPredictions saved to: {output_path}")

    # Compute metrics if labels available
    if labels:
        accuracy = sum(r["correct"] for r in results) / len(results)
        logger.info(f"\nAccuracy: {accuracy:.4f}")

        # Save metrics
        metrics_path = output_path.parent / "evaluation_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({"accuracy": accuracy, "num_examples": len(results)}, f, indent=2)
        logger.info(f"Metrics saved to: {metrics_path}")

    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION COMPLETED")
    logger.info("=" * 80)

    return str(output_path)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="Language Model Fine-tuning for Text Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train classifier")

    # Configuration loading
    config_group = train_parser.add_argument_group("Configuration")
    config_group.add_argument(
        "-c", "--config-file", help="Path to JSON/YAML config file"
    )

    # Input data
    data_group = train_parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "-i", "--input-file", help="Path to training data (CSV or TXT)"
    )
    data_group.add_argument("--text-column", default="text", help="Text column name")
    data_group.add_argument(
        "--target-column", default="label", help="Target column name"
    )
    data_group.add_argument("--validation-file", help="Path to validation data")

    # HuggingFace dataset
    data_group.add_argument("--dataset", help="HuggingFace dataset name")
    data_group.add_argument("--dataset-config", help="Dataset configuration")
    data_group.add_argument("--dataset-split", default="train", help="Dataset split")
    data_group.add_argument("--validation-split", help="Validation split")

    # Model configuration
    model_group = train_parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model-name", default="bert-base-uncased", help="Model name or path"
    )
    model_group.add_argument("--num-labels", type=int, help="Number of labels")
    model_group.add_argument(
        "--freeze-base-model",
        action="store_true",
        help="Freeze base model (train only classification head)",
    )

    # Training hyperparameters
    train_group = train_parser.add_argument_group("Training Hyperparameters")
    train_group.add_argument(
        "--num-epochs", type=int, default=3, help="Number of epochs"
    )
    train_group.add_argument("--batch-size", type=int, default=16, help="Batch size")
    train_group.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate"
    )
    train_group.add_argument("--warmup-steps", type=int, default=0, help="Warmup steps")
    train_group.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay"
    )
    train_group.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )

    # Tokenizer settings
    tokenizer_group = train_parser.add_argument_group("Tokenizer Settings")
    tokenizer_group.add_argument(
        "--max-length", type=int, default=128, help="Max sequence length"
    )
    tokenizer_group.add_argument(
        "--truncation", action="store_true", default=True, help="Enable truncation"
    )
    tokenizer_group.add_argument(
        "--padding", default="max_length", help="Padding strategy"
    )

    # Training options
    options_group = train_parser.add_argument_group("Training Options")
    options_group.add_argument(
        "--use-class-weights", action="store_true", help="Use class weights"
    )
    options_group.add_argument(
        "--early-stopping-patience", type=int, help="Early stopping patience"
    )
    options_group.add_argument(
        "--eval-steps", type=int, default=500, help="Evaluation steps"
    )
    options_group.add_argument("--save-steps", type=int, default=500, help="Save steps")
    options_group.add_argument(
        "--save-total-limit", type=int, default=3, help="Max checkpoints to keep"
    )
    options_group.add_argument(
        "--logging-steps", type=int, default=100, help="Logging steps"
    )

    # Device settings
    device_group = train_parser.add_argument_group("Device Settings")
    device_group.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "mps"], help="Device"
    )
    device_group.add_argument("--fp16", action="store_true", help="Use mixed precision")
    device_group.add_argument("--n-gpu", type=int, default=1, help="Number of GPUs")

    # Reproducibility
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output
    output_group = train_parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "-o",
        "--output-dir",
        default="experiments/lm-classifier",
        help="Output directory",
    )
    output_group.add_argument(
        "--tensorboard", action="store_true", help="Use TensorBoard"
    )
    output_group.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")

    # Predict subcommand
    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint"
    )
    predict_parser.add_argument(
        "-i", "--input-file", required=True, help="Path to input file"
    )
    predict_parser.add_argument(
        "--text-column", default="text", help="Text column name"
    )
    predict_parser.add_argument(
        "--target-column", help="Target column name (for evaluation)"
    )
    predict_parser.add_argument(
        "-o", "--output-file", required=True, help="Path to save predictions"
    )
    predict_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    predict_parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "mps"], help="Device"
    )
    predict_parser.add_argument(
        "--output-probabilities", action="store_true", help="Output class probabilities"
    )

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "train":
        # Build configuration
        if args.config_file:
            config_path = Path(args.config_file)
            if config_path.suffix in [".yaml", ".yml"]:
                config = GlobalLMConfig.from_yaml_file(str(config_path))
            elif config_path.suffix == ".json":
                config = GlobalLMConfig.from_json_file(str(config_path))
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
        else:
            # Build from CLI args
            lm_config = LMClassifierConfig(
                model_name=args.model_name,
                num_labels=args.num_labels,
                freeze_base_model=args.freeze_base_model,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                warmup_steps=args.warmup_steps,
                weight_decay=args.weight_decay,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_length=args.max_length,
                truncation=args.truncation,
                padding=args.padding,
                use_class_weights=args.use_class_weights,
                early_stopping_patience=args.early_stopping_patience,
                eval_steps=args.eval_steps,
                save_steps=args.save_steps,
                save_total_limit=args.save_total_limit,
                logging_steps=args.logging_steps,
                device=args.device,
                fp16=args.fp16,
                n_gpu=args.n_gpu,
                seed=args.seed,
                tensorboard=args.tensorboard,
                verbose=not args.quiet,
            )

            data_config = DataConfig(
                input_file=args.input_file,
                text_column=args.text_column,
                target_column=args.target_column,
                validation_file=args.validation_file,
                dataset_name=args.dataset,
                dataset_config=args.dataset_config,
                dataset_split=args.dataset_split,
                validation_split=args.validation_split,
            )

            output_config = OutputConfig(
                output_dir=args.output_dir, verbose=not args.quiet
            )

            config = GlobalLMConfig(lm_config, data_config, output_config)

        # Train
        best_checkpoint = train_lm_classifier(config)
        logger.info(f"\nBest checkpoint: {best_checkpoint}")

    elif args.command == "predict":
        # Predict
        predictions_path = predict_lm_classifier(
            checkpoint_dir=args.checkpoint,
            input_file=args.input_file,
            text_column=args.text_column,
            target_column=args.target_column,
            output_file=args.output_file,
            batch_size=args.batch_size,
            device=args.device,
            output_probabilities=args.output_probabilities,
        )
        logger.info(f"\nPredictions saved to: {predictions_path}")


if __name__ == "__main__":
    main()
