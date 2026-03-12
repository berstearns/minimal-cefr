# Step 1: Training Artificial Learner Language Models

This step trains the GPT-2-based Artificial Learner (AL) models on the EFCAMDAT
remainder corpus. These models are then used as perplexity-based feature
extractors in Step 2.

## Overview

The paper uses 7 language models for perplexity feature extraction:

| Model | Training Data | Purpose |
|---|---|---|
| Native GPT-2 | Pre-trained (OpenAI) | Baseline native-speaker reference |
| General AL | All EFCAMDAT remainder (~623k) | Full learner developmental trajectory |
| A1-specific AL | Remainder filtered to A1 (~294k) | A1-level linguistic patterns |
| A2-specific AL | Remainder filtered to A2 (~186k) | A2-level linguistic patterns |
| B1-specific AL | Remainder filtered to B1 (~100k) | B1-level linguistic patterns |
| B2-specific AL | Remainder filtered to B2 (~35k) | B2-level linguistic patterns |
| C1-specific AL | Remainder filtered to C1 (~9k) | C1-level linguistic patterns |

## Pre-requisites

The level-separated text files already exist at:

```
DATA=/home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits

$DATA/norm_andrew100k_remainder_A1_texts.txt  (66 MB)
$DATA/norm_andrew100k_remainder_A2_texts.txt  (67 MB)
$DATA/norm_andrew100k_remainder_B1_texts.txt  (53 MB)
$DATA/norm_andrew100k_remainder_B2_texts.txt  (27 MB)
$DATA/norm_andrew100k_remainder_C1_texts.txt  (8.4 MB)
$DATA/norm-EFCAMDAT-remainder.csv             (221 MB, all levels combined)
```

## Training the AL Models

AL training uses the HuggingFace Transformers `run_clm.py` script (or
equivalent fine-tuning code) to fine-tune GPT-2 on Next Token Prediction.

This step is performed **outside** the `minimal-cefr` repo. The trained model
checkpoints are then referenced by path when extracting perplexity features.

### General AL

```bash
# Fine-tune GPT-2 on all EFCAMDAT remainder texts
python run_clm.py \
    --model_name_or_path gpt2 \
    --train_file $DATA/norm-EFCAMDAT-remainder.csv \
    --text_column text \
    --output_dir models/al-general \
    --do_train \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --save_strategy epoch \
    --logging_steps 500 \
    --fp16
```

### Level-Specific ALs

```bash
for LEVEL in A1 A2 B1 B2 C1; do
    python run_clm.py \
        --model_name_or_path gpt2 \
        --train_file $DATA/norm_andrew100k_remainder_${LEVEL}_texts.txt \
        --output_dir models/al-${LEVEL} \
        --do_train \
        --num_train_epochs 3 \
        --per_device_train_batch_size 8 \
        --save_strategy epoch \
        --logging_steps 500 \
        --fp16
done
```

## Output

After training, you should have model directories:

```
models/
├── al-general/       # General AL checkpoint
├── al-A1/            # A1-specific AL
├── al-A2/            # A2-specific AL
├── al-B1/            # B1-specific AL
├── al-B2/            # B2-specific AL
└── al-C1/            # C1-specific AL
```

Each directory contains a HuggingFace-compatible model that can be loaded with
`AutoModelForCausalLM.from_pretrained("models/al-general")`.

The pre-trained GPT-2 (native model) does not need training -- it is loaded
directly as `gpt2` from HuggingFace Hub.

## Notes

- No C2-specific AL is trained because EFCAMDAT contains no C2 samples.
- Training hyperparameters (epochs, batch size, learning rate) should match
  those reported in the paper's appendix / methodology section.
- GPU training is strongly recommended given corpus sizes.
