"""
Extract Per-Token Perplexity Features from Language Models

This module computes hyper-granular per-token perplexity features for texts using
either HuggingFace or native PyTorch language models.

For each token in the input text, we compute a detailed Perplexity dictionary containing:
- token: The actual token string
- token_id: The token ID
- position: Position in the sequence
- logit: Raw logit score for the token
- prob: Probability of the token
- perplexity: Token-level perplexity (2^(-log2(prob)))
- entropy: Token entropy (-log2(prob))
- rank: Rank of the true token among all possible tokens
- top_k_tokens: Top-K alternative tokens and their probabilities
- context_window: Surrounding tokens for context

USAGE PATTERNS
==============

Input Modes:
1. Single text string
2. CSV file with text column
3. Plain text file (.txt)
4. HuggingFace dataset

Model Types:
- HuggingFace transformers models (e.g., gpt2, bert, llama)
- Native PyTorch models with custom tokenizer

EXAMPLES
========

1. Single text with HuggingFace model:
   $ python -m src.extract_perplexity_features \\
       --text "The student writes fluently in English." \\
       --model gpt2 \\
       --output results.json

2. CSV file with custom column:
   $ python -m src.extract_perplexity_features \\
       -i data.csv \\
       --text-column essay \\
       --model facebook/opt-350m \\
       --output perplexity_features/

3. Text file:
   $ python -m src.extract_perplexity_features \\
       -i document.txt \\
       --model gpt2 \\
       --output doc_perplexity.json

4. HuggingFace dataset:
   $ python -m src.extract_perplexity_features \\
       --dataset glue \\
       --dataset-split validation \\
       --text-column sentence \\
       --model distilgpt2 \\
       --output hf_perplexity/

5. Custom PyTorch model:
   $ python -m src.extract_perplexity_features \\
       -i data.csv \\
       --model-type pytorch \\
       --model-path /models/custom_lm.pt \\
       --tokenizer-path /models/tokenizer.json \\
       --output custom_perplexity/

6. Batch processing with GPU:
   $ python -m src.extract_perplexity_features \\
       -i large_dataset.csv \\
       --model meta-llama/Llama-2-7b-hf \\
       --batch-size 16 \\
       --device cuda \\
       --output llama_perplexity/

Advanced Options:
7. Save detailed features with top-10 alternatives:
   $ python -m src.extract_perplexity_features \\
       -i essays.csv \\
       --model gpt2 \\
       --top-k 10 \\
       --context-window 5 \\
       --save-format jsonl \\
       --output detailed_perplexity/

8. Aggregate statistics only (no per-token):
   $ python -m src.extract_perplexity_features \\
       -i data.csv \\
       --model gpt2 \\
       --aggregate-only \\
       --output aggregated.csv

OUTPUT FORMATS
==============

Per-Token Output (JSON):
{
  "text": "The student writes well.",
  "model": "gpt2",
  "tokens": [
    {
      "token": "The",
      "token_id": 464,
      "position": 0,
      "logit": 8.45,
      "prob": 0.85,
      "perplexity": 1.18,
      "entropy": 0.23,
      "rank": 1,
      "top_k_tokens": [
        {"token": "The", "prob": 0.85},
        {"token": "A", "prob": 0.10},
        {"token": "This", "prob": 0.03}
      ],
      "context_before": [],
      "context_after": ["student", "writes"]
    },
    ...
  ],
  "aggregate": {
    "mean_perplexity": 15.2,
    "median_perplexity": 12.1,
    "std_perplexity": 8.4,
    "mean_entropy": 3.9,
    "total_tokens": 5
  }
}

Aggregate Output (CSV):
text,model,mean_perplexity,median_perplexity,std_perplexity,mean_entropy,total_tokens
"The student writes well.",gpt2,15.2,12.1,8.4,3.9,5
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


@dataclass
class TokenPerplexity:
    """Detailed per-token perplexity information."""
    token: str
    token_id: int
    position: int
    logit: float
    prob: float
    perplexity: float
    entropy: float
    rank: int
    top_k_tokens: List[Dict[str, Any]]
    context_before: List[str]
    context_after: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TextPerplexityResult:
    """Complete perplexity analysis for a text."""
    text: str
    model: str
    tokens: List[TokenPerplexity]
    aggregate: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'text': self.text,
            'model': self.model,
            'tokens': [t.to_dict() for t in self.tokens],
            'aggregate': self.aggregate
        }


class LanguageModel(ABC):
    """Abstract interface for language models."""

    @abstractmethod
    def compute_token_perplexities(
        self,
        text: str,
        top_k: int = 5,
        context_window: int = 3
    ) -> TextPerplexityResult:
        """
        Compute per-token perplexity for input text.

        Args:
            text: Input text
            top_k: Number of top alternative tokens to return
            context_window: Number of context tokens before/after

        Returns:
            TextPerplexityResult with detailed per-token information
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name/identifier."""
        pass


class HuggingFaceLanguageModel(LanguageModel):
    """Wrapper for HuggingFace causal language models."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        trust_remote_code: bool = False,
        max_tokens: Optional[int] = None,
        padding_strategy: str = "sliding_window"
    ):
        """
        Initialize HuggingFace model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cpu', 'cuda', 'mps')
            trust_remote_code: Trust remote code for custom models
            max_tokens: Maximum tokens per pass (default: model's max_position_embeddings)
            padding_strategy: How to handle long texts ('truncate', 'sliding_window')
        """
        if not HF_AVAILABLE:
            raise ImportError("transformers not installed. Install with: pip install transformers")
        if not TORCH_AVAILABLE:
            raise ImportError("torch not installed. Install with: pip install torch")

        self.model_name = model_name
        self.device = device
        self.padding_strategy = padding_strategy

        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        ).to(device)
        self.model.eval()

        # Set pad token if not available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Determine max_tokens
        if max_tokens is None:
            # Use model's max position embeddings
            if hasattr(self.model.config, 'max_position_embeddings'):
                self.max_tokens = self.model.config.max_position_embeddings
            elif hasattr(self.model.config, 'n_positions'):
                self.max_tokens = self.model.config.n_positions
            else:
                # Fallback to 512
                self.max_tokens = 512
                print(f"⚠ Could not determine model max_length, using default: {self.max_tokens}")
        else:
            self.max_tokens = max_tokens

        print(f"✓ Model loaded on {device}")
        print(f"  Max tokens per pass: {self.max_tokens}")
        print(f"  Padding strategy: {padding_strategy}")

    def _compute_chunk_perplexities(
        self,
        input_ids: torch.Tensor,
        start_idx: int,
        top_k: int,
        context_window: int
    ) -> List[TokenPerplexity]:
        """
        Compute perplexities for a chunk of tokens.

        Args:
            input_ids: Token IDs tensor [seq_len]
            start_idx: Global starting position (for position tracking)
            top_k: Number of top alternatives
            context_window: Context window size

        Returns:
            List of TokenPerplexity objects
        """
        token_perplexities = []

        with torch.no_grad():
            # Get model outputs
            inputs = {"input_ids": input_ids.unsqueeze(0).to(self.device)}
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            logits = outputs.logits[0]  # [seq_len, vocab_size]

            # Process each token (skip first token as it has no previous context)
            for i in range(1, len(input_ids)):
                token_logits = logits[i - 1]
                token_id = input_ids[i].item()

                # Compute probabilities
                probs = F.softmax(token_logits, dim=-1)
                token_prob = probs[token_id].item()

                # Compute perplexity and entropy
                perplexity = 2 ** (-np.log2(token_prob + 1e-10))
                entropy = -np.log2(token_prob + 1e-10)

                # Get logit value
                logit = token_logits[token_id].item()

                # Get rank of true token
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item() + 1

                # Get top-k alternative tokens
                top_k_probs, top_k_indices = torch.topk(probs, min(top_k, len(probs)))
                top_k_tokens = [
                    {
                        "token": self.tokenizer.decode([idx.item()]),
                        "token_id": idx.item(),
                        "prob": prob.item()
                    }
                    for prob, idx in zip(top_k_probs, top_k_indices)
                ]

                # Get context
                context_start = max(0, i - context_window)
                context_end = min(len(input_ids), i + context_window + 1)

                context_before = [
                    self.tokenizer.decode([input_ids[j].item()])
                    for j in range(context_start, i)
                ]
                context_after = [
                    self.tokenizer.decode([input_ids[j].item()])
                    for j in range(i + 1, context_end)
                ]

                # Decode token
                token_str = self.tokenizer.decode([token_id])

                token_perplexities.append(TokenPerplexity(
                    token=token_str,
                    token_id=token_id,
                    position=start_idx + i,  # Global position
                    logit=logit,
                    prob=token_prob,
                    perplexity=perplexity,
                    entropy=entropy,
                    rank=rank,
                    top_k_tokens=top_k_tokens,
                    context_before=context_before,
                    context_after=context_after
                ))

        return token_perplexities

    def compute_token_perplexities(
        self,
        text: str,
        top_k: int = 5,
        context_window: int = 3
    ) -> TextPerplexityResult:
        """Compute per-token perplexity using HuggingFace model."""

        # Tokenize full text
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"][0]

        # Check if we need to use sliding window
        if len(input_ids) <= self.max_tokens:
            # Text fits in one pass
            token_perplexities = self._compute_chunk_perplexities(
                input_ids=input_ids,
                start_idx=0,
                top_k=top_k,
                context_window=context_window
            )
        else:
            # Use sliding window strategy
            if self.padding_strategy == "truncate":
                # Simple truncation
                truncated_ids = input_ids[:self.max_tokens]
                token_perplexities = self._compute_chunk_perplexities(
                    input_ids=truncated_ids,
                    start_idx=0,
                    top_k=top_k,
                    context_window=context_window
                )

            elif self.padding_strategy == "sliding_window":
                # Non-overlapping sliding window
                # Ensures every token gets perplexity with maximum context
                token_perplexities = []

                # Process in non-overlapping windows
                position = 0
                while position < len(input_ids):
                    # Determine window size
                    window_end = min(position + self.max_tokens, len(input_ids))
                    chunk_ids = input_ids[position:window_end]

                    # Compute perplexities for this chunk
                    chunk_perplexities = self._compute_chunk_perplexities(
                        input_ids=chunk_ids,
                        start_idx=position,
                        top_k=top_k,
                        context_window=context_window
                    )

                    token_perplexities.extend(chunk_perplexities)

                    # Move to next non-overlapping window
                    position = window_end

            else:
                raise ValueError(f"Unknown padding strategy: {self.padding_strategy}")

        # Compute aggregate statistics
        if token_perplexities:
            perplexities = [tp.perplexity for tp in token_perplexities]
            entropies = [tp.entropy for tp in token_perplexities]

            aggregate = {
                "mean_perplexity": float(np.mean(perplexities)),
                "median_perplexity": float(np.median(perplexities)),
                "std_perplexity": float(np.std(perplexities)),
                "min_perplexity": float(np.min(perplexities)),
                "max_perplexity": float(np.max(perplexities)),
                "mean_entropy": float(np.mean(entropies)),
                "std_entropy": float(np.std(entropies)),
                "total_tokens": len(token_perplexities)
            }
        else:
            aggregate = {
                "mean_perplexity": 0.0,
                "median_perplexity": 0.0,
                "std_perplexity": 0.0,
                "min_perplexity": 0.0,
                "max_perplexity": 0.0,
                "mean_entropy": 0.0,
                "std_entropy": 0.0,
                "total_tokens": 0
            }

        return TextPerplexityResult(
            text=text,
            model=self.model_name,
            tokens=token_perplexities,
            aggregate=aggregate
        )

    def get_model_name(self) -> str:
        """Get model name."""
        return self.model_name


class PyTorchLanguageModel(LanguageModel):
    """Wrapper for custom PyTorch language models."""

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: str = "cpu",
        max_tokens: Optional[int] = None,
        padding_strategy: str = "sliding_window"
    ):
        """
        Initialize custom PyTorch model.

        Args:
            model_path: Path to saved PyTorch model
            tokenizer_path: Path to tokenizer (pickle or json)
            device: Device to use
            max_tokens: Maximum tokens per pass (default: model's max_length)
            padding_strategy: How to handle long texts ('truncate', 'sliding_window')
        """
        if not TORCH_AVAILABLE:
            raise ImportError("torch not installed. Install with: pip install torch")

        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.padding_strategy = padding_strategy

        print(f"Loading custom PyTorch model from: {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)

        # Try to reconstruct model from checkpoint
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # Model saved as state dict - reconstruct
            from src.mock_pytorch_lm import SimpleCausalLM
            self.model = SimpleCausalLM(
                vocab_size=checkpoint['vocab_size'],
                embed_dim=checkpoint['embed_dim'],
                hidden_dim=checkpoint['hidden_dim'],
                max_length=checkpoint['max_length']
            )
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model = self.model.to(device)
            model_max_length = checkpoint['max_length']
        else:
            # Model saved as full object (legacy)
            self.model = checkpoint
            model_max_length = self.model.max_length

        self.model.eval()

        # Determine max_tokens
        if max_tokens is None:
            self.max_tokens = model_max_length
        else:
            self.max_tokens = max_tokens

        print(f"Loading tokenizer from: {tokenizer_path}...")
        if tokenizer_path.endswith('.pkl'):
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
        elif tokenizer_path.endswith('.json'):
            with open(tokenizer_path, 'r') as f:
                tokenizer_data = json.load(f)
                # Reconstruct tokenizer from JSON
                from src.mock_pytorch_lm import SimpleTokenizer
                self.tokenizer = SimpleTokenizer.from_json(tokenizer_data)
        else:
            raise ValueError(f"Unsupported tokenizer format: {tokenizer_path}")

        print(f"✓ Model loaded on {device}")
        print(f"  Max tokens per pass: {self.max_tokens}")
        print(f"  Padding strategy: {padding_strategy}")

    def _compute_chunk_perplexities(
        self,
        input_ids: torch.Tensor,
        start_idx: int,
        top_k: int,
        context_window: int
    ) -> List[TokenPerplexity]:
        """Compute perplexities for a chunk of tokens."""
        token_perplexities = []

        with torch.no_grad():
            # Get model outputs
            inputs_dict = {"input_ids": input_ids.unsqueeze(0).to(self.device)}
            outputs = self.model(**inputs_dict, labels=inputs_dict["input_ids"])
            logits = outputs.logits[0]  # [seq_len, vocab_size]

            # Process each token (skip first token)
            for i in range(1, len(input_ids)):
                token_logits = logits[i - 1]
                token_id = input_ids[i].item()

                # Compute probabilities
                probs = F.softmax(token_logits, dim=-1)
                token_prob = probs[token_id].item()

                # Compute perplexity and entropy
                perplexity = 2 ** (-np.log2(token_prob + 1e-10))
                entropy = -np.log2(token_prob + 1e-10)

                # Get logit value
                logit = token_logits[token_id].item()

                # Get rank of true token
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item() + 1

                # Get top-k alternative tokens
                top_k_probs, top_k_indices = torch.topk(probs, min(top_k, len(probs)))
                top_k_tokens = [
                    {
                        "token": self.tokenizer.decode([idx.item()]),
                        "token_id": idx.item(),
                        "prob": prob.item()
                    }
                    for prob, idx in zip(top_k_probs, top_k_indices)
                ]

                # Get context
                context_start = max(0, i - context_window)
                context_end = min(len(input_ids), i + context_window + 1)

                context_before = [
                    self.tokenizer.decode([input_ids[j].item()])
                    for j in range(context_start, i)
                ]
                context_after = [
                    self.tokenizer.decode([input_ids[j].item()])
                    for j in range(i + 1, context_end)
                ]

                # Decode token
                token_str = self.tokenizer.decode([token_id])

                token_perplexities.append(TokenPerplexity(
                    token=token_str,
                    token_id=token_id,
                    position=start_idx + i,  # Global position
                    logit=logit,
                    prob=token_prob,
                    perplexity=perplexity,
                    entropy=entropy,
                    rank=rank,
                    top_k_tokens=top_k_tokens,
                    context_before=context_before,
                    context_after=context_after
                ))

        return token_perplexities

    def compute_token_perplexities(
        self,
        text: str,
        top_k: int = 5,
        context_window: int = 3
    ) -> TextPerplexityResult:
        """Compute per-token perplexity using custom PyTorch model."""

        # Tokenize using custom tokenizer
        token_ids = self.tokenizer.encode(text)
        input_ids = torch.tensor(token_ids, dtype=torch.long)

        # Check if we need to use sliding window
        if len(input_ids) <= self.max_tokens:
            # Text fits in one pass
            token_perplexities = self._compute_chunk_perplexities(
                input_ids=input_ids,
                start_idx=0,
                top_k=top_k,
                context_window=context_window
            )
        else:
            # Use sliding window strategy
            if self.padding_strategy == "truncate":
                # Simple truncation
                truncated_ids = input_ids[:self.max_tokens]
                token_perplexities = self._compute_chunk_perplexities(
                    input_ids=truncated_ids,
                    start_idx=0,
                    top_k=top_k,
                    context_window=context_window
                )

            elif self.padding_strategy == "sliding_window":
                # Non-overlapping sliding window
                token_perplexities = []

                # Process in non-overlapping windows
                position = 0
                while position < len(input_ids):
                    # Determine window size
                    window_end = min(position + self.max_tokens, len(input_ids))
                    chunk_ids = input_ids[position:window_end]

                    # Compute perplexities for this chunk
                    chunk_perplexities = self._compute_chunk_perplexities(
                        input_ids=chunk_ids,
                        start_idx=position,
                        top_k=top_k,
                        context_window=context_window
                    )

                    token_perplexities.extend(chunk_perplexities)

                    # Move to next non-overlapping window
                    position = window_end

            else:
                raise ValueError(f"Unknown padding strategy: {self.padding_strategy}")

        # Compute aggregate statistics
        if token_perplexities:
            perplexities = [tp.perplexity for tp in token_perplexities]
            entropies = [tp.entropy for tp in token_perplexities]

            aggregate = {
                "mean_perplexity": float(np.mean(perplexities)),
                "median_perplexity": float(np.median(perplexities)),
                "std_perplexity": float(np.std(perplexities)),
                "min_perplexity": float(np.min(perplexities)),
                "max_perplexity": float(np.max(perplexities)),
                "mean_entropy": float(np.mean(entropies)),
                "std_entropy": float(np.std(entropies)),
                "total_tokens": len(token_perplexities)
            }
        else:
            aggregate = {
                "mean_perplexity": 0.0,
                "median_perplexity": 0.0,
                "std_perplexity": 0.0,
                "min_perplexity": 0.0,
                "max_perplexity": 0.0,
                "mean_entropy": 0.0,
                "std_entropy": 0.0,
                "total_tokens": 0
            }

        return TextPerplexityResult(
            text=text,
            model=self.get_model_name(),
            tokens=token_perplexities,
            aggregate=aggregate
        )

    def get_model_name(self) -> str:
        """Get model name."""
        return Path(self.model_path).stem


def load_model(
    model_type: str,
    model_name_or_path: str,
    device: str = "cpu",
    **kwargs
) -> LanguageModel:
    """
    Load language model based on type.

    Args:
        model_type: 'huggingface' or 'pytorch'
        model_name_or_path: Model identifier or path
        device: Device to use
        **kwargs: Additional model-specific arguments

    Returns:
        LanguageModel instance
    """
    if model_type == "huggingface":
        return HuggingFaceLanguageModel(
            model_name=model_name_or_path,
            device=device,
            trust_remote_code=kwargs.get('trust_remote_code', False),
            max_tokens=kwargs.get('max_tokens'),
            padding_strategy=kwargs.get('padding_strategy', 'sliding_window')
        )
    elif model_type == "pytorch":
        return PyTorchLanguageModel(
            model_path=model_name_or_path,
            tokenizer_path=kwargs.get('tokenizer_path'),
            device=device,
            max_tokens=kwargs.get('max_tokens'),
            padding_strategy=kwargs.get('padding_strategy', 'sliding_window')
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def process_single_text(
    text: str,
    model: LanguageModel,
    top_k: int = 5,
    context_window: int = 3
) -> TextPerplexityResult:
    """Process a single text."""
    return model.compute_token_perplexities(text, top_k, context_window)


def process_csv_file(
    csv_path: str,
    text_column: str,
    model: LanguageModel,
    top_k: int = 5,
    context_window: int = 3,
    limit: Optional[int] = None,
    verbose: bool = True
) -> List[TextPerplexityResult]:
    """Process CSV file with text column."""
    df = pd.read_csv(csv_path)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in {csv_path}")

    texts = df[text_column].fillna('').astype(str).tolist()

    if limit:
        texts = texts[:limit]

    results = []
    iterator = tqdm(texts, desc="Processing texts") if verbose else texts

    for text in iterator:
        if text.strip():  # Skip empty texts
            result = model.compute_token_perplexities(text, top_k, context_window)
            results.append(result)

    return results


def process_text_file(
    txt_path: str,
    model: LanguageModel,
    top_k: int = 5,
    context_window: int = 3
) -> TextPerplexityResult:
    """Process plain text file."""
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()

    return model.compute_token_perplexities(text, top_k, context_window)


def process_hf_dataset(
    dataset_name: str,
    split: str,
    text_column: str,
    model: LanguageModel,
    top_k: int = 5,
    context_window: int = 3,
    limit: Optional[int] = None,
    verbose: bool = True
) -> List[TextPerplexityResult]:
    """Process HuggingFace dataset."""
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets not installed. Install with: pip install datasets")

    print(f"Loading dataset: {dataset_name}, split: {split}...")
    dataset = load_dataset(dataset_name, split=split)

    if text_column not in dataset.column_names:
        raise ValueError(f"Column '{text_column}' not found in dataset")

    texts = dataset[text_column]

    if limit:
        texts = texts[:limit]

    results = []
    iterator = tqdm(texts, desc="Processing dataset") if verbose else texts

    for text in iterator:
        if text and text.strip():
            result = model.compute_token_perplexities(text, top_k, context_window)
            results.append(result)

    return results


def save_results(
    results: Union[TextPerplexityResult, List[TextPerplexityResult]],
    output_path: str,
    save_format: str = "json",
    aggregate_only: bool = False
):
    """
    Save perplexity results to file.

    Args:
        results: Single result or list of results
        output_path: Output file/directory path
        save_format: 'json', 'jsonl', or 'csv'
        aggregate_only: Save only aggregate statistics
    """
    output_path = Path(output_path)

    # Ensure results is a list
    if isinstance(results, TextPerplexityResult):
        results = [results]

    if save_format == "json":
        # Save as single JSON file
        if aggregate_only:
            data = [
                {
                    'text': r.text,
                    'model': r.model,
                    **r.aggregate
                }
                for r in results
            ]
        else:
            data = [r.to_dict() for r in results]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved to: {output_path}")

    elif save_format == "jsonl":
        # Save as JSON lines
        with open(output_path, 'w', encoding='utf-8') as f:
            for r in results:
                if aggregate_only:
                    data = {'text': r.text, 'model': r.model, **r.aggregate}
                else:
                    data = r.to_dict()
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

        print(f"✓ Saved to: {output_path}")

    elif save_format == "csv":
        # Save aggregates as CSV
        data = []
        for r in results:
            row = {
                'text': r.text,
                'model': r.model,
                **r.aggregate
            }
            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)

        print(f"✓ Saved to: {output_path}")

    else:
        raise ValueError(f"Unknown save format: {save_format}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract per-token perplexity features from language models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input configuration
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--text',
        type=str,
        help='Single text string to process'
    )
    input_group.add_argument(
        '-i', '--input',
        type=str,
        help='Input file (CSV or TXT)'
    )
    input_group.add_argument(
        '--dataset',
        type=str,
        help='HuggingFace dataset name'
    )

    # CSV/Dataset configuration
    parser.add_argument(
        '--text-column',
        type=str,
        default='text',
        help='Column name containing text (default: text)'
    )
    parser.add_argument(
        '--dataset-split',
        type=str,
        default='test',
        help='Dataset split to use (default: test)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of texts to process'
    )

    # Model configuration
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['huggingface', 'pytorch'],
        default='huggingface',
        help='Model type (default: huggingface)'
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        help='Model name (HuggingFace) or path (PyTorch)'
    )
    parser.add_argument(
        '-t', '--tokenizer-path',
        type=str,
        help='Path to tokenizer (for PyTorch models)'
    )
    parser.add_argument(
        '-d', '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to use (default: cpu)'
    )
    parser.add_argument(
        '--trust-remote-code',
        action='store_true',
        help='Trust remote code for custom HuggingFace models'
    )

    # Feature configuration
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top alternative tokens to save (default: 5)'
    )
    parser.add_argument(
        '--context-window',
        type=int,
        default=3,
        help='Number of context tokens before/after (default: 3)'
    )

    # Padding/windowing configuration
    parser.add_argument(
        '--max-tokens',
        type=int,
        help='Maximum tokens per forward pass (default: model max_position_embeddings)'
    )
    parser.add_argument(
        '--padding-strategy',
        type=str,
        choices=['truncate', 'sliding_window'],
        default='sliding_window',
        help='Strategy for handling long texts (default: sliding_window)'
    )

    # Output configuration
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output file path'
    )
    parser.add_argument(
        '-f', '--save-format',
        type=str,
        choices=['json', 'jsonl', 'csv'],
        default='json',
        help='Output format (default: json)'
    )
    parser.add_argument(
        '--aggregate-only',
        action='store_true',
        help='Save only aggregate statistics, not per-token details'
    )

    # Other options
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    # Check dependencies
    if args.model_type == 'huggingface' and not HF_AVAILABLE:
        parser.error("transformers not installed. Install with: pip install transformers")

    if args.model_type == 'pytorch' and not args.tokenizer_path:
        parser.error("--tokenizer-path required for PyTorch models")

    if not TORCH_AVAILABLE:
        parser.error("torch not installed. Install with: pip install torch")

    if args.dataset and not DATASETS_AVAILABLE:
        parser.error("datasets not installed. Install with: pip install datasets")

    # Load model
    try:
        model = load_model(
            model_type=args.model_type,
            model_name_or_path=args.model,
            device=args.device,
            tokenizer_path=args.tokenizer_path,
            trust_remote_code=args.trust_remote_code,
            max_tokens=args.max_tokens,
            padding_strategy=args.padding_strategy
        )
    except Exception as e:
        parser.error(f"Failed to load model: {e}")

    # Process input
    try:
        if args.text:
            # Single text
            result = process_single_text(
                text=args.text,
                model=model,
                top_k=args.top_k,
                context_window=args.context_window
            )
            results = [result]

        elif args.input:
            # File input
            input_path = Path(args.input)
            if input_path.suffix == '.csv':
                results = process_csv_file(
                    csv_path=str(input_path),
                    text_column=args.text_column,
                    model=model,
                    top_k=args.top_k,
                    context_window=args.context_window,
                    limit=args.limit,
                    verbose=not args.quiet
                )
            elif input_path.suffix == '.txt':
                result = process_text_file(
                    txt_path=str(input_path),
                    model=model,
                    top_k=args.top_k,
                    context_window=args.context_window
                )
                results = [result]
            else:
                parser.error(f"Unsupported file type: {input_path.suffix}")

        elif args.dataset:
            # HuggingFace dataset
            results = process_hf_dataset(
                dataset_name=args.dataset,
                split=args.dataset_split,
                text_column=args.text_column,
                model=model,
                top_k=args.top_k,
                context_window=args.context_window,
                limit=args.limit,
                verbose=not args.quiet
            )

        else:
            parser.error("Must specify --text, -i/--input, or --dataset")

        # Save results
        save_results(
            results=results,
            output_path=args.output,
            save_format=args.save_format,
            aggregate_only=args.aggregate_only
        )

    except Exception as e:
        parser.error(f"Error during processing: {e}")
        raise


if __name__ == '__main__':
    main()
