"""
Long-text processing strategies for perplexity extraction.

Strategies define how to handle texts that exceed the model's max_length:
- ImmediateContext: Compute each token's perplexity using only its immediate context window (NEW DEFAULT)
- SlidingWindow: Process text in overlapping windows (existing approach)
- Truncate: Simple truncation to max_length (fast but loses data)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
import torch


class LongTextStrategy(ABC):
    """Base class for long-text processing strategies."""

    @abstractmethod
    def process_long_text(
        self,
        model,
        tokenizer,
        input_ids: torch.Tensor,
        text: str,
        max_tokens: int,
        top_k: int = 5,
        context_window: int = 3,
    ) -> Dict[str, Any]:
        """
        Process a text that exceeds max_tokens.

        Args:
            model: Language model instance
            tokenizer: Tokenizer instance
            input_ids: Full tokenized input (1D tensor)
            text: Original text string
            max_tokens: Model's maximum sequence length
            top_k: Number of top alternatives
            context_window: Context window size for display

        Returns:
            Dict with:
            - tokens: List of TokenPerplexity dicts
            - aggregate: Dict with statistics
            - metadata: Dict with strategy info
        """
        pass


class ImmediateContextStrategy(LongTextStrategy):
    """
    Immediate Context Strategy (DEFAULT for long texts).

    For tokens beyond max_length, compute perplexity using only their
    immediate context window (last max_tokens tokens), not the entire text.

    Process:
    1. PASS 1: Forward on first max_tokens tokens -> get perplexities for tok_2...tok_max_tokens
    2. PASS 2+: For each token beyond max_tokens:
       - Forward on [tok_(n-max_tokens+1)...tok_n] (context window only)
       - Get perplexity for token n
    3. Aggregate all per-token perplexities
    """

    def process_long_text(
        self,
        model,
        tokenizer,
        input_ids: torch.Tensor,
        text: str,
        max_tokens: int,
        top_k: int = 5,
        context_window: int = 3,
    ) -> Dict[str, Any]:
        """Process long text using immediate context strategy."""

        token_perplexities = []
        num_forward_passes = 0

        # PASS 1: Process first max_tokens tokens
        first_chunk = input_ids[:max_tokens]
        chunk_results = model._compute_chunk_perplexities(
            input_ids=first_chunk,
            start_idx=0,
            top_k=top_k,
            context_window=context_window,
        )
        token_perplexities.extend(chunk_results)
        num_forward_passes += 1

        # PASS 2+: Process remaining tokens with immediate context
        for token_idx in range(max_tokens, len(input_ids)):
            # Create context window for this token
            context_start = max(0, token_idx - max_tokens + 1)
            context_ids = input_ids[context_start : token_idx + 1]

            # Forward pass on context window
            context_results = model._compute_chunk_perplexities(
                input_ids=context_ids,
                start_idx=token_idx - len(context_ids) + 1,
                top_k=top_k,
                context_window=context_window,
            )

            # Extract only the last token's perplexity (the target token)
            if context_results:
                token_perplexities.append(context_results[-1])

            num_forward_passes += 1

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
                "total_tokens": len(token_perplexities),
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
                "total_tokens": 0,
            }

        return {
            "tokens": [t.to_dict() if hasattr(t, "to_dict") else t for t in token_perplexities],
            "aggregate": aggregate,
            "metadata": {
                "strategy": "immediate_context",
                "num_forward_passes": num_forward_passes,
                "max_length_exceeded": True,
                "text_length_tokens": len(input_ids),
            },
        }


class SlidingWindowStrategy(LongTextStrategy):
    """
    Sliding Window Strategy (existing approach).

    Process text in non-overlapping windows of max_tokens.
    Every token gets computed with maximum possible context.

    Less efficient than immediate_context but may be more accurate
    for some applications where multiple context windows are desired.
    """

    def process_long_text(
        self,
        model,
        tokenizer,
        input_ids: torch.Tensor,
        text: str,
        max_tokens: int,
        top_k: int = 5,
        context_window: int = 3,
    ) -> Dict[str, Any]:
        """Process long text using sliding window strategy."""

        token_perplexities = []
        num_forward_passes = 0

        # Process in non-overlapping windows
        position = 0
        while position < len(input_ids):
            window_end = min(position + max_tokens, len(input_ids))
            chunk_ids = input_ids[position:window_end]

            # Compute perplexities for this chunk
            chunk_results = model._compute_chunk_perplexities(
                input_ids=chunk_ids,
                start_idx=position,
                top_k=top_k,
                context_window=context_window,
            )

            token_perplexities.extend(chunk_results)
            num_forward_passes += 1

            # Move to next non-overlapping window
            position = window_end

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
                "total_tokens": len(token_perplexities),
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
                "total_tokens": 0,
            }

        return {
            "tokens": [t.to_dict() if hasattr(t, "to_dict") else t for t in token_perplexities],
            "aggregate": aggregate,
            "metadata": {
                "strategy": "sliding_window",
                "num_forward_passes": num_forward_passes,
                "max_length_exceeded": True,
                "text_length_tokens": len(input_ids),
            },
        }


class TruncateStrategy(LongTextStrategy):
    """
    Truncate Strategy (fast mode).

    Simply truncate text to max_tokens and process normally.
    Fastest but loses all information about tail tokens.

    Use only when speed is critical and tail token information is not needed.
    """

    def process_long_text(
        self,
        model,
        tokenizer,
        input_ids: torch.Tensor,
        text: str,
        max_tokens: int,
        top_k: int = 5,
        context_window: int = 3,
    ) -> Dict[str, Any]:
        """Process long text using truncate strategy."""

        # Simple truncation
        truncated_ids = input_ids[:max_tokens]

        chunk_results = model._compute_chunk_perplexities(
            input_ids=truncated_ids,
            start_idx=0,
            top_k=top_k,
            context_window=context_window,
        )

        token_perplexities = chunk_results
        num_forward_passes = 1

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
                "total_tokens": len(token_perplexities),
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
                "total_tokens": 0,
            }

        return {
            "tokens": [t.to_dict() if hasattr(t, "to_dict") else t for t in token_perplexities],
            "aggregate": aggregate,
            "metadata": {
                "strategy": "truncate",
                "num_forward_passes": num_forward_passes,
                "max_length_exceeded": True,
                "text_length_tokens": len(input_ids),
                "tokens_lost": len(input_ids) - max_tokens,
            },
        }


def get_strategy(strategy_name: str) -> LongTextStrategy:
    """Get strategy instance by name."""
    strategies = {
        "immediate_context": ImmediateContextStrategy,
        "sliding_window": SlidingWindowStrategy,
        "truncate": TruncateStrategy,
    }

    if strategy_name not in strategies:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Available: {', '.join(strategies.keys())}"
        )

    return strategies[strategy_name]()
