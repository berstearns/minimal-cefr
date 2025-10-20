"""
Mock PyTorch Language Model for Testing Perplexity Extraction

This module provides a simple random-weight causal language model that can be used
for testing the perplexity extraction pipeline without downloading large models.

USAGE
=====

1. Create mock model and tokenizer:
   $ python -m src.mock_pytorch_lm --vocab-size 1000 --output models/mock_lm

2. Test with perplexity extraction:
   $ python -m src.extract_perplexity_features \\
       --text "The student writes well." \\
       --model-type pytorch \\
       --model models/mock_lm/model.pt \\
       --tokenizer-path models/mock_lm/tokenizer.json \\
       --output test_perplexity.json

3. Create larger vocabulary:
   $ python -m src.mock_pytorch_lm --vocab-size 5000 --max-length 256 --output models/mock_lm_large
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np  # noqa: F401

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class SimpleCausalLM(nn.Module):
    """Simple causal language model with random weights."""

    def __init__(
        self,
        vocab_size: int = 1000,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        max_length: int = 128,
    ):
        """
        Initialize simple causal LM.

        Args:
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
            max_length: Maximum sequence length
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        # Simple architecture
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(max_length, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, **kwargs):
        """Forward pass."""
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        token_embeds = self.embedding(input_ids)

        # Positional embeddings
        positions = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        pos_embeds = self.positional_embedding(positions)

        # Combine
        embeds = token_embeds + pos_embeds

        # RNN
        rnn_out, _ = self.rnn(embeds)

        # Project to vocabulary
        logits = self.output_projection(rnn_out)

        # Return in HuggingFace-style format
        class ModelOutput:
            def __init__(self, logits, loss=None):
                self.logits = logits
                self.loss = loss

        # Compute loss if labels provided
        loss = None
        if "labels" in kwargs and kwargs["labels"] is not None:
            labels = kwargs["labels"]
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Compute cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.vocab_size), shift_labels.view(-1)
            )

        return ModelOutput(logits=logits, loss=loss)


class SimpleTokenizer:
    """Simple character-level or word-level tokenizer for testing."""

    def __init__(self, vocab_size: int = 1000, tokenizer_type: str = "char"):
        """
        Initialize tokenizer.

        Args:
            vocab_size: Size of vocabulary
            tokenizer_type: 'char' for character-level, 'word' for word-level
        """
        self.vocab_size = vocab_size
        self.tokenizer_type = tokenizer_type

        # Create vocabulary
        if tokenizer_type == "char":
            # Character-level: ASCII + special tokens
            chars = [chr(i) for i in range(32, 127)]  # Printable ASCII
            special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
            self.vocab = special_tokens + chars
            # Pad with dummy tokens if needed
            while len(self.vocab) < vocab_size:
                self.vocab.append(f"<DUMMY_{len(self.vocab)}>")
        else:
            # Word-level: Common English words + special tokens
            common_words = [
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "from",
                "as",
                "is",
                "was",
                "are",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "should",
                "could",
                "may",
                "might",
                "must",
                "can",
                "this",
                "that",
                "these",
                "those",
                "he",
                "she",
                "it",
                "they",
                "we",
                "you",
                "I",
                "me",
                "him",
                "her",
                "us",
                "them",
                "my",
                "your",
                "his",
                "our",
                "student",
                "write",
                "read",
                "learn",
                "study",
                "test",
                "good",
                "bad",
                "well",
                "better",
                "best",
                "English",
                "language",
                "word",
                "sentence",
                ".",
                ",",
                "!",
                "?",
                "'",
                '"',
                "-",
                ":",
                ";",
            ]
            special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
            self.vocab = special_tokens + common_words
            # Pad with dummy tokens
            while len(self.vocab) < vocab_size:
                self.vocab.append(f"word_{len(self.vocab)}")

        # Create mappings
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}

        # Special token IDs
        self.pad_token_id = self.token_to_id["<PAD>"]
        self.unk_token_id = self.token_to_id["<UNK>"]
        self.bos_token_id = self.token_to_id.get("<BOS>", None)
        self.eos_token_id = self.token_to_id.get("<EOS>", None)

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if self.tokenizer_type == "char":
            # Character-level tokenization
            tokens = list(text)
            ids = [self.token_to_id.get(token, self.unk_token_id) for token in tokens]
        else:
            # Word-level tokenization (simple whitespace split)
            tokens = text.split()
            ids = [self.token_to_id.get(token, self.unk_token_id) for token in tokens]

        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        tokens = [self.id_to_token.get(id_, "<UNK>") for id_ in ids]

        if self.tokenizer_type == "char":
            return "".join(tokens)
        else:
            return " ".join(tokens)

    def __call__(self, text: str, return_tensors: Optional[str] = None):
        """Tokenize text (HuggingFace-style interface)."""
        ids = self.encode(text)

        if return_tensors == "pt":
            import torch

            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        else:
            return {"input_ids": [ids]}

    def to_json(self) -> Dict:
        """Export tokenizer to JSON format."""
        return {
            "vocab_size": self.vocab_size,
            "tokenizer_type": self.tokenizer_type,
            "vocab": self.vocab,
            "token_to_id": self.token_to_id,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }

    @classmethod
    def from_json(cls, data: Dict):
        """Load tokenizer from JSON format."""
        tokenizer = cls(
            vocab_size=data["vocab_size"], tokenizer_type=data["tokenizer_type"]
        )
        tokenizer.vocab = data["vocab"]
        tokenizer.token_to_id = {k: int(v) for k, v in data["token_to_id"].items()}
        tokenizer.id_to_token = {
            int(k): v
            for k, v in {
                idx: token for token, idx in tokenizer.token_to_id.items()
            }.items()
        }
        tokenizer.pad_token_id = data["pad_token_id"]
        tokenizer.unk_token_id = data["unk_token_id"]
        tokenizer.bos_token_id = data["bos_token_id"]
        tokenizer.eos_token_id = data["eos_token_id"]
        return tokenizer


def create_mock_model_and_tokenizer(
    vocab_size: int = 1000,
    embed_dim: int = 128,
    hidden_dim: int = 256,
    max_length: int = 128,
    tokenizer_type: str = "word",
    output_dir: str = "models/mock_lm",
):
    """
    Create and save mock language model and tokenizer.

    Args:
        vocab_size: Vocabulary size
        embed_dim: Embedding dimension
        hidden_dim: Hidden layer dimension
        max_length: Maximum sequence length
        tokenizer_type: 'char' or 'word'
        output_dir: Output directory
    """
    if not TORCH_AVAILABLE:
        raise ImportError("torch not installed. Install with: pip install torch")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Creating mock language model...")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Embed dim: {embed_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Max length: {max_length}")
    print(f"  Tokenizer type: {tokenizer_type}")

    # Create model with random weights
    model = SimpleCausalLM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        max_length=max_length,
    )
    model.eval()

    # Save model state dict and architecture info
    model_path = output_path / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "max_length": max_length,
        },
        model_path,
    )
    print(f"✓ Model saved to: {model_path}")

    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=vocab_size, tokenizer_type=tokenizer_type)

    # Save tokenizer as JSON
    tokenizer_path = output_path / "tokenizer.json"
    with open(tokenizer_path, "w") as f:
        json.dump(tokenizer.to_json(), f, indent=2)
    print(f"✓ Tokenizer saved to: {tokenizer_path}")

    # Save tokenizer as pickle (alternative format)
    tokenizer_pkl_path = output_path / "tokenizer.pkl"
    with open(tokenizer_pkl_path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"✓ Tokenizer (pickle) saved to: {tokenizer_pkl_path}")

    # Save metadata
    metadata = {
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "max_length": max_length,
        "tokenizer_type": tokenizer_type,
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path),
    }
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to: {metadata_path}")

    print(f"\n{'='*60}")
    print("Mock model created successfully!")
    print(f"{'='*60}")
    print("\nTest with perplexity extraction:")
    print("  python -m src.extract_perplexity_features \\")
    print('    --text "The student writes well." \\')
    print("    --model-type pytorch \\")
    print(f"    --model {model_path} \\")
    print(f"    --tokenizer-path {tokenizer_path} \\")
    print("    --output test_perplexity.json")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create mock PyTorch language model for testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--vocab-size", type=int, default=1000, help="Vocabulary size (default: 1000)"
    )
    parser.add_argument(
        "--embed-dim", type=int, default=128, help="Embedding dimension (default: 128)"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden layer dimension (default: 256)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length (default: 128)",
    )
    parser.add_argument(
        "--tokenizer-type",
        type=str,
        choices=["char", "word"],
        default="word",
        help="Tokenizer type (default: word)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/mock_lm",
        help="Output directory (default: models/mock_lm)",
    )

    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        parser.error("torch not installed. Install with: pip install torch")

    create_mock_model_and_tokenizer(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        max_length=args.max_length,
        tokenizer_type=args.tokenizer_type,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
