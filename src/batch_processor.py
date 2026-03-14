"""
Batch processing utility for perplexity extraction with checkpointing and resume support.

Supports:
- Separation of texts into SHORT (<max_tokens) and LONG (>=max_tokens) categories
- Variable-length batching for short texts (batch_size=16)
- Individual batching for long texts (batch_size=1) to manage memory
- Checkpoint saving after each batch
- Resume from last completed batch in either category
- Final merge of short and long batch results
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from tqdm import tqdm


class BatchProcessor:
    """Manages batch processing with checkpointing, resume support, and short/long separation."""

    def __init__(self, checkpoint_dir: Path, batch_size: int = 16, max_tokens: int = 1024):
        """
        Initialize batch processor.

        Args:
            checkpoint_dir: Directory to save batch checkpoints
            batch_size: Number of texts per batch (for short texts)
            max_tokens: Model's maximum token length for separating short vs long
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.metadata_file = self.checkpoint_dir / "metadata.json"

        # Separate directories for short and long batches
        self.short_batches_dir = self.checkpoint_dir / "short_batches"
        self.long_batches_dir = self.checkpoint_dir / "long_batches"
        self.short_batches_dir.mkdir(exist_ok=True)
        self.long_batches_dir.mkdir(exist_ok=True)

        # Manifests for tracking indices
        self.short_manifest_file = self.checkpoint_dir / "short_texts_manifest.json"
        self.long_manifest_file = self.checkpoint_dir / "long_texts_manifest.json"

    def get_last_completed_batch(self, batch_type: str = "short") -> Optional[int]:
        """Get the index of last completed batch for given type, or None if none completed."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            if batch_type == "short":
                return metadata.get("short_texts", {}).get("last_completed_batch")
            else:
                return metadata.get("long_texts", {}).get("last_completed_batch")
        return None

    def save_batch(self, batch_idx: int, results: List[Dict[str, Any]], batch_type: str = "short"):
        """Save batch results to checkpoint file."""
        if batch_type == "short":
            batch_dir = self.short_batches_dir
        else:
            batch_dir = self.long_batches_dir

        batch_file = batch_dir / f"batch_{batch_idx:06d}.json"
        with open(batch_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Update metadata
        metadata = self._load_metadata()

        if batch_type == "short":
            if "short_texts" not in metadata:
                metadata["short_texts"] = {}
            metadata["short_texts"]["last_completed_batch"] = batch_idx
            metadata["short_texts"]["completed_batches"] = batch_idx + 1
            metadata["short_texts"]["num_completed_batches"] = batch_idx + 1
            completed = batch_idx + 1
            total = metadata["short_texts"].get("total_batches", "?")
        else:
            if "long_texts" not in metadata:
                metadata["long_texts"] = {}
            metadata["long_texts"]["last_completed_batch"] = batch_idx
            metadata["long_texts"]["completed_batches"] = batch_idx + 1
            metadata["long_texts"]["num_completed_batches"] = batch_idx + 1
            completed = batch_idx + 1
            total = metadata["long_texts"].get("total_batches", "?")

        self._save_metadata(metadata)

        # Log batch save
        print(f"  [CHECKPOINT] {batch_type.upper():5s} batch {batch_idx:6d} saved: {len(results):3d} results → {batch_file.name} ({completed}/{total})")

    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self, metadata: Dict[str, Any]):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def create_batches(
        self,
        csv_path: str,
        text_column: str = "text",
        limit: Optional[int] = None,
    ) -> Tuple[List[List[int]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Create batches from CSV, separating SHORT and LONG texts.

        Args:
            csv_path: Path to input CSV
            text_column: Column name containing text
            limit: Limit number of rows (for testing)

        Returns:
            Tuple of:
            - List of short batch indices
            - List of short text manifests
            - List of long text manifests
        """
        df = pd.read_csv(csv_path)

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in {csv_path}")

        # Get text lengths (character count for initial filtering)
        texts = df[text_column].fillna("").astype(str)
        if limit:
            texts = texts.iloc[:limit]
            df = df.iloc[:limit]

        # Note: We'll do token-based filtering in the extraction script
        # For now, use character count as a proxy (~1 token per 4 chars)
        char_lengths = texts.str.len()
        token_lengths = (char_lengths / 4).astype(int)  # Rough estimate

        # Separate short and long
        short_indices = []
        long_indices = []
        short_manifest = []
        long_manifest = []

        for idx, token_len in enumerate(token_lengths):
            if token_len < self.max_tokens:
                short_indices.append(idx)
                short_manifest.append({
                    "original_row_idx": idx,
                    "est_token_length": token_len,
                })
            else:
                long_indices.append(idx)
                long_manifest.append({
                    "original_row_idx": idx,
                    "est_token_length": token_len,
                })

        print(f"Estimated split: {len(short_indices)} SHORT, {len(long_indices)} LONG")

        # Create batches for short texts
        short_batches = []
        for i in range(0, len(short_indices), self.batch_size):
            batch = short_indices[i : i + self.batch_size]
            short_batches.append(batch)

        # Create batches for long texts (one per batch for memory safety)
        long_batches = [[idx] for idx in long_indices]

        # Save manifests
        with open(self.short_manifest_file, 'w') as f:
            json.dump(short_manifest, f, indent=2)
        with open(self.long_manifest_file, 'w') as f:
            json.dump(long_manifest, f, indent=2)

        # Initialize metadata
        metadata = self._load_metadata()
        metadata["status"] = "processing"
        metadata["input_file"] = csv_path
        metadata["total_texts"] = len(df)
        metadata["short_texts"] = {
            "count": len(short_indices),
            "batch_size": self.batch_size,
            "total_batches": len(short_batches),
            "completed_batches": 0,
            "last_completed_batch": None,
        }
        metadata["long_texts"] = {
            "count": len(long_indices),
            "batch_size": 1,
            "total_batches": len(long_batches),
            "completed_batches": 0,
            "last_completed_batch": None,
        }
        self._save_metadata(metadata)

        return short_batches, long_batches, short_manifest, long_manifest

    def merge_batches(self, output_file: str) -> int:
        """
        Merge all short and long batch files into single output file.

        Args:
            output_file: Path to output merged file

        Returns:
            Total number of results merged
        """
        all_results = []

        # Merge short batches
        short_batch_files = sorted(self.short_batches_dir.glob("batch_*.json"))
        if short_batch_files:
            for batch_file in tqdm(short_batch_files, desc="Merging short batches"):
                with open(batch_file, 'r') as f:
                    results = json.load(f)
                all_results.extend(results)

        # Merge long batches
        long_batch_files = sorted(self.long_batches_dir.glob("batch_*.json"))
        if long_batch_files:
            for batch_file in tqdm(long_batch_files, desc="Merging long batches"):
                with open(batch_file, 'r') as f:
                    results = json.load(f)
                all_results.extend(results)

        # Sort by original_row_idx to restore original order
        if all_results and "original_row_idx" in all_results[0]:
            all_results.sort(key=lambda x: x.get("original_row_idx", 0))

        # Save merged output
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        # Update metadata
        metadata = self._load_metadata()
        metadata["merged"] = True
        metadata["total_results"] = len(all_results)
        metadata["output_file"] = str(output_file)
        metadata["status"] = "completed"
        self._save_metadata(metadata)

        return len(all_results)


def group_indices_by_batch(all_indices: List[int], batch_size: int) -> List[List[int]]:
    """Helper to group indices into batches."""
    batches = []
    for i in range(0, len(all_indices), batch_size):
        batches.append(all_indices[i : i + batch_size])
    return batches
