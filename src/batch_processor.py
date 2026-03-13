"""
Batch processing utility for perplexity extraction with checkpointing and resume support.

Supports:
- Variable-length batching (texts of similar lengths grouped together)
- Checkpoint saving after each batch
- Resume from last completed batch
- Final merge of all batches
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm


class BatchProcessor:
    """Manages batch processing with checkpointing and resume."""

    def __init__(self, checkpoint_dir: Path, batch_size: int = 16):
        """
        Initialize batch processor.

        Args:
            checkpoint_dir: Directory to save batch checkpoints
            batch_size: Number of texts per batch
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.metadata_file = self.checkpoint_dir / "metadata.json"
        self.batches_dir = self.checkpoint_dir / "batches"
        self.batches_dir.mkdir(exist_ok=True)

    def get_last_completed_batch(self) -> Optional[int]:
        """Get the index of last completed batch, or None if none completed."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            return metadata.get('last_completed_batch')
        return None

    def save_batch(self, batch_idx: int, results: List[Dict[str, Any]]):
        """Save batch results to checkpoint file."""
        batch_file = self.batches_dir / f"batch_{batch_idx:06d}.json"
        with open(batch_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Update metadata
        metadata = {}
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)

        metadata['last_completed_batch'] = batch_idx
        metadata['num_completed_batches'] = batch_idx + 1

        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def create_batches(self, csv_path: str, text_column: str = 'text',
                      limit: Optional[int] = None) -> List[List[int]]:
        """
        Create variable-length batches from CSV.
        Texts are grouped by similar length to minimize padding.

        Args:
            csv_path: Path to input CSV
            text_column: Column name containing text
            limit: Limit number of rows (for testing)

        Returns:
            List of batches, where each batch is a list of row indices
        """
        df = pd.read_csv(csv_path)

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in {csv_path}")

        # Get text lengths
        texts = df[text_column].fillna("").astype(str)
        if limit:
            texts = texts.iloc[:limit]

        lengths = texts.str.len()

        # Sort by length to group similar texts
        sorted_indices = lengths.argsort().tolist()

        # Create batches
        batches = []
        for i in range(0, len(sorted_indices), self.batch_size):
            batch = sorted_indices[i:i + self.batch_size]
            batches.append(batch)

        return batches

    def merge_batches(self, output_file: str) -> int:
        """
        Merge all batch files into single output file.

        Args:
            output_file: Path to output merged file

        Returns:
            Total number of results merged
        """
        batch_files = sorted(self.batches_dir.glob("batch_*.json"))

        if not batch_files:
            raise FileNotFoundError(f"No batch files found in {self.batches_dir}")

        all_results = []
        for batch_file in tqdm(batch_files, desc="Merging batches"):
            with open(batch_file, 'r') as f:
                results = json.load(f)
            all_results.extend(results)

        # Save merged output
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        # Update metadata
        metadata = {}
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)

        metadata['merged'] = True
        metadata['total_results'] = len(all_results)
        metadata['output_file'] = str(output_file)

        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        return len(all_results)


def group_indices_by_batch(all_indices: List[int], batch_size: int) -> List[List[int]]:
    """Helper to group indices into batches."""
    batches = []
    for i in range(0, len(all_indices), batch_size):
        batches.append(all_indices[i:i + batch_size])
    return batches
