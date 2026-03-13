#!/usr/bin/env python3
"""
Batched perplexity extraction with checkpointing and resume support.

Key improvements over standard extraction:
- Variable-length batching (texts grouped by length)
- Checkpoint saving after each batch
- Resume from last completed batch
- Final merge of all batches into single output

Usage:
    python -m src.extract_perplexity_features_batched \\
        -i data.csv \\
        --text-column text \\
        -m gpt2 \\
        -d cuda \\
        --aggregate-only \\
        --batch-size 16 \\
        --checkpoint-dir ./checkpoints/ \\
        -f csv \\
        -o output.csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional
import pandas as pd
from tqdm import tqdm

from src.extract_perplexity_features import (
    load_model,
    process_csv_file,
    save_results,
)
from src.batch_processor import BatchProcessor


def extract_batch(
    csv_path: str,
    batch_indices: List[int],
    text_column: str,
    model,
    top_k: int = 5,
    context_window: int = 3,
) -> List[dict]:
    """
    Extract perplexity features for a batch of texts.

    Args:
        csv_path: Path to input CSV
        batch_indices: Indices of texts to process in this batch
        text_column: Column name containing text
        model: Loaded language model
        top_k: Number of top alternatives
        context_window: Context window size

    Returns:
        List of perplexity results
    """
    df = pd.read_csv(csv_path)
    texts = df[text_column].fillna("").astype(str)

    results = []
    for idx in batch_indices:
        text = texts.iloc[idx]
        if text.strip():
            result = model.compute_token_perplexities(text, top_k, context_window)
            results.append(result.to_dict())

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batched perplexity extraction with checkpointing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input
    parser.add_argument("-i", "--input", type=str, required=True, help="Input CSV file")
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column name containing text (default: text)",
    )
    parser.add_argument("--limit", type=int, help="Limit number of rows to process")

    # Model
    parser.add_argument("-m", "--model", type=str, required=True, help="Model name")
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to use (default: cpu)",
    )

    # Batching
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16, reduce for low VRAM)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory for batch checkpoints (default: ./checkpoints)",
    )

    # Features
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top alternatives (default: 5)",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=3,
        help="Context window size (default: 3)",
    )

    # Output
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file")
    parser.add_argument(
        "-f",
        "--save-format",
        type=str,
        choices=["json", "jsonl", "csv"],
        default="csv",
        help="Output format (default: csv)",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Save only aggregate statistics",
    )

    # Other
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--no-merge", action="store_true", help="Don't merge batches at end")

    args = parser.parse_args()

    # Setup
    checkpoint_dir = Path(args.checkpoint_dir)
    batch_processor = BatchProcessor(checkpoint_dir, batch_size=args.batch_size)

    print(f"Loading model: {args.model}...")
    model = load_model(
        model_type="huggingface",
        model_name_or_path=args.model,
        device=args.device,
    )

    print(f"Creating batches from {args.input}...")
    batches = batch_processor.create_batches(
        args.input,
        text_column=args.text_column,
        limit=args.limit,
    )

    print(f"Total batches: {len(batches)}")

    # Determine start batch
    start_batch = 0
    if args.resume:
        last_completed = batch_processor.get_last_completed_batch()
        if last_completed is not None:
            start_batch = last_completed + 1
            print(f"Resuming from batch {start_batch}")

    # Process batches
    for batch_idx in range(start_batch, len(batches)):
        batch_indices = batches[batch_idx]
        print(f"\nBatch {batch_idx + 1}/{len(batches)} ({len(batch_indices)} texts)")

        results = extract_batch(
            args.input,
            batch_indices,
            args.text_column,
            model,
            top_k=args.top_k,
            context_window=args.context_window,
        )

        # Filter for aggregate-only if requested
        if args.aggregate_only:
            results = [
                {
                    "text": r["text"],
                    "model": r["model"],
                    **r["aggregate"]
                }
                for r in results
            ]

        batch_processor.save_batch(batch_idx, results)
        print(f"  Saved {len(results)} results")

    # Merge batches
    if not args.no_merge:
        print(f"\nMerging {len(batches)} batches...")
        total = batch_processor.merge_batches(args.output)
        print(f"Merged {total} results -> {args.output}")

        # Convert format if needed
        if args.save_format != "json":
            print(f"Converting to {args.save_format}...")
            df = pd.read_json(args.output)

            if args.save_format == "csv":
                df.to_csv(args.output.replace(".json", ".csv"), index=False)
                print(f"Saved as CSV: {args.output.replace('.json', '.csv')}")
            elif args.save_format == "jsonl":
                with open(args.output.replace(".json", ".jsonl"), "w") as f:
                    for _, row in df.iterrows():
                        f.write(json.dumps(row.to_dict()) + "\n")
                print(f"Saved as JSONL: {args.output.replace('.json', '.jsonl')}")
    else:
        print("Skipping merge (--no-merge flag set)")
        print(f"Batch files saved in: {batch_processor.batches_dir}")

    print("\nDone!")


if __name__ == "__main__":
    main()
