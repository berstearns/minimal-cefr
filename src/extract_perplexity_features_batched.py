#!/usr/bin/env python3
"""
Batched perplexity extraction with long-text strategy support and checkpointing.

Key features:
- Separates texts into SHORT (<max_length) and LONG (>=max_length)
- Batches short texts together (batch_size=16) for efficiency
- Processes long texts individually with immediate_context strategy
- Checkpoint saving after each batch
- Resume from last completed batch
- Final merge of all batches into single output

Usage:
    python -m src.extract_perplexity_features_batched \\
        -i data.csv \\
        --text-column text \\
        -m gpt2 \\
        -d cuda \\
        --batch-size 16 \\
        --checkpoint-dir ./checkpoints/ \\
        --long-text-strategy immediate_context \\
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
            result_dict = result.to_dict()
            # Add original row index for traceability
            result_dict["original_row_idx"] = idx
            results.append(result_dict)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batched perplexity extraction with long-text strategy support",
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
        help="Batch size for short texts (default: 16, reduce for low VRAM)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory for batch checkpoints (default: ./checkpoints)",
    )

    # Long-text strategy
    parser.add_argument(
        "--long-text-strategy",
        type=str,
        choices=["immediate_context", "sliding_window", "truncate"],
        default="immediate_context",
        help="Strategy for texts exceeding max_tokens (default: immediate_context)",
    )
    parser.add_argument(
        "--skip-short-texts",
        action="store_true",
        help="Skip processing short texts (useful for resume)",
    )
    parser.add_argument(
        "--skip-long-texts",
        action="store_true",
        help="Skip processing long texts (fast mode)",
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
        help="Flatten to aggregate stats in final output (batch JSONs always keep per-token data)",
    )

    # GPU optimization
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Use FP16 mixed precision inference (requires CUDA)",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Apply torch.compile() optimization to model",
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
        long_text_strategy=args.long_text_strategy,
        mixed_precision=args.mixed_precision,
        torch_compile=args.torch_compile,
    )

    print(f"Creating batches from {args.input}...")
    short_batches, long_batches, short_manifest, long_manifest = batch_processor.create_batches(
        args.input,
        text_column=args.text_column,
        limit=args.limit,
    )

    # Process short texts
    if not args.skip_short_texts:
        print(f"\nProcessing {len(short_batches)} short text batches...")
        start_batch = 0
        if args.resume:
            last_completed = batch_processor.get_last_completed_batch("short")
            if last_completed is not None:
                start_batch = last_completed + 1
                print(f"Resuming short texts from batch {start_batch}")

        for batch_idx in range(start_batch, len(short_batches)):
            batch_indices = short_batches[batch_idx]
            print(f"\nShort batch {batch_idx + 1}/{len(short_batches)} ({len(batch_indices)} texts)")

            results = extract_batch(
                args.input,
                batch_indices,
                args.text_column,
                model,
                top_k=args.top_k,
                context_window=args.context_window,
            )

            batch_processor.save_batch(batch_idx, results, batch_type="short")
            print(f"  Saved {len(results)} results (per-token + aggregate)")
    else:
        print("Skipping short texts")

    # Free GPU memory between short and long text processing
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    # Process long texts
    if not args.skip_long_texts:
        print(f"\nProcessing {len(long_batches)} long text batches...")
        start_batch = 0
        if args.resume:
            last_completed = batch_processor.get_last_completed_batch("long")
            if last_completed is not None:
                start_batch = last_completed + 1
                print(f"Resuming long texts from batch {start_batch}")

        for batch_idx in range(start_batch, len(long_batches)):
            batch_indices = long_batches[batch_idx]
            print(f"\nLong batch {batch_idx + 1}/{len(long_batches)} ({len(batch_indices)} texts)")

            results = extract_batch(
                args.input,
                batch_indices,
                args.text_column,
                model,
                top_k=args.top_k,
                context_window=args.context_window,
            )

            batch_processor.save_batch(batch_idx, results, batch_type="long")
            print(f"  Saved {len(results)} results (per-token + aggregate)")
    else:
        print("Skipping long texts")

    # Merge batches
    if not args.no_merge:
        print(f"\nMerging batches...")
        total = batch_processor.merge_batches(args.output)
        print(f"✓ Merged {total} results -> {args.output}")

        # Flatten to aggregate-only for final output if requested
        # (batch JSONs always keep full per-token data for analysis)
        if args.aggregate_only:
            print("Flattening to aggregate-only for output...")
            with open(args.output) as f:
                all_results = json.load(f)
            flat_results = []
            for r in all_results:
                if "aggregate" in r:
                    flat_results.append({
                        "text": r["text"],
                        "model": r["model"],
                        "original_row_idx": r.get("original_row_idx"),
                        **r["aggregate"],
                    })
                else:
                    # Already flat (from older aggregate-only batches)
                    flat_results.append(r)
            with open(args.output, 'w') as f:
                json.dump(flat_results, f, indent=2)
            print(f"  Flattened {len(flat_results)} results")

        # Convert format if needed
        if args.save_format != "json":
            print(f"Converting to {args.save_format}...")
            df = pd.read_json(args.output)

            if args.save_format == "csv":
                csv_path = args.output.replace(".json", ".csv")
                df.to_csv(csv_path, index=False)
                print(f"Saved as CSV: {csv_path}")
            elif args.save_format == "jsonl":
                jsonl_path = args.output.replace(".json", ".jsonl")
                with open(jsonl_path, "w") as f:
                    for _, row in df.iterrows():
                        f.write(json.dumps(row.to_dict()) + "\n")
                print(f"Saved as JSONL: {jsonl_path}")
    else:
        print("Skipping merge (--no-merge flag set)")
        print(f"Short batch files: {batch_processor.short_batches_dir}")
        print(f"Long batch files: {batch_processor.long_batches_dir}")

    print("\nDone!")


if __name__ == "__main__":
    main()
