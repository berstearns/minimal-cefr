#!/usr/bin/env python3
"""
Merge batch JSON files from batched extraction into single output file.

Supports merging both short_batches/ and long_batches/ directories
and restoring original CSV order via original_row_idx.

Usage:
    python scripts/merge_batch_outputs.py \\
        --checkpoint-dir ./checkpoints/ \\
        -o output.csv \\
        -f csv
"""

import argparse
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def merge_batches(
    checkpoint_dir: str,
    output_file: str,
    output_format: str = "csv",
    short_batches_dir: str = None,
    long_batches_dir: str = None,
):
    """
    Merge all batch files (short and long) into single output.

    Args:
        checkpoint_dir: Directory containing batch checkpoints
        output_file: Output file path
        output_format: Output format (csv, json, jsonl)
        short_batches_dir: Path to short_batches directory (default: checkpoint_dir/short_batches)
        long_batches_dir: Path to long_batches directory (default: checkpoint_dir/long_batches)
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Determine batch directories
    if short_batches_dir is None:
        short_batches_dir = checkpoint_dir / "short_batches"
    else:
        short_batches_dir = Path(short_batches_dir)

    if long_batches_dir is None:
        long_batches_dir = checkpoint_dir / "long_batches"
    else:
        long_batches_dir = Path(long_batches_dir)

    all_results = []

    # Merge short batches
    short_batch_files = sorted(short_batches_dir.glob("batch_*.json")) if short_batches_dir.exists() else []
    if short_batch_files:
        print(f"Found {len(short_batch_files)} short batch files")
        for batch_file in tqdm(short_batch_files, desc="Merging short batches"):
            with open(batch_file, 'r') as f:
                results = json.load(f)
            all_results.extend(results)
        print(f"  ✓ Merged {len(short_batch_files)} short batches")

    # Merge long batches
    long_batch_files = sorted(long_batches_dir.glob("batch_*.json")) if long_batches_dir.exists() else []
    if long_batch_files:
        print(f"Found {len(long_batch_files)} long batch files")
        for batch_file in tqdm(long_batch_files, desc="Merging long batches"):
            with open(batch_file, 'r') as f:
                results = json.load(f)
            all_results.extend(results)
        print(f"  ✓ Merged {len(long_batch_files)} long batches")

    if not all_results:
        raise FileNotFoundError(
            f"No batch files found in {short_batches_dir} or {long_batches_dir}"
        )

    print(f"Total results: {len(all_results)}")

    # Sort by original_row_idx to restore original order
    if all_results and "original_row_idx" in all_results[0]:
        all_results.sort(key=lambda x: x.get("original_row_idx", 0))
        print("✓ Sorted by original_row_idx")

    # Save in requested format
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "json":
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved JSON: {output_path}")

    elif output_format == "csv":
        df = pd.DataFrame(all_results)
        output_path = output_path.with_suffix('.csv')
        df.to_csv(output_path, index=False)
        print(f"Saved CSV: {output_path}")

    elif output_format == "jsonl":
        output_path = output_path.with_suffix('.jsonl')
        with open(output_path, 'w') as f:
            for result in all_results:
                f.write(json.dumps(result) + '\n')
        print(f"Saved JSONL: {output_path}")

    # Update metadata
    metadata_file = checkpoint_dir / "metadata.json"
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

    metadata['merged'] = True
    metadata['total_results'] = len(all_results)
    metadata['output_file'] = str(output_path)

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    return len(all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge batch outputs from batched extraction")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--short-batches-dir",
        type=str,
        default=None,
        help="Short batches directory (default: checkpoint-dir/short_batches)",
    )
    parser.add_argument(
        "--long-batches-dir",
        type=str,
        default=None,
        help="Long batches directory (default: checkpoint-dir/long_batches)",
    )
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file")
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=["json", "csv", "jsonl"],
        default="csv",
        help="Output format",
    )

    args = parser.parse_args()

    total = merge_batches(
        args.checkpoint_dir,
        args.output,
        args.format,
        short_batches_dir=args.short_batches_dir,
        long_batches_dir=args.long_batches_dir,
    )
    print(f"\n✓ Merged {total} results")
