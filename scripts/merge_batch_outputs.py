#!/usr/bin/env python3
"""
Merge batch JSON files from batched extraction into single output file.

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


def merge_batches(checkpoint_dir: str, output_file: str, output_format: str = "csv"):
    """
    Merge all batch files into single output.

    Args:
        checkpoint_dir: Directory containing batch checkpoints
        output_file: Output file path
        output_format: Output format (csv, json, jsonl)
    """
    checkpoint_dir = Path(checkpoint_dir)
    batches_dir = checkpoint_dir / "batches"

    if not batches_dir.exists():
        raise FileNotFoundError(f"Batches directory not found: {batches_dir}")

    batch_files = sorted(batches_dir.glob("batch_*.json"))
    if not batch_files:
        raise FileNotFoundError(f"No batch files found in {batches_dir}")

    print(f"Found {len(batch_files)} batch files")

    # Merge all results
    all_results = []
    for batch_file in tqdm(batch_files, desc="Merging batches"):
        with open(batch_file, 'r') as f:
            results = json.load(f)
        all_results.extend(results)

    print(f"Total results: {len(all_results)}")

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
    parser = argparse.ArgumentParser(description="Merge batch outputs")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory",
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

    total = merge_batches(args.checkpoint_dir, args.output, args.format)
    print(f"\n✓ Merged {total} results")
