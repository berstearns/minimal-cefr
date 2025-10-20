"""
Dataset Train-Test Splitter

Split a CSV dataset into train and test sets using scikit-learn's train_test_split.
Supports stratification to maintain class distribution.

Usage:
    python -m utils.ds_split -i data.csv -o output_dir --test-size 0.2 --stratify-column cefr_label

Examples:
    1. Basic 80/20 split:
       python -m utils.ds_split -i dataset.csv -o splits/

    2. Stratified split (recommended for classification):
       python -m utils.ds_split -i dataset.csv -o splits/ \\
           --stratify-column cefr_label --test-size 0.2

    3. Custom output names:
       python -m utils.ds_split -i dataset.csv -o splits/ \\
           --train-name my-train.csv --test-name my-test.csv

    4. Multiple splits (create validation set):
       # First split: 80% train, 20% temp
       python -m utils.ds_split -i dataset.csv -o splits/ \\
           --train-name train.csv --test-name temp.csv --test-size 0.2

       # Second split: 50% of temp = 10% val, 10% test
       python -m utils.ds_split -i splits/temp.csv -o splits/ \\
           --train-name val.csv --test-name test.csv --test-size 0.5
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    input_csv: str,
    output_dir: str,
    test_size: float = 0.2,
    stratify_column: Optional[str] = None,
    train_name: str = "train.csv",
    test_name: str = "test.csv",
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[Path, Path]:
    """
    Split a CSV dataset into train and test sets.

    Args:
        input_csv: Path to input CSV file
        output_dir: Directory to save train and test files
        test_size: Proportion of dataset for test set (default: 0.2)
        stratify_column: Column name for stratified splitting (default: None)
        train_name: Filename for training set (default: "train.csv")
        test_name: Filename for test set (default: "test.csv")
        random_state: Random seed for reproducibility (default: 42)
        verbose: Print progress information (default: True)

    Returns:
        Tuple of (train_path, test_path)

    Raises:
        FileNotFoundError: If input CSV doesn't exist
        ValueError: If stratify column doesn't exist in the dataset
    """
    # Validate input
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("DATASET TRAIN-TEST SPLIT")
        print("=" * 70)
        print(f"Input: {input_csv}")
        print(f"Output directory: {output_dir}")
        print(f"Test size: {test_size:.1%}")
        print(f"Random state: {random_state}")
        if stratify_column:
            print(f"Stratify by: {stratify_column}")
        print("=" * 70)

    # Load dataset
    if verbose:
        print(f"\nLoading dataset from {input_csv}...")

    df = pd.read_csv(input_csv)

    if verbose:
        print(f"✓ Loaded {len(df)} samples")
        print(f"  Columns: {list(df.columns)}")

    # Check stratify column exists
    if stratify_column:
        if stratify_column not in df.columns:
            raise ValueError(
                f"Stratify column '{stratify_column}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )

        if verbose:
            print(f"\nClass distribution in '{stratify_column}':")
            class_counts = df[stratify_column].value_counts().sort_index()
            for label, count in class_counts.items():
                pct = count / len(df) * 100
                print(f"  {label}: {count} ({pct:.1f}%)")

    # Perform split
    if verbose:
        print(f"\nSplitting dataset (test_size={test_size})...")

    stratify_data = df[stratify_column] if stratify_column else None

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify_data
    )

    if verbose:
        print(f"✓ Train set: {len(train_df)} samples ({len(train_df)/len(df):.1%})")
        print(f"✓ Test set: {len(test_df)} samples ({len(test_df)/len(df):.1%})")

    # Verify stratification worked
    if stratify_column and verbose:
        print("\nTrain set distribution:")
        train_counts = train_df[stratify_column].value_counts().sort_index()
        for label, count in train_counts.items():
            pct = count / len(train_df) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")

        print("\nTest set distribution:")
        test_counts = test_df[stratify_column].value_counts().sort_index()
        for label, count in test_counts.items():
            pct = count / len(test_df) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")

    # Save splits
    train_path = output_path / train_name
    test_path = output_path / test_name

    if verbose:
        print("\nSaving splits...")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    if verbose:
        print(f"✓ Train set saved to: {train_path}")
        print(f"✓ Test set saved to: {test_path}")
        print("\n" + "=" * 70)
        print("SPLIT COMPLETE!")
        print("=" * 70)

    return train_path, test_path


def main():
    parser = argparse.ArgumentParser(
        description="Split a CSV dataset into train and test sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  1. Basic 80/20 split:
     python -m utils.ds_split -i dataset.csv -o splits/

  2. Stratified split (recommended for classification):
     python -m utils.ds_split -i dataset.csv -o splits/ \\
         --stratify-column cefr_label --test-size 0.2

  3. Custom output names:
     python -m utils.ds_split -i dataset.csv -o splits/ \\
         --train-name my-train.csv --test-name my-test.csv

  4. 70/30 split with custom seed:
     python -m utils.ds_split -i dataset.csv -o splits/ \\
         --test-size 0.3 --random-state 123

  5. Create train/val/test splits:
     # First: 80% train, 20% temp
     python -m utils.ds_split -i dataset.csv -o splits/ \\
         --train-name train.csv --test-name temp.csv \\
         --test-size 0.2 --stratify-column label

     # Second: 50% val, 50% test (each 10% of original)
     python -m utils.ds_split -i splits/temp.csv -o splits/ \\
         --train-name val.csv --test-name test.csv \\
         --test-size 0.5 --stratify-column label

  6. Quiet mode (minimal output):
     python -m utils.ds_split -i dataset.csv -o splits/ -q

Usage Notes:
- Use --stratify-column for classification tasks to maintain class balance
- Default test_size is 0.2 (80% train, 20% test)
- Default random_state is 42 for reproducibility
- Output directory is created automatically if it doesn't exist
        """,
    )

    parser.add_argument(
        "-i", "--input", required=True, help="Path to input CSV file to split"
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Directory to save train and test CSV files",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of dataset for test set (default: 0.2)",
    )

    parser.add_argument(
        "--stratify-column",
        help="Column name for stratified splitting (maintains class distribution)",
    )

    parser.add_argument(
        "--train-name",
        default="train.csv",
        help="Filename for train set (default: train.csv)",
    )

    parser.add_argument(
        "--test-name",
        default="test.csv",
        help="Filename for test set (default: test.csv)",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output (default)",
    )

    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress verbose output"
    )

    args = parser.parse_args()

    # Handle quiet mode
    verbose = args.verbose and not args.quiet

    # Validate test_size
    if not 0 < args.test_size < 1:
        print(f"Error: --test-size must be between 0 and 1, got {args.test_size}")
        sys.exit(1)

    try:
        split_dataset(
            input_csv=args.input,
            output_dir=args.output_dir,
            test_size=args.test_size,
            stratify_column=args.stratify_column,
            train_name=args.train_name,
            test_name=args.test_name,
            random_state=args.random_state,
            verbose=verbose,
        )

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
