"""
Concatenate Multiple Feature Directories

This script combines multiple feature directories into a single directory,
preserving all files with source-specific prefixes.

Usage:
    python -m src.concat_features -e data/experiments/zero-shot-2 \\
        --features db3a2b11_tfidf_grouped e89a99e6_tfidf_grouped \\
        -o features/combined

Output structure:
    features/
    └── db3a2b11_tfidf_grouped++e89a99e6_tfidf_grouped/
        ├── norm-CELVA-SP/
        │   ├── db3a2b11_tfidf_grouped-config.json
        │   ├── db3a2b11_tfidf_grouped-feature_names.csv
        │   ├── db3a2b11_tfidf_grouped-features_dense.csv
        │   ├── e89a99e6_tfidf_grouped-config.json
        │   ├── e89a99e6_tfidf_grouped-feature_names.csv
        │   └── e89a99e6_tfidf_grouped-features_dense.csv
        └── ...
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import List


def validate_feature_directories(
    experiment_dir: Path, feature_dirs: List[str]
) -> List[Path]:
    """
    Validate that all feature directories exist in the experiment.

    Args:
        experiment_dir: Path to experiment directory
        feature_dirs: List of feature directory names

    Returns:
        List of validated Path objects

    Raises:
        FileNotFoundError: If any feature directory doesn't exist
    """
    features_base = experiment_dir / "features"

    if not features_base.exists():
        raise FileNotFoundError(f"Features directory not found: {features_base}")

    validated_paths = []

    for feature_dir in feature_dirs:
        feature_path = features_base / feature_dir

        if not feature_path.exists():
            raise FileNotFoundError(f"Feature directory not found: {feature_path}")

        if not feature_path.is_dir():
            raise ValueError(f"Not a directory: {feature_path}")

        validated_paths.append(feature_path)

    return validated_paths


def get_all_datasets(feature_paths: List[Path]) -> List[str]:
    """
    Get union of all dataset subdirectories across feature directories.

    Args:
        feature_paths: List of feature directory paths

    Returns:
        Sorted list of unique dataset names
    """
    all_datasets = set()

    for feature_path in feature_paths:
        # Find all subdirectories (datasets)
        datasets = [d.name for d in feature_path.iterdir() if d.is_dir()]
        all_datasets.update(datasets)

    return sorted(all_datasets)


def concat_features(
    experiment_dir: Path,
    feature_dirs: List[str],
    output_dir: str = None,
    verbose: bool = False,
    dry_run: bool = False,
) -> Path:
    """
    Concatenate multiple feature directories into one.

    Args:
        experiment_dir: Path to experiment directory
        feature_dirs: List of feature directory names to concatenate
        output_dir: Custom output directory name (optional)
        verbose: Print detailed progress
        dry_run: Show what would be done without actually copying files

    Returns:
        Path to created output directory

    Raises:
        FileNotFoundError: If experiment or feature directories don't exist
        ValueError: If less than 2 feature directories provided
    """
    if len(feature_dirs) < 2:
        raise ValueError(
            f"Need at least 2 feature directories to concatenate, got {len(feature_dirs)}"
        )

    # Validate input directories
    feature_paths = validate_feature_directories(experiment_dir, feature_dirs)

    # Determine output directory name
    if output_dir:
        output_name = output_dir
    else:
        output_name = "++".join(feature_dirs)

    output_path = experiment_dir / "features" / output_name

    if verbose:
        print(f"Input feature directories ({len(feature_dirs)}):")
        for i, fdir in enumerate(feature_dirs, 1):
            print(f"  {i}. {fdir}")
        print(f"\nOutput directory: {output_path}")
        print(f"Dry run: {dry_run}")

    # Get all unique datasets across all feature directories
    all_datasets = get_all_datasets(feature_paths)

    if verbose:
        print(f"\nFound {len(all_datasets)} unique datasets:")
        for dataset in all_datasets:
            print(f"  - {dataset}")

    # Create output directory structure
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)

    # Process each dataset
    files_copied = 0
    files_skipped = 0

    for dataset in all_datasets:
        if verbose:
            print(f"\nProcessing dataset: {dataset}")

        # Create dataset subdirectory in output
        dataset_output_dir = output_path / dataset

        if not dry_run:
            dataset_output_dir.mkdir(parents=True, exist_ok=True)

        # Copy files from each feature directory
        for feature_dir, feature_path in zip(feature_dirs, feature_paths):
            dataset_source_dir = feature_path / dataset

            if not dataset_source_dir.exists():
                if verbose:
                    print(f"  ⚠ Skipping {feature_dir}/{dataset} (does not exist)")
                continue

            # Get all files in this dataset directory
            files = [f for f in dataset_source_dir.iterdir() if f.is_file()]

            if verbose:
                print(f"  Copying from {feature_dir}/{dataset} ({len(files)} files)")

            for file in files:
                # Create new filename with prefix
                new_filename = f"{feature_dir}-{file.name}"
                output_file = dataset_output_dir / new_filename

                if dry_run:
                    print(f"    [DRY RUN] Would copy: {file} -> {output_file}")
                    files_copied += 1
                else:
                    # Copy file
                    shutil.copy2(file, output_file)

                    if verbose:
                        print(f"    ✓ {file.name} -> {new_filename}")

                    files_copied += 1

    # Summary
    if verbose or dry_run:
        print("\n" + "=" * 70)
        print("CONCATENATION SUMMARY")
        print("=" * 70)
        print(f"Input directories: {len(feature_dirs)}")
        print(f"Datasets processed: {len(all_datasets)}")
        print(f"Files copied: {files_copied}")
        if files_skipped > 0:
            print(f"Files skipped: {files_skipped}")
        print(f"Output directory: {output_path}")

        if dry_run:
            print("\n⚠ DRY RUN - No files were actually copied")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate multiple feature directories into one",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  1. Concatenate two feature directories:
     python -m src.concat_features -e data/experiments/zero-shot-2 \\
         --features db3a2b11_tfidf_grouped e89a99e6_tfidf_grouped

     Output: features/db3a2b11_tfidf_grouped++e89a99e6_tfidf_grouped/

  2. Concatenate with custom output name:
     python -m src.concat_features -e data/experiments/zero-shot-2 \\
         --features db3a2b11_tfidf_grouped e89a99e6_tfidf_grouped \\
         -o combined_features

     Output: features/combined_features/

  3. Concatenate three directories:
     python -m src.concat_features -e data/experiments/zero-shot-2 \\
         --features dir1 dir2 dir3

     Output: features/dir1++dir2++dir3/

  4. Dry run (preview without copying):
     python -m src.concat_features -e data/experiments/zero-shot-2 \\
         --features db3a2b11_tfidf_grouped e89a99e6_tfidf_grouped \\
         --dry-run

  5. Verbose output:
     python -m src.concat_features -e data/experiments/zero-shot-2 \\
         --features db3a2b11_tfidf_grouped e89a99e6_tfidf_grouped \\
         -v

Output Structure:
-----------------
For each dataset subdirectory found in any input feature directory,
the script creates a corresponding subdirectory in the output directory
and copies all files from each source with a prefix indicating their origin.

Input:
  features/db3a2b11_tfidf_grouped/norm-CELVA-SP/features_dense.csv
  features/e89a99e6_tfidf_grouped/norm-CELVA-SP/features_dense.csv

Output:
  features/db3a2b11_tfidf_grouped++e89a99e6_tfidf_grouped/
    norm-CELVA-SP/
      db3a2b11_tfidf_grouped-features_dense.csv
      e89a99e6_tfidf_grouped-features_dense.csv
""",
    )

    parser.add_argument(
        "-e",
        "--experiment-dir",
        required=True,
        help="Path to experiment directory",
    )

    parser.add_argument(
        "--features",
        nargs="+",
        required=True,
        metavar="FEATURE_DIR",
        help="Feature directory names to concatenate (minimum 2)",
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Custom output directory name (default: concatenated input names with ++)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually copying files",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Validate experiment directory
    experiment_dir = Path(args.experiment_dir)

    if not experiment_dir.exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        sys.exit(1)

    # Run concatenation
    try:
        output_path = concat_features(
            experiment_dir=experiment_dir,
            feature_dirs=args.features,
            output_dir=args.output,
            verbose=args.verbose,
            dry_run=args.dry_run,
        )

        if not args.dry_run:
            print("\n✓ Features concatenated successfully!")
            print(f"Output: {output_path}")
        else:
            print("\n✓ Dry run completed (no files copied)")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
