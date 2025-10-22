"""
Step 2: Extract Features for Test Sets using Trained TF-IDF Model

This module extracts TF-IDF features for test sets and saves them in the
required format: features_dense.csv, feature_names.csv, config.json

Expected File Structure
-----------------------
INPUT:
experiment-dir/
├── ml-test-data/                   # For data_source='test' (default)
│   ├── test-set-1/
│   │   └── test-set-1.csv          # Must have 'text' column
│   └── test-set-2/
│       └── test-set-2.csv
├── ml-training-data/               # For data_source='training'
│   └── train-data.csv
└── feature-models/
    └── tfidf/
        └── {hash}_tfidf/
            └── tfidf_model.pkl     # Pre-trained TF-IDF model (from train_tfidf.py)

OUTPUT:
experiment-dir/
└── features/
    └── {hash}_tfidf/               # Hash matches TF-IDF config
        ├── test-set-1/
        │   ├── features_dense.csv  # TF-IDF features as dense matrix
        │   ├── feature_names.csv   # Feature vocabulary
        │   └── config.json         # Extraction metadata
        ├── test-set-2/
        │   └── ...
        └── train-data/             # If data_source='training' or 'both'
            └── ...

COMBINATORIC USAGE PATTERNS
============================

This script supports flexible configuration through three mechanisms:
1. Configuration files (JSON/YAML)
2. CLI arguments
3. Combination of config files + CLI overrides

Directory Flags:
  -e, --experiment-dir    Base experiment directory
  -o, --output-dir        Where TF-IDF models are saved (default: <experiment-dir>/feature-models)
  -p, --pretrained-dir    Where to load pre-trained TF-IDF from (default: <output-dir>/tfidf)

Processing Flags:
  -t, --test-file         Process specific file (default: all files)
  -s, --data-source       Data source: test, training, or both (default: test)
  -c, --config-file       Load configuration from file
  --text-column           Text column name (default: text)

Output Flags:
  -q, --quiet            Suppress verbose output
  --no-save-config       Skip saving config files
  --no-save-features     Skip saving feature files

USAGE EXAMPLES
==============

Basic Usage:
-----------
1. Extract features using default paths:
   $ python -m src.extract_features -e data/experiments/zero-shot

   TF-IDF loaded from: data/experiments/zero-shot/feature-models/tfidf/
   Features saved to: data/experiments/zero-shot/features/

2. Process single test file:
   $ python -m src.extract_features -e data/experiments/zero-shot -t norm-celva-sp.csv

   Processes only: norm-celva-sp.csv
   Output: data/experiments/zero-shot/features/norm-celva-sp/

Data Source Selection:
---------------------
2a. Extract features from training data:
    $ python -m src.extract_features -e data/experiments/zero-shot -s training

    Data from: data/experiments/zero-shot/ml-training-data/
    Features saved to: data/experiments/zero-shot/features/

2b. Extract features from both test and training data:
    $ python -m src.extract_features -e data/experiments/zero-shot -s both

    Data from: ml-test-data/ AND ml-training-data/
    Features saved to: data/experiments/zero-shot/features/

2c. Process specific training file:
    $ python -m src.extract_features -e data/experiments/zero-shot -s training -t train-file.csv

    Processes only: train-file.csv from ml-training-data/
    Output: data/experiments/zero-shot/features/train-file/

Custom Output Directory:
-----------------------
3. Use custom model output directory:
   $ python -m src.extract_features -e data/experiments/zero-shot -o /custom/models

   TF-IDF loaded from: /custom/models/tfidf/
   Features saved to: data/experiments/zero-shot/features/

4. Completely separate directories:
   $ python -m src.extract_features -e exp1 -o exp2/models

   Test data from: exp1/ml-test-data/
   TF-IDF from: exp2/models/tfidf/
   Features to: exp1/features/

Custom Pre-trained Directory:
-----------------------------
5. Load TF-IDF from specific location:
   $ python -m src.extract_features -e data/experiments/zero-shot -p /trained/tfidf

   TF-IDF loaded from: /trained/tfidf/
   Features saved to: data/experiments/zero-shot/features/

6. Use pre-trained from different experiment:
   $ python -m src.extract_features -e exp1 -p exp2/feature-models/tfidf

   Test data from: exp1/ml-test-data/
   TF-IDF from: exp2/feature-models/tfidf/
   Features to: exp1/features/

7. Full custom paths (cross-experiment features):
   $ python -m src.extract_features \
       -e data/experiments/90-10 \
       -o models/shared \
       -p models/shared/tfidf

   Test data from: data/experiments/90-10/ml-test-data/
   TF-IDF from: models/shared/tfidf/
   Features to: data/experiments/90-10/features/

Config File Usage:
-----------------
8. Use config file with defaults:
   $ python -m src.extract_features -c config.yaml

   All settings from: config.yaml

9. Config file + CLI overrides:
   $ python -m src.extract_features -c config.yaml -e data/experiments/custom

   Base config from: config.yaml
   Overrides: experiment_dir = data/experiments/custom

10. Config file + multiple overrides:
    $ python -m src.extract_features \
        -c config.yaml \
        -e data/experiments/zero-shot \
        -p /shared/tfidf \
        -t norm-celva-sp.csv

    Base config from: config.yaml
    Overrides: experiment_dir, pretrained_dir, single test file

Advanced Combinatorics:
----------------------
11. Shared TF-IDF across multiple experiments:
    $ python -m src.extract_features -e exp1 -p /shared/tfidf
    $ python -m src.extract_features -e exp2 -p /shared/tfidf
    $ python -m src.extract_features -e exp3 -p /shared/tfidf

    Same TF-IDF model applied to different test sets

12. Different experiments, same model directory:
    $ python -m src.extract_features -e exp1 -o /models/central
    $ python -m src.extract_features -e exp2 -o /models/central

    Models in: /models/central/tfidf/
    Features in: exp1/features/ and exp2/features/

13. Process subset with custom column name:
    $ python -m src.extract_features \
        -e data/experiments/zero-shot \
        -t custom-test.csv \
        --text-column essay \
        -p /models/tfidf

    Test file: custom-test.csv
    Text column: essay (instead of default 'text')
    TF-IDF from: /models/tfidf/

14. Quiet mode with custom paths:
    $ python -m src.extract_features -e exp1 -p /tfidf -q

    Minimal output, custom TF-IDF location

15. Testing without saving:
    $ python -m src.extract_features \
        -e data/experiments/zero-shot \
        --no-save-features \
        --no-save-config \
        -q

    Runs extraction but doesn't save any files (dry-run mode)

Real-world Scenarios:
--------------------
16. Production setup (separate train/test experiments):
    # Train on large dataset
    $ python -m src.train_tfidf -e data/experiments/train-large -o /models/production

    # Extract features from test experiments using production model
    $ python -m src.extract_features -e data/experiments/test-1 -p /models/production/tfidf
    $ python -m src.extract_features -e data/experiments/test-2 -p /models/production/tfidf

17. Cross-validation scenario:
    # Fold 1
    $ python -m src.train_tfidf -e cv/fold1 -o cv/models/fold1
    $ python -m src.extract_features -e cv/fold1 -p cv/models/fold1/tfidf

    # Fold 2
    $ python -m src.train_tfidf -e cv/fold2 -o cv/models/fold2
    $ python -m src.extract_features -e cv/fold2 -p cv/models/fold2/tfidf

18. Multi-language experiments with shared infrastructure:
    $ python -m src.extract_features -e data/experiments/english -p /models/multilingual/tfidf
    $ python -m src.extract_features -e data/experiments/spanish -p /models/multilingual/tfidf
    $ python -m src.extract_features -e data/experiments/french -p /models/multilingual/tfidf

19. A/B testing different TF-IDF models:
    # Test with model A
    $ python -m src.extract_features -e exp1 -p /models/tfidf_v1 -o /results/model_a

    # Test with model B
    $ python -m src.extract_features -e exp1 -p /models/tfidf_v2 -o /results/model_b

    Same test data, different TF-IDF models

20. Complete pipeline with custom paths:
    # Step 1: Train TF-IDF
    $ python -m src.train_tfidf -e data/experiments/zero-shot -o /shared/models

    # Step 2: Extract features from both test and training data
    $ python -m src.extract_features \
        -e data/experiments/zero-shot \
        -o /shared/models \
        -p /shared/models/tfidf \
        -s both

    # Step 3: Train classifiers
    $ python -m src.train_classifiers -e data/experiments/zero-shot -o /shared/models

Training vs Test Data:
---------------------
21. Extract features only from training data:
    $ python -m src.extract_features -e data/experiments/zero-shot -s training

    Processes: data/experiments/zero-shot/ml-training-data/*.csv
    Output: data/experiments/zero-shot/features/

22. Extract from both sources with custom TF-IDF:
    $ python -m src.extract_features \
        -e data/experiments/90-10 \
        -p /models/shared/tfidf \
        -s both

    Processes: ml-training-data/*.csv AND ml-test-data/*.csv
    TF-IDF from: /models/shared/tfidf/
    Output: data/experiments/90-10/features/

23. Training data with separate model directory:
    $ python -m src.extract_features \
        -e data/experiments/zero-shot \
        -o /models/production \
        -s training

    Processes: data/experiments/zero-shot/ml-training-data/*.csv
    TF-IDF from: /models/production/tfidf/
    Output: data/experiments/zero-shot/features/

PATH RESOLUTION LOGIC
=====================

The script resolves paths in the following order:

1. TF-IDF Model Location:
   if --pretrained-dir specified:
       use --pretrained-dir
   else if --output-dir specified:
       use --output-dir/tfidf
   else:
       use --experiment-dir/feature-models/tfidf

2. Input Data Location:
   if --data-source is "test":
       use --experiment-dir/ml-test-data/
   elif --data-source is "training":
       use --experiment-dir/ml-training-data/
   elif --data-source is "both":
       use both directories

3. Features Output Location:
   always: --experiment-dir/features/<data-file-name>/

COMMON PATTERNS
===============

Pattern 1: Default Everything
  $ python -m src.extract_features -e data/experiments/zero-shot
  TF-IDF: data/experiments/zero-shot/feature-models/tfidf/
  Test:   data/experiments/zero-shot/ml-test-data/
  Output: data/experiments/zero-shot/features/

Pattern 2: Custom Model Directory
  $ python -m src.extract_features -e data/experiments/zero-shot -o /models
  TF-IDF: /models/tfidf/
  Test:   data/experiments/zero-shot/ml-test-data/
  Output: data/experiments/zero-shot/features/

Pattern 3: Custom Pre-trained Directory
  $ python -m src.extract_features -e data/experiments/zero-shot -p /tfidf
  TF-IDF: /tfidf/
  Test:   data/experiments/zero-shot/ml-test-data/
  Output: data/experiments/zero-shot/features/

Pattern 4: Both Custom Directories
  $ python -m src.extract_features -e exp1 -o /models -p /pretrained
  TF-IDF: /pretrained/
  Test:   exp1/ml-test-data/
  Output: exp1/features/
  Note: -o is ignored when -p is specified (explicit override)

TIPS & BEST PRACTICES
======================

1. Use -p when loading TF-IDF from a different experiment
2. Use -o when you want all models in a centralized location
3. Use -t to test on a single file before processing all files
4. Use -s to specify which data to process (test, training, or both)
5. Use config files for complex setups with many parameters
6. Use --no-save-features for quick validation without disk writes
7. Combine -q with scripts for cleaner logs in batch processing
8. The -p flag always takes precedence over -o for TF-IDF location
9. Use -s both to extract features from both training and test data
10. Features always save to <experiment-dir>/features/
11. Use absolute paths to avoid ambiguity in complex workflows
12. Extract training features when you need them for classifier training

"""

import argparse
import json
import pickle
from pathlib import Path
from typing import List

import pandas as pd

from src.config import DataConfig, ExperimentConfig, GlobalConfig, OutputConfig

# Import GroupedTfidfVectorizer so pickle can load it
from src.train_tfidf_groupby import GroupedTfidfVectorizer  # noqa: F401


def extract_features_for_file(  # noqa: C901
    config: GlobalConfig, file_name: str, data_source: str = "test"
) -> Path:
    """
    Extract TF-IDF features for a specific data file.

    Args:
        config: GlobalConfig containing all configuration
        file_name: Name of file in data source folder
        data_source: Data source ('test' or 'training')

    Returns:
        Path to features directory
    """
    verbose = config.output_config.verbose
    exp_config = config.experiment_config
    tfidf_config = config.tfidf_config
    data_config = config.data_config

    # Load TF-IDF model from directory
    if exp_config.pretrained_tfidf_dir:
        tfidf_model_dir = Path(exp_config.pretrained_tfidf_dir)
    else:
        tfidf_model_dir = Path(exp_config.get_tfidf_model_dir(tfidf_config))

    tfidf_model_path = tfidf_model_dir / "tfidf_model.pkl"
    if not tfidf_model_path.exists():
        raise FileNotFoundError(
            f"TF-IDF model not found: {tfidf_model_path}. Run train_tfidf.py first."
        )

    # Load model config to get feature_type
    config_path = tfidf_model_dir / "config.json"
    feature_type = "tfidf"  # Default
    config_hash = tfidf_config.get_hash()  # Default

    if config_path.exists():
        with open(config_path, "r") as f:
            model_config = json.load(f)
            # Get feature_type, with backward compatibility
            feature_type = model_config.get("feature_type")
            if feature_type is None:
                # Old config format - infer from model_type
                if model_config.get("model_type") == "GroupedTfidfVectorizer":
                    feature_type = "tfidf_grouped"
                else:
                    feature_type = "tfidf"

            # Get config_hash (old configs use "tfidf_hash")
            config_hash = model_config.get("config_hash") or model_config.get(
                "tfidf_hash", config_hash
            )

    if verbose:
        print(f"Loading TF-IDF model: {tfidf_model_path}")
        print(f"Feature type: {feature_type}")

    with open(tfidf_model_path, "rb") as f:
        tfidf = pickle.load(f)

    # Load data from appropriate source
    if data_source == "test":
        data_dir = exp_config.ml_test_dir
    elif data_source == "training":
        data_dir = exp_config.ml_training_dir
    else:
        raise ValueError(
            f"Invalid data_source: {data_source}. Must be 'test' or 'training'"
        )

    data_file_path = Path(data_dir) / file_name
    if not data_file_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_file_path}")

    if verbose:
        print(f"Loading data from: {data_file_path}")

    df_data = pd.read_csv(data_file_path)

    if data_config.text_column not in df_data.columns:
        raise ValueError(
            f"'{data_config.text_column}' column required in {data_file_path}"
        )

    X_data = df_data[data_config.text_column].fillna("").astype(str)

    if verbose:
        print(f"Samples: {len(X_data)}")

    # Extract features
    if verbose:
        print("Extracting TF-IDF features...")

    X_data_tfidf = tfidf.transform(X_data)

    # Handle both sparse matrices (standard TF-IDF) and dense arrays (grouped TF-IDF)
    if hasattr(X_data_tfidf, "toarray"):
        X_data_dense = X_data_tfidf.toarray()
    else:
        X_data_dense = X_data_tfidf

    # Create output folder using feature_type (e.g., features/252cd532_tfidf/dataset)
    data_name = data_file_path.stem
    features_base_dir = Path(exp_config.get_features_dir(config_hash, feature_type))
    features_dir = features_base_dir / data_name
    features_dir.mkdir(parents=True, exist_ok=True)

    if config.output_config.save_features:
        # Save features_dense.csv
        features_dense_path = features_dir / "features_dense.csv"
        pd.DataFrame(X_data_dense).to_csv(features_dense_path, index=False)

        # Save feature_names.csv
        feature_names = tfidf.get_feature_names_out()
        feature_names_path = features_dir / "feature_names.csv"
        pd.DataFrame({"feature_name": feature_names}).to_csv(
            feature_names_path, index=False
        )

        if verbose:
            print(f"\n✓ Features saved to: {features_dir}")
            print(f"  - features_dense.csv: {X_data_dense.shape}")
            print(f"  - feature_names.csv: {len(feature_names)} features")

    # Save config
    if config.output_config.save_config:
        feature_config = {
            "data_file": file_name,
            "data_source": data_source,
            "feature_type": feature_type,
            "config_hash": config_hash,
            "n_samples": int(len(X_data)),
            "n_features": int(X_data_dense.shape[1]),
            "model_path": str(tfidf_model_path.relative_to(exp_config.experiment_dir)),
        }

        if config.output_config.save_json:
            config_path = features_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(feature_config, f, indent=2)

            if verbose:
                print("  - config.json")

    return features_dir


def extract_all_from_source(  # noqa: C901
    config: GlobalConfig, data_source: str = "test"
) -> List[Path]:
    """
    Extract features for all files in specified data source.

    Args:
        config: GlobalConfig containing all configuration
        data_source: Data source to process ('test', 'training', or 'both')

    Returns:
        List of paths to created feature directories
    """
    verbose = config.output_config.verbose
    exp_config = config.experiment_config

    feature_dirs = []

    # Determine which sources to process
    sources_to_process = []
    if data_source == "both":
        sources_to_process = ["test", "training"]
    else:
        sources_to_process = [data_source]

    for source in sources_to_process:
        if source == "test":
            data_dir = Path(exp_config.ml_test_dir)
            source_label = "test"
        else:
            data_dir = Path(exp_config.ml_training_dir)
            source_label = "training"

        if not data_dir.exists():
            print(f"Warning: {source_label} data directory not found: {data_dir}")
            continue

        data_files = sorted(data_dir.glob("*.csv"))
        if not data_files:
            print(f"No {source_label} files found in {data_dir}")
            continue

        if verbose:
            print(f"\nProcessing {source_label.upper()} data")
            print(f"Found {len(data_files)} {source_label} files\n")
            print("=" * 70)

        for data_file in data_files:
            if verbose:
                print(f"\nProcessing: {data_file.name}")
                print("-" * 70)

            try:
                feature_dir = extract_features_for_file(config, data_file.name, source)
                feature_dirs.append(feature_dir)
            except Exception as e:
                print(f"✗ Error: {e}")

        if verbose:
            print("\n" + "=" * 70)
            print(f"{source_label.capitalize()} feature extraction complete!")

    return feature_dirs


def extract_all_testsets(config: GlobalConfig) -> List[Path]:
    """
    Extract features for all test sets in ml-test-data.

    Backward compatibility wrapper for extract_all_from_source.

    Args:
        config: GlobalConfig containing all configuration

    Returns:
        List of paths to created feature directories
    """
    return extract_all_from_source(config, data_source="test")


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for extract_features."""
    parser = argparse.ArgumentParser(
        description="Extract TF-IDF features for CEFR test sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Config loading
    config_group = parser.add_argument_group("Configuration Loading")
    config_method = config_group.add_mutually_exclusive_group()
    config_method.add_argument(
        "-c", "--config-file", help="Path to JSON or YAML config file"
    )
    config_method.add_argument(
        "--config-json", help="JSON string containing full configuration"
    )

    # Experiment configuration
    exp_group = parser.add_argument_group("Experiment Configuration")
    exp_group.add_argument(
        "-e",
        "--experiment-dir",
        help="Path to experiment directory (e.g., data/experiments/zero-shot)",
    )
    exp_group.add_argument(
        "-o",
        "--output-dir",
        help="Custom output directory for models (default: <experiment-dir>/feature-models)",
    )
    exp_group.add_argument(
        "-p",
        "--pretrained-dir",
        help="Directory containing pre-trained TF-IDF model (default: <output-dir>/tfidf)",
    )

    # Data file selection
    data_group = parser.add_argument_group("Data File Selection")
    data_group.add_argument(
        "-t",
        "--test-file",
        help="Specific file name to process (e.g., norm-celva-sp.csv). If not provided, processes all files from data source.",
    )
    data_group.add_argument(
        "-s",
        "--data-source",
        choices=["test", "training", "both"],
        default="test",
        help="Data source to process: 'test' (ml-test-data), 'training' (ml-training-data), or 'both' (default: test)",
    )

    # Data configuration
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--text-column",
        default="text",
        help="Column name containing text (default: text)",
    )

    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--no-save-config", action="store_true", help="Skip saving configuration files"
    )
    output_group.add_argument(
        "--no-save-features",
        action="store_true",
        help="Skip saving feature files (for testing)",
    )
    output_group.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress verbose output"
    )

    return parser


def args_to_config(args: argparse.Namespace) -> GlobalConfig:  # noqa: C901
    """Convert argparse namespace to GlobalConfig."""
    # Check if config file or json string provided
    if args.config_file:
        config_path = Path(args.config_file)
        if config_path.suffix in [".yaml", ".yml"]:
            config = GlobalConfig.from_yaml_file(str(config_path))
        elif config_path.suffix == ".json":
            config = GlobalConfig.from_json_file(str(config_path))
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

        # CLI args override config file
        if args.experiment_dir:
            config.experiment_config = ExperimentConfig(
                experiment_dir=args.experiment_dir,
                models_dir=args.output_dir if args.output_dir else None,
                pretrained_tfidf_dir=(
                    args.pretrained_dir if args.pretrained_dir else None
                ),
            )
        elif args.output_dir or args.pretrained_dir:
            if args.output_dir:
                config.experiment_config.models_dir = args.output_dir
            if args.pretrained_dir:
                config.experiment_config.pretrained_tfidf_dir = args.pretrained_dir

        if args.text_column != "text":
            config.data_config.text_column = args.text_column

        if args.no_save_config:
            config.output_config.save_config = False
        if args.no_save_features:
            config.output_config.save_features = False
        if args.quiet:
            config.output_config.verbose = False

        return config

    elif args.config_json:
        config = GlobalConfig.from_json_string(args.config_json)
        # Apply CLI overrides
        if args.output_dir:
            config.experiment_config.models_dir = args.output_dir
        if args.pretrained_dir:
            config.experiment_config.pretrained_tfidf_dir = args.pretrained_dir
        return config

    else:
        # Build config from CLI args
        experiment_config = ExperimentConfig(
            experiment_dir=args.experiment_dir or "data/experiments/zero-shot",
            models_dir=args.output_dir if args.output_dir else None,
            pretrained_tfidf_dir=args.pretrained_dir if args.pretrained_dir else None,
        )

        data_config = DataConfig(text_column=args.text_column)

        output_config = OutputConfig(
            save_config=not args.no_save_config,
            save_features=not args.no_save_features,
            verbose=not args.quiet,
        )

        return GlobalConfig(
            experiment_config, data_config=data_config, output_config=output_config
        )


def main():  # noqa: C901
    """Main entry point for feature extraction."""
    parser = create_parser()
    args = parser.parse_args()

    # Build configuration
    try:
        config = args_to_config(args)
    except Exception as e:
        parser.error(f"Configuration error: {e}")

    if config.output_config.verbose:
        print("Configuration:")
        print(json.dumps(config.to_dict(), indent=2))

        # Show TF-IDF model location
        if config.experiment_config.pretrained_tfidf_dir:
            tfidf_dir = Path(config.experiment_config.pretrained_tfidf_dir)
        else:
            tfidf_dir = Path(
                config.experiment_config.get_tfidf_model_dir(config.tfidf_config)
            )

        # Load model config to get feature_type
        model_config_path = tfidf_dir / "config.json"
        feature_type = "tfidf"  # Default
        config_hash = config.tfidf_config.get_hash()  # Default

        if model_config_path.exists():
            with open(model_config_path, "r") as f:
                model_cfg = json.load(f)
                # Get feature_type, with backward compatibility
                feature_type = model_cfg.get("feature_type")
                if feature_type is None:
                    # Old config format - infer from model_type
                    if model_cfg.get("model_type") == "GroupedTfidfVectorizer":
                        feature_type = "tfidf_grouped"
                    else:
                        feature_type = "tfidf"

                # Get config_hash (old configs use "tfidf_hash")
                config_hash = model_cfg.get("config_hash") or model_cfg.get(
                    "tfidf_hash", config_hash
                )

        print(f"\nTF-IDF model directory: {tfidf_dir}")
        print(f"Config hash: {config_hash}")
        print(
            f"Features output directory: {config.experiment_config.get_features_dir(config_hash, feature_type)}"
        )
        print()

    # Extract features
    try:
        if args.test_file:
            # Extract features for specific file
            if args.data_source == "both":
                # Check both directories for the file
                sources_to_try = ["test", "training"]
                for source in sources_to_try:
                    try:
                        extract_features_for_file(config, args.test_file, source)
                    except FileNotFoundError:
                        continue
            else:
                extract_features_for_file(config, args.test_file, args.data_source)
        else:
            # Extract features for all files from specified source(s)
            extract_all_from_source(config, args.data_source)
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        raise


if __name__ == "__main__":
    main()
