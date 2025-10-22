"""
Step 1: Train TF-IDF Vectorizer for CEFR Classification Pipeline

This module trains a TF-IDF vectorizer on features-training-data and saves
the fitted model for reuse in the sklearn pipeline.

Expected File Structure
-----------------------
INPUT:
experiment-dir/
└── ml-training-data/
    └── train-data.csv              # Must have 'text' column (configurable)

OUTPUT:
experiment-dir/
└── feature-models/
    └── tfidf/
        └── {hash}_tfidf/           # Hash based on TF-IDF config
            ├── tfidf_model.pkl     # Trained sklearn TfidfVectorizer
            └── config.json         # TF-IDF configuration metadata

Input CSV Format
----------------
- text_column (default: "text"): Text data for vocabulary extraction
- Any other columns are ignored during TF-IDF training
"""

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    DataConfig,
    ExperimentConfig,
    GlobalConfig,
    OutputConfig,
    TfidfConfig,
)


def train_tfidf(config: GlobalConfig) -> Path:  # noqa: C901
    """
    Train TF-IDF vectorizer on features-training-data.

    Args:
        config: GlobalConfig containing all configuration

    Returns:
        Path to saved TF-IDF model directory
    """
    verbose = config.output_config.verbose
    exp_config = config.experiment_config
    tfidf_config = config.tfidf_config
    data_config = config.data_config

    features_training_dir = Path(exp_config.features_training_dir)

    # Find training file in features-training-data folder
    csv_files = list(features_training_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {features_training_dir}")

    training_file = csv_files[0]
    if verbose:
        print(f"Loading features training data: {training_file}")

    # Load data
    df = pd.read_csv(training_file)
    if data_config.text_column not in df.columns:
        raise ValueError(
            f"'{data_config.text_column}' column required in {training_file}"
        )

    X_train = df[data_config.text_column].fillna("").astype(str)

    # Apply filtering
    if data_config.min_text_length > 0:
        X_train = X_train[X_train.str.len() >= data_config.min_text_length]

    if verbose:
        print(f"Training samples: {len(X_train)}")

    # Train TF-IDF vectorizer
    if verbose:
        print("Training TfidfVectorizer...")

    tfidf = TfidfVectorizer(
        max_features=tfidf_config.max_features,
        ngram_range=tfidf_config.ngram_range,
        min_df=tfidf_config.min_df,
        max_df=tfidf_config.max_df,
        sublinear_tf=tfidf_config.sublinear_tf,
    )
    tfidf.fit(X_train)

    # Save model to hashed directory (prevents overwrites with different TF-IDF configs)
    output_dir = Path(exp_config.get_tfidf_model_dir(tfidf_config))
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.output_config.save_models:
        model_path = output_dir / "tfidf_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(tfidf, f)

        if verbose:
            print(f"\n✓ TF-IDF model saved to: {model_path}")
            print(f"✓ Vocabulary size: {len(tfidf.vocabulary_)}")
            print(f"✓ Config hash: {tfidf_config.get_hash()}")
            print(f"✓ Readable name: {tfidf_config.get_readable_name()}")

    # Save config
    if config.output_config.save_config:
        model_config = {
            "model_type": "TfidfVectorizer",
            "feature_type": "tfidf",
            "config_hash": tfidf_config.get_hash(),
            "readable_name": tfidf_config.get_readable_name(),
            "max_features": tfidf_config.max_features,
            "ngram_range": list(tfidf_config.ngram_range),
            "min_df": tfidf_config.min_df,
            "max_df": tfidf_config.max_df,
            "sublinear_tf": tfidf_config.sublinear_tf,
            "vocabulary_size": len(tfidf.vocabulary_),
            "training_file": training_file.name,
            "training_samples": len(X_train),
        }

        if config.output_config.save_json:
            config_path = output_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(model_config, f, indent=2)

            if verbose:
                print(f"✓ Config saved to: {config_path}")

    return output_dir


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for train_tfidf."""
    parser = argparse.ArgumentParser(
        description="Train TF-IDF Vectorizer for CEFR classification",
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

    # TF-IDF configuration
    tfidf_group = parser.add_argument_group("TF-IDF Configuration")
    tfidf_group.add_argument(
        "--max-features",
        type=int,
        help="Maximum number of TF-IDF features (default: 5000)",
    )
    tfidf_group.add_argument(
        "--ngram-min", type=int, default=1, help="Minimum n-gram size (default: 1)"
    )
    tfidf_group.add_argument(
        "--ngram-max", type=int, default=2, help="Maximum n-gram size (default: 2)"
    )
    tfidf_group.add_argument(
        "--min-df", type=int, help="Minimum document frequency (default: 2)"
    )
    tfidf_group.add_argument(
        "--max-df", type=float, help="Maximum document frequency (default: 0.95)"
    )
    tfidf_group.add_argument(
        "--no-sublinear-tf", action="store_true", help="Disable sublinear TF scaling"
    )

    # Data configuration
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--text-column",
        default="text",
        help="Column name containing text (default: text)",
    )
    data_group.add_argument(
        "--min-length", type=int, help="Minimum text length filter (default: 0)"
    )

    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--no-save-config", action="store_true", help="Skip saving configuration files"
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
            )
        elif args.output_dir:
            config.experiment_config.models_dir = args.output_dir

        if args.max_features:
            config.tfidf_config.max_features = args.max_features
        if args.ngram_min or args.ngram_max:
            config.tfidf_config.ngram_range = (
                (
                    args.ngram_min
                    if args.ngram_min
                    else config.tfidf_config.ngram_range[0]
                ),
                (
                    args.ngram_max
                    if args.ngram_max
                    else config.tfidf_config.ngram_range[1]
                ),
            )
        if args.min_df:
            config.tfidf_config.min_df = args.min_df
        if args.max_df:
            config.tfidf_config.max_df = args.max_df
        if args.no_sublinear_tf:
            config.tfidf_config.sublinear_tf = False

        if args.text_column != "text":
            config.data_config.text_column = args.text_column
        if args.min_length:
            config.data_config.min_text_length = args.min_length

        if args.no_save_config:
            config.output_config.save_config = False
        if args.quiet:
            config.output_config.verbose = False

        return config

    elif args.config_json:
        config = GlobalConfig.from_json_string(args.config_json)
        # Apply CLI overrides
        if args.output_dir:
            config.experiment_config.models_dir = args.output_dir
        return config

    else:
        # Build config from CLI args
        experiment_config = ExperimentConfig(
            experiment_dir=args.experiment_dir or "data/experiments/zero-shot",
            models_dir=args.output_dir if args.output_dir else None,
        )

        tfidf_config = TfidfConfig(
            max_features=args.max_features or 5000,
            ngram_range=(args.ngram_min, args.ngram_max),
            min_df=args.min_df or 2,
            max_df=args.max_df or 0.95,
            sublinear_tf=not args.no_sublinear_tf,
        )

        data_config = DataConfig(
            text_column=args.text_column, min_text_length=args.min_length or 0
        )

        output_config = OutputConfig(
            save_config=not args.no_save_config, verbose=not args.quiet
        )

        return GlobalConfig(
            experiment_config,
            tfidf_config,
            data_config=data_config,
            output_config=output_config,
        )


def main():
    """Main entry point for TF-IDF training."""
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
        output_dir = Path(
            config.experiment_config.get_tfidf_model_dir(config.tfidf_config)
        )
        print(f"\nOutput directory: {output_dir}")
        print(f"Config hash: {config.tfidf_config.get_hash()}")
        print(f"Readable name: {config.tfidf_config.get_readable_name()}")
        print()

    # Train TF-IDF
    try:
        train_tfidf(config)
    except Exception as e:
        print(f"Error during TF-IDF training: {e}")
        raise


if __name__ == "__main__":
    main()
