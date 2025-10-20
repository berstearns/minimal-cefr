"""
Step 1: Train Grouped TF-IDF Vectorizers for CEFR Classification Pipeline

This module trains separate TF-IDF vectorizers for each group (e.g., CEFR level)
and creates a composite transformer that concatenates all features.

Use case: Train one TF-IDF per CEFR level to capture level-specific vocabulary.
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    DataConfig,
    ExperimentConfig,
    GlobalConfig,
    OutputConfig,
    TfidfConfig,
)


class GroupedTfidfVectorizer(BaseEstimator, TransformerMixin):
    """
    Composite TF-IDF vectorizer that trains separate models per group
    and concatenates their features.
    """

    def __init__(
        self,
        group_column: str,
        max_features: int = 5000,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
    ):
        self.group_column = group_column
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf

        # Will be populated during fit
        self.group_vectorizers_: Dict[str, TfidfVectorizer] = {}
        self.groups_: List[str] = []
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit separate TF-IDF vectorizers for each group.

        Args:
            X: DataFrame with text column and group column
            y: Ignored

        Returns:
            self
        """
        if self.group_column not in X.columns:
            raise ValueError(
                f"Group column '{self.group_column}' not found in DataFrame"
            )

        if "text" not in X.columns:
            raise ValueError("'text' column required in DataFrame")

        # Get unique groups (sorted for consistency)
        self.groups_ = sorted(X[self.group_column].unique().astype(str))

        # Train one vectorizer per group
        for group in self.groups_:
            # Get texts for this group
            group_mask = X[self.group_column].astype(str) == group
            group_texts = X.loc[group_mask, "text"].fillna("").astype(str)

            if len(group_texts) == 0:
                continue

            # Train TF-IDF for this group
            vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                sublinear_tf=self.sublinear_tf,
            )

            vectorizer.fit(group_texts)
            self.group_vectorizers_[group] = vectorizer

        # Build feature names (group_feature format)
        self.feature_names_ = []
        for group in self.groups_:
            if group in self.group_vectorizers_:
                vectorizer = self.group_vectorizers_[group]
                for feature in vectorizer.get_feature_names_out():
                    self.feature_names_.append(f"{group}_{feature}")

        return self

    def transform(self, X) -> np.ndarray:
        """
        Transform texts using all group-specific vectorizers and concatenate.

        Args:
            X: DataFrame with text column, Series, or array-like of texts

        Returns:
            Concatenated feature matrix
        """
        # Handle different input types
        if isinstance(X, pd.DataFrame):
            if "text" not in X.columns:
                raise ValueError("'text' column required in DataFrame")
            texts = X["text"].fillna("").astype(str)
        elif isinstance(X, pd.Series):
            texts = X.fillna("").astype(str)
        else:
            # Assume array-like
            texts = pd.Series(X).fillna("").astype(str)

        # Transform with each group's vectorizer
        feature_matrices = []

        for group in self.groups_:
            if group not in self.group_vectorizers_:
                continue

            vectorizer = self.group_vectorizers_[group]
            group_features = vectorizer.transform(texts)
            feature_matrices.append(group_features)

        # Concatenate all features horizontally
        if feature_matrices:
            return np.hstack([f.toarray() for f in feature_matrices])
        else:
            return np.array([]).reshape(len(texts), 0)

    def get_feature_names_out(self, input_features=None):
        """Get feature names for output features."""
        return np.array(self.feature_names_)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            "group_column": self.group_column,
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "sublinear_tf": self.sublinear_tf,
        }

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


def train_grouped_tfidf(config: GlobalConfig, group_column: str) -> Path:  # noqa: C901
    """
    Train grouped TF-IDF vectorizer on features-training-data.

    Args:
        config: GlobalConfig containing all configuration
        group_column: Column name to group by (e.g., 'cefr_level')

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

    if group_column not in df.columns:
        raise ValueError(f"Group column '{group_column}' not found in {training_file}")

    # Prepare DataFrame with text and group columns
    df_filtered = df[[data_config.text_column, group_column]].copy()
    df_filtered.columns = ["text", group_column]

    # Apply filtering
    if data_config.min_text_length > 0:
        df_filtered = df_filtered[
            df_filtered["text"].fillna("").str.len() >= data_config.min_text_length
        ]

    if verbose:
        print(f"Training samples: {len(df_filtered)}")
        print(f"Grouping by: {group_column}")
        group_counts = df_filtered[group_column].value_counts().sort_index()
        print(f"Groups found: {len(group_counts)}")
        for group, count in group_counts.items():
            print(f"  - {group}: {count} samples")

    # Train Grouped TF-IDF vectorizer
    if verbose:
        print("\nTraining Grouped TfidfVectorizer...")

    grouped_tfidf = GroupedTfidfVectorizer(
        group_column=group_column,
        max_features=tfidf_config.max_features,
        ngram_range=tfidf_config.ngram_range,
        min_df=tfidf_config.min_df,
        max_df=tfidf_config.max_df,
        sublinear_tf=tfidf_config.sublinear_tf,
    )

    grouped_tfidf.fit(df_filtered)

    # Save model using new naming scheme: {hash}_tfidf_grouped
    tfidf_hash = tfidf_config.get_hash()
    feature_type = "tfidf_grouped"

    # Use the new get_feature_model_dir method
    output_dir = Path(exp_config.get_feature_model_dir(tfidf_hash, feature_type))
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.output_config.save_models:
        model_path = output_dir / "tfidf_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(grouped_tfidf, f)

        if verbose:
            print(f"\n✓ Grouped TF-IDF model saved to: {model_path}")
            print(f"✓ Number of groups: {len(grouped_tfidf.groups_)}")
            print(f"✓ Total vocabulary size: {len(grouped_tfidf.feature_names_)}")
            print("✓ Feature dimensions per group:")
            for group in grouped_tfidf.groups_:
                if group in grouped_tfidf.group_vectorizers_:
                    vocab_size = len(
                        grouped_tfidf.group_vectorizers_[group].vocabulary_
                    )
                    print(f"    - {group}: {vocab_size} features")
            print(f"✓ Config hash: {tfidf_hash}")
            print(f"✓ Feature type: {feature_type}")
            print(f"✓ Readable name: {tfidf_config.get_readable_name()}_{feature_type}")

    # Save config
    if config.output_config.save_config:
        model_config = {
            "model_type": "GroupedTfidfVectorizer",
            "feature_type": feature_type,
            "group_column": group_column,
            "groups": grouped_tfidf.groups_,
            "config_hash": tfidf_hash,
            "readable_name": f"{tfidf_config.get_readable_name()}_{feature_type}",
            "max_features": tfidf_config.max_features,
            "ngram_range": list(tfidf_config.ngram_range),
            "min_df": tfidf_config.min_df,
            "max_df": tfidf_config.max_df,
            "sublinear_tf": tfidf_config.sublinear_tf,
            "total_vocabulary_size": len(grouped_tfidf.feature_names_),
            "group_vocabulary_sizes": {
                group: len(grouped_tfidf.group_vectorizers_[group].vocabulary_)
                for group in grouped_tfidf.groups_
                if group in grouped_tfidf.group_vectorizers_
            },
            "training_file": training_file.name,
            "training_samples": len(df_filtered),
        }

        if config.output_config.save_json:
            config_path = output_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(model_config, f, indent=2)

            if verbose:
                print(f"✓ Config saved to: {config_path}")

    return output_dir


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for train_tfidf_groupby."""
    parser = argparse.ArgumentParser(
        description="Train Grouped TF-IDF Vectorizer for CEFR classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  1. Train TF-IDF grouped by CEFR level:
     python -m src.train_tfidf_groupby \\
         -e data/experiments/zero-shot \\
         --group-by cefr_level

  2. With custom TF-IDF parameters:
     python -m src.train_tfidf_groupby \\
         -e data/experiments/zero-shot \\
         --group-by cefr_level \\
         --max-features 1000 \\
         --ngram-min 1 --ngram-max 2

  3. Using config file:
     python -m src.train_tfidf_groupby \\
         -c config.yaml \\
         --group-by cefr_level

Note: This creates one TF-IDF model per group value and concatenates
their features. For CEFR levels, this captures level-specific vocabulary.
""",
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

    # Grouping configuration (REQUIRED)
    group_group = parser.add_argument_group("Grouping Configuration")
    group_group.add_argument(
        "--group-by",
        required=True,
        help="Column name to group by (e.g., 'cefr_level') - REQUIRED",
    )

    # TF-IDF configuration
    tfidf_group = parser.add_argument_group("TF-IDF Configuration")
    tfidf_group.add_argument(
        "--max-features",
        type=int,
        help="Maximum number of TF-IDF features per group (default: 5000)",
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
    """Main entry point for Grouped TF-IDF training."""
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
        print(f"\nGrouping by: {args.group_by}")
        print()

    # Train Grouped TF-IDF
    try:
        train_grouped_tfidf(config, args.group_by)
    except Exception as e:
        print(f"Error during Grouped TF-IDF training: {e}")
        raise


if __name__ == "__main__":
    main()
