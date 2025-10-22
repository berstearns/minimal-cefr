"""
Step 3: Train ML Classifiers for CEFR Classification

This module trains ML classifiers on pre-extracted features. It expects:
- Features file: CSV with flat features (one row per sample)
- Feature names file: CSV or TXT with feature names
- Labels: Either from CSV column or separate file
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import mord

    MORD_AVAILABLE = True
except ImportError:
    MORD_AVAILABLE = False

from src.config import (
    ClassifierConfig,
    DataConfig,
    ExperimentConfig,
    GlobalConfig,
    OutputConfig,
)

# Fixed CEFR classes (all 6 levels)
CEFR_CLASSES = ["A1", "A2", "B1", "B2", "C1", "C2"]


def get_cefr_label_encoder() -> LabelEncoder:
    """
    Create a fixed label encoder for all 6 CEFR classes.

    This ensures the model always expects all 6 CEFR levels,
    even if the training data doesn't contain all classes.

    Returns:
        LabelEncoder fitted with all CEFR classes
    """
    encoder = LabelEncoder()
    encoder.fit(CEFR_CLASSES)
    return encoder


def get_classifier(config: ClassifierConfig):
    """
    Get classifier instance based on configuration.

    Args:
        config: ClassifierConfig

    Returns:
        Sklearn classifier instance
    """
    if config.classifier_type == "multinomialnb":
        return MultinomialNB()

    elif config.classifier_type == "logistic":
        return LogisticRegression(
            max_iter=config.logistic_max_iter,
            random_state=config.random_state,
            class_weight=config.logistic_class_weight,
        )

    elif config.classifier_type == "randomforest":
        return RandomForestClassifier(
            n_estimators=config.rf_n_estimators,
            random_state=config.random_state,
            class_weight=config.rf_class_weight,
        )

    elif config.classifier_type == "svm":
        return LinearSVC(
            max_iter=config.svm_max_iter,
            random_state=config.random_state,
            class_weight=config.svm_class_weight,
        )

    elif config.classifier_type == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is not installed. Install it with: pip install xgboost"
            )

        # Configure tree method based on GPU usage
        tree_method = config.xgb_tree_method
        if config.xgb_use_gpu:
            if tree_method == "auto":
                tree_method = "gpu_hist"
            device = "cuda"
        else:
            if tree_method == "auto":
                tree_method = "hist"
            device = "cpu"

        # Determine objective and eval_metric based on objective type
        objective = config.xgb_objective

        # For regression objectives, use XGBRegressor and adjust eval_metric
        if objective.startswith("reg:"):
            # Use regressor for ordinal regression
            return xgb.XGBRegressor(
                n_estimators=config.xgb_n_estimators,
                max_depth=config.xgb_max_depth,
                learning_rate=config.xgb_learning_rate,
                tree_method=tree_method,
                device=device,
                random_state=config.random_state,
                objective=objective,
                eval_metric="rmse",
            )
        else:
            # Classification objectives
            eval_metric = "mlogloss" if "multi" in objective else "auc"
            return xgb.XGBClassifier(
                n_estimators=config.xgb_n_estimators,
                max_depth=config.xgb_max_depth,
                learning_rate=config.xgb_learning_rate,
                tree_method=tree_method,
                device=device,
                random_state=config.random_state,
                objective=objective,
                eval_metric=eval_metric,
                num_class=len(CEFR_CLASSES),  # Always expect 6 CEFR classes
            )

    elif config.classifier_type == "mord-lr":
        if not MORD_AVAILABLE:
            raise ImportError(
                "Mord library is not installed. Install it with: pip install mord\n"
                "Or use conda: conda install -c conda-forge mord"
            )

        # Use LogisticAT (All-Threshold variant) for ordinal logistic regression
        # This is well-suited for ordinal CEFR classification
        return mord.LogisticAT(alpha=config.mord_alpha, max_iter=1000)

    else:
        raise ValueError(f"Unknown classifier type: {config.classifier_type}")


def load_features_and_labels(
    features_file: str,
    feature_names_file: Optional[str] = None,
    labels_file: Optional[str] = None,
    labels_csv: Optional[str] = None,
    cefr_column: str = "cefr_label",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.Series]:
    """
    Load pre-extracted features and labels.

    Args:
        features_file: Path to features CSV (flat features, one row per sample)
        feature_names_file: Path to feature names file (CSV or TXT)
        labels_file: Path to separate labels file (one label per line)
        labels_csv: Path to CSV containing labels in a column
        cefr_column: Column name for labels in labels_csv
        verbose: Print loading information

    Returns:
        Tuple of (X_train, y_train, feature_names, labels_series)
    """
    # Load features
    if verbose:
        print(f"Loading features from: {features_file}")

    features_df = pd.read_csv(features_file)
    X_train = features_df.values

    # Load feature names if provided
    feature_names = []
    if feature_names_file:
        if verbose:
            print(f"Loading feature names from: {feature_names_file}")

        feature_names_path = Path(feature_names_file)
        if feature_names_path.suffix == ".txt":
            with open(feature_names_file, "r") as f:
                feature_names = [line.strip() for line in f if line.strip()]
        else:  # CSV
            fn_df = pd.read_csv(feature_names_file)
            feature_names = fn_df.iloc[:, 0].tolist()
    else:
        # Use column names from features file
        feature_names = features_df.columns.tolist()

    # Load labels
    if labels_file:
        if verbose:
            print(f"Loading labels from: {labels_file}")

        with open(labels_file, "r") as f:
            y_train = np.array([line.strip() for line in f if line.strip()])
        y_train_series = pd.Series(y_train)

    elif labels_csv:
        if verbose:
            print(f"Loading labels from CSV: {labels_csv}")

        labels_df = pd.read_csv(labels_csv)
        if cefr_column not in labels_df.columns:
            raise ValueError(f"Column '{cefr_column}' not found in {labels_csv}")

        y_train_series = labels_df[cefr_column]
        y_train = y_train_series.values

    else:
        raise ValueError("Must provide either labels_file or labels_csv")

    # Validate shapes match
    if len(X_train) != len(y_train):
        raise ValueError(
            f"Features and labels length mismatch: {len(X_train)} vs {len(y_train)}"
        )

    if verbose:
        print(f"Loaded {len(X_train)} samples with {len(feature_names)} features")
        print(f"Classes: {sorted(y_train_series.unique())}")
        print(f"Class distribution:\n{y_train_series.value_counts().sort_index()}")

    return X_train, y_train, feature_names, y_train_series


def train_classifier(
    config: GlobalConfig,
    features_file: str,
    feature_names_file: Optional[str] = None,
    labels_file: Optional[str] = None,
    labels_csv: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Path:
    """
    Train a classifier on pre-extracted features.

    Args:
        config: GlobalConfig containing all configuration
        features_file: Path to features CSV file
        feature_names_file: Path to feature names file (optional)
        labels_file: Path to labels file (optional, one per line)
        labels_csv: Path to CSV with labels column (optional)
        model_name: Custom model name (optional, derived from features_file if not provided)

    Returns:
        Path to saved classifier model directory
    """
    verbose = config.output_config.verbose
    exp_config = config.experiment_config
    classifier_config = config.classifier_config
    data_config = config.data_config

    # Load features and labels
    X_train, y_train, feature_names, y_train_series = load_features_and_labels(
        features_file=features_file,
        feature_names_file=feature_names_file,
        labels_file=labels_file,
        labels_csv=labels_csv,
        cefr_column=data_config.cefr_column,
        verbose=verbose,
    )

    # Create fixed CEFR label encoder (ensures all 6 classes even if training data lacks some)
    label_encoder = get_cefr_label_encoder()

    # Encode labels for training
    y_train_encoded = label_encoder.transform(y_train)

    if verbose:
        print("\nLabel encoding:")
        for label, encoded in zip(CEFR_CLASSES, range(len(CEFR_CLASSES))):
            print(f"  {label} → {encoded}")

    # Initialize classifier
    if verbose:
        print(f"\nTraining {classifier_config.classifier_type} classifier...")
        if classifier_config.classifier_type == "xgboost":
            print(f"  XGBoost objective: {classifier_config.xgb_objective}")

    clf = get_classifier(classifier_config)

    # Determine if this is an ordinal regression model (uses integer labels directly)
    is_ordinal_regression = classifier_config.classifier_type == "mord-lr" or (
        classifier_config.classifier_type == "xgboost"
        and classifier_config.xgb_objective.startswith("reg:")
    )

    # Train the classifier
    if is_ordinal_regression:
        # Ordinal regression: use integer-encoded labels (0-5)
        clf.fit(X_train, y_train_encoded)
    else:
        # Standard classification: use label-encoded values
        clf.fit(X_train, y_train_encoded)

    # Evaluate on training data
    if verbose:
        if is_ordinal_regression:
            # For regression models, predictions are continuous - round to integers
            y_pred_continuous = clf.predict(X_train)
            y_pred_encoded = np.clip(
                np.round(y_pred_continuous), 0, len(CEFR_CLASSES) - 1
            ).astype(int)
        else:
            y_pred_encoded = clf.predict(X_train)

        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        print("\nTraining performance:")
        print(classification_report(y_train, y_pred, zero_division=0))

    # Determine model name
    if model_name is None:
        features_path = Path(features_file)

        # Try to load feature config to get feature_type and config_hash
        feature_dir = features_path.parent
        feature_config_path = feature_dir / "config.json"

        feature_type = "tfidf"  # Default
        config_hash = None

        if feature_config_path.exists():
            try:
                with open(feature_config_path, "r") as f:
                    feature_cfg = json.load(f)
                    feature_type = feature_cfg.get("feature_type", "tfidf")
                    config_hash = feature_cfg.get("config_hash")
            except (json.JSONDecodeError, KeyError, IOError):
                pass

        # Get dataset name from parent directory structure
        # features/84cbc90c_tfidf_grouped/norm-EFCAMDAT-train/features_dense.csv
        # → dataset_name = norm-EFCAMDAT-train
        dataset_name = feature_dir.name

        # Build model name: {dataset}_{classifier}_{hash}_{feature_type}
        if config_hash:
            model_name = f"{dataset_name}_{classifier_config.classifier_type}_{config_hash}_{feature_type}"
        else:
            # Fallback if no config found
            model_name = f"{dataset_name}_{classifier_config.classifier_type}"

    model_dir = Path(exp_config.models_dir) / "classifiers" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model and label encoder
    if config.output_config.save_models:
        model_path = model_dir / "classifier.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)

        # Save label encoder
        encoder_path = model_dir / "label_encoder.pkl"
        with open(encoder_path, "wb") as f:
            pickle.dump(label_encoder, f)

        if verbose:
            print(f"\n✓ Classifier saved to: {model_path}")
            print(f"✓ Label encoder saved to: {encoder_path}")

    # Save config
    if config.output_config.save_config:
        tfidf_config = config.tfidf_config

        # Extract feature_type and config_hash from model_name if available
        # model_name format: {dataset}_{classifier}_{hash}_{feature_type}
        parts = model_name.split("_")
        feature_type_from_name = parts[-1] if len(parts) >= 4 else "tfidf"
        config_hash_from_name = (
            parts[-2] if len(parts) >= 4 else tfidf_config.get_hash()
        )

        model_config = {
            "model_name": model_name,
            "classifier_type": classifier_config.classifier_type,
            "feature_type": feature_type_from_name,
            "config_hash": config_hash_from_name,
            "tfidf_readable_name": tfidf_config.get_readable_name(),
            "tfidf_max_features": tfidf_config.max_features,
            "features_file": str(features_file),
            "feature_names_file": (
                str(feature_names_file) if feature_names_file else None
            ),
            "labels_source": str(labels_file or labels_csv),
            "n_samples": int(len(X_train)),
            "n_features": int(X_train.shape[1]),
            "classes_in_training": sorted([str(c) for c in y_train_series.unique()]),
            "n_classes_in_training": int(len(y_train_series.unique())),
            "all_cefr_classes": CEFR_CLASSES,
            "n_cefr_classes": len(CEFR_CLASSES),
            "label_encoder": "label_encoder.pkl",
        }

        if config.output_config.save_json:
            config_path = model_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(model_config, f, indent=2)

            if verbose:
                print(f"✓ Config saved to: {config_path}")

    return model_dir


def train_all_classifiers(
    config: GlobalConfig,
    features_dir: Optional[str] = None,
    labels_csv_dir: Optional[str] = None,
) -> List[Path]:
    """
    Train classifiers for all feature sets in features directory.

    Args:
        config: GlobalConfig containing all configuration
        features_dir: Directory containing features subdirectories (default: hashed features dir)
        labels_csv_dir: Directory containing label CSV files (default: ml_training_dir)

    Returns:
        List of paths to created classifier model directories
    """
    verbose = config.output_config.verbose
    exp_config = config.experiment_config
    tfidf_config = config.tfidf_config

    # Use default directories if not specified
    if features_dir is None:
        # Use hashed features directory for this TF-IDF config
        config_hash = tfidf_config.get_hash()
        feature_type = "tfidf"  # Default type for standard pipeline
        features_dir = exp_config.get_features_dir(config_hash, feature_type)

    if labels_csv_dir is None:
        labels_csv_dir = exp_config.ml_training_dir

    features_base_dir = Path(features_dir)
    if not features_base_dir.exists():
        raise FileNotFoundError(f"Features directory not found: {features_base_dir}")

    # Find all feature subdirectories
    feature_subdirs = sorted([d for d in features_base_dir.iterdir() if d.is_dir()])

    if not feature_subdirs:
        print(f"No feature subdirectories found in {features_base_dir}")
        return []

    if verbose:
        print(f"Found {len(feature_subdirs)} feature sets")
        print(f"Classifier type: {config.classifier_config.classifier_type}\n")
        print("=" * 70)

    model_dirs = []
    for feature_subdir in feature_subdirs:
        if verbose:
            print(f"\nProcessing: {feature_subdir.name}")
            print("-" * 70)

        # Expected files
        features_file = feature_subdir / "features_dense.csv"
        feature_names_file = feature_subdir / "feature_names.csv"

        # Look for corresponding labels CSV
        labels_csv = Path(labels_csv_dir) / f"{feature_subdir.name}.csv"

        if not features_file.exists():
            print(f"✗ Features file not found: {features_file}")
            continue

        if not labels_csv.exists():
            print(f"✗ Labels CSV not found: {labels_csv}")
            continue

        try:
            # Read feature config to get feature_type and config_hash
            feature_config_path = feature_subdir / "config.json"
            feature_config_hash = tfidf_config.get_hash()
            feature_type_str = "tfidf"

            if feature_config_path.exists():
                with open(feature_config_path, "r") as f:
                    feature_cfg = json.load(f)
                    feature_config_hash = feature_cfg.get(
                        "config_hash", feature_config_hash
                    )
                    feature_type_str = feature_cfg.get("feature_type", feature_type_str)

            # Include config hash and feature type in model name
            model_name = f"{feature_subdir.name}_{config.classifier_config.classifier_type}_{feature_config_hash}_{feature_type_str}"

            model_dir = train_classifier(
                config,
                features_file=str(features_file),
                feature_names_file=(
                    str(feature_names_file) if feature_names_file.exists() else None
                ),
                labels_csv=str(labels_csv),
                model_name=model_name,
            )
            model_dirs.append(model_dir)
        except Exception as e:
            print(f"✗ Error: {e}")
            if verbose:
                import traceback

                traceback.print_exc()

    if verbose:
        print("\n" + "=" * 70)
        print("Classifier training complete!")

    return model_dirs


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for train_classifiers."""
    parser = argparse.ArgumentParser(
        description="Train ML classifiers for CEFR classification",
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

    # Input data selection
    input_group = parser.add_argument_group("Input Data Selection")

    # Simple use case: single feature directory
    input_group.add_argument(
        "-d",
        "--feature-dir",
        help="Path to TF-IDF feature directory (e.g., features/norm-EFCAMDAT-train/). "
        "Must contain features_dense.csv. Optional: feature_names.csv",
    )

    # Advanced: explicit file paths
    input_group.add_argument(
        "-f",
        "--features-file",
        help="Path to pre-extracted features CSV file (flat features, one row per sample)",
    )
    input_group.add_argument(
        "--feature-names-file",
        help="Path to feature names file (CSV or TXT). Optional, uses column names if not provided.",
    )
    input_group.add_argument(
        "--labels-file",
        help="Path to labels file (one label per line, matching features row order)",
    )
    input_group.add_argument(
        "--labels-csv",
        help="Path to CSV file containing labels in a column (use with --cefr-column)",
    )

    # Batch processing: multiple feature directories
    input_group.add_argument(
        "--batch-features-dir",
        help="Directory containing multiple feature subdirectories (for batch processing all)",
    )
    input_group.add_argument(
        "--labels-csv-dir",
        help="Directory containing label CSV files (for batch processing, default: ml-training-data)",
    )

    input_group.add_argument(
        "--model-name",
        help="Custom model name (optional, derived from feature directory name if not provided)",
    )

    # Classifier configuration
    clf_group = parser.add_argument_group("Classifier Configuration")
    clf_group.add_argument(
        "--classifier",
        choices=[
            "multinomialnb",
            "logistic",
            "randomforest",
            "svm",
            "xgboost",
            "mord-lr",
        ],
        help="Classifier type (default: multinomialnb). 'mord-lr' is ordinal logistic regression",
    )
    clf_group.add_argument(
        "--logistic-max-iter",
        type=int,
        help="Max iterations for logistic regression (default: 1000)",
    )
    clf_group.add_argument(
        "--rf-n-estimators",
        type=int,
        help="Number of trees for random forest (default: 100)",
    )
    clf_group.add_argument(
        "--svm-max-iter", type=int, help="Max iterations for SVM (default: 2000)"
    )
    clf_group.add_argument(
        "--xgb-n-estimators",
        type=int,
        help="Number of boosting rounds for XGBoost (default: 100)",
    )
    clf_group.add_argument(
        "--xgb-max-depth", type=int, help="Max tree depth for XGBoost (default: 6)"
    )
    clf_group.add_argument(
        "--xgb-learning-rate",
        type=float,
        help="Learning rate for XGBoost (default: 0.3)",
    )
    clf_group.add_argument(
        "--xgb-use-gpu", action="store_true", help="Use GPU for XGBoost training"
    )
    clf_group.add_argument(
        "--xgb-tree-method",
        choices=["auto", "gpu_hist", "hist", "exact"],
        help="Tree method for XGBoost (default: auto)",
    )
    clf_group.add_argument(
        "--xgb-objective",
        choices=[
            "multi:softprob",
            "multi:softmax",
            "reg:squarederror",
            "reg:pseudohubererror",
            "rank:pairwise",
        ],
        help="XGBoost objective function (default: multi:softprob). "
        "Use 'reg:squarederror' for ordinal regression.",
    )
    clf_group.add_argument(
        "--mord-alpha",
        type=float,
        help="Regularization strength for mord-lr (default: 1.0)",
    )
    clf_group.add_argument("--random-state", type=int, help="Random seed (default: 42)")

    # Data configuration
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--text-column",
        default="text",
        help="Column name containing text (default: text)",
    )
    data_group.add_argument(
        "--label-column",
        default="label",
        help="Column name containing labels (default: label)",
    )
    data_group.add_argument(
        "--cefr-column",
        default="cefr_label",
        help="Column name containing CEFR labels (default: cefr_label)",
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


def args_to_config(args: argparse.Namespace) -> GlobalConfig:
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

        if args.classifier:
            config.classifier_config.classifier_type = args.classifier
        if args.logistic_max_iter:
            config.classifier_config.logistic_max_iter = args.logistic_max_iter
        if args.rf_n_estimators:
            config.classifier_config.rf_n_estimators = args.rf_n_estimators
        if args.svm_max_iter:
            config.classifier_config.svm_max_iter = args.svm_max_iter
        if args.xgb_n_estimators:
            config.classifier_config.xgb_n_estimators = args.xgb_n_estimators
        if args.xgb_max_depth:
            config.classifier_config.xgb_max_depth = args.xgb_max_depth
        if args.xgb_learning_rate:
            config.classifier_config.xgb_learning_rate = args.xgb_learning_rate
        if args.xgb_use_gpu:
            config.classifier_config.xgb_use_gpu = args.xgb_use_gpu
        if args.xgb_tree_method:
            config.classifier_config.xgb_tree_method = args.xgb_tree_method
        if args.xgb_objective:
            config.classifier_config.xgb_objective = args.xgb_objective
        if args.mord_alpha:
            config.classifier_config.mord_alpha = args.mord_alpha
        if args.random_state:
            config.classifier_config.random_state = args.random_state

        if args.text_column != "text":
            config.data_config.text_column = args.text_column
        if args.label_column != "label":
            config.data_config.label_column = args.label_column
        if args.cefr_column != "cefr_label":
            config.data_config.cefr_column = args.cefr_column

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

        classifier_config = ClassifierConfig(
            classifier_type=args.classifier or "multinomialnb",
            logistic_max_iter=args.logistic_max_iter or 1000,
            rf_n_estimators=args.rf_n_estimators or 100,
            svm_max_iter=args.svm_max_iter or 2000,
            xgb_n_estimators=args.xgb_n_estimators or 100,
            xgb_max_depth=args.xgb_max_depth or 6,
            xgb_learning_rate=args.xgb_learning_rate or 0.3,
            xgb_use_gpu=args.xgb_use_gpu,
            xgb_tree_method=args.xgb_tree_method or "auto",
            xgb_objective=args.xgb_objective or "multi:softprob",
            mord_alpha=args.mord_alpha or 1.0,
            random_state=args.random_state or 42,
        )

        data_config = DataConfig(
            text_column=args.text_column,
            label_column=args.label_column,
            cefr_column=args.cefr_column,
        )

        output_config = OutputConfig(
            save_config=not args.no_save_config, verbose=not args.quiet
        )

        return GlobalConfig(
            experiment_config,
            classifier_config=classifier_config,
            data_config=data_config,
            output_config=output_config,
        )


def main():
    """Main entry point for classifier training."""
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
        print()

    # Train classifiers
    try:
        if args.feature_dir:
            # Simple use case: train on single feature directory
            feature_dir_path = Path(args.feature_dir)
            if not feature_dir_path.exists():
                parser.error(f"Feature directory not found: {args.feature_dir}")

            # Expected files in feature directory
            features_file = feature_dir_path / "features_dense.csv"
            feature_names_file = feature_dir_path / "feature_names.csv"

            if not features_file.exists():
                parser.error(f"features_dense.csv not found in {args.feature_dir}")

            # Derive dataset name from directory name
            dataset_name = feature_dir_path.name

            # Look for corresponding label CSV in ml-training-data
            labels_csv_path = (
                Path(config.experiment_config.ml_training_dir) / f"{dataset_name}.csv"
            )

            if not labels_csv_path.exists():
                parser.error(
                    f"Label CSV not found: {labels_csv_path}\n"
                    f"Expected: ml-training-data/{dataset_name}.csv to match {args.feature_dir}"
                )

            if config.output_config.verbose:
                print(f"Training on feature directory: {args.feature_dir}")
                print(f"Using labels from: {labels_csv_path}")
                print(f"CEFR column: {config.data_config.cefr_column}")

            train_classifier(
                config,
                features_file=str(features_file),
                feature_names_file=(
                    str(feature_names_file) if feature_names_file.exists() else None
                ),
                labels_csv=str(labels_csv_path),
                model_name=args.model_name,
            )

        elif args.features_file:
            # Advanced: train on specific features file
            if not (args.labels_file or args.labels_csv):
                parser.error(
                    "Must provide either --labels-file or --labels-csv with --features-file"
                )

            train_classifier(
                config,
                features_file=args.features_file,
                feature_names_file=args.feature_names_file,
                labels_file=args.labels_file,
                labels_csv=args.labels_csv,
                model_name=args.model_name,
            )

        elif args.batch_features_dir:
            # Batch: train all feature sets in directory
            train_all_classifiers(
                config,
                features_dir=args.batch_features_dir,
                labels_csv_dir=args.labels_csv_dir,
            )

        else:
            # Default: batch process features from experiment directory
            train_all_classifiers(
                config,
                features_dir=None,  # Will use default from config
                labels_csv_dir=args.labels_csv_dir,
            )

    except Exception as e:
        print(f"Error during classifier training: {e}")
        raise


if __name__ == "__main__":
    main()
