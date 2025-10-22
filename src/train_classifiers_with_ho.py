"""
Step 3: Train ML Classifiers with Hyperparameter Optimization (Optuna)

This module trains ML classifiers on pre-extracted features using Optuna for hyperparameter tuning.
It supports XGBoost and Logistic Regression with automatic hyperparameter search.

Expected File Structure
-----------------------
INPUT:
experiment-dir/
├── features/
│   └── {hash}_tfidf/               # Features from extract_features.py
│       └── train-data/
│           ├── features_dense.csv  # Training features (N samples × M features)
│           └── feature_names.csv
└── ml-training-data/
    └── train-data.csv              # Must have 'cefr_label' column

OUTPUT:
experiment-dir/
└── feature-models/
    └── classifiers/
        └── train-data_{classifier}_{hash}_tfidf_ho/
            ├── classifier.pkl       # Trained classifier with optimized hyperparameters
            ├── label_encoder.pkl    # LabelEncoder for class mapping
            ├── config.json          # Training configuration + best hyperparameters
            └── optuna_study.pkl     # Optuna study object (optional)

Hyperparameter Search Spaces
-----------------------------
XGBoost:
- n_estimators: [50, 100, 200, 300]
- max_depth: [3, 6, 10, 15]
- learning_rate: [0.01, 0.05, 0.1, 0.3]
- subsample: [0.6, 0.8, 1.0]
- colsample_bytree: [0.6, 0.8, 1.0]

Logistic Regression:
- C: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
- max_iter: [100, 500, 1000]
- solver: ['lbfgs', 'saga']
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from src.config import GlobalConfig

# Fixed CEFR classes (all 6 levels)
CEFR_CLASSES = ["A1", "A2", "B1", "B2", "C1", "C2"]


def get_cefr_label_encoder() -> LabelEncoder:
    """
    Create a fixed label encoder for all 6 CEFR classes.

    Returns:
        LabelEncoder fitted with all CEFR classes
    """
    encoder = LabelEncoder()
    encoder.fit(CEFR_CLASSES)
    return encoder


def load_features_and_labels(  # noqa: C901
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


def optimize_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    use_gpu: bool = False,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Optimize XGBoost hyperparameters using Optuna.

    Args:
        X_train: Training features
        y_train: Training labels (encoded)
        X_val: Validation features
        y_val: Validation labels (encoded)
        n_trials: Number of Optuna trials
        use_gpu: Whether to use GPU
        random_state: Random state for reproducibility
        verbose: Print progress

    Returns:
        Tuple of (best_params, tuning_history)
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is not installed. Install it with: pip install optuna"
        )

    if not XGBOOST_AVAILABLE:
        raise ImportError(
            "XGBoost is not installed. Install it with: pip install xgboost"
        )

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    tuning_history = []

    def objective(trial):
        """Optuna objective function for XGBoost."""
        params = {
            "objective": "multi:softmax",
            "num_class": len(CEFR_CLASSES),
            "eval_metric": "mlogloss",
            "tree_method": "gpu_hist" if use_gpu else "hist",
            "device": "cuda" if use_gpu else "cpu",
            "random_state": random_state,
            # Hyperparameters to tune
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }

        # Train model
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=params["n_estimators"],
            evals=[(dval, "validation")],
            verbose_eval=False,
        )

        # Predict on validation set
        y_pred = booster.predict(dval)
        accuracy = accuracy_score(y_val, y_pred)

        # Record trial
        trial_info = {
            "trial_number": trial.number,
            "params": params.copy(),
            "accuracy": accuracy,
        }
        tuning_history.append(trial_info)

        return accuracy  # Optuna maximizes by default

    # Create study and optimize
    study = optuna.create_study(direction="maximize", study_name="xgboost_cefr")

    if verbose:
        print(f"\nStarting Optuna hyperparameter optimization ({n_trials} trials)...")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    best_params = study.best_params
    best_params.update(
        {
            "objective": "multi:softmax",
            "num_class": len(CEFR_CLASSES),
            "eval_metric": "mlogloss",
            "tree_method": "gpu_hist" if use_gpu else "hist",
            "device": "cuda" if use_gpu else "cpu",
            "random_state": random_state,
        }
    )

    if verbose:
        print(f"\nBest trial accuracy: {study.best_value:.4f}")
        print(f"Best parameters: {best_params}")

    return best_params, tuning_history


def optimize_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 30,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Optimize Logistic Regression hyperparameters using Optuna.

    Args:
        X_train: Training features
        y_train: Training labels (encoded)
        X_val: Validation features
        y_val: Validation labels (encoded)
        n_trials: Number of Optuna trials
        random_state: Random state for reproducibility
        verbose: Print progress

    Returns:
        Tuple of (best_params, tuning_history)
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is not installed. Install it with: pip install optuna"
        )

    tuning_history = []

    def objective(trial):
        """Optuna objective function for Logistic Regression."""
        params = {
            "C": trial.suggest_float("C", 0.001, 100.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"]),
            "solver": trial.suggest_categorical(
                "solver", ["saga"]
            ),  # saga supports all penalties
            "max_iter": trial.suggest_int("max_iter", 500, 3000),
            "class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", None]
            ),
            "random_state": random_state,
        }

        # Add l1_ratio for elasticnet
        if params["penalty"] == "elasticnet":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)

        # Train model
        try:
            clf = LogisticRegression(**params)
            clf.fit(X_train, y_train)

            # Predict on validation set
            y_pred = clf.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)

            # Record trial
            trial_info = {
                "trial_number": trial.number,
                "params": params.copy(),
                "accuracy": accuracy,
            }
            tuning_history.append(trial_info)

            return accuracy
        except Exception:
            # Return low score if training fails
            return 0.0

    # Create study and optimize
    study = optuna.create_study(direction="maximize", study_name="logistic_cefr")

    if verbose:
        print(f"\nStarting Optuna hyperparameter optimization ({n_trials} trials)...")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    best_params = study.best_params

    if verbose:
        print(f"\nBest trial accuracy: {study.best_value:.4f}")
        print(f"Best parameters: {best_params}")

    return best_params, tuning_history


def train_classifier_with_ho(  # noqa: C901
    config: GlobalConfig,
    features_file: str,
    feature_names_file: Optional[str] = None,
    labels_file: Optional[str] = None,
    labels_csv: Optional[str] = None,
    model_name: Optional[str] = None,
    n_trials: int = 50,
    val_split: float = 0.2,
) -> Path:
    """
    Train a classifier with hyperparameter optimization on pre-extracted features.

    Args:
        config: GlobalConfig containing all configuration
        features_file: Path to features CSV file
        feature_names_file: Path to feature names file (optional)
        labels_file: Path to labels file (optional, one per line)
        labels_csv: Path to CSV with labels column (optional)
        model_name: Custom model name (optional, derived from features_file if not provided)
        n_trials: Number of Optuna trials for hyperparameter optimization
        val_split: Validation split ratio (e.g., 0.2 = 20% validation)

    Returns:
        Path to saved classifier model directory
    """
    verbose = config.output_config.verbose
    exp_config = config.experiment_config
    classifier_config = config.classifier_config
    data_config = config.data_config

    # Check Optuna availability
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is required for hyperparameter optimization. "
            "Install it with: pip install optuna"
        )

    # Load features and labels
    X, y, feature_names, y_series = load_features_and_labels(
        features_file=features_file,
        feature_names_file=feature_names_file,
        labels_file=labels_file,
        labels_csv=labels_csv,
        cefr_column=data_config.cefr_column,
        verbose=verbose,
    )

    # Create fixed CEFR label encoder
    label_encoder = get_cefr_label_encoder()
    y_encoded = label_encoder.transform(y)

    if verbose:
        print("\nLabel encoding:")
        for label, encoded in zip(CEFR_CLASSES, range(len(CEFR_CLASSES))):
            print(f"  {label} → {encoded}")

    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y_encoded,
        test_size=val_split,
        random_state=classifier_config.random_state,
        stratify=y_encoded,
    )

    if verbose:
        print("\nSplit data:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")

    # Optimize hyperparameters based on classifier type
    if classifier_config.classifier_type == "xgboost":
        best_params, tuning_history = optimize_xgboost(
            X_train,
            y_train,
            X_val,
            y_val,
            n_trials=n_trials,
            use_gpu=classifier_config.xgb_use_gpu,
            random_state=classifier_config.random_state,
            verbose=verbose,
        )

        # Train final model on full dataset with best params
        if verbose:
            print("\nTraining final XGBoost model on full dataset...")

        dtrain = xgb.DMatrix(X, label=y_encoded)
        num_boost_round = best_params.pop("n_estimators")
        booster = xgb.train(best_params, dtrain, num_boost_round=num_boost_round)

        # Restore n_estimators to best_params for saving
        best_params["n_estimators"] = num_boost_round

        model = booster  # For XGBoost, we save the booster directly

    elif classifier_config.classifier_type == "logistic":
        best_params, tuning_history = optimize_logistic(
            X_train,
            y_train,
            X_val,
            y_val,
            n_trials=n_trials,
            random_state=classifier_config.random_state,
            verbose=verbose,
        )

        # Train final model on full dataset with best params
        if verbose:
            print("\nTraining final Logistic Regression model on full dataset...")

        model = LogisticRegression(**best_params)
        model.fit(X, y_encoded)

    else:
        raise ValueError(
            f"Hyperparameter optimization only supports 'xgboost' and 'logistic', "
            f"got: {classifier_config.classifier_type}"
        )

    # Determine model name
    if model_name is None:
        # Extract dataset name from features_file path
        features_path = Path(features_file)
        # Try to get parent directory name (dataset name)
        if features_path.parent.name:
            dataset_name = features_path.parent.name
        else:
            dataset_name = "features_dense"

        # Check if features_file is in a feature directory with config
        feature_dir = features_path.parent.parent
        feature_config_path = feature_dir / "config.json"

        if feature_config_path.exists():
            with open(feature_config_path, "r") as f:
                feature_cfg = json.load(f)
                feature_type = feature_cfg.get("feature_type", "tfidf")
                config_hash = feature_cfg.get("config_hash")
        else:
            feature_type = "tfidf"
            config_hash = None

        # Build model name: {dataset}_{classifier}_ho_{hash}_{feature_type}
        if config_hash:
            model_name = f"{dataset_name}_{classifier_config.classifier_type}_ho_{config_hash}_{feature_type}"
        else:
            model_name = f"{dataset_name}_{classifier_config.classifier_type}_ho"

    # Create output directory
    classifiers_dir = Path(exp_config.models_dir) / "classifiers"
    classifiers_dir.mkdir(parents=True, exist_ok=True)

    output_dir = classifiers_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nSaving model to: {output_dir}")

    # Save model
    model_file = output_dir / "model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    # Save label encoder
    encoder_file = output_dir / "label_encoder.pkl"
    with open(encoder_file, "wb") as f:
        pickle.dump(label_encoder, f)

    # Save feature names
    feature_names_file_out = output_dir / "feature_names.txt"
    with open(feature_names_file_out, "w") as f:
        for name in feature_names:
            f.write(f"{name}\n")

    # Save configuration
    config_dict = {
        "model_type": classifier_config.classifier_type,
        "model_name": model_name,
        "n_features": len(feature_names),
        "n_classes": len(CEFR_CLASSES),
        "classes": CEFR_CLASSES,
        "hyperparameter_optimization": True,
        "n_trials": n_trials,
        "val_split": val_split,
        "best_params": best_params,
        "random_state": classifier_config.random_state,
    }

    # Save config as JSON
    config_file = output_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Save tuning history
    tuning_file = output_dir / "tuning_history.json"
    with open(tuning_file, "w") as f:
        json.dump(tuning_history, f, indent=2)

    if verbose:
        print(f"✓ Model saved: {model_file}")
        print(f"✓ Label encoder saved: {encoder_file}")
        print(f"✓ Feature names saved: {feature_names_file_out}")
        print(f"✓ Config saved: {config_file}")
        print(f"✓ Tuning history saved: {tuning_file}")
        print("\nBest hyperparameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")

    return output_dir


def batch_train_classifiers_with_ho(  # noqa: C901
    config: GlobalConfig,
    batch_features_dir: str,
    labels_csv_dir: Optional[str] = None,
    n_trials: int = 50,
    val_split: float = 0.2,
) -> List[Path]:
    """
    Train classifiers with hyperparameter optimization on multiple datasets in a batch directory.

    Args:
        config: GlobalConfig
        batch_features_dir: Directory containing multiple dataset subdirectories
        labels_csv_dir: Directory containing label CSV files (optional)
        n_trials: Number of Optuna trials
        val_split: Validation split ratio

    Returns:
        List of paths to trained model directories
    """
    verbose = config.output_config.verbose
    batch_path = Path(batch_features_dir)

    if not batch_path.exists():
        raise ValueError(f"Batch features directory not found: {batch_features_dir}")

    # Find all subdirectories with features_dense.csv
    dataset_dirs = []
    for subdir in batch_path.iterdir():
        if subdir.is_dir() and (subdir / "features_dense.csv").exists():
            dataset_dirs.append(subdir)

    if not dataset_dirs:
        raise ValueError(
            f"No dataset directories with features_dense.csv found in {batch_features_dir}"
        )

    if verbose:
        print(f"Found {len(dataset_dirs)} datasets to process")

    trained_models = []

    for dataset_dir in sorted(dataset_dirs):
        dataset_name = dataset_dir.name

        if verbose:
            print(f"\n{'='*70}")
            print(f"Processing dataset: {dataset_name}")
            print(f"{'='*70}")

        features_file = dataset_dir / "features_dense.csv"
        feature_names_file = dataset_dir / "feature_names.txt"

        # Determine labels CSV path
        if labels_csv_dir:
            labels_csv = Path(labels_csv_dir) / dataset_name / "data.csv"
        else:
            # Try to find in standard experiment structure
            labels_csv = (
                Path(config.experiment_config.ml_training_dir)
                / dataset_name
                / "data.csv"
            )
            if not labels_csv.exists():
                # Try test data
                labels_csv = (
                    Path(config.experiment_config.ml_test_dir)
                    / dataset_name
                    / "data.csv"
                )

        if not labels_csv.exists():
            print(f"⚠ Warning: Labels CSV not found for {dataset_name}, skipping")
            continue

        # Train classifier with hyperparameter optimization
        try:
            model_dir = train_classifier_with_ho(
                config=config,
                features_file=str(features_file),
                feature_names_file=(
                    str(feature_names_file) if feature_names_file.exists() else None
                ),
                labels_csv=str(labels_csv),
                n_trials=n_trials,
                val_split=val_split,
            )
            trained_models.append(model_dir)

        except Exception as e:
            print(f"✗ Error training classifier for {dataset_name}: {e}")
            continue

    if verbose:
        print(f"\n{'='*70}")
        print(
            f"Batch training complete: {len(trained_models)}/{len(dataset_dirs)} models trained"
        )
        print(f"{'='*70}")

    return trained_models


def main():  # noqa: C901
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Train ML classifiers with Optuna hyperparameter optimization for CEFR classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Configuration options
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument(
        "-c", "--config-file", type=str, help="Path to config file (YAML or JSON)"
    )
    config_group.add_argument(
        "--config-json", type=str, help="JSON string with configuration"
    )

    # Experiment configuration
    parser.add_argument(
        "-e", "--experiment-dir", type=str, help="Experiment directory path"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, help="Custom output directory for models"
    )

    # Input files (single dataset mode)
    parser.add_argument(
        "-d",
        "--feature-dir",
        type=str,
        help="Feature directory (contains features_dense.csv)",
    )
    parser.add_argument("-f", "--features-file", type=str, help="Features CSV file")
    parser.add_argument(
        "--feature-names-file", type=str, help="Feature names file (TXT or CSV)"
    )
    parser.add_argument("--labels-file", type=str, help="Labels file (one per line)")
    parser.add_argument("--labels-csv", type=str, help="CSV file containing labels")

    # Batch mode
    parser.add_argument(
        "--batch-features-dir",
        type=str,
        help="Directory with multiple feature subdirectories",
    )
    parser.add_argument(
        "--labels-csv-dir",
        type=str,
        help="Directory with label CSV files for batch mode",
    )

    # Model configuration
    parser.add_argument("--model-name", type=str, help="Custom model name")
    parser.add_argument(
        "--classifier",
        type=str,
        choices=["xgboost", "logistic"],
        default="xgboost",
        help="Classifier type (only xgboost and logistic supported for HO)",
    )

    # Hyperparameter optimization settings
    parser.add_argument(
        "--n-trials", type=int, default=50, help="Number of Optuna trials (default: 50)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)",
    )

    # XGBoost specific
    parser.add_argument(
        "--xgb-use-gpu", action="store_true", help="Use GPU for XGBoost"
    )

    # Random state
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random state for reproducibility"
    )

    # Data configuration
    parser.add_argument(
        "--text-column", type=str, default="text", help="Text column name"
    )
    parser.add_argument(
        "--label-column", type=str, default="label", help="Label column name"
    )
    parser.add_argument(
        "--cefr-column", type=str, default="cefr_label", help="CEFR column name"
    )

    # Output options
    parser.add_argument(
        "--no-save-config",
        action="store_true",
        help="Do not save config to output directory",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress verbose output"
    )

    args = parser.parse_args()

    # Load or create configuration
    if args.config_file:
        config = GlobalConfig.from_file(args.config_file)
    elif args.config_json:
        config = GlobalConfig.from_json(args.config_json)
    else:
        config = GlobalConfig()

    # Override with CLI arguments
    if args.experiment_dir:
        config.experiment_config.experiment_dir = args.experiment_dir
        config.experiment_config.__post_init__()

    if args.output_dir:
        config.experiment_config.models_dir = args.output_dir

    # Classifier config
    config.classifier_config.classifier_type = args.classifier
    config.classifier_config.random_state = args.random_state
    config.classifier_config.xgb_use_gpu = args.xgb_use_gpu

    # Data config
    config.data_config.text_column = args.text_column
    config.data_config.label_column = args.label_column
    config.data_config.cefr_column = args.cefr_column

    # Output config
    config.output_config.verbose = not args.quiet

    # Validate Optuna is available
    if not OPTUNA_AVAILABLE:
        print("ERROR: Optuna is not installed. Install it with: pip install optuna")
        return 1

    # Batch mode
    if args.batch_features_dir:
        batch_train_classifiers_with_ho(
            config=config,
            batch_features_dir=args.batch_features_dir,
            labels_csv_dir=args.labels_csv_dir,
            n_trials=args.n_trials,
            val_split=args.val_split,
        )
        return 0

    # Single dataset mode
    if args.feature_dir:
        # Feature directory mode
        feature_dir = Path(args.feature_dir)
        features_file = feature_dir / "features_dense.csv"
        feature_names_file = feature_dir / "feature_names.txt"

        if not features_file.exists():
            print(f"ERROR: features_dense.csv not found in {args.feature_dir}")
            return 1

        # Try to find labels CSV
        if args.labels_csv:
            labels_csv = args.labels_csv
        else:
            # Try to infer from experiment structure
            dataset_name = feature_dir.name
            labels_csv = (
                Path(config.experiment_config.ml_training_dir)
                / dataset_name
                / "data.csv"
            )
            if not labels_csv.exists():
                labels_csv = (
                    Path(config.experiment_config.ml_test_dir)
                    / dataset_name
                    / "data.csv"
                )
            if not labels_csv.exists():
                print(f"ERROR: Could not find labels CSV for {dataset_name}")
                print("Please specify --labels-csv explicitly")
                return 1

        train_classifier_with_ho(
            config=config,
            features_file=str(features_file),
            feature_names_file=(
                str(feature_names_file) if feature_names_file.exists() else None
            ),
            labels_csv=str(labels_csv),
            model_name=args.model_name,
            n_trials=args.n_trials,
            val_split=args.val_split,
        )

    elif args.features_file:
        # Direct file mode
        train_classifier_with_ho(
            config=config,
            features_file=args.features_file,
            feature_names_file=args.feature_names_file,
            labels_file=args.labels_file,
            labels_csv=args.labels_csv,
            model_name=args.model_name,
            n_trials=args.n_trials,
            val_split=args.val_split,
        )

    else:
        print(
            "ERROR: Must specify either --batch-features-dir, --feature-dir, or --features-file"
        )
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
