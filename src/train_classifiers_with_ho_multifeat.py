"""
Two-Stage Hyperparameter Optimization with Feature Configuration Selection

Stage 1: Quick screening of all feature configurations (with early pruning)
Stage 2: Deep optimization on top-K feature configurations

This script uses Optuna to:
1. Quickly evaluate all 24 feature configs with baseline hyperparameters
2. Prune obviously bad configs early (configurable threshold)
3. Run extensive hyperparameter tuning on top performers

Expected File Structure
-----------------------
INPUT:
experiment-dir/
├── features/
│   ├── {hash1}_tfidf/              # Multiple TF-IDF configs
│   │   └── train-data/
│   │       ├── features_dense.csv
│   │       └── feature_names.csv
│   ├── {hash2}_tfidf/
│   │   └── train-data/
│   │       └── ...
│   └── ... (multiple feature configs)
└── ml-training-data/
    └── train-data.csv              # Must have 'cefr_label' column

OUTPUT:
experiment-dir/
└── feature-models/
    └── classifiers/
        └── train-data_{classifier}_{best_hash}_tfidf_ho_multifeat/
            ├── classifier.pkl       # Best model from best feature config
            ├── label_encoder.pkl
            ├── config.json          # Contains best_feature_hash and hyperparameters
            ├── stage1_results.json  # Screening results for all configs
            └── stage2_results.json  # Deep optimization results

Two-Stage Process
-----------------
Stage 1 (Screening):
- Evaluates ALL feature configurations quickly
- Uses baseline hyperparameters
- Prunes bad configs early (configurable threshold)
- Outputs: Top-K feature configurations

Stage 2 (Deep Optimization):
- Takes top-K configs from Stage 1
- Runs full hyperparameter optimization on each
- Selects best model overall
- Saves comprehensive results
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

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
    """Create a fixed label encoder for all 6 CEFR classes."""
    encoder = LabelEncoder()
    encoder.fit(CEFR_CLASSES)
    return encoder


def compute_sample_weights(
    y: np.ndarray, strategy: str = "equal", alpha: float = 1.0
) -> np.ndarray:
    """
    Compute sample weights based on class distribution.

    Args:
        y: Labels (encoded)
        strategy: Weighting strategy
            - 'equal': All samples have equal weight (default)
            - 'inverse': Weight inversely proportional to class frequency
            - 'inverse_sqrt': Weight inversely proportional to sqrt of class frequency
        alpha: Calibration parameter (0 = equal weights, 1 = full inverse weighting)
               Only used for 'inverse' and 'inverse_sqrt' strategies

    Returns:
        Array of sample weights
    """
    n_samples = len(y)

    if strategy == "equal":
        return np.ones(n_samples)

    # Compute class weights
    classes = np.unique(y)
    n_classes = len(classes)

    if strategy == "inverse":
        # weight = n_samples / (n_classes * class_count)
        class_weights = compute_class_weight("balanced", classes=classes, y=y)
    elif strategy == "inverse_sqrt":
        # Custom: weight = sqrt(n_samples / class_count)
        class_counts = np.bincount(y)
        class_weights = np.sqrt(n_samples / (class_counts[classes] * n_classes))
    else:
        raise ValueError(f"Unknown weighting strategy: {strategy}")

    # Apply alpha calibration: interpolate between equal (alpha=0) and full inverse (alpha=1)
    if alpha < 1.0:
        # Equal weight is 1.0 for all classes
        equal_weight = np.ones(n_classes)
        class_weights = (1 - alpha) * equal_weight + alpha * class_weights

    # Map class weights to sample weights
    sample_weights = class_weights[y]

    return sample_weights


def compute_ordinal_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    tolerance: int = 1,
) -> float:
    """
    Compute ordinal accuracy allowing for adjacent level errors.

    For CEFR classification: A1=0, A2=1, B1=2, B2=3, C1=4, C2=5

    Args:
        y_true: True labels (encoded)
        y_pred: Predicted labels (encoded)
        sample_weight: Sample weights (optional)
        tolerance: Number of levels of tolerance (default: 1 = adjacent accuracy)

    Returns:
        Ordinal accuracy (proportion of predictions within tolerance)
    """
    correct = np.abs(y_true - y_pred) <= tolerance

    if sample_weight is not None:
        return np.average(correct, weights=sample_weight)
    else:
        return np.mean(correct)


def compute_ordinal_distance_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    penalty: str = "quadratic",
) -> float:
    """
    Compute ordinal distance-based score (higher is better).

    Penalizes predictions based on distance from true label.
    Returns a score normalized to [0, 1] range where 1 is perfect.

    Args:
        y_true: True labels (encoded)
        y_pred: Predicted labels (encoded)
        sample_weight: Sample weights (optional)
        penalty: Penalty function
            - 'linear': score = 1 - |distance| / max_distance
            - 'quadratic': score = 1 - (distance^2) / max_distance^2

    Returns:
        Mean ordinal distance score (0 to 1, higher is better)
    """
    distances = np.abs(y_true - y_pred)
    max_distance = len(CEFR_CLASSES) - 1  # 5 for A1-C2

    if penalty == "linear":
        # Linear penalty: 0 distance = 1.0, max distance = 0.0
        scores = 1.0 - (distances / max_distance)
    elif penalty == "quadratic":
        # Quadratic penalty: penalizes far predictions more heavily
        scores = 1.0 - (distances**2) / (max_distance**2)
    else:
        raise ValueError(f"Unknown penalty: {penalty}")

    if sample_weight is not None:
        return np.average(scores, weights=sample_weight)
    else:
        return np.mean(scores)


def compute_ordinal_mse_score(
    y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None
) -> float:
    """
    Compute ordinal MSE-based score (higher is better).

    Treats labels as numerical indices and computes MSE, then converts to
    a score in [0, 1] range where 1 is perfect and 0 is worst possible.

    For CEFR: A1=0, A2=1, B1=2, B2=3, C1=4, C2=5
    MSE = mean((y_true - y_pred)^2)
    Max MSE = 25 (predicting 0 when true is 5, or vice versa)
    Score = 1 - (MSE / max_MSE)

    Args:
        y_true: True labels (encoded as 0-5)
        y_pred: Predicted labels (encoded as 0-5)
        sample_weight: Sample weights (optional)

    Returns:
        MSE-based score (0 to 1, higher is better)

    Examples:
        Perfect prediction (0→0): MSE=0, Score=1.0
        Adjacent error (0→1): MSE=1, Score=0.96
        2-level error (0→2): MSE=4, Score=0.84
        3-level error (0→3): MSE=9, Score=0.64
        4-level error (0→4): MSE=16, Score=0.36
        Worst error (0→5): MSE=25, Score=0.0
    """
    # Compute squared differences
    squared_errors = (y_true - y_pred) ** 2

    # Maximum possible MSE is when predicting 0 for 5 or 5 for 0
    max_mse = (len(CEFR_CLASSES) - 1) ** 2  # 5^2 = 25

    # Compute MSE
    if sample_weight is not None:
        mse = np.average(squared_errors, weights=sample_weight)
    else:
        mse = np.mean(squared_errors)

    # Convert to score (1 - normalized MSE)
    score = 1.0 - (mse / max_mse)

    return score


def compute_metric(  # noqa: C901
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "f1_macro",
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """
    Compute evaluation metric.

    Args:
        y_true: True labels (encoded for ordinal metrics)
        y_pred: Predicted labels (encoded for ordinal metrics)
        metric: Metric to compute
            - 'accuracy': Standard accuracy
            - 'f1_macro': Macro-averaged F1 (default)
            - 'f1_weighted': Weighted F1
            - 'f1_micro': Micro-averaged F1
            - 'precision_macro': Macro-averaged precision
            - 'recall_macro': Macro-averaged recall
            - 'ordinal_accuracy': Adjacent accuracy (allows ±1 level error)
            - 'ordinal_distance_linear': Linear distance penalty (0-1, higher better)
            - 'ordinal_distance_quadratic': Quadratic distance penalty (penalizes far errors more)
            - 'ordinal_mse': MSE-based score treating labels as numerical (0-5)
        sample_weight: Sample weights (optional)

    Returns:
        Metric value
    """
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred, sample_weight=sample_weight)

    elif metric == "f1_macro":
        return f1_score(y_true, y_pred, average="macro", sample_weight=sample_weight)

    elif metric == "f1_weighted":
        return f1_score(y_true, y_pred, average="weighted", sample_weight=sample_weight)

    elif metric == "f1_micro":
        return f1_score(y_true, y_pred, average="micro", sample_weight=sample_weight)

    elif metric == "precision_macro":
        return precision_score(
            y_true, y_pred, average="macro", sample_weight=sample_weight
        )

    elif metric == "recall_macro":
        return recall_score(
            y_true, y_pred, average="macro", sample_weight=sample_weight
        )

    elif metric == "ordinal_accuracy":
        return compute_ordinal_accuracy(
            y_true, y_pred, sample_weight=sample_weight, tolerance=1
        )

    elif metric == "ordinal_distance_linear":
        return compute_ordinal_distance_score(
            y_true, y_pred, sample_weight=sample_weight, penalty="linear"
        )

    elif metric == "ordinal_distance_quadratic":
        return compute_ordinal_distance_score(
            y_true, y_pred, sample_weight=sample_weight, penalty="quadratic"
        )

    elif metric == "ordinal_mse":
        return compute_ordinal_mse_score(y_true, y_pred, sample_weight=sample_weight)

    else:
        raise ValueError(f"Unknown metric: {metric}")


def load_features_and_labels_from_dir(  # noqa: C901
    feature_dir: Path,
    cefr_column: str = "cefr_label",
    ml_training_dir: Optional[str] = None,
    ml_test_dir: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """
    Load features and labels from a feature directory.

    Args:
        feature_dir: Path to feature directory
        cefr_column: CEFR column name
        ml_training_dir: Training data directory
        ml_test_dir: Test data directory
        verbose: Print info

    Returns:
        Tuple of (X, y_encoded, feature_names, dataset_name)
    """
    dataset_name = feature_dir.name
    features_file = feature_dir / "features_dense.csv"
    feature_names_file = feature_dir / "feature_names.txt"

    if not features_file.exists():
        raise ValueError(f"features_dense.csv not found in {feature_dir}")

    # Load features
    features_df = pd.read_csv(features_file)
    X = features_df.values

    # Load feature names
    if feature_names_file.exists():
        with open(feature_names_file, "r") as f:
            feature_names = [line.strip() for line in f if line.strip()]
    else:
        feature_names = features_df.columns.tolist()

    # Find labels CSV
    labels_csv = None
    if ml_training_dir:
        candidate = Path(ml_training_dir) / dataset_name / "data.csv"
        if candidate.exists():
            labels_csv = candidate

    if labels_csv is None and ml_test_dir:
        candidate = Path(ml_test_dir) / dataset_name / "data.csv"
        if candidate.exists():
            labels_csv = candidate

    if labels_csv is None:
        raise ValueError(f"Could not find labels CSV for dataset: {dataset_name}")

    # Load labels
    labels_df = pd.read_csv(labels_csv)
    if cefr_column not in labels_df.columns:
        raise ValueError(f"Column '{cefr_column}' not found in {labels_csv}")

    y = labels_df[cefr_column].values

    # Validate
    if len(X) != len(y):
        raise ValueError(f"Features/labels mismatch: {len(X)} vs {len(y)}")

    # Encode labels
    label_encoder = get_cefr_label_encoder()
    y_encoded = label_encoder.transform(y)

    if verbose:
        print(
            f"  Loaded {dataset_name}: {len(X)} samples, {len(feature_names)} features"
        )

    return X, y_encoded, feature_names, dataset_name


def quick_evaluate_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric: str = "f1_macro",
    weight_strategy: str = "equal",
    weight_alpha: float = 1.0,
    use_gpu: bool = False,
    random_state: int = 42,
) -> float:
    """
    Quick evaluation with baseline XGBoost hyperparameters.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        metric: Evaluation metric
        weight_strategy: Sample weighting strategy
        weight_alpha: Weight calibration parameter
        use_gpu: Use GPU
        random_state: Random state

    Returns:
        Validation metric score
    """
    # Compute sample weights
    train_weights = compute_sample_weights(y_train, weight_strategy, weight_alpha)

    params = {
        "objective": "multi:softmax",
        "num_class": len(CEFR_CLASSES),
        "eval_metric": "mlogloss",
        "tree_method": "gpu_hist" if use_gpu else "hist",
        "device": "cuda" if use_gpu else "cpu",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "random_state": random_state,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=train_weights)
    dval = xgb.DMatrix(X_val, label=y_val)

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=params["n_estimators"],
        evals=[(dval, "validation")],
        verbose_eval=False,
    )

    y_pred = booster.predict(dval)
    val_weights = compute_sample_weights(y_val, weight_strategy, weight_alpha)
    return compute_metric(y_val, y_pred, metric, sample_weight=val_weights)


def quick_evaluate_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric: str = "f1_macro",
    weight_strategy: str = "equal",
    weight_alpha: float = 1.0,
    random_state: int = 42,
) -> float:
    """
    Quick evaluation with baseline Logistic Regression hyperparameters.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        metric: Evaluation metric
        weight_strategy: Sample weighting strategy
        weight_alpha: Weight calibration parameter
        random_state: Random state

    Returns:
        Validation metric score
    """
    # Compute sample weights
    train_weights = compute_sample_weights(y_train, weight_strategy, weight_alpha)

    clf = LogisticRegression(
        C=1.0, max_iter=1000, random_state=random_state, class_weight="balanced"
    )
    clf.fit(X_train, y_train, sample_weight=train_weights)
    y_pred = clf.predict(X_val)

    val_weights = compute_sample_weights(y_val, weight_strategy, weight_alpha)
    return compute_metric(y_val, y_pred, metric, sample_weight=val_weights)


def stage1_screen_features(  # noqa: C901
    feature_dirs: List[Path],
    classifier_type: str,
    config: GlobalConfig,
    val_split: float = 0.2,
    metric: str = "f1_macro",
    weight_strategy: str = "equal",
    weight_alpha: float = 1.0,
    early_discard_threshold: Optional[float] = None,
    early_discard_percentile: Optional[float] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Stage 1: Screen all feature configurations with baseline hyperparameters.

    Args:
        feature_dirs: List of feature directory paths
        classifier_type: 'xgboost' or 'logistic'
        config: GlobalConfig
        val_split: Validation split ratio
        metric: Evaluation metric (f1_macro, accuracy, etc.)
        weight_strategy: Sample weighting strategy (equal, inverse, inverse_sqrt)
        weight_alpha: Weight calibration parameter (0-1)
        early_discard_threshold: Absolute metric threshold (e.g., 0.3 = discard if metric < 0.3)
        early_discard_percentile: Relative percentile threshold (e.g., 25 = discard bottom 25%)
        verbose: Print progress

    Returns:
        List of dicts with feature config evaluation results, sorted by metric (descending)
    """
    if verbose:
        print("\n" + "=" * 80)
        print("STAGE 1: Feature Configuration Screening")
        print("=" * 80)
        print(f"Evaluating {len(feature_dirs)} feature configurations...")
        print(f"Classifier: {classifier_type}")
        print(f"Metric: {metric}")
        print(f"Weight strategy: {weight_strategy} (alpha={weight_alpha})")
        print(f"Validation split: {val_split}")
        if early_discard_threshold:
            print(f"Early discard threshold: {metric} < {early_discard_threshold:.4f}")
        if early_discard_percentile:
            print(f"Early discard percentile: bottom {early_discard_percentile}%")
        print()

    results = []

    for i, feature_dir in enumerate(feature_dirs, 1):
        feature_config_name = feature_dir.parent.name  # e.g., "252cd532_tfidf"

        if verbose:
            print(
                f"[{i}/{len(feature_dirs)}] Evaluating {feature_config_name}/{feature_dir.name}...",
                end=" ",
            )

        try:
            # Load data
            X, y_encoded, feature_names, dataset_name = (
                load_features_and_labels_from_dir(
                    feature_dir,
                    cefr_column=config.data_config.cefr_column,
                    ml_training_dir=config.experiment_config.ml_training_dir,
                    ml_test_dir=config.experiment_config.ml_test_dir,
                    verbose=False,
                )
            )

            # Split train/val
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y_encoded,
                test_size=val_split,
                random_state=config.classifier_config.random_state,
                stratify=y_encoded,
            )

            # Quick evaluation
            if classifier_type == "xgboost":
                score = quick_evaluate_xgboost(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    metric=metric,
                    weight_strategy=weight_strategy,
                    weight_alpha=weight_alpha,
                    use_gpu=config.classifier_config.xgb_use_gpu,
                    random_state=config.classifier_config.random_state,
                )
            elif classifier_type == "logistic":
                score = quick_evaluate_logistic(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    metric=metric,
                    weight_strategy=weight_strategy,
                    weight_alpha=weight_alpha,
                    random_state=config.classifier_config.random_state,
                )
            else:
                raise ValueError(f"Unsupported classifier: {classifier_type}")

            result = {
                "feature_config": feature_config_name,
                "dataset": dataset_name,
                "feature_dir": str(feature_dir),
                "n_features": len(feature_names),
                "n_samples": len(X),
                "metric": metric,
                "score": score,
                "passed_screening": True,  # May be updated below
            }
            results.append(result)

            if verbose:
                print(f"{metric}: {score:.4f}")

        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")
            results.append(
                {
                    "feature_config": feature_config_name,
                    "dataset": dataset_name,
                    "feature_dir": str(feature_dir),
                    "metric": metric,
                    "score": 0.0,
                    "error": str(e),
                    "passed_screening": False,
                }
            )

    # Sort by score (descending)
    results.sort(key=lambda x: x["score"], reverse=True)

    # Apply early discard filters
    if early_discard_threshold is not None:
        for result in results:
            if result["score"] < early_discard_threshold:
                result["passed_screening"] = False
                result["discard_reason"] = (
                    f"{metric} {result['score']:.4f} < threshold {early_discard_threshold:.4f}"
                )

    if early_discard_percentile is not None:
        n_discard = int(len(results) * early_discard_percentile / 100)
        for i in range(len(results) - n_discard, len(results)):
            if results[i]["passed_screening"]:
                results[i]["passed_screening"] = False
                results[i][
                    "discard_reason"
                ] = f"Bottom {early_discard_percentile}% percentile"

    # Summary
    passed = [r for r in results if r["passed_screening"]]
    failed = [r for r in results if not r["passed_screening"]]

    if verbose:
        print("\n" + "-" * 80)
        print("Stage 1 Results:")
        print("-" * 80)
        print(f"Total evaluated: {len(results)}")
        print(f"Passed screening: {len(passed)}")
        print(f"Discarded: {len(failed)}")
        print()

        print("Top 10 feature configurations:")
        for i, result in enumerate(results[:10], 1):
            status = "✓" if result["passed_screening"] else "✗"
            print(
                f"  {i:2d}. [{status}] {result['feature_config']:25s} "
                f"{metric}={result['score']:.4f} ({result.get('n_features', 0):5d} features)"
            )

        if failed:
            print()
            print(f"Discarded configurations ({len(failed)}):")
            for result in failed:
                reason = result.get(
                    "discard_reason", result.get("error", "Failed screening")
                )
                print(
                    f"  ✗ {result['feature_config']:25s} {metric}={result['score']:.4f} - {reason}"
                )
        print()

    return results


def stage2_optimize_top_features(  # noqa: C901
    stage1_results: List[Dict[str, Any]],
    top_k: int,
    classifier_type: str,
    config: GlobalConfig,
    n_trials_per_config: int = 50,
    val_split: float = 0.2,
    metric: str = "f1_macro",
    weight_strategy: str = "equal",
    weight_alpha: float = 1.0,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Stage 2: Deep hyperparameter optimization on top-K feature configurations.

    Args:
        stage1_results: Results from stage 1
        top_k: Number of top configs to optimize
        classifier_type: 'xgboost' or 'logistic'
        config: GlobalConfig
        n_trials_per_config: Optuna trials per feature config
        val_split: Validation split ratio
        metric: Evaluation metric
        weight_strategy: Sample weighting strategy
        weight_alpha: Weight calibration parameter
        verbose: Print progress

    Returns:
        List of optimization results with best hyperparameters
    """
    if verbose:
        print("\n" + "=" * 80)
        print("STAGE 2: Deep Hyperparameter Optimization")
        print("=" * 80)
        print(f"Optimizing top {top_k} feature configurations")
        print(f"Trials per configuration: {n_trials_per_config}")
        print(f"Metric: {metric}")
        print(f"Weight strategy: {weight_strategy} (alpha={weight_alpha})")
        print()

    # Get top-k that passed screening
    passed_results = [r for r in stage1_results if r["passed_screening"]]
    top_configs = passed_results[:top_k]

    if len(top_configs) < top_k:
        print(
            f"Warning: Only {len(top_configs)} configs passed screening (requested top-{top_k})"
        )

    optimization_results = []

    for i, stage1_result in enumerate(top_configs, 1):
        feature_dir = Path(stage1_result["feature_dir"])
        feature_config = stage1_result["feature_config"]

        if verbose:
            print(f"\n[{i}/{len(top_configs)}] Optimizing: {feature_config}")
            print(f"  Stage 1 {metric}: {stage1_result['score']:.4f}")
            print(f"  Features: {stage1_result.get('n_features', 'N/A')}")
            print(f"  Running {n_trials_per_config} trials...")

        try:
            # Load data
            X, y_encoded, feature_names, dataset_name = (
                load_features_and_labels_from_dir(
                    feature_dir,
                    cefr_column=config.data_config.cefr_column,
                    ml_training_dir=config.experiment_config.ml_training_dir,
                    ml_test_dir=config.experiment_config.ml_test_dir,
                    verbose=False,
                )
            )

            # Split train/val
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y_encoded,
                test_size=val_split,
                random_state=config.classifier_config.random_state,
                stratify=y_encoded,
            )

            # Run Optuna optimization
            if classifier_type == "xgboost":
                best_params, tuning_history = optimize_xgboost(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    n_trials=n_trials_per_config,
                    metric=metric,
                    weight_strategy=weight_strategy,
                    weight_alpha=weight_alpha,
                    use_gpu=config.classifier_config.xgb_use_gpu,
                    random_state=config.classifier_config.random_state,
                    verbose=verbose,
                )
            elif classifier_type == "logistic":
                best_params, tuning_history = optimize_logistic(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    n_trials=n_trials_per_config,
                    metric=metric,
                    weight_strategy=weight_strategy,
                    weight_alpha=weight_alpha,
                    random_state=config.classifier_config.random_state,
                    verbose=verbose,
                )
            else:
                raise ValueError(f"Unsupported classifier: {classifier_type}")

            # Get best score from tuning history
            best_score = max(trial["score"] for trial in tuning_history)

            result = {
                "feature_config": feature_config,
                "dataset": dataset_name,
                "feature_dir": str(feature_dir),
                "n_features": len(feature_names),
                "metric": metric,
                "stage1_score": stage1_result["score"],
                "stage2_best_score": best_score,
                "improvement": best_score - stage1_result["score"],
                "best_params": best_params,
                "tuning_history": tuning_history,
                "n_trials": len(tuning_history),
            }
            optimization_results.append(result)

            if verbose:
                print(
                    f"  ✓ Best {metric}: {best_score:.4f} "
                    f"(+{result['improvement']:.4f} improvement)"
                )

        except Exception as e:
            if verbose:
                print(f"  ✗ FAILED: {e}")
            optimization_results.append(
                {
                    "feature_config": feature_config,
                    "dataset": dataset_name,
                    "metric": metric,
                    "error": str(e),
                    "stage1_score": stage1_result["score"],
                }
            )

    # Sort by stage2 score
    optimization_results.sort(key=lambda x: x.get("stage2_best_score", 0), reverse=True)

    if verbose:
        print("\n" + "=" * 80)
        print("STAGE 2 COMPLETE - Final Rankings:")
        print("=" * 80)
        print(
            f"{'Rank':<6} {'Feature Config':<25} {'Stage1':<10} {'Stage2':<10} {'Improve':<10}"
        )
        print("-" * 80)
        for rank, result in enumerate(optimization_results, 1):
            if "stage2_best_score" in result:
                print(
                    f"{rank:<6} {result['feature_config']:<25} "
                    f"{result['stage1_score']:<10.4f} "
                    f"{result['stage2_best_score']:<10.4f} "
                    f"+{result['improvement']:<9.4f}"
                )
        print()

    return optimization_results


def optimize_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    metric: str = "f1_macro",
    weight_strategy: str = "equal",
    weight_alpha: float = 1.0,
    use_gpu: bool = False,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Optimize XGBoost hyperparameters using Optuna."""
    # Compute sample weights
    train_weights = compute_sample_weights(y_train, weight_strategy, weight_alpha)
    val_weights = compute_sample_weights(y_val, weight_strategy, weight_alpha)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=train_weights)
    dval = xgb.DMatrix(X_val, label=y_val)

    tuning_history = []

    def objective(trial):
        params = {
            "objective": "multi:softmax",
            "num_class": len(CEFR_CLASSES),
            "eval_metric": "mlogloss",
            "tree_method": "gpu_hist" if use_gpu else "hist",
            "device": "cuda" if use_gpu else "cpu",
            "random_state": random_state,
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

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=params["n_estimators"],
            evals=[(dval, "validation")],
            verbose_eval=False,
        )

        y_pred = booster.predict(dval)
        score = compute_metric(y_val, y_pred, metric, sample_weight=val_weights)

        tuning_history.append(
            {
                "trial_number": trial.number,
                "params": params.copy(),
                "metric": metric,
                "score": score,
            }
        )

        return score

    study = optuna.create_study(direction="maximize")
    optuna.logging.set_verbosity(
        optuna.logging.WARNING if not verbose else optuna.logging.INFO
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

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

    return best_params, tuning_history


def optimize_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 30,
    metric: str = "f1_macro",
    weight_strategy: str = "equal",
    weight_alpha: float = 1.0,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Optimize Logistic Regression hyperparameters using Optuna."""
    # Compute sample weights
    train_weights = compute_sample_weights(y_train, weight_strategy, weight_alpha)
    val_weights = compute_sample_weights(y_val, weight_strategy, weight_alpha)

    tuning_history = []

    def objective(trial):
        params = {
            "C": trial.suggest_float("C", 0.001, 100.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"]),
            "solver": "saga",
            "max_iter": trial.suggest_int("max_iter", 500, 3000),
            "class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", None]
            ),
            "random_state": random_state,
        }

        if params["penalty"] == "elasticnet":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)

        try:
            clf = LogisticRegression(**params)
            clf.fit(X_train, y_train, sample_weight=train_weights)
            y_pred = clf.predict(X_val)
            score = compute_metric(y_val, y_pred, metric, sample_weight=val_weights)

            tuning_history.append(
                {
                    "trial_number": trial.number,
                    "params": params.copy(),
                    "metric": metric,
                    "score": score,
                }
            )

            return score
        except Exception:
            return 0.0

    study = optuna.create_study(direction="maximize")
    optuna.logging.set_verbosity(
        optuna.logging.WARNING if not verbose else optuna.logging.INFO
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params, tuning_history


def save_results(
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    output_dir: Path,
    classifier_type: str,
    verbose: bool = True,
):
    """Save two-stage optimization results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save stage 1 results
    stage1_file = output_dir / "stage1_screening_results.json"
    with open(stage1_file, "w") as f:
        json.dump(stage1_results, f, indent=2)

    # Save stage 2 results (without tuning_history to keep file smaller)
    stage2_summary = []
    for result in stage2_results:
        summary = {k: v for k, v in result.items() if k != "tuning_history"}
        stage2_summary.append(summary)

    stage2_file = output_dir / "stage2_optimization_results.json"
    with open(stage2_file, "w") as f:
        json.dump(stage2_summary, f, indent=2)

    # Save detailed results with tuning history
    stage2_detailed_file = output_dir / "stage2_detailed_results.json"
    with open(stage2_detailed_file, "w") as f:
        json.dump(stage2_results, f, indent=2)

    # Create summary report
    report_file = output_dir / "optimization_summary.md"
    with open(report_file, "w") as f:
        f.write("# Two-Stage Hyperparameter Optimization Summary\n\n")
        f.write(f"**Classifier:** {classifier_type}\n\n")

        f.write("## Stage 1: Feature Configuration Screening\n\n")
        passed = [r for r in stage1_results if r["passed_screening"]]
        f.write(f"- Total configurations evaluated: {len(stage1_results)}\n")
        f.write(f"- Passed screening: {len(passed)}\n")
        f.write(f"- Discarded: {len(stage1_results) - len(passed)}\n\n")

        f.write("### Top 10 Configurations (Stage 1)\n\n")
        f.write("| Rank | Feature Config | Score | Features | Status |\n")
        f.write("|------|---------------|-------|----------|--------|\n")
        for i, result in enumerate(stage1_results[:10], 1):
            status = "✓ Passed" if result["passed_screening"] else "✗ Discarded"
            # metric_name = result.get("metric", "score")
            f.write(
                f"| {i} | {result['feature_config']} | {result['score']:.4f} | "
                f"{result.get('n_features', 'N/A')} | {status} |\n"
            )

        f.write("\n## Stage 2: Deep Hyperparameter Optimization\n\n")
        f.write(f"- Configurations optimized: {len(stage2_results)}\n\n")

        f.write("### Final Rankings\n\n")
        metric_label = (
            stage2_results[0].get("metric", "Score") if stage2_results else "Score"
        )
        f.write(
            f"| Rank | Feature Config | Stage 1 {metric_label} | Stage 2 {metric_label} | Improvement |\n"
        )
        f.write("|------|---------------|-------------|-------------|-------------|\n")
        for i, result in enumerate(stage2_results, 1):
            if "stage2_best_score" in result:
                f.write(
                    f"| {i} | {result['feature_config']} | "
                    f"{result['stage1_score']:.4f} | "
                    f"{result['stage2_best_score']:.4f} | "
                    f"+{result['improvement']:.4f} |\n"
                )

        if stage2_results and "best_params" in stage2_results[0]:
            f.write("\n### Best Configuration\n\n")
            best = stage2_results[0]
            f.write(f"**Feature Configuration:** {best['feature_config']}\n\n")
            metric_label = best.get("metric", "Score")
            f.write(f"**Best {metric_label}:** {best['stage2_best_score']:.4f}\n\n")
            f.write("**Best Hyperparameters:**\n\n")
            f.write("```json\n")
            f.write(json.dumps(best["best_params"], indent=2))
            f.write("\n```\n")

    if verbose:
        print(f"\n✓ Results saved to: {output_dir}")
        print(f"  - {stage1_file.name}")
        print(f"  - {stage2_file.name}")
        print(f"  - {stage2_detailed_file.name}")
        print(f"  - {report_file.name}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Two-stage hyperparameter optimization with feature configuration selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize all feature configs in directory, keep top 5 for stage 2
  python -m src.train_classifiers_with_ho_multifeat \\
      -e data/experiments/zero-shot \\
      --features-base-dir data/experiments/zero-shot/features \\
      --top-k 5 --stage2-trials 100

  # With early discarding: drop configs with accuracy < 40%
  python -m src.train_classifiers_with_ho_multifeat \\
      -e data/experiments/zero-shot \\
      --features-base-dir data/experiments/zero-shot/features \\
      --early-discard-threshold 0.40 \\
      --top-k 3

  # Discard bottom 30% after stage 1
  python -m src.train_classifiers_with_ho_multifeat \\
      -e data/experiments/zero-shot \\
      --features-base-dir data/experiments/zero-shot/features \\
      --early-discard-percentile 30 \\
      --top-k 5
        """,
    )

    # Configuration
    parser.add_argument(
        "-e",
        "--experiment-dir",
        type=str,
        required=True,
        help="Experiment directory path",
    )

    # Feature configurations
    parser.add_argument(
        "--features-base-dir",
        type=str,
        required=True,
        help="Base directory containing feature configuration folders",
    )
    parser.add_argument(
        "--feature-pattern",
        type=str,
        default="*_tfidf*",
        help="Glob pattern to match feature directories (default: *_tfidf*)",
    )

    # Stage 1 settings
    parser.add_argument(
        "--early-discard-threshold",
        type=float,
        help="Discard configs with accuracy below this threshold (e.g., 0.40 = 40%%)",
    )
    parser.add_argument(
        "--early-discard-percentile",
        type=float,
        help="Discard bottom N%% of configs (e.g., 25 = bottom 25%%)",
    )

    # Stage 2 settings
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top configs to deeply optimize in stage 2 (default: 5)",
    )
    parser.add_argument(
        "--stage2-trials",
        type=int,
        default=50,
        help="Number of Optuna trials per config in stage 2 (default: 50)",
    )

    # Optimization goal
    parser.add_argument(
        "--metric",
        type=str,
        choices=[
            "accuracy",
            "f1_macro",
            "f1_weighted",
            "f1_micro",
            "precision_macro",
            "recall_macro",
            "ordinal_accuracy",
            "ordinal_distance_linear",
            "ordinal_distance_quadratic",
            "ordinal_mse",
        ],
        default="f1_macro",
        help="Optimization metric (default: f1_macro). "
        "Ordinal metrics account for CEFR level ordering: "
        "ordinal_accuracy allows ±1 level error, "
        "ordinal_distance_linear uses linear penalty, "
        "ordinal_distance_quadratic penalizes far predictions more heavily, "
        "ordinal_mse treats labels as numerical and uses MSE",
    )

    # Sample weighting strategy
    parser.add_argument(
        "--weight-strategy",
        type=str,
        choices=["equal", "inverse", "inverse_sqrt"],
        default="equal",
        help="Sample weighting strategy (default: equal). "
        "inverse = weight inversely to class size, "
        "inverse_sqrt = weight inversely to sqrt of class size",
    )
    parser.add_argument(
        "--weight-alpha",
        type=float,
        default=1.0,
        help="Weight calibration parameter (0=equal weights, 1=full inverse weighting). "
        "Default: 1.0",
    )

    # Common settings
    parser.add_argument(
        "--classifier",
        type=str,
        choices=["xgboost", "logistic"],
        default="xgboost",
        help="Classifier type (default: xgboost)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random state for reproducibility"
    )
    parser.add_argument(
        "--xgb-use-gpu", action="store_true", help="Use GPU for XGBoost"
    )

    # Data config
    parser.add_argument(
        "--cefr-column", type=str, default="cefr_level", help="CEFR column name"
    )

    # Output
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="Output directory for results (default: experiment_dir/ho_multifeat_results)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress verbose output"
    )

    args = parser.parse_args()

    # Check dependencies
    if not OPTUNA_AVAILABLE:
        print("ERROR: Optuna is not installed. Install it with: pip install optuna")
        return 1

    if args.classifier == "xgboost" and not XGBOOST_AVAILABLE:
        print("ERROR: XGBoost is not installed. Install it with: pip install xgboost")
        return 1

    # Create config
    config = GlobalConfig()
    config.experiment_config.experiment_dir = args.experiment_dir
    config.experiment_config.__post_init__()

    config.classifier_config.classifier_type = args.classifier
    config.classifier_config.random_state = args.random_state
    config.classifier_config.xgb_use_gpu = args.xgb_use_gpu

    config.data_config.cefr_column = args.cefr_column
    config.output_config.verbose = not args.quiet

    # Find all feature directories
    features_base = Path(args.features_base_dir)
    if not features_base.exists():
        print(f"ERROR: Features base directory not found: {args.features_base_dir}")
        return 1

    # Find all feature config directories
    feature_config_dirs = sorted(features_base.glob(args.feature_pattern))
    if not feature_config_dirs:
        print(
            f"ERROR: No feature directories found matching pattern: {args.feature_pattern}"
        )
        return 1

    # For each feature config, find dataset subdirectories
    all_feature_dirs = []
    for feature_config_dir in feature_config_dirs:
        for dataset_dir in feature_config_dir.iterdir():
            if dataset_dir.is_dir() and (dataset_dir / "features_dense.csv").exists():
                all_feature_dirs.append(dataset_dir)

    if not all_feature_dirs:
        print("ERROR: No dataset directories with features_dense.csv found")
        return 1

    print(
        f"Found {len(all_feature_dirs)} feature directories across "
        f"{len(feature_config_dirs)} feature configurations"
    )

    # Stage 1: Screen all features
    stage1_results = stage1_screen_features(
        feature_dirs=all_feature_dirs,
        classifier_type=args.classifier,
        config=config,
        val_split=args.val_split,
        metric=args.metric,
        weight_strategy=args.weight_strategy,
        weight_alpha=args.weight_alpha,
        early_discard_threshold=args.early_discard_threshold,
        early_discard_percentile=args.early_discard_percentile,
        verbose=not args.quiet,
    )

    # Stage 2: Optimize top-K
    stage2_results = stage2_optimize_top_features(
        stage1_results=stage1_results,
        top_k=args.top_k,
        classifier_type=args.classifier,
        config=config,
        n_trials_per_config=args.stage2_trials,
        val_split=args.val_split,
        metric=args.metric,
        weight_strategy=args.weight_strategy,
        weight_alpha=args.weight_alpha,
        verbose=not args.quiet,
    )

    # Save results
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            Path(config.experiment_config.experiment_dir) / "ho_multifeat_results"
        )

    save_results(
        stage1_results=stage1_results,
        stage2_results=stage2_results,
        output_dir=output_dir,
        classifier_type=args.classifier,
        verbose=not args.quiet,
    )

    return 0


if __name__ == "__main__":
    exit(main())
