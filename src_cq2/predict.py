"""
Step 4: Make Predictions using Pretrained Classifiers

This module loads pretrained classifiers and makes predictions on:
- Pre-extracted features (default, recommended)
- Raw text with TF-IDF preprocessing (optional, use --preprocess-text)

With custom CEFR classification and calibration reports.

Expected Experiment Folder Structure:
=====================================

experiment_dir/
├── ml-training-data/           # Training data CSVs (used for label lookup if needed)
│   └── *.csv                   # Training datasets
├── ml-test-data/               # Test data CSVs (contains ground truth labels)
│   └── *.csv                   # Test datasets (e.g., norm-CELVA-SP.csv)
├── feature-models/             # TF-IDF models and classifiers
│   ├── {config_hash}_tfidf/    # TF-IDF model directories (for --preprocess-text mode)
│   │   ├── config.json         # TF-IDF configuration
│   │   └── tfidf_model.pkl     # Trained TF-IDF vectorizer
│   └── classifiers/            # Trained classifier models
│       └── {model_name}/       # Classifier directory (e.g., norm-EFCAMDAT-train_logistic_005ebc16_tfidf)
│           ├── classifier.pkl  # Trained classifier
│           ├── config.json     # Classifier configuration
│           └── label_encoder.pkl  # Label encoder
├── features/                   # Extracted TF-IDF features (default mode)
│   └── {config_hash}_tfidf/    # Feature directory per TF-IDF config
│       └── {dataset_name}/     # Dataset subdirectories (e.g., norm-CELVA-SP)
│           ├── config.json     # Feature extraction config
│           ├── feature_names.csv  # Feature names
│           └── features_dense.csv # Dense feature matrix
└── results/                    # Prediction results (output)
    └── {model_name}/           # Results per model
        └── {dataset_name}/     # Results per dataset
            ├── soft_predictions.json
            ├── argmax_predictions.json
            ├── rounded_avg_predictions.json
            └── evaluation_report.md

For detailed structure documentation, see: src/experiment_structure.py
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline

from src.config import DataConfig, ExperimentConfig, GlobalConfig, OutputConfig
from src.experiment_structure import validate_experiment_structure

# Import GroupedTfidfVectorizer so pickle can load it
from src.train_tfidf_groupby import GroupedTfidfVectorizer  # noqa: F401


class PretrainedTfidfWrapper:
    """
    Wrapper for pretrained TfidfVectorizer to work with sklearn make_pipeline API.
    Used with --preprocess-text option.
    """

    def __init__(self, model_path: str):
        with open(model_path, "rb") as f:
            self.vectorizer = pickle.load(f)

    def fit(self, X, y=None):
        # Already fitted, return self for pipeline compatibility
        return self

    def transform(self, X):
        return self.vectorizer.transform(X)

    def fit_transform(self, X, y=None):
        return self.transform(X)


class PretrainedClassifierWrapper:
    """
    Wrapper for pretrained classifier to work with sklearn make_pipeline API.
    Handles label encoding/decoding for classifiers that use encoded labels.
    Used with --preprocess-text option.
    """

    def __init__(self, classifier_dir: Path):
        classifier_path = classifier_dir / "classifier.pkl"
        with open(classifier_path, "rb") as f:
            self.classifier = pickle.load(f)

        # Try to load label encoder if it exists
        encoder_path = classifier_dir / "label_encoder.pkl"
        if encoder_path.exists():
            with open(encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)
        else:
            # Backward compatibility: old models without label encoder
            self.label_encoder = None

    def fit(self, X, y=None):
        # Already fitted, return self for pipeline compatibility
        return self

    def predict(self, X):
        y_pred_encoded = self.classifier.predict(X)

        # Decode labels if encoder exists
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(y_pred_encoded)
        else:
            return y_pred_encoded

    def predict_proba(self, X):
        if hasattr(self.classifier, "predict_proba"):
            return self.classifier.predict_proba(X)
        raise AttributeError("Classifier does not support predict_proba")


def load_classifier_and_encoder(classifier_dir: Path) -> Tuple:
    """
    Load trained classifier and label encoder.

    Args:
        classifier_dir: Path to classifier model directory

    Returns:
        Tuple of (classifier, label_encoder)
    """
    classifier_path = classifier_dir / "classifier.pkl"
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier not found: {classifier_path}")

    with open(classifier_path, "rb") as f:
        classifier = pickle.load(f)

    # Load label encoder if it exists
    encoder_path = classifier_dir / "label_encoder.pkl"
    label_encoder = None
    if encoder_path.exists():
        with open(encoder_path, "rb") as f:
            label_encoder = pickle.load(f)

    return classifier, label_encoder


def load_features_and_labels(
    features_file: str,
    labels_file: Optional[str] = None,
    labels_csv: Optional[str] = None,
    cefr_column: str = "cefr_label",
    verbose: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[pd.Series]]:
    """
    Load pre-extracted features and optional labels for evaluation.

    Args:
        features_file: Path to features CSV (flat features, one row per sample)
        labels_file: Path to separate labels file (one label per line)
        labels_csv: Path to CSV containing labels in a column
        cefr_column: Column name for labels in labels_csv
        verbose: Print loading information

    Returns:
        Tuple of (X_test, y_test, y_test_series) - y_test is None if no labels provided
    """
    # Load features
    if verbose:
        print(f"Loading features from: {features_file}")

    features_df = pd.read_csv(features_file)
    X_test = features_df.values

    # Load labels if provided
    y_test = None
    y_test_series = None

    if labels_file:
        if verbose:
            print(f"Loading labels from: {labels_file}")

        with open(labels_file, "r") as f:
            y_test = np.array([line.strip() for line in f if line.strip()])
        y_test_series = pd.Series(y_test)

    elif labels_csv:
        if verbose:
            print(f"Loading labels from CSV: {labels_csv}")

        labels_df = pd.read_csv(labels_csv)
        if cefr_column not in labels_df.columns:
            raise ValueError(f"Column '{cefr_column}' not found in {labels_csv}")

        y_test_series = labels_df[cefr_column]
        y_test = y_test_series.values

    # Validate shapes match if labels provided
    if y_test is not None and len(X_test) != len(y_test):
        raise ValueError(
            f"Features and labels length mismatch: {len(X_test)} vs {len(y_test)}"
        )

    if verbose:
        print(f"Loaded {len(X_test)} samples with {X_test.shape[1]} features")
        if y_test is not None:
            print(f"True classes: {sorted(y_test_series.unique())}")

    return X_test, y_test, y_test_series


def cefr_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List] = None,
    target_names: Optional[List[str]] = None,
    digits: int = 2,
) -> str:
    """
    Custom CEFR multiclass classification report.

    Similar interface to sklearn and imblearn classification reports but with
    CEFR-specific metrics focusing on ordered class performance.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of labels to include in report
        target_names: Display names for labels
        digits: Number of decimal places

    Returns:
        Formatted classification report string
    """
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    if target_names is None:
        target_names = [str(label) for label in labels]

    # Calculate precision, recall, f1, support for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    # Calculate macro and weighted averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        )
    )

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate CEFR-specific metrics: adjacent accuracy (off-by-one)
    # For CEFR levels, being off by one level is less severe
    cefr_order = {label: idx for idx, label in enumerate(labels)}
    y_true_idx = np.array([cefr_order.get(y, -1) for y in y_true])
    y_pred_idx = np.array([cefr_order.get(y, -1) for y in y_pred])
    adjacent_correct = np.abs(y_true_idx - y_pred_idx) <= 1
    adjacent_accuracy = np.mean(adjacent_correct[y_true_idx >= 0])

    # Build report string
    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
    report = head_fmt.format(
        "", *headers, width=max(len(name) for name in target_names)
    )
    report += "\n\n"

    row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"

    # Per-class rows
    for i, label in enumerate(labels):
        report += row_fmt.format(
            target_names[i],
            precision[i],
            recall[i],
            f1[i],
            int(support[i]),
            width=max(len(name) for name in target_names),
            digits=digits,
        )

    report += "\n"

    # Macro average
    report += row_fmt.format(
        "macro avg",
        macro_precision,
        macro_recall,
        macro_f1,
        int(np.sum(support)),
        width=max(len(name) for name in target_names),
        digits=digits,
    )

    # Weighted average
    report += row_fmt.format(
        "weighted avg",
        weighted_precision,
        weighted_recall,
        weighted_f1,
        int(np.sum(support)),
        width=max(len(name) for name in target_names),
        digits=digits,
    )

    # Add CEFR-specific metrics
    report += "\n"
    report += f"{'accuracy':{max(len(name) for name in target_names)}s} "
    report += f"{accuracy:>9.{digits}f}"
    report += f" {int(np.sum(support)):>9}\n"

    report += f"{'adjacent accuracy':{max(len(name) for name in target_names)}s} "
    report += f"{adjacent_accuracy:>9.{digits}f}"
    report += f" {int(np.sum(support)):>9}\n"

    return report


def cefr_calibration_report(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    labels: Optional[List] = None,
    target_names: Optional[List[str]] = None,
    n_bins: int = 10,
    digits: int = 2,
) -> str:
    """
    Custom CEFR calibration report for soft probabilities.

    Evaluates how well predicted probabilities match actual outcomes,
    particularly important for CEFR classification confidence scores.

    Args:
        y_true: Ground truth labels
        y_pred_proba: Predicted probabilities (n_samples, n_classes)
        labels: List of labels
        target_names: Display names for labels
        n_bins: Number of bins for calibration
        digits: Number of decimal places

    Returns:
        Formatted calibration report string
    """
    if labels is None:
        labels = sorted(list(set(y_true)))

    if target_names is None:
        target_names = [str(label) for label in labels]

    # Map labels to indices (not used but kept for reference)
    # label_to_idx = {label: idx for idx, label in enumerate(labels)}

    # Get predicted class (argmax of probabilities)
    y_pred_idx = np.argmax(y_pred_proba, axis=1)
    y_pred = np.array([labels[idx] for idx in y_pred_idx])

    # Get confidence (max probability)
    confidence = np.max(y_pred_proba, axis=1)

    # Calculate Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_stats = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this confidence bin
        in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin] == y_pred[in_bin])
            avg_confidence_in_bin = np.mean(confidence[in_bin])

            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            bin_stats.append(
                {
                    "range": f"({bin_lower:.2f}, {bin_upper:.2f}]",
                    "count": int(np.sum(in_bin)),
                    "accuracy": accuracy_in_bin,
                    "confidence": avg_confidence_in_bin,
                    "calibration_gap": avg_confidence_in_bin - accuracy_in_bin,
                }
            )
        else:
            bin_stats.append(
                {
                    "range": f"({bin_lower:.2f}, {bin_upper:.2f}]",
                    "count": 0,
                    "accuracy": 0.0,
                    "confidence": 0.0,
                    "calibration_gap": 0.0,
                }
            )

    # Build report
    report = "CEFR CALIBRATION REPORT (Soft Probabilities)\n"
    report += "=" * 80 + "\n\n"

    report += f"Expected Calibration Error (ECE): {ece:.{digits}f}\n\n"

    # Bin statistics
    report += "Confidence Bins:\n"
    report += (
        f"{'Range':<20} {'Count':>10} {'Accuracy':>12} {'Confidence':>12} {'Gap':>12}\n"
    )
    report += "-" * 80 + "\n"

    for stat in bin_stats:
        if stat["count"] > 0:
            report += f"{stat['range']:<20} {stat['count']:>10} "
            report += f"{stat['accuracy']:>12.{digits}f} "
            report += f"{stat['confidence']:>12.{digits}f} "
            report += f"{stat['calibration_gap']:>12.{digits}f}\n"

    report += "\n"

    # Per-class calibration
    report += "Per-Class Calibration:\n"
    report += f"{'Class':<15} {'Avg Prob':>12} {'Accuracy':>12} {'Count':>10}\n"
    report += "-" * 80 + "\n"

    for i, label in enumerate(labels):
        # Get samples where this class was predicted
        predicted_as_class = y_pred == label
        if np.sum(predicted_as_class) > 0:
            avg_prob = np.mean(y_pred_proba[predicted_as_class, i])
            accuracy = np.mean(y_true[predicted_as_class] == label)
            count = int(np.sum(predicted_as_class))

            report += f"{target_names[i]:<15} {avg_prob:>12.{digits}f} "
            report += f"{accuracy:>12.{digits}f} {count:>10}\n"

    return report


def predict_on_features(  # noqa: C901
    config: GlobalConfig,
    classifier_model_name: str,
    features_file: str,
    labels_file: Optional[str] = None,
    labels_csv: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Make predictions on pre-extracted features.

    Args:
        config: GlobalConfig containing all configuration
        classifier_model_name: Name of trained classifier model
        features_file: Path to features CSV file
        labels_file: Path to labels file (optional, for evaluation)
        labels_csv: Path to CSV with labels column (optional, for evaluation)

    Returns:
        Tuple of (y_test, y_pred) - y_test is None if no labels provided
    """
    verbose = config.output_config.verbose
    exp_config = config.experiment_config
    data_config = config.data_config

    # Load classifier and label encoder
    classifier_dir = Path(exp_config.models_dir) / "classifiers" / classifier_model_name
    if not classifier_dir.exists():
        raise FileNotFoundError(f"Classifier directory not found: {classifier_dir}")

    # Load classifier config to get feature_type and config_hash
    classifier_config_path = classifier_dir / "config.json"
    expected_feature_type = None
    expected_config_hash = None

    if classifier_config_path.exists():
        try:
            with open(classifier_config_path, "r") as f:
                clf_cfg = json.load(f)
                expected_feature_type = clf_cfg.get("feature_type")
                expected_config_hash = clf_cfg.get("config_hash")
        except Exception:
            pass

    if verbose:
        print(f"Classifier model: {classifier_model_name}")
        if expected_feature_type:
            print(f"Expected feature type: {expected_feature_type}")
        if expected_config_hash:
            print(f"Expected config hash: {expected_config_hash}")
        print("-" * 70)

    classifier, label_encoder = load_classifier_and_encoder(classifier_dir)

    # Load features and labels
    X_test, y_test, y_test_series = load_features_and_labels(
        features_file=features_file,
        labels_file=labels_file,
        labels_csv=labels_csv,
        cefr_column=data_config.cefr_column,
        verbose=verbose,
    )

    # Make predictions
    if verbose:
        print("\nMaking predictions...")

    y_pred_encoded = classifier.predict(X_test)

    # Decode predictions if encoder exists
    if label_encoder is not None:
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
    else:
        y_pred = y_pred_encoded

    # Get probabilities if available
    y_pred_proba = None
    if hasattr(classifier, "predict_proba"):
        try:
            y_pred_proba = classifier.predict_proba(X_test)
        except (AttributeError, NotImplementedError):
            if verbose:
                print(
                    "Note: Classifier does not support predict_proba, skipping calibration report"
                )

    # Get full model class list (needed for calibration and results saving)
    if label_encoder is not None:
        # Use all classes from the encoder
        model_classes = label_encoder.classes_.tolist()
    elif y_test is not None:
        # Fallback to test labels
        model_classes = sorted(y_test_series.unique())
    else:
        # No labels available, use predicted classes
        model_classes = sorted(np.unique(y_pred))

    # Print evaluation reports if labels available
    if y_test is not None and verbose:
        labels_list = sorted(y_test_series.unique())

        print("\n" + "=" * 80)
        print("CEFR CLASSIFICATION REPORT (Multiclass)")
        print("=" * 80)
        print(
            cefr_classification_report(
                y_test,
                y_pred,
                labels=labels_list,
                target_names=[str(label) for label in labels_list],
            )
        )

        print("\n" + "=" * 80)
        print("STANDARD CLASSIFICATION REPORT")
        print("=" * 80)
        print(classification_report(y_test, y_pred, zero_division=0))

        print("\n" + "=" * 80)
        print("CONFUSION MATRIX")
        print("=" * 80)
        cm = confusion_matrix(y_test, y_pred, labels=labels_list)
        print(f"Labels (rows=true, cols=pred): {labels_list}")
        print(cm)

        # Print calibration report if probabilities available
        if y_pred_proba is not None:
            print("\n" + "=" * 80)
            print(
                cefr_calibration_report(
                    y_test,
                    y_pred_proba,
                    labels=model_classes,  # Use full model classes, not just test labels
                    target_names=[str(label) for label in model_classes],
                )
            )
            print("=" * 80)

    # Save results organized by model, then by dataset
    if config.output_config.save_results:
        # Derive dataset name from features file
        features_path = Path(features_file)
        if features_path.name == "features_dense.csv":
            dataset_name = features_path.parent.name
        else:
            dataset_name = features_path.stem

        # Organize results by: model/dataset/
        # This prevents overwrites when same dataset is used with different TF-IDF configs or classifiers
        results_dir = (
            Path(exp_config.results_dir) / classifier_model_name / dataset_name
        )
        results_dir.mkdir(parents=True, exist_ok=True)

        # Determine actual classes from probability array shape
        if y_pred_proba is not None:
            n_proba_classes = y_pred_proba.shape[1]
            # Use only the classes that match the probability array size
            proba_classes = model_classes[:n_proba_classes]
        else:
            proba_classes = model_classes

        # Save soft predictions (probabilities)
        if y_pred_proba is not None and config.output_config.save_json:
            soft_predictions = []
            for i in range(len(y_pred_proba)):
                pred_dict = {
                    "sample_id": i,
                    "probabilities": {
                        str(cls): float(y_pred_proba[i][j])
                        for j, cls in enumerate(proba_classes)
                    },
                }
                if y_test is not None:
                    pred_dict["true_label"] = str(y_test[i])
                soft_predictions.append(pred_dict)

            soft_pred_path = results_dir / "soft_predictions.json"
            with open(soft_pred_path, "w") as f:
                json.dump(soft_predictions, f, indent=2)

        # Save argmax predictions
        if config.output_config.save_json:
            argmax_predictions = []
            for i in range(len(y_pred)):
                pred_dict = {"sample_id": i, "predicted_label": str(y_pred[i])}
                if y_test is not None:
                    pred_dict["true_label"] = str(y_test[i])
                if y_pred_proba is not None:
                    pred_dict["confidence"] = float(np.max(y_pred_proba[i]))
                argmax_predictions.append(pred_dict)

            argmax_pred_path = results_dir / "argmax_predictions.json"
            with open(argmax_pred_path, "w") as f:
                json.dump(argmax_predictions, f, indent=2)

        # Save rounded average predictions (regression-style)
        if y_pred_proba is not None and config.output_config.save_json:
            rounded_avg_predictions = []

            # Map classes to numeric values (based on actual proba classes)
            # class_to_idx = {cls: idx for idx, cls in enumerate(proba_classes)}
            idx_to_class = {idx: cls for idx, cls in enumerate(proba_classes)}

            y_pred_rounded_avg = []
            for i in range(len(y_pred_proba)):
                # Calculate expected value (weighted average of class indices)
                expected_idx = np.sum(
                    [j * y_pred_proba[i][j] for j in range(len(proba_classes))]
                )
                # Round to nearest integer index
                rounded_idx = int(np.round(expected_idx))
                # Clip to valid range
                rounded_idx = np.clip(rounded_idx, 0, len(proba_classes) - 1)
                # Map back to class label
                pred_label = idx_to_class[rounded_idx]
                y_pred_rounded_avg.append(pred_label)

                pred_dict = {
                    "sample_id": i,
                    "predicted_label": str(pred_label),
                    "expected_value": float(expected_idx),
                    "rounded_index": int(rounded_idx),
                }
                if y_test is not None:
                    pred_dict["true_label"] = str(y_test[i])
                rounded_avg_predictions.append(pred_dict)

            rounded_avg_path = results_dir / "rounded_avg_predictions.json"
            with open(rounded_avg_path, "w") as f:
                json.dump(rounded_avg_predictions, f, indent=2)

            # Generate reports for rounded average strategy
            if y_test is not None and verbose:
                y_pred_rounded_avg = np.array(y_pred_rounded_avg)

                print("\n" + "=" * 80)
                print("ROUNDED AVERAGE STRATEGY RESULTS")
                print("=" * 80)

                print("\nCEFR CLASSIFICATION REPORT (Rounded Avg):")
                print(
                    cefr_classification_report(
                        y_test,
                        y_pred_rounded_avg,
                        labels=labels_list,
                        target_names=[str(label) for label in labels_list],
                    )
                )

        # Save markdown reports
        if y_test is not None:
            report_path = results_dir / "evaluation_report.md"
            with open(report_path, "w") as f:
                f.write(f"# Evaluation Report: {dataset_name}\n\n")
                f.write(f"**Classifier**: {classifier_model_name}\n")
                f.write(f"**Dataset**: {dataset_name}\n")
                f.write(f"**Samples**: {len(y_test)}\n")
                f.write(
                    f"**Classes in test set**: {', '.join(map(str, labels_list))}\n\n"
                )

                # Argmax strategy
                f.write("## Strategy 1: Argmax Predictions\n\n")
                f.write(
                    "Standard argmax strategy: predict class with highest probability.\n\n"
                )
                f.write("### CEFR Classification Report\n\n")
                f.write("```\n")
                f.write(
                    cefr_classification_report(
                        y_test,
                        y_pred,
                        labels=labels_list,
                        target_names=[str(label) for label in labels_list],
                    )
                )
                f.write("```\n\n")

                f.write("### Standard Classification Report\n\n")
                f.write("```\n")
                f.write(classification_report(y_test, y_pred, zero_division=0))
                f.write("```\n\n")

                f.write("### Confusion Matrix\n\n")
                f.write(f"Labels (rows=true, cols=pred): {labels_list}\n\n")
                f.write("```\n")
                cm = confusion_matrix(y_test, y_pred, labels=labels_list)
                f.write(str(cm))
                f.write("\n```\n\n")

                # Calibration report
                if y_pred_proba is not None:
                    f.write("### Calibration Report\n\n")
                    f.write("```\n")
                    f.write(
                        cefr_calibration_report(
                            y_test,
                            y_pred_proba,
                            labels=model_classes,
                            target_names=[str(label) for label in model_classes],
                        )
                    )
                    f.write("```\n\n")

                # Rounded average strategy
                if y_pred_proba is not None:
                    f.write("## Strategy 2: Rounded Average Predictions\n\n")
                    f.write(
                        "Regression-style strategy: calculate expected class index from probabilities, "
                    )
                    f.write("round to nearest integer, map back to class label.\n\n")
                    f.write("### CEFR Classification Report\n\n")
                    f.write("```\n")
                    f.write(
                        cefr_classification_report(
                            y_test,
                            y_pred_rounded_avg,
                            labels=labels_list,
                            target_names=[str(label) for label in labels_list],
                        )
                    )
                    f.write("```\n\n")

                    f.write("### Standard Classification Report\n\n")
                    f.write("```\n")
                    f.write(
                        classification_report(
                            y_test, y_pred_rounded_avg, zero_division=0
                        )
                    )
                    f.write("```\n\n")

                    f.write("### Confusion Matrix\n\n")
                    f.write(f"Labels (rows=true, cols=pred): {labels_list}\n\n")
                    f.write("```\n")
                    cm_rounded = confusion_matrix(
                        y_test, y_pred_rounded_avg, labels=labels_list
                    )
                    f.write(str(cm_rounded))
                    f.write("\n```\n\n")

        if verbose:
            print(f"\n✓ Results saved to: {results_dir}/")
            if config.output_config.save_json:
                if y_pred_proba is not None:
                    print("  - soft_predictions.json")
                print("  - argmax_predictions.json")
                if y_pred_proba is not None:
                    print("  - rounded_avg_predictions.json")
            if y_test is not None:
                print("  - evaluation_report.md")

    return y_test, y_pred


def extract_config_hash_from_model_name(model_name: str) -> Optional[str]:
    """
    Extract config hash and feature type from model name.

    Examples:
        'train_data_logistic_e2752d18_tfidf' -> 'e2752d18_tfidf'
        'norm-EFCAMDAT-train_logistic_005ebc16_tfidf' -> '005ebc16_tfidf'
        'norm-EFCAMDAT-train_logistic_10b21d1b_tfidf_grouped' -> '10b21d1b_tfidf_grouped'

    Args:
        model_name: Name of the classifier model

    Returns:
        Config hash with feature type suffix, or None if pattern doesn't match
    """
    import re

    # Pattern: anything followed by underscore, 8-char hex hash, underscore, feature type
    # Feature type can be: tfidf, tfidf_grouped, etc.
    pattern = r"_([0-9a-f]{8})_(tfidf(?:_grouped)?)"
    match = re.search(pattern, model_name)
    if match:
        config_hash = match.group(1)
        feature_type = match.group(2)
        return f"{config_hash}_{feature_type}"
    return None


def predict_all_models_batch(  # noqa: C901
    config: GlobalConfig,
    models_dir: str,
    features_dir: str,
    labels_csv_dir: Optional[str] = None,
) -> List[Tuple]:
    """
    Run predictions for all models in a directory, automatically matching feature directories.

    For each model, processes all dataset subdirectories within the matching feature directory.
    For example, if feature_dir is '005ebc16_tfidf/' containing subdirectories
    'norm-CELVA-SP/', 'norm-EFCAMDAT-test/', etc., each will be processed separately.

    Args:
        config: GlobalConfig containing all configuration
        models_dir: Directory containing classifier model subdirectories
        features_dir: Directory containing feature subdirectories (with dataset subdirectories inside)
        labels_csv_dir: Directory containing label CSV files (optional)

    Returns:
        List of (model_name, dataset_name, y_test, y_pred) tuples
    """
    verbose = config.output_config.verbose
    exp_config = config.experiment_config

    if labels_csv_dir is None:
        labels_csv_dir = exp_config.ml_test_dir

    models_base_dir = Path(models_dir)
    features_base_dir = Path(features_dir)

    if not models_base_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_base_dir}")

    if not features_base_dir.exists():
        raise FileNotFoundError(f"Features directory not found: {features_base_dir}")

    # Find all model subdirectories
    model_subdirs = sorted([d for d in models_base_dir.iterdir() if d.is_dir()])

    if not model_subdirs:
        print(f"No model subdirectories found in {models_base_dir}")
        return []

    # Build a mapping of config_hash_feature_type -> feature_dir
    feature_dirs_map = {}
    for feature_dir in features_base_dir.iterdir():
        if feature_dir.is_dir():
            feature_dirs_map[feature_dir.name] = feature_dir

    if verbose:
        print(f"Found {len(model_subdirs)} models")
        print(f"Found {len(feature_dirs_map)} feature directories")
        print("=" * 70)

    results = []
    matched_count = 0
    unmatched_count = 0

    for model_subdir in model_subdirs:
        model_name = model_subdir.name

        # Extract config hash from model name
        feature_key = extract_config_hash_from_model_name(model_name)

        if not feature_key:
            if verbose:
                print(
                    f"\n✗ Skipping {model_name}: Could not extract config hash from model name"
                )
            unmatched_count += 1
            continue

        # Find matching feature directory
        if feature_key not in feature_dirs_map:
            if verbose:
                print(
                    f"\n✗ Skipping {model_name}: No matching feature directory for '{feature_key}'"
                )
            unmatched_count += 1
            continue

        feature_dir = feature_dirs_map[feature_key]

        # Check if features are in nested dataset directories
        dataset_subdirs = [d for d in feature_dir.iterdir() if d.is_dir()]

        if not dataset_subdirs:
            if verbose:
                print(
                    f"\n✗ Skipping {model_name}: No dataset subdirectories found in {feature_dir}"
                )
            unmatched_count += 1
            continue

        matched_count += 1

        if verbose:
            print(f"\n{'=' * 70}")
            print(
                f"Processing model {matched_count}/{len(model_subdirs)}: {model_name}"
            )
            print(f"  Config hash: {feature_key}")
            print(f"  Feature dir: {feature_dir.name}")
            print(
                f"  Found {len(dataset_subdirs)} dataset(s): {[d.name for d in dataset_subdirs]}"
            )
            print("-" * 70)

        # Process each dataset subdirectory
        for dataset_subdir in sorted(dataset_subdirs):
            dataset_name = dataset_subdir.name
            features_file = dataset_subdir / "features_dense.csv"

            if not features_file.exists():
                if verbose:
                    print(
                        f"\n  ✗ Skipping dataset {dataset_name}: features_dense.csv not found"
                    )
                continue

            if verbose:
                print(f"\n  Processing dataset: {dataset_name}")

            # Look for corresponding labels CSV
            labels_csv = Path(labels_csv_dir) / f"{dataset_name}.csv"

            if not labels_csv.exists():
                # Try training data directory
                training_labels_csv = (
                    Path(exp_config.ml_training_dir) / f"{dataset_name}.csv"
                )
                if training_labels_csv.exists():
                    labels_csv = training_labels_csv
                    if verbose:
                        print(
                            f"    Using labels from training data: {training_labels_csv.name}"
                        )
                else:
                    if verbose:
                        print(
                            "    ⚠ Labels CSV not found in test or training data (predictions only, no evaluation)"
                        )
                    labels_csv = None

            try:
                y_test, y_pred = predict_on_features(
                    config,
                    classifier_model_name=model_name,
                    features_file=str(features_file),
                    labels_csv=str(labels_csv) if labels_csv else None,
                )
                results.append((model_name, dataset_name, y_test, y_pred))
            except Exception as e:
                print(f"    ✗ Error processing {dataset_name}: {e}")
                if verbose:
                    import traceback

                    traceback.print_exc()

    if verbose:
        print(f"\n{'=' * 70}")
        print("Batch processing complete:")
        print(f"  Successfully processed: {matched_count}")
        print(f"  Skipped/Failed: {unmatched_count}")
        print("=" * 70)

    return results


def predict_all_feature_sets(  # noqa: C901
    config: GlobalConfig,
    classifier_model_name: str,
    features_dir: str,
    labels_csv_dir: Optional[str] = None,
) -> List[Tuple]:
    """
    Run predictions for all feature sets in a directory.

    Args:
        config: GlobalConfig containing all configuration
        classifier_model_name: Name of trained classifier model
        features_dir: Directory containing feature subdirectories
        labels_csv_dir: Directory containing label CSV files (optional)

    Returns:
        List of (y_test, y_pred) tuples
    """
    verbose = config.output_config.verbose
    exp_config = config.experiment_config

    if labels_csv_dir is None:
        labels_csv_dir = exp_config.ml_test_dir

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
        print("=" * 70)

    results = []
    for feature_subdir in feature_subdirs:
        if verbose:
            print(f"\nProcessing: {feature_subdir.name}")
            print("-" * 70)

        # Expected files
        features_file = feature_subdir / "features_dense.csv"

        if not features_file.exists():
            print(f"✗ Features file not found: {features_file}")
            continue

        # Look for corresponding labels CSV
        # Try test data directory first, then training data directory
        labels_csv = Path(labels_csv_dir) / f"{feature_subdir.name}.csv"

        if not labels_csv.exists():
            # Try training data directory
            training_labels_csv = (
                Path(exp_config.ml_training_dir) / f"{feature_subdir.name}.csv"
            )
            if training_labels_csv.exists():
                labels_csv = training_labels_csv
                if verbose:
                    print(
                        f"  Using labels from training data: {training_labels_csv.name}"
                    )
            else:
                if verbose:
                    print(
                        "⚠ Labels CSV not found in test or training data (predictions only, no evaluation)"
                    )
                labels_csv = None

        try:
            y_test, y_pred = predict_on_features(
                config,
                classifier_model_name=classifier_model_name,
                features_file=str(features_file),
                labels_csv=str(labels_csv) if labels_csv else None,
            )
            results.append((y_test, y_pred))
        except Exception as e:
            print(f"✗ Error: {e}")
            if verbose:
                import traceback

                traceback.print_exc()

    return results


def predict_with_text_pipeline(  # noqa: C901
    config: GlobalConfig, classifier_model_name: str, test_file: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions using TF-IDF pipeline with raw text (legacy mode).
    Requires --preprocess-text flag.

    Args:
        config: GlobalConfig containing all configuration
        classifier_model_name: Name of trained classifier model
        test_file: Path or name of test CSV file with text column

    Returns:
        Tuple of (y_test, y_pred)
    """
    verbose = config.output_config.verbose
    exp_config = config.experiment_config
    data_config = config.data_config

    # Load TF-IDF model
    tfidf_model_path = Path(exp_config.models_dir) / "tfidf" / "tfidf_model.pkl"
    if not tfidf_model_path.exists():
        raise FileNotFoundError(f"TF-IDF model not found: {tfidf_model_path}")

    # Load classifier
    classifier_dir = Path(exp_config.models_dir) / "classifiers" / classifier_model_name
    if not classifier_dir.exists():
        raise FileNotFoundError(f"Classifier directory not found: {classifier_dir}")

    # Load test data
    test_file_path = Path(test_file)
    if not test_file_path.is_absolute():
        # Try ml_test_dir if relative path
        test_file_path = Path(exp_config.ml_test_dir) / test_file

    if not test_file_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_file_path}")

    if verbose:
        print(f"Classifier model: {classifier_model_name}")
        print(f"Test file: {test_file_path}")
        print("-" * 70)

    df_test = pd.read_csv(test_file_path)

    # Validate columns
    if data_config.text_column not in df_test.columns:
        raise ValueError(
            f"Text column '{data_config.text_column}' not found in {test_file_path}"
        )

    if data_config.label_column not in df_test.columns:
        raise ValueError(
            f"Label column '{data_config.label_column}' not found in {test_file_path}"
        )

    X_test = df_test[data_config.text_column].fillna("").astype(str)
    y_test = df_test[data_config.label_column].values

    if verbose:
        print(f"Loaded {len(X_test)} samples")
        print(f"True classes: {sorted(pd.Series(y_test).unique())}")

    # Create sklearn pipeline with pretrained models
    if verbose:
        print("\nCreating sklearn pipeline with TF-IDF + classifier...")

    pipeline = make_pipeline(
        PretrainedTfidfWrapper(str(tfidf_model_path)),
        PretrainedClassifierWrapper(classifier_dir),
    )

    # Make predictions
    if verbose:
        print("Making predictions...")

    y_pred = pipeline.predict(X_test)

    # Get probabilities if available
    y_pred_proba = None
    try:
        y_pred_proba = pipeline.predict_proba(X_test)
    except (AttributeError, NotImplementedError):
        if verbose:
            print(
                "Note: Classifier does not support predict_proba, skipping calibration report"
            )

    # Print evaluation reports
    if verbose:
        labels_list = sorted(pd.Series(y_test).unique())

        # Get full model class list from the classifier wrapper (for calibration report)
        classifier_wrapper = pipeline.steps[-1][1]
        if (
            hasattr(classifier_wrapper, "label_encoder")
            and classifier_wrapper.label_encoder is not None
        ):
            model_classes = classifier_wrapper.label_encoder.classes_.tolist()
        else:
            model_classes = labels_list

        print("\n" + "=" * 80)
        print("CEFR CLASSIFICATION REPORT (Multiclass)")
        print("=" * 80)
        print(
            cefr_classification_report(
                y_test,
                y_pred,
                labels=labels_list,
                target_names=[str(label) for label in labels_list],
            )
        )

        print("\n" + "=" * 80)
        print("STANDARD CLASSIFICATION REPORT")
        print("=" * 80)
        print(classification_report(y_test, y_pred, zero_division=0))

        print("\n" + "=" * 80)
        print("CONFUSION MATRIX")
        print("=" * 80)
        cm = confusion_matrix(y_test, y_pred, labels=labels_list)
        print(f"Labels (rows=true, cols=pred): {labels_list}")
        print(cm)

        # Print calibration report if probabilities available
        if y_pred_proba is not None:
            print("\n" + "=" * 80)
            print(
                cefr_calibration_report(
                    y_test,
                    y_pred_proba,
                    labels=model_classes,  # Use full model classes, not just test labels
                    target_names=[str(label) for label in model_classes],
                )
            )
            print("=" * 80)

    # Save results organized by dataset
    if config.output_config.save_results:
        dataset_name = Path(test_file).stem
        results_dir = Path(exp_config.results_dir) / dataset_name
        results_dir.mkdir(parents=True, exist_ok=True)

        # Determine actual classes from probability array shape
        if y_pred_proba is not None:
            n_proba_classes = y_pred_proba.shape[1]
            # Use only the classes that match the probability array size
            proba_classes = model_classes[:n_proba_classes]
        else:
            proba_classes = model_classes

        # Save soft predictions (probabilities)
        if y_pred_proba is not None and config.output_config.save_json:
            soft_predictions = []
            for i in range(len(y_pred_proba)):
                pred_dict = {
                    "sample_id": i,
                    "probabilities": {
                        str(cls): float(y_pred_proba[i][j])
                        for j, cls in enumerate(proba_classes)
                    },
                }
                pred_dict["true_label"] = str(y_test[i])
                soft_predictions.append(pred_dict)

            soft_pred_path = results_dir / "soft_predictions.json"
            with open(soft_pred_path, "w") as f:
                json.dump(soft_predictions, f, indent=2)

        # Save argmax predictions
        if config.output_config.save_json:
            argmax_predictions = []
            for i in range(len(y_pred)):
                pred_dict = {
                    "sample_id": i,
                    "predicted_label": str(y_pred[i]),
                    "true_label": str(y_test[i]),
                }
                if y_pred_proba is not None:
                    pred_dict["confidence"] = float(np.max(y_pred_proba[i]))
                argmax_predictions.append(pred_dict)

            argmax_pred_path = results_dir / "argmax_predictions.json"
            with open(argmax_pred_path, "w") as f:
                json.dump(argmax_predictions, f, indent=2)

        # Save rounded average predictions (regression-style)
        if y_pred_proba is not None and config.output_config.save_json:
            rounded_avg_predictions = []

            # Map classes to numeric values (based on actual proba classes)
            # class_to_idx = {cls: idx for idx, cls in enumerate(proba_classes)}
            idx_to_class = {idx: cls for idx, cls in enumerate(proba_classes)}

            y_pred_rounded_avg = []
            for i in range(len(y_pred_proba)):
                # Calculate expected value (weighted average of class indices)
                expected_idx = np.sum(
                    [j * y_pred_proba[i][j] for j in range(len(proba_classes))]
                )
                # Round to nearest integer index
                rounded_idx = int(np.round(expected_idx))
                # Clip to valid range
                rounded_idx = np.clip(rounded_idx, 0, len(proba_classes) - 1)
                # Map back to class label
                pred_label = idx_to_class[rounded_idx]
                y_pred_rounded_avg.append(pred_label)

                pred_dict = {
                    "sample_id": i,
                    "predicted_label": str(pred_label),
                    "expected_value": float(expected_idx),
                    "rounded_index": int(rounded_idx),
                    "true_label": str(y_test[i]),
                }
                rounded_avg_predictions.append(pred_dict)

            rounded_avg_path = results_dir / "rounded_avg_predictions.json"
            with open(rounded_avg_path, "w") as f:
                json.dump(rounded_avg_predictions, f, indent=2)

            # Generate reports for rounded average strategy
            if verbose:
                y_pred_rounded_avg = np.array(y_pred_rounded_avg)

                print("\n" + "=" * 80)
                print("ROUNDED AVERAGE STRATEGY RESULTS")
                print("=" * 80)

                print("\nCEFR CLASSIFICATION REPORT (Rounded Avg):")
                print(
                    cefr_classification_report(
                        y_test,
                        y_pred_rounded_avg,
                        labels=labels_list,
                        target_names=[str(label) for label in labels_list],
                    )
                )

        # Save markdown reports
        report_path = results_dir / "evaluation_report.md"
        with open(report_path, "w") as f:
            f.write(f"# Evaluation Report: {dataset_name}\n\n")
            f.write(f"**Classifier**: {classifier_model_name}\n")
            f.write(f"**Dataset**: {dataset_name}\n")
            f.write(f"**Samples**: {len(y_test)}\n")
            f.write(f"**Classes in test set**: {', '.join(map(str, labels_list))}\n\n")

            # Argmax strategy
            f.write("## Strategy 1: Argmax Predictions\n\n")
            f.write(
                "Standard argmax strategy: predict class with highest probability.\n\n"
            )
            f.write("### CEFR Classification Report\n\n")
            f.write("```\n")
            f.write(
                cefr_classification_report(
                    y_test,
                    y_pred,
                    labels=labels_list,
                    target_names=[str(label) for label in labels_list],
                )
            )
            f.write("```\n\n")

            f.write("### Standard Classification Report\n\n")
            f.write("```\n")
            f.write(classification_report(y_test, y_pred, zero_division=0))
            f.write("```\n\n")

            f.write("### Confusion Matrix\n\n")
            f.write(f"Labels (rows=true, cols=pred): {labels_list}\n\n")
            f.write("```\n")
            cm = confusion_matrix(y_test, y_pred, labels=labels_list)
            f.write(str(cm))
            f.write("\n```\n\n")

            # Calibration report
            if y_pred_proba is not None:
                f.write("### Calibration Report\n\n")
                f.write("```\n")
                f.write(
                    cefr_calibration_report(
                        y_test,
                        y_pred_proba,
                        labels=model_classes,
                        target_names=[str(label) for label in model_classes],
                    )
                )
                f.write("```\n\n")

            # Rounded average strategy
            if y_pred_proba is not None:
                f.write("## Strategy 2: Rounded Average Predictions\n\n")
                f.write(
                    "Regression-style strategy: calculate expected class index from probabilities, "
                )
                f.write("round to nearest integer, map back to class label.\n\n")
                f.write("### CEFR Classification Report\n\n")
                f.write("```\n")
                f.write(
                    cefr_classification_report(
                        y_test,
                        y_pred_rounded_avg,
                        labels=labels_list,
                        target_names=[str(label) for label in labels_list],
                    )
                )
                f.write("```\n\n")

                f.write("### Standard Classification Report\n\n")
                f.write("```\n")
                f.write(
                    classification_report(y_test, y_pred_rounded_avg, zero_division=0)
                )
                f.write("```\n\n")

                f.write("### Confusion Matrix\n\n")
                f.write(f"Labels (rows=true, cols=pred): {labels_list}\n\n")
                f.write("```\n")
                cm_rounded = confusion_matrix(
                    y_test, y_pred_rounded_avg, labels=labels_list
                )
                f.write(str(cm_rounded))
                f.write("\n```\n\n")

        if verbose:
            print(f"\n✓ Results saved to: {results_dir}/")
            if config.output_config.save_json:
                if y_pred_proba is not None:
                    print("  - soft_predictions.json")
                print("  - argmax_predictions.json")
                if y_pred_proba is not None:
                    print("  - rounded_avg_predictions.json")
            print("  - evaluation_report.md")

    return y_test, y_pred


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for predict."""
    parser = argparse.ArgumentParser(
        description="Make predictions using pretrained classifiers (default: pre-extracted features, optional: --preprocess-text for raw text)",
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

    # Classifier selection
    clf_group = parser.add_argument_group("Classifier Selection")
    clf_group.add_argument(
        "-m",
        "--classifier-model",
        help="Classifier model name (e.g., norm-EFCAMDAT-train_xgboost). Required unless using --batch-models-dir.",
    )

    # Processing mode
    mode_group = parser.add_argument_group("Processing Mode")
    mode_group.add_argument(
        "--preprocess-text",
        action="store_true",
        help="Use TF-IDF preprocessing pipeline with raw text (legacy mode). "
        "Requires -t/--test-file with text CSV. "
        "Default is to use pre-extracted features.",
    )

    # Input data selection (features mode - default)
    input_group = parser.add_argument_group(
        "Input Data Selection (Features Mode - Default)"
    )

    # Simple use case: single feature directory
    input_group.add_argument(
        "-d",
        "--feature-dir",
        help="[Features mode] Path to TF-IDF feature directory (e.g., features/norm-CELVA-SP/). "
        "Must contain features_dense.csv. Script will look for matching labels CSV.",
    )

    # Advanced: explicit file paths
    input_group.add_argument(
        "-f",
        "--features-file",
        help="[Features mode] Path to pre-extracted features CSV file (flat features, one row per sample)",
    )
    input_group.add_argument(
        "--labels-file",
        help="[Features mode] Path to labels file (one label per line, matching features row order)",
    )
    input_group.add_argument(
        "--labels-csv",
        help="[Features mode] Path to CSV file containing labels in a column (use with --cefr-column)",
    )

    # Batch processing: multiple feature directories
    input_group.add_argument(
        "--batch-features-dir",
        help="[Features mode] Directory containing multiple feature subdirectories (for batch processing all)",
    )
    input_group.add_argument(
        "--labels-csv-dir",
        help="[Features mode] Directory containing label CSV files (for batch processing, default: ml-test-data)",
    )

    # Batch processing: multiple models with automatic feature matching
    input_group.add_argument(
        "--batch-models-dir",
        help="[Features mode] Directory containing multiple classifier models (e.g., models/classifiers/). "
        "Requires --batch-features-dir. Script will extract config hash from model name and match to feature directories.",
    )

    # Text mode input
    text_group = parser.add_argument_group("Text Mode Input (with --preprocess-text)")
    text_group.add_argument(
        "-t",
        "--test-file",
        help="[Text mode] Test CSV file with raw text (use with --preprocess-text). "
        "File must contain text and label columns.",
    )

    # Data configuration
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--text-column",
        default="text",
        help="[Text mode] Column name containing text (default: text)",
    )
    data_group.add_argument(
        "--label-column",
        default="label",
        help="[Text mode] Column name containing labels (default: label)",
    )
    data_group.add_argument(
        "--cefr-column",
        default="cefr_label",
        help="[Features mode] Column name containing CEFR labels (default: cefr_label)",
    )

    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--no-save-results", action="store_true", help="Skip saving prediction results"
    )
    output_group.add_argument(
        "--no-save-csv", action="store_true", help="Skip saving CSV outputs"
    )
    output_group.add_argument(
        "--no-save-json", action="store_true", help="Skip saving JSON outputs"
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

        if args.cefr_column != "cefr_label":
            config.data_config.cefr_column = args.cefr_column
        if args.text_column != "text":
            config.data_config.text_column = args.text_column
        if args.label_column != "label":
            config.data_config.label_column = args.label_column

        if args.no_save_results:
            config.output_config.save_results = False
        if args.no_save_csv:
            config.output_config.save_csv = False
        if args.no_save_json:
            config.output_config.save_json = False
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

        data_config = DataConfig(
            text_column=args.text_column,
            label_column=args.label_column,
            cefr_column=args.cefr_column,
        )

        output_config = OutputConfig(
            save_results=not args.no_save_results,
            save_csv=not args.no_save_csv,
            save_json=not args.no_save_json,
            verbose=not args.quiet,
        )

        return GlobalConfig(
            experiment_config, data_config=data_config, output_config=output_config
        )


def main():  # noqa: C901
    """Main entry point for predictions."""
    parser = create_parser()
    args = parser.parse_args()

    print(args)
    input()
    # Build configuration
    try:
        config = args_to_config(args)
    except Exception as e:
        parser.error(f"Configuration error: {e}")
    print(config)
    input()

    if config.output_config.verbose:
        print("Configuration:")
        print(json.dumps(config.to_dict(), indent=2))
        print()

    # Validate experiment structure
    required_dirs = ["ml-test-data", "feature-models", "features"]
    if args.preprocess_text:
        # Text mode needs TF-IDF models
        pass  # feature-models already in required_dirs
    is_valid, errors = validate_experiment_structure(
        config.experiment_config.experiment_dir,
        required_dirs=required_dirs,
        verbose=config.output_config.verbose,
    )
    if not is_valid:
        print("\n✗ Experiment structure validation failed!")
        for error in errors:
            print(f"  {error}")
        print("\nSee docstring or src/experiment_structure.py for expected structure.")
        raise SystemExit(1)

    # Validate classifier model is provided when needed
    if not args.batch_models_dir and not args.classifier_model:
        parser.error("--classifier-model is required unless using --batch-models-dir")

    # Make predictions
    try:
        if args.preprocess_text:
            # Text mode: use TF-IDF pipeline with raw text
            if not args.test_file:
                parser.error("--preprocess-text requires -t/--test-file")

            if config.output_config.verbose:
                print("Mode: Text preprocessing with TF-IDF pipeline")
                print(f"Text column: {config.data_config.text_column}")
                print(f"Label column: {config.data_config.label_column}")
                print()

            predict_with_text_pipeline(
                config,
                classifier_model_name=args.classifier_model,
                test_file=args.test_file,
            )

        else:
            # Features mode (default): use pre-extracted features
            if config.output_config.verbose:
                print("Mode: Pre-extracted features (default)")
                print()

            if args.feature_dir:
                # Simple use case: predict on single feature directory
                feature_dir_path = Path(args.feature_dir)
                if not feature_dir_path.exists():
                    parser.error(f"Feature directory not found: {args.feature_dir}")

                # Expected files in feature directory
                features_file = feature_dir_path / "features_dense.csv"

                if not features_file.exists():
                    parser.error(f"features_dense.csv not found in {args.feature_dir}")

                # Derive dataset name from directory name
                dataset_name = feature_dir_path.name

                # Look for corresponding label CSV in ml-test-data
                labels_csv_path = (
                    Path(config.experiment_config.ml_test_dir) / f"{dataset_name}.csv"
                )

                if not labels_csv_path.exists():
                    if config.output_config.verbose:
                        print(f"⚠ Label CSV not found: {labels_csv_path}")
                        print("  Making predictions without evaluation")
                    labels_csv_path = None

                if config.output_config.verbose:
                    print(f"Predicting on feature directory: {args.feature_dir}")
                    if labels_csv_path:
                        print(f"Using labels from: {labels_csv_path}")
                        print(f"CEFR column: {config.data_config.cefr_column}")

                predict_on_features(
                    config,
                    classifier_model_name=args.classifier_model,
                    features_file=str(features_file),
                    labels_csv=str(labels_csv_path) if labels_csv_path else None,
                )

            elif args.features_file:
                # Advanced: predict on specific features file
                predict_on_features(
                    config,
                    classifier_model_name=args.classifier_model,
                    features_file=args.features_file,
                    labels_file=args.labels_file,
                    labels_csv=args.labels_csv,
                )

            elif args.batch_models_dir:
                # Batch: predict with all models in directory, auto-matching features
                if not args.batch_features_dir:
                    parser.error("--batch-models-dir requires --batch-features-dir")

                predict_all_models_batch(
                    config,
                    models_dir=args.batch_models_dir,
                    features_dir=args.batch_features_dir,
                    labels_csv_dir=args.labels_csv_dir,
                )

            elif args.batch_features_dir:
                # Batch: predict on all feature sets in directory
                predict_all_feature_sets(
                    config,
                    classifier_model_name=args.classifier_model,
                    features_dir=args.batch_features_dir,
                    labels_csv_dir=args.labels_csv_dir,
                )

            else:
                parser.error(
                    "Must provide either -d/--feature-dir, -f/--features-file, --batch-features-dir, or --batch-models-dir (or use --preprocess-text with -t/--test-file)"
                )

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise


if __name__ == "__main__":
    main()
