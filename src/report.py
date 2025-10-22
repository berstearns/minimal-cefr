"""
Results Summarization and Model Ranking Tool

Analyzes all models in an experiment's results directory, extracts performance
metrics, and ranks models according to various criteria.

Expected File Structure
-----------------------
INPUT:
experiment-dir/
└── results/
    ├── model-1_xgboost_abc123_tfidf/
    │   ├── test-set-1/
    │   │   └── evaluation_report.md    # Contains metrics
    │   └── test-set-2/
    │       └── evaluation_report.md
    └── model-2_logistic_def456_tfidf/
        └── ...

OUTPUT:
experiment-dir/
└── results_summary.md                  # Ranked model comparison table

The tool parses evaluation_report.md files from all model/dataset combinations
and generates a summary table ranked by performance metrics.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


@dataclass
class ModelMetrics:
    """Performance metrics for a single model on a single dataset."""

    model_name: str
    dataset_name: str
    strategy: str  # "argmax" or "rounded_avg"

    # Metrics
    accuracy: Optional[float] = None
    adjacent_accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    weighted_f1: Optional[float] = None

    # Model configuration
    tfidf_hash: Optional[str] = None
    tfidf_max_features: Optional[int] = None
    tfidf_readable_name: Optional[str] = None
    classifier_type: Optional[str] = None

    # Additional info
    n_samples: Optional[int] = None
    classes_in_test: Optional[List[str]] = field(default_factory=list)


def parse_evaluation_report(
    report_path: Path,
) -> Dict[str, Dict[str, float]]:  # noqa: C901
    """
    Parse evaluation_report.md to extract metrics for both strategies.

    Returns:
        Dict with keys "argmax" and "rounded_avg", each containing metrics dict
    """
    if not report_path.exists():
        return {}

    with open(report_path, "r") as f:
        content = f.read()

    results = {}

    # Parse both strategies
    for strategy_name, section_title in [
        ("argmax", "Strategy 1: Argmax Predictions"),
        ("rounded_avg", "Strategy 2: Rounded Average Predictions"),
    ]:
        metrics = {}

        # Find the section
        if section_title in content:
            # Extract accuracy from CEFR Classification Report
            # Pattern: "accuracy      0.XX      1742"
            accuracy_pattern = r"^accuracy\s+(0?\.\d+)\s+\d+$"
            accuracy_match = re.search(
                accuracy_pattern, content[content.find(section_title) :], re.MULTILINE
            )
            if accuracy_match:
                metrics["accuracy"] = float(accuracy_match.group(1))

            # Extract adjacent accuracy
            # Pattern: "adjacent accuracy      0.XX      1742"
            adj_acc_pattern = r"^adjacent accuracy\s+(0?\.\d+)\s+\d+$"
            adj_acc_match = re.search(
                adj_acc_pattern, content[content.find(section_title) :], re.MULTILINE
            )
            if adj_acc_match:
                metrics["adjacent_accuracy"] = float(adj_acc_match.group(1))

            # Extract macro avg F1 from standard classification report
            # Pattern: "macro avg       0.XX      0.XX      0.XX      1742"
            # We want the f1-score (3rd column)
            macro_f1_pattern = r"^macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+$"
            macro_f1_match = re.search(
                macro_f1_pattern, content[content.find(section_title) :], re.MULTILINE
            )
            if macro_f1_match:
                metrics["macro_f1"] = float(
                    macro_f1_match.group(3)
                )  # 3rd group is f1-score

            # Extract weighted avg F1
            weighted_f1_pattern = (
                r"^weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+$"
            )
            weighted_f1_match = re.search(
                weighted_f1_pattern,
                content[content.find(section_title) :],
                re.MULTILINE,
            )
            if weighted_f1_match:
                metrics["weighted_f1"] = float(
                    weighted_f1_match.group(3)
                )  # 3rd group is f1-score

        results[strategy_name] = metrics

    # Extract dataset info (from header)
    samples_pattern = r"\*\*Samples\*\*:\s+(\d+)"
    samples_match = re.search(samples_pattern, content)
    if samples_match:
        for strategy in results.values():
            strategy["n_samples"] = int(samples_match.group(1))

    classes_pattern = r"\*\*Classes in test set\*\*:\s+(.+)"
    classes_match = re.search(classes_pattern, content)
    if classes_match:
        classes = [c.strip() for c in classes_match.group(1).split(",")]
        for strategy in results.values():
            strategy["classes_in_test"] = classes

    return results


def parse_manual_predictions(
    predictions_json_path: Path,
) -> Tuple[List[str], List[Dict[str, float]]]:
    """
    Parse manual prediction JSON file.

    Expected format:
    {
      "sample_id_1": {"A1": 0.0, "A2": 0.1, "B1": 0.8, "B2": 0.1, "C1": 0.0, "C2": 0.0},
      "sample_id_2": {...},
      ...
    }

    Returns:
        Tuple of (sample_ids, probabilities_list)
    """
    with open(predictions_json_path, "r") as f:
        data = json.load(f)

    sample_ids = []
    probabilities = []

    for sample_id, probs in data.items():
        sample_ids.append(sample_id)
        probabilities.append(probs)

    return sample_ids, probabilities


def load_ground_truth_labels(
    labels_csv_path: Path, id_column: str = "writing_id", label_column: str = "cefr_level"
) -> Dict[str, str]:
    """
    Load ground truth labels from CSV file.

    Args:
        labels_csv_path: Path to CSV file with ground truth labels
        id_column: Name of column containing sample IDs
        label_column: Name of column containing CEFR labels

    Returns:
        Dict mapping sample_id -> true_label
    """
    df = pd.read_csv(labels_csv_path)

    if id_column not in df.columns:
        raise ValueError(
            f"Column '{id_column}' not found in {labels_csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    if label_column not in df.columns:
        raise ValueError(
            f"Column '{label_column}' not found in {labels_csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    # Convert IDs to strings to ensure consistent matching with JSON keys
    labels_dict = {}
    for idx, label in zip(df[id_column], df[label_column]):
        # Convert to string for consistent matching
        labels_dict[str(idx)] = str(label)

    return labels_dict


def compute_metrics_from_predictions(
    sample_ids: List[str],
    probabilities: List[Dict[str, float]],
    ground_truth: Dict[str, str],
    strategy: str = "argmax",
) -> Dict[str, float]:
    """
    Compute metrics from predictions and ground truth.

    Args:
        sample_ids: List of sample IDs
        probabilities: List of probability dictionaries
        ground_truth: Dict mapping sample_id -> true_label
        strategy: "argmax" or "rounded_avg"

    Returns:
        Dict with computed metrics
    """
    CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
    CEFR_TO_IDX = {level: idx for idx, level in enumerate(CEFR_LEVELS)}

    # Filter to samples that exist in ground truth
    valid_indices = [
        i for i, sid in enumerate(sample_ids) if sid in ground_truth
    ]

    if not valid_indices:
        return {}

    filtered_ids = [sample_ids[i] for i in valid_indices]
    filtered_probs = [probabilities[i] for i in valid_indices]

    # Get true labels
    y_true = [ground_truth[sid] for sid in filtered_ids]

    # Generate predictions based on strategy
    if strategy == "argmax":
        # Standard argmax
        y_pred = []
        for probs in filtered_probs:
            pred_label = max(probs.items(), key=lambda x: x[1])[0]
            y_pred.append(pred_label)

    elif strategy == "rounded_avg":
        # Regression-style: expected class index, rounded
        y_pred = []
        for probs in filtered_probs:
            expected_idx = sum(
                CEFR_TO_IDX[level] * prob
                for level, prob in probs.items()
                if level in CEFR_TO_IDX
            )
            rounded_idx = int(round(expected_idx))
            rounded_idx = max(0, min(5, rounded_idx))  # Clamp to valid range
            y_pred.append(CEFR_LEVELS[rounded_idx])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Compute standard metrics
    accuracy = accuracy_score(y_true, y_pred)

    # Compute adjacent accuracy (CEFR-specific)
    adjacent_correct = 0
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label not in CEFR_TO_IDX or pred_label not in CEFR_TO_IDX:
            continue
        true_idx = CEFR_TO_IDX[true_label]
        pred_idx = CEFR_TO_IDX[pred_label]
        if abs(true_idx - pred_idx) <= 1:
            adjacent_correct += 1

    adjacent_accuracy = adjacent_correct / len(y_true) if y_true else 0.0

    # Compute F1 scores
    # Get unique labels present in y_true
    labels_in_data = sorted(set(y_true) & set(CEFR_LEVELS), key=lambda x: CEFR_TO_IDX[x])

    if len(labels_in_data) > 1:
        macro_f1 = f1_score(y_true, y_pred, labels=labels_in_data, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, labels=labels_in_data, average="weighted", zero_division=0)
    else:
        # Not enough classes for meaningful F1
        macro_f1 = 0.0
        weighted_f1 = 0.0

    return {
        "accuracy": accuracy,
        "adjacent_accuracy": adjacent_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "n_samples": len(y_true),
        "classes_in_test": labels_in_data,
    }


def load_model_config(experiment_dir: Path, model_name: str) -> Dict:
    """Load model config.json to get TF-IDF and classifier parameters."""
    config_path = (
        experiment_dir / "feature-models" / "classifiers" / model_name / "config.json"
    )

    if not config_path.exists():
        return {}

    with open(config_path, "r") as f:
        return json.load(f)


def detect_results_structure(results_dir: Path) -> str:
    """
    Detect the type of results structure.

    Returns:
        "standard" - Has evaluation_report.md files
        "manual" - Has prediction JSON files without evaluation reports
    """
    if not results_dir.exists():
        return "standard"

    # Check a few model directories
    model_dirs = [d for d in results_dir.iterdir() if d.is_dir()][:3]

    for model_dir in model_dirs:
        dataset_dirs = [d for d in model_dir.iterdir() if d.is_dir()][:3]

        for dataset_dir in dataset_dirs:
            # Check for evaluation_report.md
            if (dataset_dir / "evaluation_report.md").exists():
                return "standard"

            # Check for JSON prediction files
            json_files = list(dataset_dir.glob("*.json"))
            if json_files:
                return "manual"

    # Default to standard
    return "standard"


def collect_metrics_from_manual_predictions(
    experiment_dir: Path,
    labels_dir: Optional[Path] = None,
    id_column: str = "writing_id",
    label_column: str = "cefr_level",
    verbose: bool = False,
) -> List[ModelMetrics]:
    """
    Collect metrics from manual prediction JSON files.

    Args:
        experiment_dir: Path to experiment directory (or results directory)
        labels_dir: Directory containing ground truth CSV files
        id_column: Column name for sample IDs in CSV
        label_column: Column name for labels in CSV
        verbose: Print progress

    Returns:
        List of ModelMetrics objects
    """
    # Determine if experiment_dir is the results dir or contains it
    if (experiment_dir / "results").exists():
        results_dir = experiment_dir / "results"
        # Try to find labels in ml-test-data if not specified
        if labels_dir is None and (experiment_dir / "ml-test-data").exists():
            labels_dir = experiment_dir / "ml-test-data"
    else:
        results_dir = experiment_dir

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return []

    if labels_dir is None:
        print("Warning: No labels directory specified. Use --labels-dir to specify ground truth labels.")
        print("Will attempt to match dataset names to CSV files in current directory.")
        labels_dir = Path(".")

    all_metrics = []

    # Iterate through model directories
    model_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])

    if verbose:
        print(f"Found {len(model_dirs)} model directories")

    for model_dir in model_dirs:
        model_name = model_dir.name

        # Load model configuration (may not exist for manual predictions)
        model_config = {}
        if (experiment_dir / "feature-models").exists():
            model_config = load_model_config(experiment_dir, model_name)

        # Iterate through dataset subdirectories
        dataset_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir()])

        for dataset_dir in dataset_dirs:
            dataset_name = dataset_dir.name

            # Find prediction JSON file(s)
            json_files = list(dataset_dir.glob("*.json"))

            if not json_files:
                if verbose:
                    print(f"  ⚠ No JSON files: {model_name}/{dataset_name}")
                continue

            # Use first JSON file found (or look for specific pattern)
            predictions_file = json_files[0]
            if verbose:
                print(f"  Processing: {model_name}/{dataset_name} ({predictions_file.name})")

            # Find corresponding ground truth CSV
            # Try different naming patterns
            possible_csv_names = [
                f"{dataset_name}.csv",
                f"norm-{dataset_name}.csv",
                f"{dataset_name.upper()}.csv",
                f"norm-{dataset_name.upper()}.csv",
            ]

            labels_csv = None
            for csv_name in possible_csv_names:
                csv_path = labels_dir / csv_name
                if csv_path.exists():
                    labels_csv = csv_path
                    break

            # If not found, try fuzzy matching (case-insensitive substring match)
            if labels_csv is None:
                all_csvs = list(labels_dir.glob("*.csv"))
                dataset_lower = dataset_name.lower()

                for csv_path in all_csvs:
                    csv_name_lower = csv_path.stem.lower()
                    # Check if dataset name is a substring of CSV name
                    if dataset_lower in csv_name_lower:
                        labels_csv = csv_path
                        if verbose:
                            print(f"     Matched {dataset_name} -> {csv_path.name}")
                        break

            if labels_csv is None:
                if verbose:
                    print(f"  ⚠ No ground truth CSV found for {dataset_name}")
                    print(f"     Tried: {possible_csv_names}")
                continue

            try:
                # Load predictions
                sample_ids, probabilities = parse_manual_predictions(predictions_file)

                # Load ground truth
                ground_truth = load_ground_truth_labels(
                    labels_csv, id_column=id_column, label_column=label_column
                )

                # Compute metrics for both strategies
                for strategy in ["argmax", "rounded_avg"]:
                    metrics = compute_metrics_from_predictions(
                        sample_ids, probabilities, ground_truth, strategy=strategy
                    )

                    if not metrics:
                        continue

                    model_metrics = ModelMetrics(
                        model_name=model_name,
                        dataset_name=dataset_name,
                        strategy=strategy,
                        accuracy=metrics.get("accuracy"),
                        adjacent_accuracy=metrics.get("adjacent_accuracy"),
                        macro_f1=metrics.get("macro_f1"),
                        weighted_f1=metrics.get("weighted_f1"),
                        n_samples=metrics.get("n_samples"),
                        classes_in_test=metrics.get("classes_in_test", []),
                        tfidf_hash=model_config.get("tfidf_hash"),
                        tfidf_max_features=model_config.get("tfidf_max_features"),
                        tfidf_readable_name=model_config.get("tfidf_readable_name"),
                        classifier_type=model_config.get("classifier_type"),
                    )

                    all_metrics.append(model_metrics)

            except Exception as e:
                print(f"  ✗ Error processing {model_name}/{dataset_name}: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
                continue

    if verbose:
        print(
            f"Collected {len(all_metrics)} metric records "
            f"({len(all_metrics)//2 if all_metrics else 0} per strategy)"
        )

    return all_metrics


def collect_all_metrics(
    experiment_dir: Path,
    mode: str = "auto",
    labels_dir: Optional[Path] = None,
    id_column: str = "writing_id",
    label_column: str = "cefr_level",
    verbose: bool = False,
) -> List[ModelMetrics]:
    """
    Scan results directory and collect all model metrics.

    Args:
        experiment_dir: Path to experiment directory
        mode: "auto", "standard", or "manual"
        labels_dir: For manual mode, directory with ground truth CSVs
        id_column: For manual mode, column name for sample IDs
        label_column: For manual mode, column name for labels
        verbose: Print progress

    Returns:
        List of ModelMetrics objects
    """
    results_dir = experiment_dir / "results"

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return []

    # Detect or use specified mode
    if mode == "auto":
        detected_mode = detect_results_structure(results_dir)
        if verbose:
            print(f"Auto-detected mode: {detected_mode}")
        mode = detected_mode

    # Route to appropriate collection function
    if mode == "manual":
        return collect_metrics_from_manual_predictions(
            experiment_dir=experiment_dir,
            labels_dir=labels_dir,
            id_column=id_column,
            label_column=label_column,
            verbose=verbose,
        )

    # Standard mode (original implementation)
    all_metrics = []

    # Iterate through model directories
    model_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])

    if verbose:
        print(f"Found {len(model_dirs)} model directories")

    for model_dir in model_dirs:
        model_name = model_dir.name

        # Load model configuration
        model_config = load_model_config(experiment_dir, model_name)

        # Iterate through dataset subdirectories
        dataset_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir()])

        for dataset_dir in dataset_dirs:
            dataset_name = dataset_dir.name

            # Parse evaluation report
            report_path = dataset_dir / "evaluation_report.md"
            if not report_path.exists():
                if verbose:
                    print(f"  ⚠ No evaluation report: {model_name}/{dataset_name}")
                continue

            metrics_by_strategy = parse_evaluation_report(report_path)

            # Create ModelMetrics for each strategy
            for strategy, metrics in metrics_by_strategy.items():
                if not metrics:  # Skip if no metrics extracted
                    continue

                model_metrics = ModelMetrics(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    strategy=strategy,
                    accuracy=metrics.get("accuracy"),
                    adjacent_accuracy=metrics.get("adjacent_accuracy"),
                    macro_f1=metrics.get("macro_f1"),
                    weighted_f1=metrics.get("weighted_f1"),
                    n_samples=metrics.get("n_samples"),
                    classes_in_test=metrics.get("classes_in_test", []),
                    tfidf_hash=model_config.get("tfidf_hash"),
                    tfidf_max_features=model_config.get("tfidf_max_features"),
                    tfidf_readable_name=model_config.get("tfidf_readable_name"),
                    classifier_type=model_config.get("classifier_type"),
                )

                all_metrics.append(model_metrics)

    if verbose:
        print(
            f"Collected {len(all_metrics)} metric records ({len(all_metrics)//2} per strategy)"
        )

    return all_metrics


def rank_models(
    metrics_list: List[ModelMetrics],
    criterion: str = "accuracy",
    dataset_filter: Optional[str] = None,
    strategy_filter: Optional[str] = None,
) -> List[ModelMetrics]:
    """
    Rank models according to a criterion.

    Args:
        metrics_list: List of ModelMetrics
        criterion: Ranking criterion ("accuracy", "adjacent_accuracy", "macro_f1", "weighted_f1")
        dataset_filter: Only include specific dataset (e.g., "norm-CELVA-SP")
        strategy_filter: Only include specific strategy ("argmax" or "rounded_avg")

    Returns:
        Sorted list of ModelMetrics (best first)
    """
    # Filter
    filtered = metrics_list

    if dataset_filter:
        filtered = [m for m in filtered if m.dataset_name == dataset_filter]

    if strategy_filter:
        filtered = [m for m in filtered if m.strategy == strategy_filter]

    # Get criterion value
    def get_criterion_value(m: ModelMetrics) -> float:
        value = getattr(m, criterion, None)
        return value if value is not None else -1.0

    # Sort descending (best first)
    sorted_metrics = sorted(filtered, key=get_criterion_value, reverse=True)

    return sorted_metrics


def print_ranking_table(
    ranked_metrics: List[ModelMetrics],
    criterion: str,
    top_n: Optional[int] = None,
    show_config: bool = True,
):
    """Print ranked models as a formatted table."""
    if not ranked_metrics:
        print("No results to display.")
        return

    if top_n:
        ranked_metrics = ranked_metrics[:top_n]

    print(f"\n{'='*120}")
    print(f"RANKING BY: {criterion.upper().replace('_', ' ')}")
    print(f"{'='*120}")

    # Header
    if show_config:
        print(
            f"{'Rank':<6} {'Model':<45} {'Dataset':<20} {'Strategy':<12} {criterion.replace('_', ' ').title():<10} {'TF-IDF':<20} {'Classifier':<12}"
        )
        print(f"{'-'*6} {'-'*45} {'-'*20} {'-'*12} {'-'*10} {'-'*20} {'-'*12}")
    else:
        print(
            f"{'Rank':<6} {'Model':<45} {'Dataset':<20} {'Strategy':<12} {criterion.replace('_', ' ').title():<10}"
        )
        print(f"{'-'*6} {'-'*45} {'-'*20} {'-'*12} {'-'*10}")

    # Rows
    for i, m in enumerate(ranked_metrics, 1):
        value = getattr(m, criterion, None)
        value_str = f"{value:.4f}" if value is not None else "N/A"

        # Truncate model name if too long
        model_display = (
            m.model_name[:43] + ".." if len(m.model_name) > 45 else m.model_name
        )

        if show_config:
            tfidf_display = (
                m.tfidf_readable_name or f"hash:{m.tfidf_hash[:8]}"
                if m.tfidf_hash
                else "N/A"
            )
            clf_display = m.classifier_type or "N/A"
            print(
                f"{i:<6} {model_display:<45} {m.dataset_name:<20} {m.strategy:<12} {value_str:<10} {tfidf_display:<20} {clf_display:<12}"
            )
        else:
            print(
                f"{i:<6} {model_display:<45} {m.dataset_name:<20} {m.strategy:<12} {value_str:<10}"
            )

    print(f"{'='*120}\n")


def print_ranking_grouped_by_dataset(
    metrics_list: List[ModelMetrics],
    criterion: str,
    strategy_filter: Optional[str] = None,
    top_n: Optional[int] = None,
    show_config: bool = True,
):
    """Print ranked models grouped by dataset."""
    if not metrics_list:
        print("No results to display.")
        return

    # Group metrics by dataset
    datasets = {}
    for m in metrics_list:
        if m.dataset_name not in datasets:
            datasets[m.dataset_name] = []
        datasets[m.dataset_name].append(m)

    # Sort and rank within each dataset
    print(f"\n{'='*120}")
    print(f"RANKING BY: {criterion.upper().replace('_', ' ')} (Grouped by Dataset)")
    print(f"{'='*120}\n")

    for dataset_name in sorted(datasets.keys()):
        dataset_metrics = datasets[dataset_name]

        # Rank within this dataset
        ranked = rank_models(
            dataset_metrics, criterion=criterion, strategy_filter=strategy_filter
        )

        if not ranked:
            continue

        # Apply top_n limit per dataset
        if top_n:
            ranked = ranked[:top_n]

        # Print dataset header
        print(f"📊 Dataset: {dataset_name}")
        print(f"{'-'*120}")

        # Header
        if show_config:
            print(
                f"{'Rank':<6} {'Model':<45} {'Strategy':<12} {criterion.replace('_', ' ').title():<10} {'TF-IDF':<20} {'Classifier':<12}"
            )
            print(f"{'-'*6} {'-'*45} {'-'*12} {'-'*10} {'-'*20} {'-'*12}")
        else:
            print(
                f"{'Rank':<6} {'Model':<45} {'Strategy':<12} {criterion.replace('_', ' ').title():<10}"
            )
            print(f"{'-'*6} {'-'*45} {'-'*12} {'-'*10}")

        # Rows
        for i, m in enumerate(ranked, 1):
            value = getattr(m, criterion, None)
            value_str = f"{value:.4f}" if value is not None else "N/A"

            # Truncate model name if too long
            model_display = (
                m.model_name[:43] + ".." if len(m.model_name) > 45 else m.model_name
            )

            if show_config:
                tfidf_display = (
                    m.tfidf_readable_name or f"hash:{m.tfidf_hash[:8]}"
                    if m.tfidf_hash
                    else "N/A"
                )
                clf_display = m.classifier_type or "N/A"
                print(
                    f"{i:<6} {model_display:<45} {m.strategy:<12} {value_str:<10} {tfidf_display:<20} {clf_display:<12}"
                )
            else:
                print(f"{i:<6} {model_display:<45} {m.strategy:<12} {value_str:<10}")

        print()  # Blank line between datasets

    print(f"{'='*120}\n")


def generate_summary_report(  # noqa: C901
    experiment_dir: Path,
    metrics_list: List[ModelMetrics],
    output_path: Optional[Path] = None,
):
    """Generate comprehensive markdown summary report."""
    if not metrics_list:
        print("No metrics to summarize.")
        return

    # Aggregate statistics
    unique_models = set(m.model_name for m in metrics_list)
    unique_datasets = set(m.dataset_name for m in metrics_list)
    unique_strategies = set(m.strategy for m in metrics_list)

    report_lines = []
    report_lines.append(f"# Experiment Results Summary: {experiment_dir.name}")
    report_lines.append("")
    report_lines.append(f"**Experiment Directory:** `{experiment_dir}`")
    report_lines.append(f"**Generated:** {Path.cwd()}")
    report_lines.append("")
    report_lines.append("## Overview")
    report_lines.append("")
    report_lines.append(f"- **Total Models:** {len(unique_models)}")
    report_lines.append(f"- **Datasets Evaluated:** {len(unique_datasets)}")
    report_lines.append(f"- **Prediction Strategies:** {len(unique_strategies)}")
    report_lines.append(f"- **Total Evaluations:** {len(metrics_list)}")
    report_lines.append("")

    # Top models by different criteria
    criteria = [
        ("accuracy", "Accuracy"),
        ("adjacent_accuracy", "Adjacent Accuracy"),
        ("macro_f1", "Macro F1-Score"),
        ("weighted_f1", "Weighted F1-Score"),
    ]

    for criterion, criterion_name in criteria:
        report_lines.append(f"## Top 10 Models by {criterion_name}")
        report_lines.append("")

        for strategy in ["argmax", "rounded_avg"]:
            ranked = rank_models(
                metrics_list, criterion=criterion, strategy_filter=strategy
            )[:10]

            if ranked:
                report_lines.append(
                    f"### Strategy: {strategy.replace('_', ' ').title()}"
                )
                report_lines.append("")
                report_lines.append(
                    f"| Rank | Model | Dataset | {criterion_name} | TF-IDF Config | Classifier |"
                )
                report_lines.append(
                    "|------|-------|---------|----------|---------------|------------|"
                )

                for i, m in enumerate(ranked, 1):
                    value = getattr(m, criterion, None)
                    value_str = f"{value:.4f}" if value is not None else "N/A"
                    tfidf_display = (
                        m.tfidf_readable_name or f"`{m.tfidf_hash[:8]}`"
                        if m.tfidf_hash
                        else "N/A"
                    )
                    clf_display = m.classifier_type or "N/A"

                    report_lines.append(
                        f"| {i} | `{m.model_name}` | {m.dataset_name} | {value_str} | {tfidf_display} | {clf_display} |"
                    )

                report_lines.append("")

    # Performance by dataset
    report_lines.append("## Performance by Dataset")
    report_lines.append("")

    for dataset in sorted(unique_datasets):
        report_lines.append(f"### {dataset}")
        report_lines.append("")

        dataset_metrics = [
            m
            for m in metrics_list
            if m.dataset_name == dataset and m.strategy == "argmax"
        ]

        if dataset_metrics:
            # Find best model for this dataset
            best_by_acc = max(dataset_metrics, key=lambda m: m.accuracy or 0)
            best_by_adj = max(dataset_metrics, key=lambda m: m.adjacent_accuracy or 0)

            report_lines.append(
                f"- **Best Accuracy:** {best_by_acc.accuracy:.4f} (`{best_by_acc.model_name}`)"
            )
            report_lines.append(
                f"- **Best Adjacent Accuracy:** {best_by_adj.adjacent_accuracy:.4f} (`{best_by_adj.model_name}`)"
            )
            report_lines.append(f"- **Models Evaluated:** {len(dataset_metrics)}")
            report_lines.append("")

    # TF-IDF configuration comparison
    report_lines.append("## TF-IDF Configuration Analysis")
    report_lines.append("")

    tfidf_configs = {}
    for m in metrics_list:
        if m.tfidf_hash and m.strategy == "argmax":
            if m.tfidf_hash not in tfidf_configs:
                tfidf_configs[m.tfidf_hash] = {
                    "readable_name": m.tfidf_readable_name,
                    "max_features": m.tfidf_max_features,
                    "accuracies": [],
                }
            if m.accuracy is not None:
                tfidf_configs[m.tfidf_hash]["accuracies"].append(m.accuracy)

    if tfidf_configs:
        report_lines.append(
            "| TF-IDF Config | Max Features | Avg Accuracy | Min Accuracy | Max Accuracy | Evaluations |"
        )
        report_lines.append(
            "|---------------|--------------|--------------|--------------|--------------|-------------|"
        )

        for hash_id, config in sorted(
            tfidf_configs.items(),
            key=lambda x: (
                sum(x[1]["accuracies"]) / len(x[1]["accuracies"])
                if x[1]["accuracies"]
                else 0
            ),
            reverse=True,
        ):
            accs = config["accuracies"]
            if accs:
                avg_acc = sum(accs) / len(accs)
                min_acc = min(accs)
                max_acc = max(accs)

                name_display = config["readable_name"] or f"`{hash_id[:8]}`"
                features_display = config["max_features"] or "N/A"

                report_lines.append(
                    f"| {name_display} | {features_display} | {avg_acc:.4f} | {min_acc:.4f} | {max_acc:.4f} | {len(accs)} |"
                )

        report_lines.append("")

    # Write or print report
    report_content = "\n".join(report_lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report_content)
        print(f"Summary report saved to: {output_path}")
    else:
        print(report_content)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize and rank CEFR classification model results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  1. Generate summary report:
     python -m src.report -e data/experiments/zero-shot

  2. Rank by accuracy (top 20):
     python -m src.report -e data/experiments/zero-shot --rank accuracy --top 20

  3. Rank by adjacent accuracy for specific dataset:
     python -m src.report -e data/experiments/zero-shot \\
         --rank adjacent_accuracy \\
         --dataset norm-CELVA-SP

  4. Compare argmax vs rounded_avg strategies:
     python -m src.report -e data/experiments/zero-shot \\
         --rank accuracy \\
         --strategy argmax

  5. Save summary to file:
     python -m src.report -e data/experiments/zero-shot \\
         --summary-report results_summary.md

  6. Rank by macro F1 for rounded_avg strategy:
     python -m src.report -e data/experiments/zero-shot \\
         --rank macro_f1 \\
         --strategy rounded_avg \\
         --top 10

  7. Manual predictions mode (auto-detect):
     python -m src.report -e data/experiments/prompting \\
         --labels-dir data/experiments/zero-shot/ml-test-data \\
         --rank accuracy \\
         -v

  8. Manual predictions mode (explicit):
     python -m src.report -e data/experiments/prompting \\
         --mode manual \\
         --labels-dir /path/to/ground-truth-csvs \\
         --rank adjacent_accuracy
""",
    )

    parser.add_argument(
        "-e", "--experiment-dir", required=True, help="Path to experiment directory"
    )

    parser.add_argument(
        "--mode",
        choices=["auto", "standard", "manual"],
        default="auto",
        help="Results structure mode: 'auto' (detect), 'standard' (evaluation_report.md), 'manual' (prediction JSON files)",
    )

    parser.add_argument(
        "--labels-dir",
        help="(Manual mode) Directory containing ground truth CSV files",
    )

    parser.add_argument(
        "--id-column",
        default="writing_id",
        help="(Manual mode) Column name for sample IDs in CSV (default: writing_id)",
    )

    parser.add_argument(
        "--label-column",
        default="cefr_level",
        help="(Manual mode) Column name for labels in CSV (default: cefr_level)",
    )

    parser.add_argument(
        "--rank",
        choices=["accuracy", "adjacent_accuracy", "macro_f1", "weighted_f1"],
        help="Rank models by criterion",
    )

    parser.add_argument(
        "--dataset", help="Filter by specific dataset (e.g., norm-CELVA-SP)"
    )

    parser.add_argument(
        "--strategy",
        choices=["argmax", "rounded_avg"],
        help="Filter by prediction strategy",
    )

    parser.add_argument("--top", type=int, help="Show only top N results")

    parser.add_argument(
        "--summary-report",
        help="Generate comprehensive summary report (markdown file path)",
    )

    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Don't show TF-IDF/classifier config in ranking table",
    )

    parser.add_argument(
        "--no-group",
        action="store_true",
        help="Don't group rankings by dataset (show flat ranking instead)",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)

    if not experiment_dir.exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        sys.exit(1)

    # Collect all metrics
    print(f"Analyzing results in: {experiment_dir}")

    labels_dir_path = Path(args.labels_dir) if args.labels_dir else None

    metrics_list = collect_all_metrics(
        experiment_dir=experiment_dir,
        mode=args.mode,
        labels_dir=labels_dir_path,
        id_column=args.id_column,
        label_column=args.label_column,
        verbose=args.verbose,
    )

    if not metrics_list:
        print("No metrics found.")
        sys.exit(0)

    # Generate summary report
    if args.summary_report:
        output_path = Path(args.summary_report)
        generate_summary_report(experiment_dir, metrics_list, output_path)

    # Rank models
    if args.rank:
        # Use grouped display by default, unless --dataset filter is used or --no-group is specified
        use_grouped = not args.dataset and not args.no_group

        if use_grouped:
            # Group by dataset (default behavior)
            print_ranking_grouped_by_dataset(
                metrics_list,
                criterion=args.rank,
                strategy_filter=args.strategy,
                top_n=args.top,
                show_config=not args.no_config,
            )
        else:
            # Flat ranking (when filtering by specific dataset or --no-group)
            ranked = rank_models(
                metrics_list,
                criterion=args.rank,
                dataset_filter=args.dataset,
                strategy_filter=args.strategy,
            )

            print_ranking_table(
                ranked,
                criterion=args.rank,
                top_n=args.top,
                show_config=not args.no_config,
            )

    # If neither ranking nor summary requested, show brief overview
    if not args.rank and not args.summary_report:
        print("\nQuick Overview:")
        print(f"  Total evaluations: {len(metrics_list)}")
        print(f"  Unique models: {len(set(m.model_name for m in metrics_list))}")
        print(f"  Datasets: {len(set(m.dataset_name for m in metrics_list))}")
        print("\nUse --rank or --summary-report to see detailed results.")
        print("Run with -h for usage examples.")


if __name__ == "__main__":
    main()
