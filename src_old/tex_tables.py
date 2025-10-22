"""
Generate LaTeX/Markdown Tables from Experiment Results

This script scans experiment results and generates publication-ready tables:
1. F1 Summary Table: F1 scores across all test datasets
2. Dataset-Specific Results Table: Detailed metrics for a specific dataset

Usage:
    python -m src.tex_tables -e data/experiments/zero-shot-2 -o tables/f1-summary.md --table-type f1-summary
    python -m src.tex_tables -e data/experiments/zero-shot-2 -o tables/celva-sp.md --table-type dataset-detail --dataset norm-CELVA-SP
"""

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# Import from existing report module
from src.report import ModelMetrics, collect_all_metrics


@dataclass
class TableRow:
    """A row in the output table."""

    model_display_name: str
    model_full_name: str
    features_description: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    is_header_row: bool = False


def extract_model_display_name(model_name: str) -> str:
    """
    Extract a readable model name from the full model name.

    Example:
        norm-EFCAMDAT-train_logistic_005ebc16_tfidf -> Logistic Reg. tf-idf
        norm-EFCAMDAT-train_xgboost_005ebc16_tfidf -> XGBoost tf-idf
    """
    # Extract classifier type
    classifier_map = {
        "logistic": "Logistic Reg.",
        "xgboost": "XGBoost",
        "randomforest": "Random Forest",
        "multinomialnb": "Naive Bayes",
        "svm": "SVM",
    }

    classifier_type = None
    for key in classifier_map:
        if key in model_name.lower():
            classifier_type = classifier_map[key]
            break

    # Extract feature type
    if "tfidf" in model_name.lower():
        features = "tf-idf"
    elif "perplexity" in model_name.lower() or "perp" in model_name.lower():
        features = "perplexities"
    else:
        features = "unknown"

    if classifier_type:
        return f"{classifier_type} {features}"
    else:
        # Fallback to abbreviated model name
        return model_name[:30]


def generate_f1_summary_table(
    metrics_list: List[ModelMetrics],
    datasets: List[str],
    strategy: str = "argmax",
    format: str = "markdown",
) -> str:
    """
    Generate F1 summary table showing F1 scores across datasets.

    Table structure:
    | Model & Setup | Dataset1 | Dataset2 | Dataset3 | Avg |

    Args:
        metrics_list: List of ModelMetrics from all models/datasets
        datasets: List of dataset names to include as columns
        strategy: "argmax" or "rounded_avg"
        format: "markdown" or "latex"

    Returns:
        Formatted table string
    """
    # Group metrics by model
    model_metrics = defaultdict(dict)

    for m in metrics_list:
        if m.strategy != strategy:
            continue

        # Use weighted F1 as the primary metric
        f1_score = m.weighted_f1 or m.macro_f1
        if f1_score is not None:
            model_metrics[m.model_name][m.dataset_name] = f1_score

    # Build table rows
    rows = []

    for model_name in sorted(model_metrics.keys()):
        display_name = extract_model_display_name(model_name)

        row_data = {"model": display_name, "model_full": model_name}

        # Add dataset scores
        scores = []
        for dataset in datasets:
            score = model_metrics[model_name].get(dataset)
            row_data[dataset] = score
            if score is not None:
                scores.append(score)

        # Calculate average
        if scores:
            row_data["avg"] = sum(scores) / len(scores)
        else:
            row_data["avg"] = None

        rows.append(row_data)

    # Sort rows by average F1 (descending)
    rows.sort(key=lambda x: x.get("avg") or 0, reverse=True)

    # Format output
    if format == "markdown":
        return _format_f1_table_markdown(rows, datasets)
    elif format == "latex":
        return _format_f1_table_latex(rows, datasets)
    else:
        raise ValueError(f"Unknown format: {format}")


def _format_f1_table_markdown(rows: List[Dict], datasets: List[str]) -> str:
    """Format F1 summary table as markdown."""
    lines = []

    # Header
    dataset_display_names = [d.replace("norm-", "").replace("-", " ") for d in datasets]
    header = "| Model & Setup | " + " | ".join(dataset_display_names) + " | Avg |"
    separator = "|" + "|".join(["---"] * (len(datasets) + 2)) + "|"

    lines.append(header)
    lines.append(separator)

    # Rows
    for row in rows:
        values = [row["model"]]

        for dataset in datasets:
            score = row.get(dataset)
            if score is not None:
                values.append(f"{score:.3f}")
            else:
                values.append("N/A")

        # Average
        avg = row.get("avg")
        if avg is not None:
            values.append(f"**{avg:.3f}**")
        else:
            values.append("N/A")

        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def _format_f1_table_latex(rows: List[Dict], datasets: List[str]) -> str:
    """Format F1 summary table as LaTeX."""
    lines = []

    # Table header
    lines.append("\\begin{table*}[ht]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\caption{Results across test datasets (F1 scores)}")

    # Column specification
    col_spec = "l " + "r " * (len(datasets) + 1)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")

    # Header row
    dataset_display_names = [
        d.replace("norm-", "").upper() + "$\\uparrow$" for d in datasets
    ]
    header = (
        "Model \\& Setup & "
        + " & ".join(dataset_display_names)
        + " & Avg$\\uparrow$ \\\\"
    )
    lines.append(header)
    lines.append("\\hline")

    # Data rows
    for row in rows:
        values = [row["model"]]

        for dataset in datasets:
            score = row.get(dataset)
            if score is not None:
                values.append(f"{score:.3f}")
            else:
                values.append("N/A")

        # Average
        avg = row.get("avg")
        if avg is not None:
            values.append(f"\\textbf{{{avg:.3f}}}")
        else:
            values.append("N/A")

        lines.append(" & ".join(values) + " \\\\")

    # Table footer
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\label{tab:f1_summary}")
    lines.append("\\end{table*}")

    return "\n".join(lines)


def generate_dataset_detail_table(
    metrics_list: List[ModelMetrics],
    dataset_name: str,
    strategy: str = "argmax",
    format: str = "markdown",
) -> str:
    """
    Generate detailed results table for a specific dataset.

    Table structure:
    | Model | Features | Samples | Accuracy | Adj Acc | Macro F1 | Weighted F1 |

    Args:
        metrics_list: List of ModelMetrics
        dataset_name: Dataset to filter for
        strategy: "argmax" or "rounded_avg"
        format: "markdown" or "latex"

    Returns:
        Formatted table string
    """
    # Filter metrics for this dataset and strategy
    dataset_metrics = [
        m
        for m in metrics_list
        if m.dataset_name == dataset_name and m.strategy == strategy
    ]

    if not dataset_metrics:
        return f"No results found for dataset: {dataset_name}"

    # Build table rows
    rows = []
    for m in dataset_metrics:
        display_name = extract_model_display_name(m.model_name)

        # Extract features description from TF-IDF config
        if m.tfidf_readable_name:
            features = m.tfidf_readable_name
        elif m.tfidf_max_features:
            features = f"tf-idf ({m.tfidf_max_features})"
        else:
            features = "tf-idf"

        row = {
            "model": display_name,
            "model_full": m.model_name,
            "features": features,
            "samples": m.n_samples or 0,
            "accuracy": m.accuracy,
            "adjacent_accuracy": m.adjacent_accuracy,
            "macro_f1": m.macro_f1,
            "weighted_f1": m.weighted_f1,
        }
        rows.append(row)

    # Sort by weighted F1 (or macro F1 if weighted not available)
    rows.sort(
        key=lambda x: x.get("weighted_f1") or x.get("macro_f1") or 0, reverse=True
    )

    # Format output
    if format == "markdown":
        return _format_detail_table_markdown(rows, dataset_name)
    elif format == "latex":
        return _format_detail_table_latex(rows, dataset_name)
    else:
        raise ValueError(f"Unknown format: {format}")


def _format_detail_table_markdown(rows: List[Dict], dataset_name: str) -> str:
    """Format dataset detail table as markdown."""
    lines = []

    # Title
    lines.append(f"## Results for {dataset_name}")
    lines.append("")

    # Header
    header = (
        "| Model | Features | Samples | Accuracy | Adj Acc | Macro F1 | Weighted F1 |"
    )
    separator = "|" + "|".join(["---"] * 7) + "|"

    lines.append(header)
    lines.append(separator)

    # Rows
    for row in rows:
        values = [
            row["model"],
            row["features"],
            str(row["samples"]),
            f"{row['accuracy']:.3f}" if row["accuracy"] is not None else "N/A",
            (
                f"{row['adjacent_accuracy']:.3f}"
                if row["adjacent_accuracy"] is not None
                else "N/A"
            ),
            f"{row['macro_f1']:.3f}" if row["macro_f1"] is not None else "N/A",
            (
                f"**{row['weighted_f1']:.3f}**"
                if row["weighted_f1"] is not None
                else "N/A"
            ),
        ]

        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def _format_detail_table_latex(rows: List[Dict], dataset_name: str) -> str:
    """Format dataset detail table as LaTeX."""
    lines = []

    # Table header
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4.5pt}")
    lines.append("\\renewcommand{\\arraystretch}{1}")
    lines.append("\\begin{tabular}{l l r r r r r}")
    lines.append(f"\\multicolumn{{7}}{{c}}{{Results on {dataset_name} Dataset}} \\\\")
    lines.append("\\hline")

    # Header row
    header = "Model & Features & Samples & Accuracy$\\uparrow$ & Adj Acc$\\uparrow$ & Macro F1$\\uparrow$ & Weighted F1$\\uparrow$ \\\\"
    lines.append(header)
    lines.append("\\hline")

    # Data rows
    for row in rows:
        values = [
            row["model"],
            row["features"],
            str(row["samples"]),
            f"{row['accuracy']:.3f}" if row["accuracy"] is not None else "N/A",
            (
                f"{row['adjacent_accuracy']:.3f}"
                if row["adjacent_accuracy"] is not None
                else "N/A"
            ),
            f"{row['macro_f1']:.3f}" if row["macro_f1"] is not None else "N/A",
            (
                f"\\textbf{{{row['weighted_f1']:.3f}}}"
                if row["weighted_f1"] is not None
                else "N/A"
            ),
        ]

        lines.append(" & ".join(values) + " \\\\")

    # Table footer
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{Classification results on {dataset_name}}}")
    lines.append(f"\\label{{tab:{dataset_name.lower().replace('-', '_')}_results}}")
    lines.append("\\end{table*}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX/Markdown tables from experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  1. Generate F1 summary table (markdown):
     python -m src.tex_tables -e data/experiments/zero-shot-2 \\
         -o tables/f1-summary.md \\
         --table-type f1-summary

  2. Generate F1 summary table (LaTeX):
     python -m src.tex_tables -e data/experiments/zero-shot-2 \\
         -o tables/f1-summary.tex \\
         --table-type f1-summary \\
         --format latex

  3. Generate dataset-specific detail table:
     python -m src.tex_tables -e data/experiments/zero-shot-2 \\
         -o tables/celva-sp-results.md \\
         --table-type dataset-detail \\
         --dataset norm-CELVA-SP

  4. Use rounded_avg strategy instead of argmax:
     python -m src.tex_tables -e data/experiments/zero-shot-2 \\
         -o tables/f1-summary.md \\
         --table-type f1-summary \\
         --strategy rounded_avg

  5. Print to stdout instead of file:
     python -m src.tex_tables -e data/experiments/zero-shot-2 \\
         --table-type f1-summary

  6. Specify custom datasets for F1 summary:
     python -m src.tex_tables -e data/experiments/zero-shot-2 \\
         --table-type f1-summary \\
         --datasets norm-CELVA-SP norm-KUPA-KEYS norm-EFCAMDAT-test
""",
    )

    parser.add_argument(
        "-e", "--experiment-dir", required=True, help="Path to experiment directory"
    )

    parser.add_argument(
        "-o", "--output", help="Output file path (prints to stdout if not specified)"
    )

    parser.add_argument(
        "--table-type",
        required=True,
        choices=["f1-summary", "dataset-detail"],
        help="Type of table to generate",
    )

    parser.add_argument(
        "--dataset", help="Dataset name (required for dataset-detail table type)"
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        help="List of datasets to include in F1 summary (auto-detected if not specified)",
    )

    parser.add_argument(
        "--strategy",
        choices=["argmax", "rounded_avg"],
        default="argmax",
        help="Prediction strategy to use (default: argmax)",
    )

    parser.add_argument(
        "--format",
        choices=["markdown", "latex"],
        default="markdown",
        help="Output format (default: markdown)",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Validate arguments
    if args.table_type == "dataset-detail" and not args.dataset:
        parser.error("--dataset is required for dataset-detail table type")

    # Load experiment results
    experiment_dir = Path(args.experiment_dir)

    if not experiment_dir.exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        sys.exit(1)

    if args.verbose:
        print(f"Loading results from: {experiment_dir}")

    # Collect metrics
    metrics_list = collect_all_metrics(experiment_dir, verbose=args.verbose)

    if not metrics_list:
        print("Error: No metrics found in experiment directory")
        sys.exit(1)

    if args.verbose:
        print(f"Found {len(metrics_list)} metric records")

    # Generate table
    if args.table_type == "f1-summary":
        # Auto-detect datasets if not specified
        if args.datasets:
            datasets = args.datasets
        else:
            datasets = sorted(set(m.dataset_name for m in metrics_list))

        if args.verbose:
            print(f"Generating F1 summary for datasets: {datasets}")

        table_content = generate_f1_summary_table(
            metrics_list, datasets=datasets, strategy=args.strategy, format=args.format
        )

    elif args.table_type == "dataset-detail":
        if args.verbose:
            print(f"Generating detail table for dataset: {args.dataset}")

        table_content = generate_dataset_detail_table(
            metrics_list,
            dataset_name=args.dataset,
            strategy=args.strategy,
            format=args.format,
        )

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(table_content)

        print(f"Table saved to: {output_path}")
    else:
        print(table_content)


if __name__ == "__main__":
    main()
