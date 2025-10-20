"""
Experiment Folder Structure Definition and Validation

This module defines the canonical experiment folder structure used across all scripts
in the CEFR classification pipeline. All scripts must reference this structure in their
docstrings and validate against it.

CANONICAL EXPERIMENT FOLDER STRUCTURE:
======================================

experiment_dir/
├── ml-training-data/           # Training data CSVs
│   └── *.csv                   # Training datasets (e.g., norm-EFCAMDAT-train.csv)
├── ml-test-data/               # Test data CSVs
│   └── *.csv                   # Test datasets (e.g., norm-CELVA-SP.csv, norm-EFCAMDAT-test.csv)
├── features-training-data/     # Training features (optional, for holdout validation)
│   └── *.csv                   # Training feature CSVs
├── feature-models/             # TF-IDF models and classifiers
│   ├── {config_hash}_tfidf/    # TF-IDF model directories (e.g., 005ebc16_tfidf)
│   │   ├── config.json         # TF-IDF configuration
│   │   └── tfidf_model.pkl     # Trained TF-IDF vectorizer
│   └── classifiers/            # Trained classifier models
│       └── {model_name}/       # Classifier directory (e.g., norm-EFCAMDAT-train_logistic_005ebc16_tfidf)
│           ├── classifier.pkl  # Trained classifier
│           ├── config.json     # Classifier configuration
│           └── label_encoder.pkl  # Label encoder (if used)
├── features/                   # Extracted TF-IDF features
│   └── {config_hash}_tfidf/    # Feature directory per TF-IDF config (e.g., 005ebc16_tfidf)
│       └── {dataset_name}/     # Dataset subdirectories (e.g., norm-CELVA-SP)
│           ├── config.json     # Feature extraction config
│           ├── feature_names.csv  # Feature names
│           └── features_dense.csv # Dense feature matrix
└── results/                    # Prediction results
    └── {model_name}/           # Results per model
        └── {dataset_name}/     # Results per dataset
            ├── soft_predictions.json      # Probability predictions
            ├── argmax_predictions.json    # Argmax predictions
            ├── rounded_avg_predictions.json  # Rounded average predictions
            └── evaluation_report.md       # Evaluation metrics

NAMING CONVENTIONS:
==================
- config_hash: 8-character hexadecimal hash (e.g., '005ebc16')
- feature_type: 'tfidf' or 'tfidf_grouped'
- model_name: {training_dataset}_{classifier_type}_{config_hash}_{feature_type}
  Examples:
  - norm-EFCAMDAT-train_logistic_005ebc16_tfidf
  - norm-EFCAMDAT-train_xgboost_10b21d1b_tfidf_grouped
- dataset_name: Normalized dataset name (e.g., 'norm-CELVA-SP', 'norm-EFCAMDAT-test')

SCRIPT RESPONSIBILITIES:
========================
1. train_tfidf.py / train_tfidf_groupby.py
   - Reads: ml-training-data/
   - Writes: feature-models/{config_hash}_tfidf/

2. extract_features.py
   - Reads: ml-training-data/, ml-test-data/, feature-models/{config_hash}_tfidf/
   - Writes: features/{config_hash}_tfidf/{dataset_name}/

3. train_classifiers.py / train_classifiers_with_ho.py
   - Reads: features/{config_hash}_tfidf/{dataset_name}/
   - Writes: feature-models/classifiers/{model_name}/

4. predict.py
   - Reads: features/{config_hash}_tfidf/{dataset_name}/, feature-models/classifiers/{model_name}/
   - Writes: results/{model_name}/{dataset_name}/

5. pipeline.py
   - Orchestrates all scripts above

6. report.py
   - Reads: results/
   - Generates: Summary reports
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


# Canonical structure definition
EXPERIMENT_STRUCTURE = {
    "ml-training-data": {
        "type": "directory",
        "required": True,
        "description": "Training data CSVs",
        "expected_files": ["*.csv"]
    },
    "ml-test-data": {
        "type": "directory",
        "required": True,
        "description": "Test data CSVs",
        "expected_files": ["*.csv"]
    },
    "features-training-data": {
        "type": "directory",
        "required": False,
        "description": "Training features for holdout validation",
        "expected_files": ["*.csv"]
    },
    "feature-models": {
        "type": "directory",
        "required": True,
        "description": "TF-IDF models and classifiers",
        "subdirs": {
            "{config_hash}_tfidf": {
                "type": "directory",
                "pattern": r"^[0-9a-f]{8}_tfidf(?:_grouped)?$",
                "expected_files": ["config.json", "tfidf_model.pkl"]
            },
            "classifiers": {
                "type": "directory",
                "expected_subdirs": {
                    "{model_name}": {
                        "pattern": r"^.+_[0-9a-f]{8}_tfidf(?:_grouped)?$",
                        "expected_files": ["classifier.pkl", "config.json", "label_encoder.pkl"]
                    }
                }
            }
        }
    },
    "features": {
        "type": "directory",
        "required": True,
        "description": "Extracted TF-IDF features",
        "subdirs": {
            "{config_hash}_tfidf": {
                "type": "directory",
                "pattern": r"^[0-9a-f]{8}_tfidf(?:_grouped)?$",
                "expected_subdirs": {
                    "{dataset_name}": {
                        "expected_files": ["config.json", "feature_names.csv", "features_dense.csv"]
                    }
                }
            }
        }
    },
    "results": {
        "type": "directory",
        "required": False,
        "description": "Prediction results",
        "subdirs": {
            "{model_name}": {
                "type": "directory",
                "expected_subdirs": {
                    "{dataset_name}": {
                        "expected_files": [
                            "soft_predictions.json",
                            "argmax_predictions.json",
                            "rounded_avg_predictions.json",
                            "evaluation_report.md"
                        ]
                    }
                }
            }
        }
    }
}


def validate_experiment_structure(
    experiment_dir: str,
    required_dirs: Optional[List[str]] = None,
    verbose: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate that an experiment directory follows the canonical structure.

    Args:
        experiment_dir: Path to experiment directory
        required_dirs: List of required directories to check. If None, checks all required dirs.
                      Options: ['ml-training-data', 'ml-test-data', 'features-training-data',
                               'feature-models', 'features', 'results']
        verbose: Print validation messages

    Returns:
        Tuple of (is_valid, error_messages)
    """
    exp_path = Path(experiment_dir)
    errors = []
    warnings = []

    if not exp_path.exists():
        errors.append(f"Experiment directory does not exist: {experiment_dir}")
        return False, errors

    if not exp_path.is_dir():
        errors.append(f"Experiment path is not a directory: {experiment_dir}")
        return False, errors

    # Determine which directories to check
    if required_dirs is None:
        dirs_to_check = {k: v for k, v in EXPERIMENT_STRUCTURE.items() if v.get("required", True)}
    else:
        dirs_to_check = {k: v for k, v in EXPERIMENT_STRUCTURE.items() if k in required_dirs}

    # Check required directories
    for dir_name, dir_spec in dirs_to_check.items():
        dir_path = exp_path / dir_name

        if not dir_path.exists():
            if dir_spec.get("required", True):
                errors.append(f"Required directory missing: {dir_name}/")
            else:
                warnings.append(f"Optional directory missing: {dir_name}/")
        elif not dir_path.is_dir():
            errors.append(f"Path exists but is not a directory: {dir_name}")

    # Print results if verbose
    if verbose:
        if not errors and not warnings:
            print(f"✓ Experiment structure validated: {experiment_dir}")
        else:
            print(f"Experiment structure validation for: {experiment_dir}")
            if errors:
                print(f"\n✗ Errors ({len(errors)}):")
                for error in errors:
                    print(f"  - {error}")
            if warnings:
                print(f"\n⚠ Warnings ({len(warnings)}):")
                for warning in warnings:
                    print(f"  - {warning}")

    return len(errors) == 0, errors


def get_structure_docstring() -> str:
    """
    Get the canonical structure documentation as a formatted string.
    Use this in script docstrings to ensure consistency.

    Returns:
        Formatted structure documentation string
    """
    return """
Expected Experiment Folder Structure:
=====================================

experiment_dir/
├── ml-training-data/           # Training data CSVs
│   └── *.csv                   # Training datasets
├── ml-test-data/               # Test data CSVs
│   └── *.csv                   # Test datasets
├── features-training-data/     # Training features (optional)
│   └── *.csv                   # Training feature CSVs
├── feature-models/             # TF-IDF models and classifiers
│   ├── {config_hash}_tfidf/    # TF-IDF model directories
│   │   ├── config.json         # TF-IDF configuration
│   │   └── tfidf_model.pkl     # Trained TF-IDF vectorizer
│   └── classifiers/            # Trained classifier models
│       └── {model_name}/       # Classifier directory
│           ├── classifier.pkl  # Trained classifier
│           ├── config.json     # Classifier configuration
│           └── label_encoder.pkl  # Label encoder
├── features/                   # Extracted TF-IDF features
│   └── {config_hash}_tfidf/    # Feature directory per TF-IDF config
│       └── {dataset_name}/     # Dataset subdirectories
│           ├── config.json     # Feature extraction config
│           ├── feature_names.csv  # Feature names
│           └── features_dense.csv # Dense feature matrix
└── results/                    # Prediction results
    └── {model_name}/           # Results per model
        └── {dataset_name}/     # Results per dataset
            ├── soft_predictions.json
            ├── argmax_predictions.json
            ├── rounded_avg_predictions.json
            └── evaluation_report.md

For detailed structure documentation, see: src/experiment_structure.py
"""


def create_experiment_structure(experiment_dir: str, create_subdirs: bool = False) -> None:
    """
    Create the basic experiment directory structure.

    Args:
        experiment_dir: Path to experiment directory
        create_subdirs: If True, creates example subdirectories
    """
    exp_path = Path(experiment_dir)
    exp_path.mkdir(parents=True, exist_ok=True)

    # Create main directories
    for dir_name, dir_spec in EXPERIMENT_STRUCTURE.items():
        dir_path = exp_path / dir_name
        if dir_spec.get("required", True) or create_subdirs:
            dir_path.mkdir(exist_ok=True)
            print(f"Created: {dir_name}/")

    # Create classifiers subdirectory
    classifiers_dir = exp_path / "feature-models" / "classifiers"
    classifiers_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created: feature-models/classifiers/")

    print(f"\n✓ Experiment structure created at: {experiment_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Validate: python -m src.experiment_structure validate <experiment_dir>")
        print("  Create:   python -m src.experiment_structure create <experiment_dir>")
        print("  Show:     python -m src.experiment_structure show")
        sys.exit(1)

    command = sys.argv[1]

    if command == "show":
        print(__doc__)
    elif command == "validate" and len(sys.argv) >= 3:
        experiment_dir = sys.argv[2]
        is_valid, errors = validate_experiment_structure(experiment_dir, verbose=True)
        sys.exit(0 if is_valid else 1)
    elif command == "create" and len(sys.argv) >= 3:
        experiment_dir = sys.argv[2]
        create_experiment_structure(experiment_dir)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
