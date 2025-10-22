"""
Dry Run Mode for CEFR Classification Pipeline

This module creates dummy empty files to preview the directory structure
and files that would be created by the pipeline, without running the actual
training, feature extraction, or prediction steps.
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

from src.config import GlobalConfig, ExperimentConfig, TfidfConfig


def create_dummy_file(file_path: Path, description: str = "", verbose: bool = True):
    """
    Create a dummy empty file with optional description.

    Args:
        file_path: Path to the file to create
        description: Optional description to add to the file
        verbose: Print creation messages
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.suffix == ".json":
        # Create minimal JSON structure
        dummy_content = {
            "_dry_run": True,
            "_description": description or f"Dummy file for {file_path.name}",
            "_note": "This is a dry run dummy file. Run the actual pipeline to generate real data."
        }
        with open(file_path, "w") as f:
            json.dump(dummy_content, f, indent=2)
    elif file_path.suffix == ".md":
        # Create minimal markdown
        content = f"# Dry Run Output\n\n{description}\n\nThis is a dummy file created by dry run mode.\n"
        with open(file_path, "w") as f:
            f.write(content)
    elif file_path.suffix in [".csv", ".pkl", ".txt"]:
        # Create empty file
        file_path.touch()
    else:
        # Unknown file type, create empty
        file_path.touch()

    if verbose:
        print(f"  Created: {file_path}")


def dry_run_step1_tfidf(
    config: GlobalConfig,
    tfidf_configs: Optional[List[dict]] = None,
    verbose: bool = True
) -> List[Path]:
    """
    Dry run for Step 1: Train TF-IDF Vectorizer(s).

    Args:
        config: GlobalConfig containing configuration
        tfidf_configs: List of TF-IDF config dicts
        verbose: Print detailed output

    Returns:
        List of created model directories
    """
    if tfidf_configs is None:
        tfidf_configs = [{"max_features": config.tfidf_config.max_features}]

    if verbose:
        print("\n" + "=" * 70)
        print("DRY RUN - STEP 1: Train TF-IDF Vectorizer(s)")
        print("=" * 70)

    created_dirs = []

    for i, tfidf_conf_dict in enumerate(tfidf_configs, 1):
        # Create TF-IDF config
        tfidf_config = TfidfConfig(
            max_features=tfidf_conf_dict.get("max_features", config.tfidf_config.max_features),
            ngram_range=tfidf_conf_dict.get("ngram_range", config.tfidf_config.ngram_range),
            min_df=tfidf_conf_dict.get("min_df", config.tfidf_config.min_df),
            max_df=tfidf_conf_dict.get("max_df", config.tfidf_config.max_df),
            sublinear_tf=tfidf_conf_dict.get("sublinear_tf", config.tfidf_config.sublinear_tf),
        )

        # Get output directory
        output_dir = Path(config.experiment_config.get_tfidf_model_dir(tfidf_config))

        if verbose:
            print(f"\nConfig {i}/{len(tfidf_configs)}:")
            print(f"  Hash: {tfidf_config.get_hash()}")
            print(f"  max_features: {tfidf_config.max_features}")
            print(f"  Directory: {output_dir}")

        # Create dummy files
        create_dummy_file(
            output_dir / "tfidf_model.pkl",
            f"TF-IDF model with {tfidf_config.max_features} features",
            verbose
        )
        create_dummy_file(
            output_dir / "config.json",
            f"TF-IDF configuration (hash: {tfidf_config.get_hash()})",
            verbose
        )

        created_dirs.append(output_dir)

    return created_dirs


def dry_run_step2_features(
    config: GlobalConfig,
    tfidf_configs: Optional[List[dict]] = None,
    data_source: str = "both",
    verbose: bool = True
) -> List[Path]:
    """
    Dry run for Step 2: Extract Features.

    Args:
        config: GlobalConfig containing configuration
        tfidf_configs: List of TF-IDF config dicts
        data_source: Data source ("test", "training", or "both")
        verbose: Print detailed output

    Returns:
        List of created feature directories
    """
    if tfidf_configs is None:
        tfidf_configs = [{"max_features": config.tfidf_config.max_features}]

    if verbose:
        print("\n" + "=" * 70)
        print("DRY RUN - STEP 2: Extract Features")
        print("=" * 70)

    created_dirs = []

    # Determine which sources to process
    sources_to_process = []
    if data_source == "both":
        sources_to_process = ["test", "training"]
    else:
        sources_to_process = [data_source]

    for i, tfidf_conf_dict in enumerate(tfidf_configs, 1):
        # Create TF-IDF config
        tfidf_config = TfidfConfig(
            max_features=tfidf_conf_dict.get("max_features", config.tfidf_config.max_features),
            ngram_range=tfidf_conf_dict.get("ngram_range", config.tfidf_config.ngram_range),
            min_df=tfidf_conf_dict.get("min_df", config.tfidf_config.min_df),
            max_df=tfidf_conf_dict.get("max_df", config.tfidf_config.max_df),
            sublinear_tf=tfidf_conf_dict.get("sublinear_tf", config.tfidf_config.sublinear_tf),
        )

        config_hash = tfidf_config.get_hash()
        feature_type = "tfidf"

        if verbose:
            print(f"\nTF-IDF Config {i}/{len(tfidf_configs)}:")
            print(f"  Hash: {config_hash}")

        for source in sources_to_process:
            if source == "test":
                data_dir = Path(config.experiment_config.ml_test_dir)
            else:
                data_dir = Path(config.experiment_config.ml_training_dir)

            if not data_dir.exists():
                if verbose:
                    print(f"  ⚠ Skipping {source}: directory not found: {data_dir}")
                continue

            # Find CSV files
            data_files = sorted(data_dir.glob("*.csv"))
            if not data_files:
                if verbose:
                    print(f"  ⚠ No {source} CSV files found in {data_dir}")
                continue

            if verbose:
                print(f"\n  {source.capitalize()} data ({len(data_files)} files):")

            for data_file in data_files:
                dataset_name = data_file.stem
                features_base_dir = Path(config.experiment_config.get_features_dir(config_hash, feature_type))
                features_dir = features_base_dir / dataset_name

                if verbose:
                    print(f"\n    Dataset: {dataset_name}")
                    print(f"    Directory: {features_dir}")

                # Create dummy feature files
                create_dummy_file(
                    features_dir / "features_dense.csv",
                    f"Dense feature matrix for {dataset_name}",
                    verbose
                )
                create_dummy_file(
                    features_dir / "feature_names.csv",
                    f"Feature names for {dataset_name}",
                    verbose
                )
                create_dummy_file(
                    features_dir / "config.json",
                    f"Feature extraction config for {dataset_name}",
                    verbose
                )

                created_dirs.append(features_dir)

    return created_dirs


def dry_run_step3_classifiers(
    config: GlobalConfig,
    tfidf_configs: Optional[List[dict]] = None,
    classifiers: Optional[List[str]] = None,
    verbose: bool = True
) -> List[Path]:
    """
    Dry run for Step 3: Train Classifiers.

    Args:
        config: GlobalConfig containing configuration
        tfidf_configs: List of TF-IDF config dicts
        classifiers: List of classifier types
        verbose: Print detailed output

    Returns:
        List of created classifier directories
    """
    if tfidf_configs is None:
        tfidf_configs = [{"max_features": config.tfidf_config.max_features}]

    if classifiers is None:
        classifiers = [config.classifier_config.classifier_type]

    if verbose:
        print("\n" + "=" * 70)
        print("DRY RUN - STEP 3: Train Classifiers")
        print("=" * 70)

    created_dirs = []

    # Find training datasets
    ml_training_dir = Path(config.experiment_config.ml_training_dir)
    if not ml_training_dir.exists():
        if verbose:
            print(f"  ⚠ Training directory not found: {ml_training_dir}")
        return created_dirs

    training_files = sorted(ml_training_dir.glob("*.csv"))
    if not training_files:
        if verbose:
            print(f"  ⚠ No training CSV files found in {ml_training_dir}")
        return created_dirs

    for tfidf_idx, tfidf_conf_dict in enumerate(tfidf_configs, 1):
        # Create TF-IDF config
        tfidf_config = TfidfConfig(
            max_features=tfidf_conf_dict.get("max_features", config.tfidf_config.max_features),
            ngram_range=tfidf_conf_dict.get("ngram_range", config.tfidf_config.ngram_range),
            min_df=tfidf_conf_dict.get("min_df", config.tfidf_config.min_df),
            max_df=tfidf_conf_dict.get("max_df", config.tfidf_config.max_df),
            sublinear_tf=tfidf_conf_dict.get("sublinear_tf", config.tfidf_config.sublinear_tf),
        )

        config_hash = tfidf_config.get_hash()
        feature_type = "tfidf"

        for clf_idx, classifier_type in enumerate(classifiers, 1):
            if verbose:
                print(f"\nTF-IDF {tfidf_idx}/{len(tfidf_configs)}, Classifier {clf_idx}/{len(classifiers)}:")
                print(f"  Config hash: {config_hash}")
                print(f"  Classifier: {classifier_type}")

            for training_file in training_files:
                dataset_name = training_file.stem

                # Build model name
                model_name = f"{dataset_name}_{classifier_type}_{config_hash}_{feature_type}"
                model_dir = Path(config.experiment_config.models_dir) / "classifiers" / model_name

                if verbose:
                    print(f"\n    Model: {model_name}")
                    print(f"    Directory: {model_dir}")

                # Create dummy classifier files
                create_dummy_file(
                    model_dir / "classifier.pkl",
                    f"Classifier model: {classifier_type} for {dataset_name}",
                    verbose
                )
                create_dummy_file(
                    model_dir / "label_encoder.pkl",
                    f"Label encoder for {dataset_name}",
                    verbose
                )
                create_dummy_file(
                    model_dir / "config.json",
                    f"Classifier config: {classifier_type}",
                    verbose
                )

                created_dirs.append(model_dir)

    return created_dirs


def dry_run_step4_predictions(
    config: GlobalConfig,
    tfidf_configs: Optional[List[dict]] = None,
    classifiers: Optional[List[str]] = None,
    verbose: bool = True
) -> List[Path]:
    """
    Dry run for Step 4: Make Predictions.

    Args:
        config: GlobalConfig containing configuration
        tfidf_configs: List of TF-IDF config dicts
        classifiers: List of classifier types
        verbose: Print detailed output

    Returns:
        List of created results directories
    """
    if tfidf_configs is None:
        tfidf_configs = [{"max_features": config.tfidf_config.max_features}]

    if classifiers is None:
        classifiers = [config.classifier_config.classifier_type]

    if verbose:
        print("\n" + "=" * 70)
        print("DRY RUN - STEP 4: Make Predictions")
        print("=" * 70)

    created_dirs = []

    # Find training and test datasets
    ml_training_dir = Path(config.experiment_config.ml_training_dir)
    ml_test_dir = Path(config.experiment_config.ml_test_dir)

    training_files = sorted(ml_training_dir.glob("*.csv")) if ml_training_dir.exists() else []
    test_files = sorted(ml_test_dir.glob("*.csv")) if ml_test_dir.exists() else []

    if not training_files:
        if verbose:
            print(f"  ⚠ No training CSV files found")
        return created_dirs

    for tfidf_idx, tfidf_conf_dict in enumerate(tfidf_configs, 1):
        # Create TF-IDF config
        tfidf_config = TfidfConfig(
            max_features=tfidf_conf_dict.get("max_features", config.tfidf_config.max_features),
            ngram_range=tfidf_conf_dict.get("ngram_range", config.tfidf_config.ngram_range),
            min_df=tfidf_conf_dict.get("min_df", config.tfidf_config.min_df),
            max_df=tfidf_conf_dict.get("max_df", config.tfidf_config.max_df),
            sublinear_tf=tfidf_conf_dict.get("sublinear_tf", config.tfidf_config.sublinear_tf),
        )

        config_hash = tfidf_config.get_hash()
        feature_type = "tfidf"

        for clf_idx, classifier_type in enumerate(classifiers, 1):
            for training_file in training_files:
                training_dataset_name = training_file.stem
                model_name = f"{training_dataset_name}_{classifier_type}_{config_hash}_{feature_type}"

                if verbose:
                    print(f"\nModel: {model_name}")

                # Predict on all test files
                for test_file in test_files:
                    test_dataset_name = test_file.stem
                    results_dir = Path(config.experiment_config.results_dir) / model_name / test_dataset_name

                    if verbose:
                        print(f"  Test dataset: {test_dataset_name}")
                        print(f"  Results directory: {results_dir}")

                    # Create dummy result files
                    create_dummy_file(
                        results_dir / "soft_predictions.json",
                        f"Soft predictions for {test_dataset_name}",
                        verbose
                    )
                    create_dummy_file(
                        results_dir / "argmax_predictions.json",
                        f"Argmax predictions for {test_dataset_name}",
                        verbose
                    )
                    create_dummy_file(
                        results_dir / "rounded_avg_predictions.json",
                        f"Rounded average predictions for {test_dataset_name}",
                        verbose
                    )
                    create_dummy_file(
                        results_dir / "evaluation_report.md",
                        f"Evaluation report for {test_dataset_name}",
                        verbose
                    )

                    created_dirs.append(results_dir)

    return created_dirs


def run_full_dry_run(
    config: GlobalConfig,
    steps: Optional[List[int]] = None,
    classifiers: Optional[List[str]] = None,
    tfidf_configs: Optional[List[dict]] = None,
    verbose: bool = True
) -> bool:
    """
    Run complete dry run of the pipeline.

    Args:
        config: GlobalConfig containing configuration
        steps: List of step numbers to run (default: all steps)
        classifiers: List of classifier types
        tfidf_configs: List of TF-IDF config dicts
        verbose: Print detailed output

    Returns:
        True if successful
    """
    all_steps = [1, 2, 3, 4]
    steps_to_run = steps if steps is not None else all_steps

    if classifiers is None:
        classifiers = [config.classifier_config.classifier_type]

    if tfidf_configs is None:
        tfidf_configs = [{"max_features": config.tfidf_config.max_features}]

    print("\n" + "=" * 70)
    print("DRY RUN MODE - CEFR CLASSIFICATION PIPELINE")
    print("=" * 70)
    print(f"Experiment: {Path(config.experiment_config.experiment_dir).name}")
    print(f"TF-IDF configs: {len(tfidf_configs)}")
    print(f"Classifiers: {classifiers}")
    print(f"Steps to run: {steps_to_run}")
    print("=" * 70)
    print("\nCreating dummy files to preview output structure...\n")

    try:
        # Step 1: Train TF-IDF
        if 1 in steps_to_run:
            dry_run_step1_tfidf(config, tfidf_configs, verbose)

        # Step 2: Extract Features
        if 2 in steps_to_run:
            dry_run_step2_features(config, tfidf_configs, data_source="both", verbose=verbose)

        # Step 3: Train Classifiers
        if 3 in steps_to_run:
            dry_run_step3_classifiers(config, tfidf_configs, classifiers, verbose)

        # Step 4: Predictions
        if 4 in steps_to_run:
            dry_run_step4_predictions(config, tfidf_configs, classifiers, verbose)

        # Summary
        if verbose:
            print("\n" + "=" * 70)
            print("DRY RUN COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            print("\nDummy files created. Directory structure:")
            print(f"\n  {config.experiment_config.experiment_dir}/")
            print(f"    feature-models/")
            print(f"      <hash>_tfidf/          # TF-IDF models")
            print(f"      classifiers/           # Trained classifiers")
            print(f"    features/")
            print(f"      <hash>_tfidf/          # Extracted features")
            print(f"        <dataset>/")
            print(f"    results/")
            print(f"      <model_name>/          # Prediction results")
            print(f"        <dataset>/")
            print("\nTo run the actual pipeline, remove the --dry-run flag.")
            print("=" * 70)

        return True

    except Exception as e:
        print(f"\n✗ Dry run failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for dry run."""
    # Note: Importing from pipeline causes heavy dependencies to load
    # So we create a minimal parser here instead
    parser = argparse.ArgumentParser(
        description="Dry Run Mode - Preview pipeline output structure without running actual computations",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Experiment configuration
    parser.add_argument(
        "-e", "--experiment-dir",
        required=True,
        help="Path to experiment directory"
    )
    parser.add_argument(
        "--classifier",
        default="xgboost",
        help="Classifier type (default: xgboost)"
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        help="Multiple classifiers to train"
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Max TF-IDF features (default: 5000)"
    )
    parser.add_argument(
        "--max-features-list",
        nargs="+",
        type=int,
        help="Multiple max_features values"
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        help="Pipeline steps to run (default: all)"
    )
    parser.add_argument(
        "--cefr-column",
        default="cefr_label",
        help="CEFR column name (default: cefr_label)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )

    return parser


def main():
    """Main entry point for dry run mode."""
    parser = create_parser()
    args = parser.parse_args()

    # Build configuration without complex imports
    try:
        from src.config import ExperimentConfig, TfidfConfig, ClassifierConfig, DataConfig, OutputConfig, GlobalConfig

        experiment_config = ExperimentConfig(experiment_dir=args.experiment_dir)
        tfidf_config = TfidfConfig(max_features=args.max_features)
        classifier_config = ClassifierConfig(classifier_type=args.classifier)
        data_config = DataConfig(cefr_column=args.cefr_column)
        output_config = OutputConfig(verbose=not args.quiet)

        config = GlobalConfig(
            experiment_config,
            tfidf_config,
            classifier_config,
            data_config,
            output_config
        )
    except Exception as e:
        parser.error(f"Configuration error: {e}")

    # Parse steps
    steps = None
    if args.steps:
        try:
            # Simple step parsing
            steps = []
            for step in args.steps:
                try:
                    steps.append(int(step))
                except ValueError:
                    # Map names to numbers
                    step_map = {
                        "tfidf": 1, "train-tfidf": 1,
                        "extract": 2, "features": 2, "extract-features": 2,
                        "train": 3, "classify": 3, "train-classifiers": 3,
                        "predict": 4, "inference": 4
                    }
                    if step.lower() in step_map:
                        steps.append(step_map[step.lower()])
                    else:
                        parser.error(f"Invalid step: {step}")
            steps = sorted(list(set(steps)))
        except Exception as e:
            parser.error(f"Step parsing error: {e}")

    # Determine classifiers
    classifiers = None
    if args.classifiers:
        classifiers = args.classifiers
    elif args.classifier:
        classifiers = [args.classifier]

    # Determine TF-IDF configurations
    tfidf_configs = None
    if args.max_features_list:
        tfidf_configs = [{"max_features": mf} for mf in args.max_features_list]

    if config.output_config.verbose:
        print("Dry Run Configuration:")
        if classifiers:
            print(f"  Classifiers: {classifiers}")
        if tfidf_configs:
            print(f"  TF-IDF configurations: {tfidf_configs}")
        if steps:
            print(f"  Steps: {steps}")
        print()

    # Run dry run
    success = run_full_dry_run(config, steps, classifiers, tfidf_configs, verbose=config.output_config.verbose)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
