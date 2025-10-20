"""
CEFR Classification Pipeline Runner

Orchestrates the complete CEFR classification pipeline with support for:
- Multiple TF-IDF configurations
- Multiple classifiers
- Batch processing of all training/test sets
- Pre-extracted features workflow
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from src.config import (
    ClassifierConfig,
    DataConfig,
    ExperimentConfig,
    GlobalConfig,
    OutputConfig,
    TfidfConfig,
)
from src.extract_features import extract_all_from_source
from src.predict import predict_all_feature_sets
from src.train_classifiers import train_all_classifiers
from src.train_tfidf import train_tfidf


def run_pipeline(  # noqa: C901
    config: GlobalConfig,
    steps: Optional[List[int]] = None,
    classifiers: Optional[List[str]] = None,
    tfidf_configs: Optional[List[dict]] = None,
    summarize: bool = False,
) -> bool:
    """
    Run the complete CEFR classification pipeline or specific steps.

    Args:
        config: GlobalConfig containing base configuration
        steps: List of step numbers to run (e.g., [1, 2, 3, 4]). If None, runs all steps.
        classifiers: List of classifier types to train (e.g., ["xgboost", "logistic"])
        tfidf_configs: List of TF-IDF config dicts (e.g., [{"max_features": 5000}, {"max_features": 10000}])

    Returns:
        True if successful, False otherwise
    """
    verbose = config.output_config.verbose
    all_steps = [1, 2, 3, 4]
    steps_to_run = steps if steps is not None else all_steps

    # Default to single classifier if not specified
    if classifiers is None:
        classifiers = [config.classifier_config.classifier_type]

    # Default to single TF-IDF config if not specified
    if tfidf_configs is None:
        tfidf_configs = [{"max_features": config.tfidf_config.max_features}]

    if verbose:
        print(f"\n{'='*70}")
        print("CEFR CLASSIFICATION PIPELINE")
        print(f"{'='*70}")
        print(f"Experiment: {Path(config.experiment_config.experiment_dir).name}")
        print(f"TF-IDF configs: {len(tfidf_configs)}")
        print(f"Classifiers: {classifiers}")
        print(f"Steps to run: {steps_to_run}")
        print(f"{'='*70}\n")

    try:
        # Step 1: Train TF-IDF Vectorizer(s) and Extract Training Features
        if 1 in steps_to_run:
            for i, tfidf_conf_dict in enumerate(tfidf_configs, 1):
                if verbose:
                    print("\n" + "=" * 70)
                    print(
                        f"STEP 1.{i}: Train TF-IDF Vectorizer (config {i}/{len(tfidf_configs)})"
                    )
                    print(
                        f"  max_features: {tfidf_conf_dict.get('max_features', config.tfidf_config.max_features)}"
                    )
                    print("=" * 70)

                # Create config with this TF-IDF configuration
                tfidf_config = TfidfConfig(
                    max_features=tfidf_conf_dict.get(
                        "max_features", config.tfidf_config.max_features
                    ),
                    ngram_range=tfidf_conf_dict.get(
                        "ngram_range", config.tfidf_config.ngram_range
                    ),
                    min_df=tfidf_conf_dict.get("min_df", config.tfidf_config.min_df),
                    max_df=tfidf_conf_dict.get("max_df", config.tfidf_config.max_df),
                    sublinear_tf=tfidf_conf_dict.get(
                        "sublinear_tf", config.tfidf_config.sublinear_tf
                    ),
                )

                step_config = GlobalConfig(
                    config.experiment_config,
                    tfidf_config,
                    config.classifier_config,
                    config.data_config,
                    config.output_config,
                )

                train_tfidf(step_config)

                if verbose:
                    print(f"\n✓ Step 1.{i} completed successfully")

        # Step 2: Extract Test Features for all TF-IDF configurations
        if 2 in steps_to_run:
            for i, tfidf_conf_dict in enumerate(tfidf_configs, 1):
                if verbose:
                    print("\n" + "=" * 70)
                    print(
                        f"STEP 2.{i}: Extract All Features (config {i}/{len(tfidf_configs)})"
                    )
                    print(
                        f"  max_features: {tfidf_conf_dict.get('max_features', config.tfidf_config.max_features)}"
                    )
                    print("=" * 70)

                # Create config with this TF-IDF configuration
                tfidf_config = TfidfConfig(
                    max_features=tfidf_conf_dict.get(
                        "max_features", config.tfidf_config.max_features
                    ),
                    ngram_range=tfidf_conf_dict.get(
                        "ngram_range", config.tfidf_config.ngram_range
                    ),
                    min_df=tfidf_conf_dict.get("min_df", config.tfidf_config.min_df),
                    max_df=tfidf_conf_dict.get("max_df", config.tfidf_config.max_df),
                    sublinear_tf=tfidf_conf_dict.get(
                        "sublinear_tf", config.tfidf_config.sublinear_tf
                    ),
                )

                step_config = GlobalConfig(
                    config.experiment_config,
                    tfidf_config,
                    config.classifier_config,
                    config.data_config,
                    config.output_config,
                )

                # Extract features from both training and test data
                extract_all_from_source(step_config, data_source="both")

                if verbose:
                    print(f"\n✓ Step 2.{i} completed successfully")

        # Step 3: Train ML Classifiers (all combinations)
        if 3 in steps_to_run:
            for tfidf_idx, tfidf_conf_dict in enumerate(tfidf_configs, 1):
                tfidf_max_feat = tfidf_conf_dict.get(
                    "max_features", config.tfidf_config.max_features
                )

                for clf_idx, classifier_type in enumerate(classifiers, 1):
                    if verbose:
                        print("\n" + "=" * 70)
                        print(f"STEP 3.{tfidf_idx}.{clf_idx}: Train Classifiers")
                        print(f"  TF-IDF: max_features={tfidf_max_feat}")
                        print(f"  Classifier: {classifier_type}")
                        print("=" * 70)

                    # Create config for this combination
                    tfidf_config = TfidfConfig(
                        max_features=tfidf_max_feat,
                        ngram_range=tfidf_conf_dict.get(
                            "ngram_range", config.tfidf_config.ngram_range
                        ),
                        min_df=tfidf_conf_dict.get(
                            "min_df", config.tfidf_config.min_df
                        ),
                        max_df=tfidf_conf_dict.get(
                            "max_df", config.tfidf_config.max_df
                        ),
                        sublinear_tf=tfidf_conf_dict.get(
                            "sublinear_tf", config.tfidf_config.sublinear_tf
                        ),
                    )

                    classifier_config = ClassifierConfig(
                        classifier_type=classifier_type,
                        logistic_max_iter=config.classifier_config.logistic_max_iter,
                        logistic_class_weight=config.classifier_config.logistic_class_weight,
                        rf_n_estimators=config.classifier_config.rf_n_estimators,
                        rf_class_weight=config.classifier_config.rf_class_weight,
                        svm_max_iter=config.classifier_config.svm_max_iter,
                        svm_class_weight=config.classifier_config.svm_class_weight,
                        xgb_n_estimators=config.classifier_config.xgb_n_estimators,
                        xgb_max_depth=config.classifier_config.xgb_max_depth,
                        xgb_learning_rate=config.classifier_config.xgb_learning_rate,
                        xgb_use_gpu=config.classifier_config.xgb_use_gpu,
                        xgb_tree_method=config.classifier_config.xgb_tree_method,
                        random_state=config.classifier_config.random_state,
                    )

                    step_config = GlobalConfig(
                        config.experiment_config,
                        tfidf_config,
                        classifier_config,
                        config.data_config,
                        config.output_config,
                    )

                    # Train on all feature directories in batch
                    train_all_classifiers(
                        step_config,
                        features_dir=None,  # Use default from config
                        labels_csv_dir=None,  # Use default from config
                    )

                    if verbose:
                        print(
                            f"\n✓ Step 3.{tfidf_idx}.{clf_idx} completed successfully"
                        )

        # Step 4: Make Predictions (all combinations)
        if 4 in steps_to_run:
            # Get all trained classifiers
            classifiers_dir = Path(config.experiment_config.models_dir) / "classifiers"
            if not classifiers_dir.exists():
                print(f"✗ No classifiers found in {classifiers_dir}")
                return False

            trained_models = sorted(
                [d.name for d in classifiers_dir.iterdir() if d.is_dir()]
            )

            if not trained_models:
                print(f"✗ No trained models found in {classifiers_dir}")
                return False

            if verbose:
                print("\n" + "=" * 70)
                print("STEP 4: Make Predictions")
                print(f"  Found {len(trained_models)} trained models")
                print("=" * 70)

            # Predict with each model on all test sets
            for model_idx, model_name in enumerate(trained_models, 1):
                if verbose:
                    print(
                        f"\n--- Model {model_idx}/{len(trained_models)}: {model_name} ---"
                    )

                # Load model config to get TF-IDF hash and determine features directory
                model_dir = classifiers_dir / model_name
                model_config_path = model_dir / "config.json"

                if model_config_path.exists():
                    with open(model_config_path, "r") as f:
                        model_config_data = json.load(f)
                    tfidf_hash = model_config_data.get("tfidf_hash")

                    if tfidf_hash:
                        # Use hashed features directory
                        features_dir = (
                            Path(config.experiment_config.features_output_dir)
                            / tfidf_hash
                        )
                        if verbose:
                            print(f"  Using features from TF-IDF config: {tfidf_hash}")
                            print(f"  Features directory: {features_dir}")
                    else:
                        # Backward compatibility: old models without hash
                        features_dir = Path(
                            config.experiment_config.features_output_dir
                        )
                        if verbose:
                            print(
                                "  Warning: Model config missing tfidf_hash, using default features directory"
                            )
                else:
                    # Fallback: use default features directory
                    features_dir = Path(config.experiment_config.features_output_dir)
                    if verbose:
                        print(
                            "  Warning: Model config not found, using default features directory"
                        )

                try:
                    predict_all_feature_sets(
                        config,
                        classifier_model_name=model_name,
                        features_dir=str(features_dir),
                        labels_csv_dir=str(Path(config.experiment_config.ml_test_dir)),
                    )
                except Exception as e:
                    print(f"✗ Error predicting with {model_name}: {e}")
                    if verbose:
                        import traceback

                        traceback.print_exc()

            if verbose:
                print("\n✓ Step 4 completed successfully")

        # Print summary
        if verbose:
            print("\n" + "=" * 70)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 70)

            # Summary of trained models
            classifiers_dir = Path(config.experiment_config.models_dir) / "classifiers"
            if classifiers_dir.exists():
                trained_models = sorted(
                    [d.name for d in classifiers_dir.iterdir() if d.is_dir()]
                )
                print(f"\nTrained models ({len(trained_models)}):")
                for model in trained_models:
                    print(f"  - {model}")

            # Summary of results
            results_dir = Path(config.experiment_config.results_dir)
            if results_dir.exists():
                result_dirs = sorted(
                    [d.name for d in results_dir.iterdir() if d.is_dir()]
                )
                print(f"\nResults saved ({len(result_dirs)} model directories):")
                for dataset in result_dirs:
                    print(f"  - {dataset}/")
                print(f"\nResults directory: {results_dir}")

        # Generate summary report if requested
        if summarize:
            if verbose:
                print("\n" + "=" * 70)
                print("GENERATING SUMMARY REPORT")
                print("=" * 70)

            try:
                from src.report import collect_all_metrics, generate_summary_report

                experiment_dir = Path(config.experiment_config.experiment_dir)
                metrics_list = collect_all_metrics(experiment_dir, verbose=verbose)

                if metrics_list:
                    summary_path = experiment_dir / "results_summary.md"
                    generate_summary_report(experiment_dir, metrics_list, summary_path)

                    if verbose:
                        print(f"\n✓ Summary report generated: {summary_path}")
                else:
                    if verbose:
                        print("\n⚠ No metrics found for summary report")

            except Exception as e:
                print(f"\n✗ Error generating summary report: {e}")
                if verbose:
                    import traceback

                    traceback.print_exc()

        return True

    except Exception as e:
        print(f"\n✗ Pipeline execution failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return False


def create_parser() -> argparse.ArgumentParser:
    """Create comprehensive argument parser for the pipeline."""
    parser = argparse.ArgumentParser(
        description="""
CEFR Classification Pipeline - Complete ML Training and Evaluation Workflow

This pipeline orchestrates the complete CEFR text classification workflow:
  1. Train TF-IDF vectorizer(s) on training data
  2. Extract features for all training and test sets
  3. Train ML classifier(s) on pre-extracted features
  4. Make predictions on all test sets with all trained models

Supports multiple TF-IDF configurations and multiple classifiers for
systematic experimentation and hyperparameter search.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
═══════════════════════════════════════════════════════════════════════════════
                              USAGE EXAMPLES
═══════════════════════════════════════════════════════════════════════════════

BASIC USAGE
───────────────────────────────────────────────────────────────────────────────

  1. Run full pipeline with default XGBoost classifier:

     python -m src.pipeline \\
         -e data/experiments/zero-shot \\
         --cefr-column cefr_level

  2. Run with specific classifier:

     python -m src.pipeline \\
         -e data/experiments/zero-shot \\
         --classifier logistic \\
         --cefr-column cefr_level

  3. Quiet mode (suppress verbose output):

     python -m src.pipeline \\
         -e data/experiments/zero-shot \\
         --classifier xgboost \\
         --cefr-column cefr_level \\
         -q

MULTIPLE CLASSIFIERS
───────────────────────────────────────────────────────────────────────────────

  4. Compare multiple classifiers on same features:

     python -m src.pipeline \\
         -e data/experiments/zero-shot \\
         --classifiers xgboost logistic randomforest \\
         --cefr-column cefr_level

     Result: 3 models trained and evaluated

  5. All classifiers comparison:

     python -m src.pipeline \\
         -e data/experiments/zero-shot \\
         --classifiers xgboost logistic randomforest svm multinomialnb \\
         --cefr-column cefr_level

     Result: 5 models trained for comprehensive comparison

MULTIPLE TF-IDF CONFIGURATIONS
───────────────────────────────────────────────────────────────────────────────

  6. Feature size experiments:

     python -m src.pipeline \\
         -e data/experiments/zero-shot \\
         --max-features-list 1000 5000 10000 \\
         --classifier xgboost \\
         --cefr-column cefr_level

     Result: 3 TF-IDF models + 3 classifiers
     Each TF-IDF config stored in unique hashed directory (no overwrites!)

  7. Fine-grained feature size search:

     python -m src.pipeline \\
         -e data/experiments/zero-shot \\
         --max-features-list 1000 2000 5000 10000 20000 \\
         --classifier xgboost \\
         --cefr-column cefr_level

     Result: 5 TF-IDF models + 5 classifiers

FULL GRID SEARCH (TF-IDF × CLASSIFIERS)
───────────────────────────────────────────────────────────────────────────────

  8. Small grid search (2 × 2 = 4 models):

     python -m src.pipeline \\
         -e data/experiments/zero-shot \\
         --max-features-list 1000 5000 \\
         --classifiers xgboost logistic \\
         --cefr-column cefr_level

  9. Medium grid search (3 × 3 = 9 models):

     python -m src.pipeline \\
         -e data/experiments/zero-shot \\
         --max-features-list 1000 5000 10000 \\
         --classifiers xgboost logistic randomforest \\
         --cefr-column cefr_level

  10. Large grid search (4 × 4 = 16 models):

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --max-features-list 1000 2000 5000 10000 \\
          --classifiers xgboost logistic randomforest svm \\
          --cefr-column cefr_level

XGBOOST CUSTOMIZATION
───────────────────────────────────────────────────────────────────────────────

  11. XGBoost with GPU acceleration:

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --classifier xgboost \\
          --xgb-use-gpu \\
          --cefr-column cefr_level

  12. XGBoost hyperparameter tuning:

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --classifier xgboost \\
          --xgb-n-estimators 200 \\
          --xgb-max-depth 8 \\
          --xgb-use-gpu \\
          --cefr-column cefr_level

  13. Production XGBoost model (optimized):

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --classifier xgboost \\
          --max-features 10000 \\
          --xgb-n-estimators 300 \\
          --xgb-max-depth 10 \\
          --xgb-use-gpu \\
          --cefr-column cefr_level

TF-IDF CUSTOMIZATION
───────────────────────────────────────────────────────────────────────────────

  14. Custom n-gram range:

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --ngram-min 1 \\
          --ngram-max 3 \\
          --classifier xgboost \\
          --cefr-column cefr_level

  15. Custom document frequency thresholds:

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --min-df 5 \\
          --max-df 0.8 \\
          --classifier xgboost \\
          --cefr-column cefr_level

  16. Combined TF-IDF customization:

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --max-features 10000 \\
          --ngram-min 1 \\
          --ngram-max 3 \\
          --min-df 3 \\
          --max-df 0.9 \\
          --classifier xgboost \\
          --cefr-column cefr_level

PARTIAL PIPELINE EXECUTION
───────────────────────────────────────────────────────────────────────────────

  17. Run only TF-IDF training and feature extraction:

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --steps 1 2 \\
          --cefr-column cefr_level

  18. Run only classifier training (features already extracted):

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --steps 3 \\
          --classifier xgboost \\
          --cefr-column cefr_level

  19. Run only predictions (models already trained):

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --steps 4 \\
          --cefr-column cefr_level

  20. Run training and predictions (skip feature extraction):

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --steps 3 4 \\
          --classifier xgboost \\
          --cefr-column cefr_level

DATA CONFIGURATION
───────────────────────────────────────────────────────────────────────────────

  21. Custom column names:

      python -m src.pipeline \\
          -e data/experiments/my-experiment \\
          --text-column essay \\
          --label-column level \\
          --cefr-column cefr \\
          --classifier xgboost

  22. Different CEFR column:

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --cefr-column cefr_level \\
          --classifier xgboost

SYSTEMATIC EXPERIMENTATION
───────────────────────────────────────────────────────────────────────────────

  23. Baseline experiment (fast):

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --classifier multinomialnb \\
          --max-features 1000 \\
          --cefr-column cefr_level

  24. Standard experiment:

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --max-features-list 5000 10000 \\
          --classifiers logistic xgboost \\
          --cefr-column cefr_level

  25. Comprehensive experiment:

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --max-features-list 1000 5000 10000 \\
          --classifiers multinomialnb logistic randomforest xgboost svm \\
          --cefr-column cefr_level

      Result: 15 models (3 TF-IDF × 5 classifiers)

ADVANCED SCENARIOS
───────────────────────────────────────────────────────────────────────────────

  26. Reproduce specific configuration:

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --max-features 5000 \\
          --ngram-min 1 \\
          --ngram-max 2 \\
          --min-df 2 \\
          --max-df 0.95 \\
          --classifier xgboost \\
          --xgb-n-estimators 100 \\
          --xgb-max-depth 6 \\
          --random-state 42 \\
          --cefr-column cefr_level

  27. Quick test run (minimal features, fast classifier):

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --max-features 500 \\
          --classifier multinomialnb \\
          --cefr-column cefr_level \\
          -q

  28. Production model (optimized for accuracy):

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --max-features 20000 \\
          --ngram-min 1 \\
          --ngram-max 3 \\
          --classifier xgboost \\
          --xgb-n-estimators 300 \\
          --xgb-max-depth 10 \\
          --xgb-use-gpu \\
          --cefr-column cefr_level

  29. Feature importance study:

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --max-features-list 100 500 1000 5000 10000 \\
          --classifier xgboost \\
          --cefr-column cefr_level

      # Compare performance across feature sizes to find optimal

  30. Algorithm comparison:

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --classifiers multinomialnb logistic randomforest svm xgboost \\
          --max-features 5000 \\
          --cefr-column cefr_level

      # Compare all algorithms on same features

═══════════════════════════════════════════════════════════════════════════════
                    ABLATION STUDIES & HYPERPARAMETER SEARCH
═══════════════════════════════════════════════════════════════════════════════

Systematic ablation studies help identify optimal configurations. Each TF-IDF
configuration is stored in a unique hashed directory, preventing overwrites.

TF-IDF PARAMETER ABLATIONS
───────────────────────────────────────────────────────────────────────────────

  31. Ablate max_features (vocabulary size):
      Effect: Controls number of features, impacts model capacity and speed

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --max-features-list 500 1000 2000 5000 10000 20000 50000 \\
          --classifier xgboost \\
          --cefr-column cefr_level

      Analysis: Compare accuracy vs. training time across feature sizes
      Expected: Accuracy increases then plateaus; larger = slower

  32. Ablate ngram range (linguistic context):
      Effect: Unigrams only vs. unigrams+bigrams vs. unigrams+bigrams+trigrams

      # Unigrams only
      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --ngram-min 1 --ngram-max 1 \\
          --max-features 10000 \\
          --classifier xgboost \\
          --cefr-column cefr_level

      # Unigrams + bigrams (default)
      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --ngram-min 1 --ngram-max 2 \\
          --max-features 10000 \\
          --classifier xgboost \\
          --cefr-column cefr_level

      # Unigrams + bigrams + trigrams
      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --ngram-min 1 --ngram-max 3 \\
          --max-features 10000 \\
          --classifier xgboost \\
          --cefr-column cefr_level

      # Bigrams only
      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --ngram-min 2 --ngram-max 2 \\
          --max-features 10000 \\
          --classifier xgboost \\
          --cefr-column cefr_level

      Analysis: Trigrams capture more context but may overfit
      Expected: (1,2) or (1,3) usually optimal for text classification

  33. Ablate min_df (rare term filtering):
      Effect: Higher min_df removes rare terms, reduces noise

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --min-df 1 \\
          --max-features 10000 \\
          --classifier xgboost \\
          --cefr-column cefr_level

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --min-df 2 \\
          --max-features 10000 \\
          --classifier xgboost \\
          --cefr-column cefr_level

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --min-df 5 \\
          --max-features 10000 \\
          --classifier xgboost \\
          --cefr-column cefr_level

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --min-df 10 \\
          --max-features 10000 \\
          --classifier xgboost \\
          --cefr-column cefr_level

      Analysis: Higher min_df reduces vocabulary, removes outliers
      Expected: min_df=2-5 usually optimal (balances noise vs. coverage)

  34. Ablate max_df (common term filtering):
      Effect: Lower max_df removes very common terms (stop words)

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --max-df 1.0 \\
          --max-features 10000 \\
          --classifier xgboost \\
          --cefr-column cefr_level

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --max-df 0.95 \\
          --max-features 10000 \\
          --classifier xgboost \\
          --cefr-column cefr_level

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --max-df 0.8 \\
          --max-features 10000 \\
          --classifier xgboost \\
          --cefr-column cefr_level

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --max-df 0.5 \\
          --max-features 10000 \\
          --classifier xgboost \\
          --cefr-column cefr_level

      Analysis: Too low removes informative frequent words
      Expected: max_df=0.8-0.95 usually optimal

XGBOOST PARAMETER ABLATIONS
───────────────────────────────────────────────────────────────────────────────

  35. Ablate n_estimators (number of trees):
      Effect: More trees = better fit but slower, risk of overfitting

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --classifier xgboost \\
          --xgb-n-estimators 50 \\
          --max-features 10000 \\
          --cefr-column cefr_level

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --classifier xgboost \\
          --xgb-n-estimators 100 \\
          --max-features 10000 \\
          --cefr-column cefr_level

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --classifier xgboost \\
          --xgb-n-estimators 200 \\
          --max-features 10000 \\
          --cefr-column cefr_level

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --classifier xgboost \\
          --xgb-n-estimators 500 \\
          --max-features 10000 \\
          --cefr-column cefr_level

      Analysis: Monitor training vs. validation accuracy
      Expected: Performance plateaus after 100-300 trees typically

  36. Ablate max_depth (tree complexity):
      Effect: Deeper trees capture more interactions but may overfit

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --classifier xgboost \\
          --xgb-max-depth 3 \\
          --max-features 10000 \\
          --cefr-column cefr_level

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --classifier xgboost \\
          --xgb-max-depth 6 \\
          --max-features 10000 \\
          --cefr-column cefr_level

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --classifier xgboost \\
          --xgb-max-depth 10 \\
          --max-features 10000 \\
          --cefr-column cefr_level

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --classifier xgboost \\
          --xgb-max-depth 15 \\
          --max-features 10000 \\
          --cefr-column cefr_level

      Analysis: Check for overfitting with deeper trees
      Expected: max_depth=6-10 usually optimal for text data

  37. Ablate learning_rate (shrinkage):
      Effect: Lower = more conservative, needs more trees but generalizes better

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --classifier xgboost \\
          --xgb-learning-rate 0.01 \\
          --xgb-n-estimators 500 \\
          --max-features 10000 \\
          --cefr-column cefr_level

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --classifier xgboost \\
          --xgb-learning-rate 0.05 \\
          --xgb-n-estimators 300 \\
          --max-features 10000 \\
          --cefr-column cefr_level

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --classifier xgboost \\
          --xgb-learning-rate 0.1 \\
          --xgb-n-estimators 200 \\
          --max-features 10000 \\
          --cefr-column cefr_level

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          --classifier xgboost \\
          --xgb-learning-rate 0.3 \\
          --xgb-n-estimators 100 \\
          --max-features 10000 \\
          --cefr-column cefr_level

      Analysis: Lower learning_rate needs more estimators
      Expected: 0.01-0.1 with appropriate n_estimators usually optimal

COMBINED ABLATIONS (2D Grid Search)
───────────────────────────────────────────────────────────────────────────────

  38. TF-IDF max_features × XGBoost n_estimators:
      Understand interaction between feature size and model capacity

      # Systematic grid: 3 × 3 = 9 models
      # Run multiple times with different n_estimators

      python -m src.pipeline \\
          -e data/experiments/ablation \\
          --max-features-list 1000 5000 10000 \\
          --classifier xgboost \\
          --xgb-n-estimators 50 \\
          --cefr-column cefr_level

      python -m src.pipeline \\
          -e data/experiments/ablation \\
          --max-features-list 1000 5000 10000 \\
          --classifier xgboost \\
          --xgb-n-estimators 100 \\
          --cefr-column cefr_level

      python -m src.pipeline \\
          -e data/experiments/ablation \\
          --max-features-list 1000 5000 10000 \\
          --classifier xgboost \\
          --xgb-n-estimators 200 \\
          --cefr-column cefr_level

      Analysis: Create heatmap of accuracy vs. (max_features, n_estimators)
      Question: Does optimal n_estimators depend on max_features?

  39. N-gram range × max_features:
      Understand how linguistic context interacts with vocabulary size

      python -m src.pipeline \\
          -e data/experiments/ablation \\
          --ngram-min 1 --ngram-max 1 \\
          --max-features-list 1000 5000 10000 \\
          --classifier xgboost \\
          --cefr-column cefr_level

      python -m src.pipeline \\
          -e data/experiments/ablation \\
          --ngram-min 1 --ngram-max 2 \\
          --max-features-list 1000 5000 10000 \\
          --classifier xgboost \\
          --cefr-column cefr_level

      python -m src.pipeline \\
          -e data/experiments/ablation \\
          --ngram-min 1 --ngram-max 3 \\
          --max-features-list 1000 5000 10000 \\
          --classifier xgboost \\
          --cefr-column cefr_level

      Analysis: Do bigrams/trigrams help more with smaller vocabularies?
      Expected: Bigrams more valuable when max_features is limited

  40. XGBoost max_depth × n_estimators:
      Classic bias-variance tradeoff exploration

      python -m src.pipeline \\
          -e data/experiments/ablation \\
          --classifier xgboost \\
          --xgb-max-depth 3 \\
          --xgb-n-estimators 100 \\
          --max-features 10000 \\
          --cefr-column cefr_level

      python -m src.pipeline \\
          -e data/experiments/ablation \\
          --classifier xgboost \\
          --xgb-max-depth 6 \\
          --xgb-n-estimators 100 \\
          --max-features 10000 \\
          --cefr-column cefr_level

      # Continue with depth=10, 15 and estimators=50, 200, 500
      # Creates matrix of models to analyze overfitting patterns

COMPREHENSIVE HYPERPARAMETER SEARCH
───────────────────────────────────────────────────────────────────────────────

  41. Coarse-to-fine TF-IDF search:
      Stage 1: Broad search across orders of magnitude

      python -m src.pipeline \\
          -e data/experiments/coarse-search \\
          --max-features-list 100 500 1000 5000 10000 50000 \\
          --classifier xgboost \\
          --cefr-column cefr_level

      Stage 2: Fine-grained search around best coarse result
      (Suppose 5000-10000 performed best)

      python -m src.pipeline \\
          -e data/experiments/fine-search \\
          --max-features-list 5000 6000 7000 8000 9000 10000 \\
          --classifier xgboost \\
          --cefr-column cefr_level

  42. Multi-dimensional XGBoost search (small):
      3 × 3 × 2 = 18 models

      # Depth=3
      python -m src.pipeline \\
          -e data/experiments/xgb-grid \\
          --classifier xgboost \\
          --xgb-max-depth 3 \\
          --xgb-n-estimators 100 \\
          --xgb-learning-rate 0.1 \\
          --cefr-column cefr_level

      python -m src.pipeline \\
          -e data/experiments/xgb-grid \\
          --classifier xgboost \\
          --xgb-max-depth 3 \\
          --xgb-n-estimators 200 \\
          --xgb-learning-rate 0.1 \\
          --cefr-column cefr_level

      # ... continue with all combinations of:
      # max_depth: [3, 6, 10]
      # n_estimators: [100, 200, 500]
      # learning_rate: [0.05, 0.1]

  43. Full factorial experiment (production-ready):
      TF-IDF (3) × Classifier (5) × Feature size (3) = 45 models

      python -m src.pipeline \\
          -e data/experiments/full-factorial \\
          --max-features-list 5000 10000 20000 \\
          --classifiers multinomialnb logistic randomforest svm xgboost \\
          --cefr-column cefr_level

      Analysis:
      - Compare across classifiers: Which performs best?
      - Compare across features: Diminishing returns beyond 10k?
      - Interaction effects: Do some classifiers benefit more from more features?

ABLATION ANALYSIS WORKFLOW
───────────────────────────────────────────────────────────────────────────────

  44. Step-by-step systematic ablation:

      # Baseline (defaults)
      python -m src.pipeline \\
          -e data/experiments/baseline \\
          --max-features 5000 \\
          --ngram-min 1 --ngram-max 2 \\
          --min-df 2 --max-df 0.95 \\
          --classifier xgboost \\
          --xgb-n-estimators 100 \\
          --xgb-max-depth 6 \\
          --xgb-learning-rate 0.3 \\
          --cefr-column cefr_level

      # Ablation 1: Vary ONLY max_features (keep all else constant)
      python -m src.pipeline \\
          -e data/experiments/ablation-1 \\
          --max-features-list 1000 2000 5000 10000 20000 \\
          --ngram-min 1 --ngram-max 2 \\
          --min-df 2 --max-df 0.95 \\
          --classifier xgboost \\
          --xgb-n-estimators 100 \\
          --xgb-max-depth 6 \\
          --cefr-column cefr_level

      # Ablation 2: Use best max_features, vary n_estimators
      # (Suppose max_features=10000 was best from ablation-1)
      python -m src.pipeline \\
          -e data/experiments/ablation-2 \\
          --max-features 10000 \\
          --ngram-min 1 --ngram-max 2 \\
          --classifier xgboost \\
          --xgb-n-estimators 50 \\
          --cefr-column cefr_level

      python -m src.pipeline \\
          -e data/experiments/ablation-2 \\
          --max-features 10000 \\
          --classifier xgboost \\
          --xgb-n-estimators 100 \\
          --cefr-column cefr_level

      # Continue iteratively, optimizing one parameter at a time

      # Final optimized model
      python -m src.pipeline \\
          -e data/experiments/optimized \\
          --max-features 10000 \\
          --ngram-min 1 --ngram-max 3 \\
          --min-df 3 \\
          --classifier xgboost \\
          --xgb-n-estimators 200 \\
          --xgb-max-depth 8 \\
          --xgb-learning-rate 0.1 \\
          --xgb-use-gpu \\
          --cefr-column cefr_level

RECOMMENDED ABLATION SEQUENCES
───────────────────────────────────────────────────────────────────────────────

  45. Minimal ablation (fast, 1-2 hours):
      Focus on most impactful parameters

      # TF-IDF max_features
      python -m src.pipeline \\
          -e data/experiments/minimal \\
          --max-features-list 1000 5000 10000 \\
          --classifier xgboost \\
          --cefr-column cefr_level

      # XGBoost depth
      python -m src.pipeline \\
          -e data/experiments/minimal \\
          --max-features 5000 \\
          --classifier xgboost \\
          --xgb-max-depth 3 \\
          --cefr-column cefr_level

      python -m src.pipeline \\
          -e data/experiments/minimal \\
          --max-features 5000 \\
          --classifier xgboost \\
          --xgb-max-depth 10 \\
          --cefr-column cefr_level

  46. Standard ablation (thorough, 4-8 hours):
      Covers key parameters systematically

      # Max features (5 configs)
      python -m src.pipeline \\
          -e data/experiments/standard \\
          --max-features-list 500 1000 5000 10000 20000 \\
          --classifier xgboost \\
          --cefr-column cefr_level

      # N-gram range (3 configs)
      python -m src.pipeline \\
          -e data/experiments/standard \\
          --max-features 10000 \\
          --ngram-min 1 --ngram-max 1 \\
          --classifier xgboost \\
          --cefr-column cefr_level

      python -m src.pipeline \\
          -e data/experiments/standard \\
          --max-features 10000 \\
          --ngram-min 1 --ngram-max 2 \\
          --classifier xgboost \\
          --cefr-column cefr_level

      python -m src.pipeline \\
          -e data/experiments/standard \\
          --max-features 10000 \\
          --ngram-min 1 --ngram-max 3 \\
          --classifier xgboost \\
          --cefr-column cefr_level

      # XGBoost params (9 configs: 3 depth × 3 estimators)
      # ... (see example 42)

  47. Exhaustive ablation (comprehensive, 12-24 hours):
      Publication-ready hyperparameter analysis

      # Full TF-IDF grid
      python -m src.pipeline \\
          -e data/experiments/exhaustive \\
          --max-features-list 500 1000 2000 5000 10000 20000 50000 \\
          --classifiers xgboost logistic randomforest \\
          --cefr-column cefr_level

      # N-gram ablation for each max_features
      # Document frequency thresholds
      # Full XGBoost grid (depth × estimators × learning_rate)
      # Cross-classifier comparison

      Result: 100+ models, comprehensive performance landscape

═══════════════════════════════════════════════════════════════════════════════
                           CONFIGURATION FILES
═══════════════════════════════════════════════════════════════════════════════

You can also use YAML or JSON configuration files:

  48. Using YAML config:

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          -c config.yaml

  49. Using JSON config:

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          -c config.json

  50. Config file with CLI overrides:

      python -m src.pipeline \\
          -e data/experiments/zero-shot \\
          -c config.yaml \\
          --classifier xgboost \\
          --xgb-use-gpu

YAML Configuration Example (config.yaml):
───────────────────────────────────────────────────────────────────────────────
  experiment_config:
    experiment_dir: data/experiments/zero-shot

  tfidf_config:
    max_features: 5000
    ngram_range: [1, 2]
    min_df: 2
    max_df: 0.95
    sublinear_tf: true

  classifier_config:
    classifier_type: xgboost
    xgb_use_gpu: true
    xgb_n_estimators: 200
    xgb_max_depth: 8

  data_config:
    text_column: text
    label_column: label
    cefr_column: cefr_level

  output_config:
    verbose: true
    save_config: true

JSON Configuration Example (config.json):
───────────────────────────────────────────────────────────────────────────────
  {
    "experiment_config": {
      "experiment_dir": "data/experiments/zero-shot"
    },
    "tfidf_config": {
      "max_features": 5000,
      "ngram_range": [1, 2],
      "min_df": 2,
      "max_df": 0.95
    },
    "classifier_config": {
      "classifier_type": "xgboost",
      "xgb_use_gpu": true
    },
    "data_config": {
      "cefr_column": "cefr_level"
    }
  }

═══════════════════════════════════════════════════════════════════════════════
                              OUTPUT STRUCTURE
═══════════════════════════════════════════════════════════════════════════════

After running the pipeline, you'll find:

  experiment-dir/
    feature-models/
      tfidf/
        abc12345/              ← Hashed TF-IDF config directory
          tfidf_model.pkl
          config.json
        def67890/              ← Another TF-IDF config
          tfidf_model.pkl
          config.json
      classifiers/
        {dataset}_{classifier}/
          classifier.pkl
          label_encoder.pkl    ← For XGBoost
          config.json

    features/
      abc12345/                ← Features from TF-IDF abc12345
        norm-EFCAMDAT-train/
          features_dense.csv
          feature_names.csv
        norm-CELVA-SP/
          features_dense.csv
      def67890/                ← Features from TF-IDF def67890
        norm-EFCAMDAT-train/
          features_dense.csv

    results/
      norm-CELVA-SP/           ← Results by dataset
        soft_predictions.json
        argmax_predictions.json
        rounded_avg_predictions.json
        evaluation_report.md
      norm-KUPA-KEYS/
        ...

═══════════════════════════════════════════════════════════════════════════════
                           PIPELINE STEPS EXPLAINED
═══════════════════════════════════════════════════════════════════════════════

  Step 1: Train TF-IDF Vectorizer
    - Fits TF-IDF on training data
    - Extracts training features
    - Saves model to hashed directory

  Step 2: Extract Features
    - Loads TF-IDF model
    - Extracts features for all train/test sets
    - Saves to hashed feature directories

  Step 3: Train Classifiers
    - Loads pre-extracted features
    - Trains classifier(s) with label encoding
    - Saves models + encoders + configs

  Step 4: Make Predictions
    - Loads trained classifiers
    - Predicts on all test sets
    - Generates 3 prediction strategies
    - Saves markdown evaluation reports

═══════════════════════════════════════════════════════════════════════════════
                                  TIPS
═══════════════════════════════════════════════════════════════════════════════

  • Use --max-features-list for feature size experiments
  • Use --classifiers for algorithm comparison
  • Use --steps to run partial pipeline
  • Use -q for quiet mode in automated scripts
  • Use --xgb-use-gpu for faster XGBoost training
  • Each TF-IDF config gets unique directory (no overwrites!)
  • Results organized by dataset for easy comparison

═══════════════════════════════════════════════════════════════════════════════

For more information, see documentation in the project repository.
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
        required=True,
        help="Path to experiment directory (e.g., data/experiments/zero-shot)",
    )
    exp_group.add_argument(
        "-o",
        "--output-dir",
        help="Custom output directory for models (default: <experiment-dir>/feature-models)",
    )

    # Pipeline control
    pipeline_group = parser.add_argument_group("Pipeline Control")
    pipeline_group.add_argument(
        "--steps",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4],
        help="Specific steps to run (1=TF-IDF, 2=Features, 3=Classifiers, 4=Predict). Default: all steps",
    )

    # TF-IDF configuration
    tfidf_group = parser.add_argument_group("TF-IDF Configuration")
    tfidf_group.add_argument(
        "--max-features",
        type=int,
        help="Maximum number of TF-IDF features (default: 5000)",
    )
    tfidf_group.add_argument(
        "--max-features-list",
        nargs="+",
        type=int,
        help="Train multiple TF-IDF models with different max_features (e.g., 1000 5000 10000)",
    )
    tfidf_group.add_argument(
        "--ngram-min", type=int, help="Minimum n-gram size (default: 1)"
    )
    tfidf_group.add_argument(
        "--ngram-max", type=int, help="Maximum n-gram size (default: 2)"
    )
    tfidf_group.add_argument(
        "--min-df", type=int, help="Minimum document frequency (default: 2)"
    )
    tfidf_group.add_argument(
        "--max-df", type=float, help="Maximum document frequency (default: 0.95)"
    )

    # Classifier configuration
    clf_group = parser.add_argument_group("Classifier Configuration")
    clf_group.add_argument(
        "--classifier",
        choices=["multinomialnb", "logistic", "randomforest", "svm", "xgboost"],
        help="Single classifier type (default: xgboost)",
    )
    clf_group.add_argument(
        "--classifiers",
        nargs="+",
        choices=["multinomialnb", "logistic", "randomforest", "svm", "xgboost"],
        help="Train multiple classifiers (e.g., xgboost logistic randomforest)",
    )
    clf_group.add_argument("--random-state", type=int, help="Random seed (default: 42)")
    clf_group.add_argument(
        "--xgb-use-gpu", action="store_true", help="Use GPU for XGBoost training"
    )
    clf_group.add_argument(
        "--xgb-n-estimators",
        type=int,
        help="Number of boosting rounds for XGBoost (default: 100)",
    )
    clf_group.add_argument(
        "--xgb-max-depth", type=int, help="Max tree depth for XGBoost (default: 6)"
    )

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
    output_group.add_argument(
        "--summarize",
        action="store_true",
        help="Generate summary report after pipeline completion (saves to results_summary.md)",
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
            ngram_min = (
                args.ngram_min if args.ngram_min else config.tfidf_config.ngram_range[0]
            )
            ngram_max = (
                args.ngram_max if args.ngram_max else config.tfidf_config.ngram_range[1]
            )
            config.tfidf_config.ngram_range = (ngram_min, ngram_max)
        if args.min_df:
            config.tfidf_config.min_df = args.min_df
        if args.max_df:
            config.tfidf_config.max_df = args.max_df

        if args.classifier:
            config.classifier_config.classifier_type = args.classifier
        if args.random_state:
            config.classifier_config.random_state = args.random_state
        if args.xgb_use_gpu:
            config.classifier_config.xgb_use_gpu = args.xgb_use_gpu
        if args.xgb_n_estimators:
            config.classifier_config.xgb_n_estimators = args.xgb_n_estimators
        if args.xgb_max_depth:
            config.classifier_config.xgb_max_depth = args.xgb_max_depth

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
            experiment_dir=args.experiment_dir,
            models_dir=args.output_dir if args.output_dir else None,
        )

        tfidf_config = TfidfConfig(
            max_features=args.max_features or 5000,
            ngram_range=(args.ngram_min or 1, args.ngram_max or 2),
            min_df=args.min_df or 2,
            max_df=args.max_df or 0.95,
        )

        classifier_config = ClassifierConfig(
            classifier_type=args.classifier or "xgboost",
            random_state=args.random_state or 42,
            xgb_use_gpu=args.xgb_use_gpu if args.xgb_use_gpu else False,
            xgb_n_estimators=args.xgb_n_estimators or 100,
            xgb_max_depth=args.xgb_max_depth or 6,
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
            tfidf_config,
            classifier_config,
            data_config,
            output_config,
        )


def main():
    """Main entry point for the CEFR classification pipeline."""
    parser = create_parser()
    args = parser.parse_args()

    # Build configuration
    try:
        config = args_to_config(args)
    except Exception as e:
        parser.error(f"Configuration error: {e}")

    # Determine classifiers to train
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
        print("Base Configuration:")
        print(json.dumps(config.to_dict(), indent=2))
        if classifiers:
            print(f"\nClassifiers to train: {classifiers}")
        if tfidf_configs:
            print(f"TF-IDF configurations: {tfidf_configs}")
        print()

    # Run pipeline
    success = run_pipeline(
        config, args.steps, classifiers, tfidf_configs, summarize=args.summarize
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
