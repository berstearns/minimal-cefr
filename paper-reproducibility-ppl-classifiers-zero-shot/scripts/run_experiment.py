#!/usr/bin/env python3
"""
End-to-end PPL Classifiers Zero-Shot Experiment.

Assembles pre-extracted perplexity features from gzip files,
trains LR + XGBoost classifiers on 3 feature configs,
predicts on 3 test sets, generates summary report.

No GPU, no perplexity re-extraction -- features are pre-computed.
DOES NOT MODIFY ANYTHING IN src/.
All outputs go into the experiment/ subdirectory.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]  # minimal-cefr/
REPRO_DIR = Path(__file__).resolve().parents[1]   # paper-reproducibility-ppl-classifiers-zero-shot/
EXP_DIR = REPRO_DIR / "experiment"
DATA_SRC = Path(
    "/home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits"
)

# ── Datasets ───────────────────────────────────────────────────────────────
DATASETS = {
    "norm-EFCAMDAT-train": {"src": DATA_SRC / "norm-EFCAMDAT-train.csv", "role": "train"},
    "norm-EFCAMDAT-test":  {"src": DATA_SRC / "norm-EFCAMDAT-test.csv",  "role": "test"},
    "norm-CELVA-SP":       {"src": DATA_SRC / "norm-CELVA-SP.csv",       "role": "test"},
    "norm-KUPA-KEYS":      {"src": DATA_SRC / "norm-KUPA-KEYS.csv",      "role": "test"},
}

# ── Feature configurations ─────────────────────────────────────────────────
FEATURE_CONFIGS = ["native_only", "native_general", "all_models"]

CLASSIFIERS = ["logistic", "xgboost"]
CEFR_COL = "cefr_level"


def run_cmd(cmd, desc="", cwd=None):
    """Run a shell command, print output, raise on failure."""
    print(f"\n{'='*70}")
    print(f"  {desc}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*70}")
    t0 = time.time()
    result = subprocess.run(
        cmd, cwd=cwd or str(REPO_ROOT),
        capture_output=True, text=True, timeout=7200,  # 2h max
    )
    elapsed = time.time() - t0
    if result.stdout:
        lines = result.stdout.strip().split("\n")
        if len(lines) > 40:
            print(f"  ... ({len(lines)-40} lines omitted)")
        for line in lines[-40:]:
            print(f"  {line}")
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-2000:]}")
        raise RuntimeError(f"Command failed (rc={result.returncode}): {desc}")
    print(f"  Done in {elapsed:.1f}s")
    return result


def step0_setup():
    """Copy data into experiment directory."""
    print("\n" + "#"*70)
    print("# STEP 0: Setup experiment directory")
    print("#"*70)

    for name, info in DATASETS.items():
        if info["role"] == "train":
            dest_dir = EXP_DIR / "ml-training-data"
        else:
            dest_dir = EXP_DIR / "ml-test-data"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{name}.csv"
        if not dest.exists():
            print(f"  Copying {info['src'].name} -> {dest}")
            shutil.copy2(info["src"], dest)
        else:
            print(f"  Already exists: {dest.name}")

    # Ensure other required dirs exist
    for subdir in ["feature-models", "features", "results"]:
        (EXP_DIR / subdir).mkdir(exist_ok=True)

    print("  Setup complete.")


def step1_assemble_features():
    """Assemble pre-extracted perplexity features from gzip files."""
    print("\n" + "#"*70)
    print("# STEP 1: Assemble perplexity features from gzip files")
    print("#"*70)

    # Check if features already assembled (all configs, all datasets)
    all_exist = True
    for config_name in FEATURE_CONFIGS:
        for ds_name in DATASETS:
            feat_file = EXP_DIR / "features" / config_name / ds_name / "features_dense.csv"
            if not feat_file.exists():
                all_exist = False
                break
        if not all_exist:
            break

    if all_exist:
        print("  Features already assembled for all configurations.")
        return

    assemble_script = REPRO_DIR / "scripts" / "assemble_ppl_features.py"
    run_cmd(
        [sys.executable, str(assemble_script)],
        desc="Assemble perplexity features from pre-extracted gzip files",
    )


def step2_train_classifiers():
    """Train LR + XGBoost for each feature configuration."""
    print("\n" + "#"*70)
    print("# STEP 2: Train classifiers (3 configs x 2 classifiers = 6 models)")
    print("#"*70)

    model_names = {}
    for config_name in FEATURE_CONFIGS:
        for clf_type in CLASSIFIERS:
            model_name = f"ppl_{config_name}_{clf_type}"
            model_names[(config_name, clf_type)] = model_name

            model_dir = EXP_DIR / "feature-models" / "classifiers" / model_name
            if (model_dir / "classifier.pkl").exists():
                print(f"  Already trained: {model_name}")
                continue

            features_file = (
                EXP_DIR / "features" / config_name / "norm-EFCAMDAT-train" / "features_dense.csv"
            )
            labels_csv = EXP_DIR / "ml-training-data" / "norm-EFCAMDAT-train.csv"

            if not features_file.exists():
                print(f"  SKIP {model_name}: features not found at {features_file}")
                continue

            cmd = [
                sys.executable, "-m", "src.train_classifiers",
                "-e", str(EXP_DIR),
                "--features-file", str(features_file),
                "--labels-csv", str(labels_csv),
                "--cefr-column", CEFR_COL,
                "--classifier", clf_type,
                "--model-name", model_name,
            ]
            run_cmd(cmd, desc=f"Train {clf_type} on {config_name} features")

    return model_names


def step3_predict(model_names):
    """Run prediction for all (model, test_set) pairs."""
    print("\n" + "#"*70)
    print("# STEP 3: Predict on test sets (6 models x 3 test sets = 18 evals)")
    print("#"*70)

    test_sets = [n for n, info in DATASETS.items() if info["role"] == "test"]

    for config_name in FEATURE_CONFIGS:
        for clf_type in CLASSIFIERS:
            model_name = model_names.get(
                (config_name, clf_type),
                f"ppl_{config_name}_{clf_type}",
            )

            for test_name in test_sets:
                results_dir = EXP_DIR / "results" / model_name / test_name
                if (results_dir / "evaluation_report.md").exists():
                    print(f"  Already done: {model_name} -> {test_name}")
                    continue

                features_file = (
                    EXP_DIR / "features" / config_name / test_name / "features_dense.csv"
                )
                labels_csv = EXP_DIR / "ml-test-data" / f"{test_name}.csv"

                if not features_file.exists():
                    print(f"  SKIP: features not found for {config_name}/{test_name}")
                    continue

                cmd = [
                    sys.executable, "-m", "src.predict",
                    "-e", str(EXP_DIR),
                    "-m", model_name,
                    "--features-file", str(features_file),
                    "--labels-csv", str(labels_csv),
                    "--cefr-column", CEFR_COL,
                ]
                run_cmd(cmd, desc=f"Predict: {model_name} -> {test_name}")


def step4_report():
    """Generate summary report."""
    print("\n" + "#"*70)
    print("# STEP 4: Generate results summary")
    print("#"*70)

    cmd = [
        sys.executable, "-m", "src.report",
        "-e", str(EXP_DIR),
        "--rank", "accuracy",
        "--summary-report", str(EXP_DIR / "results_summary.md"),
        "--include-all-datasets",
        "-v",
    ]
    run_cmd(cmd, desc="Generate results summary")


def main():
    t_start = time.time()
    print("="*70)
    print(" PPL Classifiers Zero-Shot Experiment")
    print(" (Pre-extracted perplexity features, no GPU)")
    print(f" Repo root:       {REPO_ROOT}")
    print(f" Experiment dir:  {EXP_DIR}")
    print(f" Data source:     {DATA_SRC}")
    print(f" Feature configs: {FEATURE_CONFIGS}")
    print(f" Classifiers:     {CLASSIFIERS}")
    print("="*70)

    # ── Step 0: Setup ──
    step0_setup()

    # ── Step 1: Assemble features ──
    step1_assemble_features()

    # ── Step 2: Train classifiers ──
    model_names = step2_train_classifiers()

    # ── Step 3: Predict on all test sets ──
    step3_predict(model_names)

    # ── Step 4: Report ──
    step4_report()

    elapsed = time.time() - t_start
    print("\n" + "="*70)
    print(f" EXPERIMENT COMPLETE in {elapsed/60:.1f} minutes")
    print(f" Results: {EXP_DIR / 'results_summary.md'}")
    print("="*70)


if __name__ == "__main__":
    main()
