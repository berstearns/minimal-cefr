#!/usr/bin/env python3
"""
End-to-end GPT-2 Native Perplexity Zero-Shot Experiment.

Extracts aggregate perplexity features from pre-trained GPT-2,
trains LR + XGBoost classifiers, predicts on 3 test sets.

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
REPRO_DIR = Path(__file__).resolve().parents[1]   # paper-reproducibility-.../
EXP_DIR = REPRO_DIR / "experiment"
DATA_SRC = Path(
    "/home/b/p/my-data/i/phd-experimental-data/cefr-classification/data/splits"
)

# Datasets
DATASETS = {
    "norm-EFCAMDAT-train": {"src": DATA_SRC / "norm-EFCAMDAT-train.csv", "role": "train"},
    "norm-EFCAMDAT-test":  {"src": DATA_SRC / "norm-EFCAMDAT-test.csv",  "role": "test"},
    "norm-CELVA-SP":       {"src": DATA_SRC / "norm-CELVA-SP.csv",       "role": "test"},
    "norm-KUPA-KEYS":      {"src": DATA_SRC / "norm-KUPA-KEYS.csv",      "role": "test"},
}

# Limits for CPU run (set to None for full data)
LIMITS = {
    "norm-EFCAMDAT-train": 2000,
    "norm-EFCAMDAT-test":  500,
    "norm-CELVA-SP":       None,  # 1742 rows, manageable
    "norm-KUPA-KEYS":      None,  # 1006 rows, manageable
}

CLASSIFIERS = ["logistic", "xgboost"]
CEFR_COL = "cefr_level"
FEATURE_DIR_NAME = "gpt2_native"


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
        # Print last 40 lines to avoid flooding
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


def step1_extract_perplexity(name):
    """Extract GPT-2 aggregate perplexity for one dataset."""
    raw_dir = REPRO_DIR / "perplexity-raw"
    raw_dir.mkdir(exist_ok=True)
    out_file = raw_dir / f"{name}.csv"

    if out_file.exists():
        print(f"  Perplexity already extracted: {out_file.name}")
        return out_file

    src_csv = DATASETS[name]["src"]
    cmd = [
        sys.executable, "-m", "src.extract_perplexity_features",
        "-i", str(src_csv),
        "--text-column", "text",
        "-m", "gpt2",
        "-d", "cpu",
        "--aggregate-only",
        "-f", "csv",
        "-o", str(out_file),
    ]
    limit = LIMITS.get(name)
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    run_cmd(cmd, desc=f"Extract GPT-2 perplexity: {name} (limit={limit})")
    return out_file


def step2_convert_features(name, raw_csv):
    """Convert raw perplexity CSV to features_dense.csv (numeric only)."""
    feat_dir = EXP_DIR / "features" / FEATURE_DIR_NAME / name
    feat_dir.mkdir(parents=True, exist_ok=True)
    out_file = feat_dir / "features_dense.csv"

    if out_file.exists():
        print(f"  Features already converted: {out_file}")
        return out_file

    print(f"  Converting {raw_csv.name} -> features_dense.csv")
    df = pd.read_csv(raw_csv)

    # Drop non-numeric columns
    drop_cols = [c for c in ["text", "model"] if c in df.columns]
    df_numeric = df.drop(columns=drop_cols)

    # Verify all columns are numeric
    for col in df_numeric.columns:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors="coerce")
    df_numeric = df_numeric.fillna(0.0)

    df_numeric.to_csv(out_file, index=False)

    # Also save feature names
    fn_file = feat_dir / "feature_names.csv"
    pd.DataFrame({"feature_name": df_numeric.columns}).to_csv(fn_file, index=False)

    print(f"  Saved {len(df_numeric)} rows x {len(df_numeric.columns)} features")
    return out_file


def step3_train_classifier(clf_type):
    """Train a classifier on EFCAMDAT-train perplexity features."""
    model_name = f"norm-EFCAMDAT-train_{clf_type}_gpt2native"
    model_dir = EXP_DIR / "feature-models" / "classifiers" / model_name

    if (model_dir / "classifier.pkl").exists():
        print(f"  Classifier already trained: {model_name}")
        return model_name

    features_file = (
        EXP_DIR / "features" / FEATURE_DIR_NAME / "norm-EFCAMDAT-train" / "features_dense.csv"
    )
    labels_csv = EXP_DIR / "ml-training-data" / "norm-EFCAMDAT-train.csv"

    # Verify row counts match (labels may have more rows than features if --limit was used)
    n_features = len(pd.read_csv(features_file))
    df_labels = pd.read_csv(labels_csv)
    if len(df_labels) != n_features:
        print(f"  Trimming labels from {len(df_labels)} to {n_features} rows to match features")
        trimmed = REPRO_DIR / "trimmed-labels" / "norm-EFCAMDAT-train.csv"
        trimmed.parent.mkdir(parents=True, exist_ok=True)
        df_labels.head(n_features).to_csv(trimmed, index=False)
        labels_csv = trimmed

    cmd = [
        sys.executable, "-m", "src.train_classifiers",
        "-e", str(EXP_DIR),
        "--features-file", str(features_file),
        "--labels-csv", str(labels_csv),
        "--cefr-column", CEFR_COL,
        "--classifier", clf_type,
        "--model-name", model_name,
    ]
    run_cmd(cmd, desc=f"Train {clf_type} on GPT-2 native features")
    return model_name


def step4_predict(model_name, test_name):
    """Run prediction for one (model, test_set) pair."""
    results_dir = EXP_DIR / "results" / model_name / test_name

    if (results_dir / "evaluation_report.md").exists():
        print(f"  Prediction already done: {model_name} -> {test_name}")
        return

    features_file = (
        EXP_DIR / "features" / FEATURE_DIR_NAME / test_name / "features_dense.csv"
    )
    labels_csv = EXP_DIR / "ml-test-data" / f"{test_name}.csv"

    # Check if we need to trim labels to match features (due to --limit)
    n_features = len(pd.read_csv(features_file))
    df_labels = pd.read_csv(labels_csv)
    if len(df_labels) != n_features:
        print(f"  Trimming test labels from {len(df_labels)} to {n_features} rows")
        trimmed = REPRO_DIR / "trimmed-labels" / f"{test_name}.csv"
        trimmed.parent.mkdir(parents=True, exist_ok=True)
        df_labels.head(n_features).to_csv(trimmed, index=False)
        labels_csv = trimmed

    cmd = [
        sys.executable, "-m", "src.predict",
        "-e", str(EXP_DIR),
        "-m", model_name,
        "--features-file", str(features_file),
        "--labels-csv", str(labels_csv),
        "--cefr-column", CEFR_COL,
    ]
    run_cmd(cmd, desc=f"Predict: {model_name} -> {test_name}")


def step5_report():
    """Generate summary report."""
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
    print(" GPT-2 Native Perplexity Zero-Shot Experiment")
    print(f" Repo root: {REPO_ROOT}")
    print(f" Experiment dir: {EXP_DIR}")
    print(f" Data source: {DATA_SRC}")
    print("="*70)

    # ── Step 0: Setup ──
    step0_setup()

    # ── Step 1+2: Extract perplexity & convert to features ──
    print("\n" + "#"*70)
    print("# STEP 1+2: Extract GPT-2 perplexity & convert to features")
    print("#"*70)
    for name in DATASETS:
        raw_csv = step1_extract_perplexity(name)
        step2_convert_features(name, raw_csv)

    # ── Step 3: Train classifiers ──
    print("\n" + "#"*70)
    print("# STEP 3: Train classifiers (LR + XGBoost)")
    print("#"*70)
    model_names = {}
    for clf_type in CLASSIFIERS:
        model_names[clf_type] = step3_train_classifier(clf_type)

    # ── Step 4: Predict on all test sets ──
    print("\n" + "#"*70)
    print("# STEP 4: Predict on test sets")
    print("#"*70)
    test_sets = [n for n, info in DATASETS.items() if info["role"] == "test"]
    for clf_type in CLASSIFIERS:
        for test_name in test_sets:
            step4_predict(model_names[clf_type], test_name)

    # ── Step 5: Report ──
    print("\n" + "#"*70)
    print("# STEP 5: Generate report")
    print("#"*70)
    step5_report()

    elapsed = time.time() - t_start
    print("\n" + "="*70)
    print(f" EXPERIMENT COMPLETE in {elapsed/60:.1f} minutes")
    print(f" Results: {EXP_DIR / 'results_summary.md'}")
    print("="*70)


if __name__ == "__main__":
    main()
