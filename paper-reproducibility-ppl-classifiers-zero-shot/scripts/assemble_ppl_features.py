#!/usr/bin/env python3
"""
Assemble pre-extracted perplexity feature matrices from gzip files.

Reads .csv.features.gzip files from gdrive-data/fe/, column-concatenates
per-model features, and writes features_dense.csv into the experiment dir.

No GPU, no model loading -- pure pandas operations.

DOES NOT MODIFY ANYTHING IN src/.
"""

import os
import sys
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]  # minimal-cefr/
GDRIVE = REPO_ROOT.parent / "gdrive-data" / "fe"
REPRO_DIR = Path(__file__).resolve().parents[1]
EXP_DIR = REPRO_DIR / "experiment"

# ── Models ─────────────────────────────────────────────────────────────────
MODELS = [
    "gpt2", "AL-all-gpt2",
    "AL-a1-gpt2", "AL-a2-gpt2", "AL-b1-gpt2", "AL-b2-gpt2", "AL-c1-gpt2",
]

# ── Feature file mapping: (dataset, model) -> gzip path ───────────────────
FILES = {
    # --- EFCAMDAT-train (andrew100ktrain) ---
    ("norm-EFCAMDAT-train", "gpt2"):
        GDRIVE / "andrew100ktrain_df-gpt2_avg+ppp+ppos-fe_2025-04-23-09-31.csv.features.gzip",
    ("norm-EFCAMDAT-train", "AL-all-gpt2"):
        GDRIVE / "andrew100ktrain_df-AL-all-gpt2_avg+ppp+ppos-fe_2025-04-23-13-31.csv.features.gzip",
    ("norm-EFCAMDAT-train", "AL-a1-gpt2"):
        GDRIVE / "andrew100ktrain_df-AL-a1-gpt2_avg+ppp+ppos-fe_2025-04-23-09-31.csv.features.gzip",
    ("norm-EFCAMDAT-train", "AL-a2-gpt2"):
        GDRIVE / "andrew100ktrain_df-AL-a2-gpt2_avg+ppp+ppos-fe_2025-04-23-09-31.csv.features.gzip",
    ("norm-EFCAMDAT-train", "AL-b1-gpt2"):
        GDRIVE / "andrew100ktrain_df-AL-b1-gpt2_avg+ppp+ppos-fe_2025-04-23-09-31.csv.features.gzip",
    ("norm-EFCAMDAT-train", "AL-b2-gpt2"):
        GDRIVE / "andrew100ktrain_df-AL-b2-gpt2_avg+ppp+ppos-fe_2025-04-23-09-31.csv.features.gzip",
    ("norm-EFCAMDAT-train", "AL-c1-gpt2"):
        GDRIVE / "andrew100ktrain_df-AL-c1-gpt2_avg+ppp+ppos-fe_2025-04-23-09-31.csv.features.gzip",

    # --- EFCAMDAT-test (andrew100ktest) -- flat variants only available ---
    ("norm-EFCAMDAT-test", "gpt2"):
        GDRIVE / "andrew100ktest_df-gpt2_avg+ppp+ppos-fe-flat_2025-04-24-00-56.csv.features.gzip",
    ("norm-EFCAMDAT-test", "AL-all-gpt2"):
        GDRIVE / "andrew100ktest_df-AL-all-gpt2_avg+ppp+ppos-fe-flat_2025-04-24-00-56.csv.features.gzip",
    ("norm-EFCAMDAT-test", "AL-a1-gpt2"):
        GDRIVE / "andrew100ktest_df-AL-a1-gpt2_avg+ppp+ppos-fe-flat_2025-04-24-00-56.csv.features.gzip",
    ("norm-EFCAMDAT-test", "AL-a2-gpt2"):
        GDRIVE / "andrew100ktest_df-AL-a2-gpt2_avg+ppp+ppos-fe-flat_2025-04-24-00-56.csv.features.gzip",
    ("norm-EFCAMDAT-test", "AL-b1-gpt2"):
        GDRIVE / "andrew100ktest_df-AL-b1-gpt2_avg+ppp+ppos-fe-flat_2025-04-24-00-56.csv.features.gzip",
    ("norm-EFCAMDAT-test", "AL-b2-gpt2"):
        GDRIVE / "andrew100ktest_df-AL-b2-gpt2_avg+ppp+ppos-fe-flat_2025-04-24-00-56.csv.features.gzip",
    ("norm-EFCAMDAT-test", "AL-c1-gpt2"):
        GDRIVE / "andrew100ktest_df-AL-c1-gpt2_avg+ppp+ppos-fe-flat_2025-04-24-00-56.csv.features.gzip",

    # --- CELVA-SP ---
    ("norm-CELVA-SP", "gpt2"):
        GDRIVE / "celva-gpt2_avg+ppp+ppos-fe_2025-04-23-08-31.csv.features.gzip",
    ("norm-CELVA-SP", "AL-all-gpt2"):
        GDRIVE / "celva-AL-all-gpt2_avg+ppp+ppos-fe_2025-04-23-08-31.csv.features.gzip",
    ("norm-CELVA-SP", "AL-a1-gpt2"):
        GDRIVE / "celva-AL-a1-gpt2_avg+ppp+ppos-fe_2025-04-23-08-31.csv.features.gzip",
    ("norm-CELVA-SP", "AL-a2-gpt2"):
        GDRIVE / "celva-AL-a2-gpt2_avg+ppp+ppos-fe_2025-04-23-08-31.csv.features.gzip",
    ("norm-CELVA-SP", "AL-b1-gpt2"):
        GDRIVE / "celva-AL-b1-gpt2_avg+ppp+ppos-fe_2025-04-23-08-31.csv.features.gzip",
    ("norm-CELVA-SP", "AL-b2-gpt2"):
        GDRIVE / "celva-AL-b2-gpt2_avg+ppp+ppos-fe_2025-04-23-08-31.csv.features.gzip",
    ("norm-CELVA-SP", "AL-c1-gpt2"):
        GDRIVE / "celva-AL-c1-gpt2_avg+ppp+ppos-fe_2025-04-23-08-31.csv.features.gzip",

    # --- KUPA-KEYS ---
    ("norm-KUPA-KEYS", "gpt2"):
        GDRIVE / "KUPA-KEYS-gpt2_avg+ppp+ppos-fe_2025-04-23-06-04.csv.features.gzip",
    ("norm-KUPA-KEYS", "AL-all-gpt2"):
        GDRIVE / "KUPA-KEYS-AL-all-gpt2_avg+ppp+ppos-fe_2025-04-23-06-04.csv.features.gzip",
    ("norm-KUPA-KEYS", "AL-a1-gpt2"):
        GDRIVE / "KUPA-KEYS-AL-a1-gpt2_avg+ppp+ppos-fe_2025-04-23-06-04.csv.features.gzip",
    ("norm-KUPA-KEYS", "AL-a2-gpt2"):
        GDRIVE / "KUPA-KEYS-AL-a2-gpt2_avg+ppp+ppos-fe_2025-04-23-06-04.csv.features.gzip",
    ("norm-KUPA-KEYS", "AL-b1-gpt2"):
        GDRIVE / "KUPA-KEYS-AL-b1-gpt2_avg+ppp+ppos-fe_2025-04-23-06-04.csv.features.gzip",
    ("norm-KUPA-KEYS", "AL-b2-gpt2"):
        GDRIVE / "KUPA-KEYS-AL-b2-gpt2_avg+ppp+ppos-fe_2025-04-23-06-04.csv.features.gzip",
    ("norm-KUPA-KEYS", "AL-c1-gpt2"):
        GDRIVE / "KUPA-KEYS-AL-c1-gpt2_avg+ppp+ppos-fe_2025-04-23-06-04.csv.features.gzip",
}

DATASETS = [
    "norm-EFCAMDAT-train", "norm-EFCAMDAT-test",
    "norm-CELVA-SP", "norm-KUPA-KEYS",
]

# Feature configurations to assemble
CONFIGS = {
    "native_only":    ["gpt2"],
    "native_general": ["gpt2", "AL-all-gpt2"],
    "all_models":     MODELS,
}


def load_features(path):
    """Load gzip features, drop index and non-numeric ID columns."""
    df = pd.read_csv(path, compression="gzip")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    # Non-flat files may contain writing_id; drop it
    if "writing_id" in df.columns:
        df = df.drop(columns=["writing_id"])
    return df


def assemble(config_name, model_list):
    """Assemble features for one configuration across all datasets."""
    for dataset in DATASETS:
        frames = []
        for model in model_list:
            key = (dataset, model)
            path = FILES.get(key)
            if path is None or not path.exists():
                print(f"  MISSING: {key} -> {path}")
                continue
            df = load_features(path)
            # Prefix columns with model name to avoid collisions
            df.columns = [f"{model}__{c}" for c in df.columns]
            frames.append(df)

        if not frames:
            print(f"  SKIP {config_name}/{dataset}: no feature files found")
            continue

        merged = pd.concat(frames, axis=1)
        out_dir = EXP_DIR / "features" / config_name / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "features_dense.csv"
        merged.to_csv(out_path, index=False)

        # Save feature names
        fn_path = out_dir / "feature_names.csv"
        pd.DataFrame({"feature_name": merged.columns}).to_csv(fn_path, index=False)
        print(f"  {config_name}/{dataset}: {merged.shape} -> {out_path}")


def main():
    print(f"GDRIVE: {GDRIVE}")
    print(f"EXP_DIR: {EXP_DIR}")

    for config_name, model_list in CONFIGS.items():
        print(f"\n=== {config_name} ({len(model_list)} model(s)) ===")
        assemble(config_name, model_list)

    print("\nDone.")


if __name__ == "__main__":
    main()
