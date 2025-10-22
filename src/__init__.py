"""
CEFR Classification Pipeline

A minimal scikit-learn compliant pipeline for CEFR text classification.

Main Modules
------------
- pipeline.py: Complete pipeline orchestration with multi-config support
- train_tfidf.py: TF-IDF vectorizer training
- extract_features.py: Feature extraction from pre-trained TF-IDF
- train_classifiers.py: ML classifier training
- predict.py: Prediction on test sets
- report.py: Results summarization and model ranking
- config.py: Centralized configuration system
- experiment_structure.py: Canonical directory structure reference

Quick Start
-----------
# Run full pipeline
python -m src.pipeline -e experiments/my-exp --cefr-column cefr_level

# Add new test set to existing experiment
python -m src.pipeline -e experiments/my-exp --add-test-set data/new-test.csv

See experiment_structure.py for complete directory structure documentation.
"""

__version__ = "1.0.0"
