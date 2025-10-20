"""
Centralized Configuration System for CEFR Classification Pipeline

This module provides a comprehensive dataclass-based configuration system
for the CEFR (Common European Framework of Reference for Languages)
text classification pipeline.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class TfidfConfig:
    """Configuration for TF-IDF vectorizer training."""

    max_features: int = 5000
    ngram_range: tuple = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    sublinear_tf: bool = True

    def __post_init__(self):
        if self.max_features <= 0:
            raise ValueError(f"max_features must be positive, got {self.max_features}")
        if (
            not isinstance(self.ngram_range, (tuple, list))
            or len(self.ngram_range) != 2
        ):
            raise ValueError("ngram_range must be a tuple of 2 integers")
        if self.min_df < 1:
            raise ValueError(f"min_df must be >= 1, got {self.min_df}")
        if not (0 < self.max_df <= 1.0):
            raise ValueError(f"max_df must be between 0 and 1, got {self.max_df}")

    def get_hash(self) -> str:
        """
        Generate a unique hash for this TF-IDF configuration.

        This ensures that different TF-IDF configurations don't overwrite each other.
        The hash is based on all configuration parameters.

        Returns:
            8-character hex string uniquely identifying this configuration
        """
        # Create a canonical string representation of the config
        config_str = (
            f"max_features={self.max_features},"
            f"ngram_range={self.ngram_range},"
            f"min_df={self.min_df},"
            f"max_df={self.max_df},"
            f"sublinear_tf={self.sublinear_tf}"
        )

        # Generate SHA256 hash and take first 8 characters
        hash_obj = hashlib.sha256(config_str.encode("utf-8"))
        return hash_obj.hexdigest()[:8]

    def get_readable_name(self) -> str:
        """
        Generate a human-readable name for this TF-IDF configuration.

        Returns:
            Readable string like "tfidf_5000_ngram1-2"
        """
        ngram_str = f"{self.ngram_range[0]}-{self.ngram_range[1]}"
        return f"tfidf_{self.max_features}_ngram{ngram_str}"


@dataclass
class ClassifierConfig:
    """Configuration for ML classifier training."""

    classifier_type: str = "multinomialnb"

    # Classifier-specific parameters
    logistic_max_iter: int = 1000
    logistic_class_weight: str = "balanced"

    rf_n_estimators: int = 100
    rf_class_weight: str = "balanced"

    svm_max_iter: int = 2000
    svm_class_weight: str = "balanced"

    # XGBoost parameters
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.3
    xgb_use_gpu: bool = False
    xgb_tree_method: str = "auto"  # auto, gpu_hist, hist, exact

    random_state: int = 42

    def __post_init__(self):
        valid_classifiers = [
            "multinomialnb",
            "logistic",
            "randomforest",
            "svm",
            "xgboost",
        ]
        if self.classifier_type not in valid_classifiers:
            raise ValueError(
                f"classifier_type must be one of {valid_classifiers}, got {self.classifier_type}"
            )


@dataclass
class ExperimentConfig:
    """Configuration for experiment directory and data paths."""

    experiment_dir: str = "data/experiments/zero-shot"

    # Auto-derived paths (computed in __post_init__)
    features_training_dir: Optional[str] = None
    ml_training_dir: Optional[str] = None
    ml_test_dir: Optional[str] = None
    models_dir: Optional[str] = None
    features_output_dir: Optional[str] = None
    results_dir: Optional[str] = None

    # Optional: custom pre-trained model directory
    pretrained_tfidf_dir: Optional[str] = None

    def __post_init__(self):
        exp_path = Path(self.experiment_dir)
        if not exp_path.exists():
            raise ValueError(f"Experiment directory not found: {self.experiment_dir}")

        # Auto-derive paths
        self.features_training_dir = str(exp_path / "features-training-data")
        self.ml_training_dir = str(exp_path / "ml-training-data")
        self.ml_test_dir = str(exp_path / "ml-test-data")
        if self.models_dir is None:
            self.models_dir = str(exp_path / "feature-models")
        self.features_output_dir = str(exp_path / "features")
        self.results_dir = str(exp_path / "results")

    def get_feature_model_dir(self, config_hash: str, feature_type: str) -> str:
        """
        Get the feature model directory for a specific configuration.

        New flexible naming: {hash}_{feature_type}
        Examples:
            - 252cd532_tfidf
            - 84cbc90c_tfidf_grouped
            - abc12345_bert
            - def67890_word2vec

        Args:
            config_hash: Configuration hash
            feature_type: Feature type identifier (e.g., "tfidf", "tfidf_grouped", "bert")

        Returns:
            Path to feature model directory (e.g., "feature-models/252cd532_tfidf")
        """
        model_name = f"{config_hash}_{feature_type}"
        return str(Path(self.models_dir) / model_name)

    def get_tfidf_model_dir(
        self, tfidf_config: "TfidfConfig", feature_type: str = "tfidf"
    ) -> str:
        """
        Get the TF-IDF model directory for a specific configuration.

        Args:
            tfidf_config: TF-IDF configuration
            feature_type: Type identifier (default: "tfidf", can be "tfidf_grouped", etc.)

        Returns:
            Path to TF-IDF model directory (e.g., "feature-models/252cd532_tfidf")
        """
        tfidf_hash = tfidf_config.get_hash()
        return self.get_feature_model_dir(tfidf_hash, feature_type)

    def get_features_dir(self, config_hash: str, feature_type: str) -> str:
        """
        Get the features directory for a specific feature configuration.

        Args:
            config_hash: Configuration hash
            feature_type: Feature type identifier

        Returns:
            Path to features directory (e.g., "features/252cd532_tfidf")
        """
        dir_name = f"{config_hash}_{feature_type}"
        return str(Path(self.features_output_dir) / dir_name)


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    text_column: str = "text"
    label_column: str = "label"
    cefr_column: str = "cefr_label"
    min_text_length: int = 0
    max_samples: Optional[int] = None
    random_sample: bool = False

    def __post_init__(self):
        if self.min_text_length < 0:
            raise ValueError(
                f"min_text_length must be non-negative, got {self.min_text_length}"
            )
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {self.max_samples}")


@dataclass
class OutputConfig:
    """Configuration for output and results."""

    save_config: bool = True
    save_models: bool = True
    save_features: bool = True
    save_results: bool = True
    verbose: bool = True
    overwrite: bool = False

    # Output format options
    save_csv: bool = True
    save_json: bool = True


@dataclass
class GlobalConfig:
    """Centralized global configuration combining all sub-configs."""

    experiment_config: ExperimentConfig = field(default_factory=ExperimentConfig)
    tfidf_config: TfidfConfig = field(default_factory=TfidfConfig)
    classifier_config: ClassifierConfig = field(default_factory=ClassifierConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    output_config: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self):
        # Validate sub-configs (their __post_init__ already ran)
        if not isinstance(self.experiment_config, ExperimentConfig):
            raise TypeError("experiment_config must be an ExperimentConfig instance")
        if not isinstance(self.tfidf_config, TfidfConfig):
            raise TypeError("tfidf_config must be a TfidfConfig instance")
        if not isinstance(self.classifier_config, ClassifierConfig):
            raise TypeError("classifier_config must be a ClassifierConfig instance")
        if not isinstance(self.data_config, DataConfig):
            raise TypeError("data_config must be a DataConfig instance")
        if not isinstance(self.output_config, OutputConfig):
            raise TypeError("output_config must be an OutputConfig instance")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GlobalConfig":
        """Create config from dictionary."""
        experiment_config = ExperimentConfig(**config_dict.get("experiment_config", {}))
        tfidf_config = TfidfConfig(**config_dict.get("tfidf_config", {}))
        classifier_config = ClassifierConfig(**config_dict.get("classifier_config", {}))
        data_config = DataConfig(**config_dict.get("data_config", {}))
        output_config = OutputConfig(**config_dict.get("output_config", {}))
        return cls(
            experiment_config,
            tfidf_config,
            classifier_config,
            data_config,
            output_config,
        )

    @classmethod
    def from_json_string(cls, json_string: str) -> "GlobalConfig":
        """Create config from JSON string."""
        config_dict = json.loads(json_string)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json_file(cls, json_path: str) -> "GlobalConfig":
        """Create config from JSON file."""
        with open(json_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_yaml_file(cls, yaml_path: str) -> "GlobalConfig":
        """Create config from YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "experiment_config": {
                "experiment_dir": self.experiment_config.experiment_dir,
            },
            "tfidf_config": {
                "max_features": self.tfidf_config.max_features,
                "ngram_range": list(self.tfidf_config.ngram_range),
                "min_df": self.tfidf_config.min_df,
                "max_df": self.tfidf_config.max_df,
                "sublinear_tf": self.tfidf_config.sublinear_tf,
            },
            "classifier_config": {
                "classifier_type": self.classifier_config.classifier_type,
                "logistic_max_iter": self.classifier_config.logistic_max_iter,
                "logistic_class_weight": self.classifier_config.logistic_class_weight,
                "rf_n_estimators": self.classifier_config.rf_n_estimators,
                "rf_class_weight": self.classifier_config.rf_class_weight,
                "svm_max_iter": self.classifier_config.svm_max_iter,
                "svm_class_weight": self.classifier_config.svm_class_weight,
                "xgb_n_estimators": self.classifier_config.xgb_n_estimators,
                "xgb_max_depth": self.classifier_config.xgb_max_depth,
                "xgb_learning_rate": self.classifier_config.xgb_learning_rate,
                "xgb_use_gpu": self.classifier_config.xgb_use_gpu,
                "xgb_tree_method": self.classifier_config.xgb_tree_method,
                "random_state": self.classifier_config.random_state,
            },
            "data_config": {
                "text_column": self.data_config.text_column,
                "label_column": self.data_config.label_column,
                "cefr_column": self.data_config.cefr_column,
                "min_text_length": self.data_config.min_text_length,
                "max_samples": self.data_config.max_samples,
                "random_sample": self.data_config.random_sample,
            },
            "output_config": {
                "save_config": self.output_config.save_config,
                "save_models": self.output_config.save_models,
                "save_features": self.output_config.save_features,
                "save_results": self.output_config.save_results,
                "verbose": self.output_config.verbose,
                "overwrite": self.output_config.overwrite,
                "save_csv": self.output_config.save_csv,
                "save_json": self.output_config.save_json,
            },
        }

    def save_json(self, path: str):
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_yaml(self, path: str):
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
