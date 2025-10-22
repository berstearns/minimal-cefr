"""
Domain Types for CEFR Classification System

This module defines the core domain model for the CEFR (Common European Framework
of Reference for Languages) text classification system using Domain-Driven Design
principles.

The domain model is organized into:
- Value Objects: Immutable objects with no identity (CEFR levels, features)
- Entities: Objects with identity (text samples, models)
- Aggregates: Clusters of entities and value objects (datasets, experiments)
- Domain Events: Significant occurrences in the domain

This module is currently isolated and not integrated into the main codebase.
It serves as a reference implementation of the domain model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, NewType, Optional, Protocol, Tuple, Union
from uuid import UUID, uuid4

import numpy as np
import numpy.typing as npt

# ============================================================================
# Type Aliases
# ============================================================================

# Primitive type aliases for better semantic meaning
TextContent = NewType("TextContent", str)
FeatureArray = npt.NDArray[np.float64]
LabelVector = npt.NDArray[np.int64]
ProbabilityVector = npt.NDArray[np.float64]
ConfusionMatrix = npt.NDArray[np.int64]


# ============================================================================
# Enums - Value Objects representing discrete choices
# ============================================================================


class CEFRLevel(str, Enum):
    """
    CEFR proficiency levels.

    The Common European Framework of Reference for Languages defines six levels
    of language proficiency from beginner (A1) to proficient (C2).
    """

    A1 = "A1"  # Beginner
    A2 = "A2"  # Elementary
    B1 = "B1"  # Intermediate
    B2 = "B2"  # Upper Intermediate
    C1 = "C1"  # Advanced
    C2 = "C2"  # Proficient

    @property
    def ordinal_value(self) -> int:
        """Return the ordinal position (0-5) for ordinal regression."""
        return list(CEFRLevel).index(self)

    @classmethod
    def from_ordinal(cls, value: int) -> CEFRLevel:
        """Create CEFR level from ordinal value (0-5)."""
        levels = list(cls)
        if not 0 <= value < len(levels):
            raise ValueError(f"Ordinal value must be 0-{len(levels)-1}, got {value}")
        return levels[value]

    def is_adjacent_to(self, other: CEFRLevel) -> bool:
        """Check if two levels are adjacent (e.g., A2 and B1)."""
        return abs(self.ordinal_value - other.ordinal_value) == 1

    def distance_from(self, other: CEFRLevel) -> int:
        """Calculate ordinal distance between levels."""
        return abs(self.ordinal_value - other.ordinal_value)


class ClassifierType(str, Enum):
    """Types of classifiers supported in the system."""

    MULTINOMIAL_NB = "multinomialnb"
    LOGISTIC_REGRESSION = "logistic"
    RANDOM_FOREST = "randomforest"
    SVM = "svm"
    XGBOOST = "xgboost"
    MORD_LOGISTIC = "mord-lr"  # Ordinal regression
    BERT = "bert"
    GPT2 = "gpt2"
    ROBERTA = "roberta"
    DISTILBERT = "distilbert"


class FeatureType(str, Enum):
    """Types of features that can be extracted from text."""

    TFIDF = "tfidf"
    TFIDF_GROUPED = "tfidf_grouped"
    BERT_EMBEDDINGS = "bert"
    GPT2_EMBEDDINGS = "gpt2"
    PERPLEXITY = "perplexity"
    LINGUISTIC = "linguistic"
    COMBINED = "combined"


class DataSplit(str, Enum):
    """Data split types for machine learning."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class PredictionStrategy(str, Enum):
    """Strategies for converting model outputs to predictions."""

    ARGMAX = "argmax"  # Take class with highest probability
    ROUNDED_AVERAGE = "rounded_avg"  # Round probability-weighted average
    SOFT = "soft"  # Keep full probability distribution
    THRESHOLD = "threshold"  # Apply custom thresholds


class MetricType(str, Enum):
    """Types of evaluation metrics."""

    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1"
    ADJACENT_ACCURACY = "adjacent_accuracy"
    QWK = "quadratic_weighted_kappa"  # Quadratic Weighted Kappa
    MSE = "mse"  # Mean Squared Error (ordinal)
    MAE = "mae"  # Mean Absolute Error (ordinal)
    CONFUSION_MATRIX = "confusion_matrix"


# ============================================================================
# Value Objects - Immutable objects with no identity
# ============================================================================


@dataclass(frozen=True)
class CEFRLabel:
    """
    Value object representing a CEFR label with metadata.

    Immutable container for a CEFR level with confidence and provenance.
    """

    level: CEFRLevel
    confidence: float = 1.0  # Confidence in the label (0.0 to 1.0)
    source: str = "manual"  # Source of the label (manual, predicted, etc.)

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.confidence}")


@dataclass(frozen=True)
class FeatureVector:
    """
    Value object representing extracted features from text.

    Contains the feature values, feature names, and metadata about extraction.
    """

    features: npt.NDArray[np.float64]
    feature_names: Tuple[str, ...]
    feature_type: FeatureType
    extraction_timestamp: datetime = field(default_factory=datetime.utcnow)
    config_hash: Optional[str] = None

    def __post_init__(self):
        if len(self.features) != len(self.feature_names):
            raise ValueError(
                f"Feature count mismatch: {len(self.features)} features "
                f"but {len(self.feature_names)} names"
            )

    @property
    def dimension(self) -> int:
        """Get the dimensionality of the feature vector."""
        return len(self.features)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary mapping feature names to values."""
        return dict(zip(self.feature_names, self.features))


@dataclass(frozen=True)
class PredictionResult:
    """
    Value object representing a prediction from a classifier.

    Contains predicted label, probabilities, and metadata.
    """

    predicted_label: CEFRLevel
    probabilities: Dict[CEFRLevel, float]
    strategy: PredictionStrategy
    confidence: float
    model_name: str
    prediction_timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        # Validate probabilities sum to ~1.0
        total_prob = sum(self.probabilities.values())
        if not (0.99 <= total_prob <= 1.01):
            raise ValueError(f"Probabilities must sum to 1.0, got {total_prob:.4f}")

        # Validate confidence
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.confidence}")

    def get_top_k_predictions(self, k: int = 3) -> List[Tuple[CEFRLevel, float]]:
        """Get top-k predicted levels with probabilities."""
        sorted_probs = sorted(
            self.probabilities.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_probs[:k]

    def is_correct(self, true_label: CEFRLevel) -> bool:
        """Check if prediction matches true label."""
        return self.predicted_label == true_label

    def is_adjacent_correct(self, true_label: CEFRLevel) -> bool:
        """Check if prediction is within one level of true label."""
        return (
            self.predicted_label == true_label
            or self.predicted_label.is_adjacent_to(true_label)
        )


@dataclass(frozen=True)
class EvaluationMetrics:
    """
    Value object containing evaluation metrics for a model.

    Comprehensive metrics including standard and CEFR-specific measures.
    """

    accuracy: float
    adjacent_accuracy: float
    macro_f1: float
    weighted_f1: float
    qwk: Optional[float] = None  # Quadratic Weighted Kappa
    mse: Optional[float] = None  # Mean Squared Error (ordinal)
    mae: Optional[float] = None  # Mean Absolute Error (ordinal)
    per_class_metrics: Optional[Dict[CEFRLevel, Dict[str, float]]] = None
    confusion_matrix: Optional[ConfusionMatrix] = None

    def __post_init__(self):
        # Validate metric ranges
        for metric_name in ["accuracy", "adjacent_accuracy", "macro_f1", "weighted_f1"]:
            value = getattr(self, metric_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{metric_name} must be 0.0-1.0, got {value}")


@dataclass(frozen=True)
class TFIDFConfiguration:
    """
    Value object for TF-IDF vectorizer configuration.

    Defines all hyperparameters for TF-IDF feature extraction.
    """

    max_features: int = 5000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    sublinear_tf: bool = True

    def __post_init__(self):
        if self.max_features <= 0:
            raise ValueError(f"max_features must be positive, got {self.max_features}")
        if self.ngram_range[0] > self.ngram_range[1]:
            raise ValueError(f"Invalid ngram_range: {self.ngram_range}")
        if not 0.0 < self.max_df <= 1.0:
            raise ValueError(f"max_df must be 0.0-1.0, got {self.max_df}")

    def get_hash(self) -> str:
        """Generate unique hash for this configuration."""
        import hashlib

        config_str = (
            f"{self.max_features}_{self.ngram_range}_{self.min_df}_"
            f"{self.max_df}_{self.sublinear_tf}"
        )
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


@dataclass(frozen=True)
class ModelConfiguration:
    """
    Value object for model training configuration.

    Contains hyperparameters and settings for classifier training.
    """

    classifier_type: ClassifierType
    random_state: int = 42
    hyperparameters: Dict[str, Union[int, float, str, bool]] = field(
        default_factory=dict
    )

    def get_identifier(self) -> str:
        """Get unique identifier for this configuration."""
        import hashlib

        config_str = (
            f"{self.classifier_type.value}_{sorted(self.hyperparameters.items())}"
        )
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


# ============================================================================
# Entities - Objects with identity
# ============================================================================


@dataclass
class TextSample:
    """
    Entity representing a single text sample with CEFR annotation.

    Has identity (sample_id) and can be mutated (e.g., label updates).
    """

    sample_id: UUID
    text: TextContent
    true_label: Optional[CEFRLabel] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    features: Optional[FeatureVector] = None
    predictions: List[PredictionResult] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def add_features(self, features: FeatureVector) -> None:
        """Add extracted features to this sample."""
        self.features = features

    def add_prediction(self, prediction: PredictionResult) -> None:
        """Add a prediction result to this sample."""
        self.predictions.append(prediction)

    def get_latest_prediction(self) -> Optional[PredictionResult]:
        """Get the most recent prediction."""
        if not self.predictions:
            return None
        return max(self.predictions, key=lambda p: p.prediction_timestamp)

    @property
    def has_label(self) -> bool:
        """Check if sample has a true label."""
        return self.true_label is not None

    @property
    def word_count(self) -> int:
        """Get word count of the text."""
        return len(self.text.split())


@dataclass
class TrainedModel:
    """
    Entity representing a trained classifier model.

    Has identity (model_id) and persisted state.
    """

    model_id: UUID
    model_name: str
    classifier_type: ClassifierType
    configuration: ModelConfiguration
    feature_type: FeatureType
    tfidf_config_hash: Optional[str] = None
    training_dataset_id: Optional[UUID] = None
    trained_at: datetime = field(default_factory=datetime.utcnow)
    training_metrics: Optional[EvaluationMetrics] = None
    model_path: Optional[Path] = None

    def get_display_name(self) -> str:
        """Get human-readable display name."""
        parts = [
            self.model_name,
            self.classifier_type.value,
        ]
        if self.tfidf_config_hash:
            parts.append(f"{self.tfidf_config_hash}_{self.feature_type.value}")
        return "_".join(parts)


@dataclass
class FeatureExtractor:
    """
    Entity representing a feature extraction model (e.g., TF-IDF vectorizer).

    Has identity and persisted state.
    """

    extractor_id: UUID
    feature_type: FeatureType
    configuration: Union[TFIDFConfiguration, Dict]
    config_hash: str
    trained_at: datetime = field(default_factory=datetime.utcnow)
    vocabulary_size: Optional[int] = None
    model_path: Optional[Path] = None


# ============================================================================
# Aggregates - Clusters of entities and value objects
# ============================================================================


@dataclass
class Dataset:
    """
    Aggregate representing a collection of text samples.

    Root entity for managing collections of samples with shared characteristics.
    """

    dataset_id: UUID
    name: str
    split: DataSplit
    samples: List[TextSample] = field(default_factory=list)
    feature_type: Optional[FeatureType] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, str] = field(default_factory=dict)

    def add_sample(self, sample: TextSample) -> None:
        """Add a sample to the dataset."""
        self.samples.append(sample)

    def remove_sample(self, sample_id: UUID) -> bool:
        """Remove a sample by ID. Returns True if found and removed."""
        initial_len = len(self.samples)
        self.samples = [s for s in self.samples if s.sample_id != sample_id]
        return len(self.samples) < initial_len

    @property
    def size(self) -> int:
        """Get number of samples in dataset."""
        return len(self.samples)

    @property
    def labeled_samples(self) -> List[TextSample]:
        """Get only samples with labels."""
        return [s for s in self.samples if s.has_label]

    @property
    def label_distribution(self) -> Dict[CEFRLevel, int]:
        """Get distribution of labels in dataset."""
        from collections import Counter

        labels = [s.true_label.level for s in self.labeled_samples]
        return dict(Counter(labels))

    def get_samples_by_level(self, level: CEFRLevel) -> List[TextSample]:
        """Get all samples with a specific CEFR level."""
        return [
            s
            for s in self.labeled_samples
            if s.true_label and s.true_label.level == level
        ]

    def has_features_extracted(self) -> bool:
        """Check if all samples have features extracted."""
        return all(s.features is not None for s in self.samples)


@dataclass
class ExperimentRun:
    """
    Aggregate representing a complete ML experiment run.

    Root entity for managing the entire lifecycle of a classification experiment.
    """

    experiment_id: UUID
    name: str
    experiment_dir: Path
    training_dataset: Dataset
    test_datasets: List[Dataset] = field(default_factory=list)
    feature_extractor: Optional[FeatureExtractor] = None
    trained_models: List[TrainedModel] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    status: str = "initialized"  # initialized, running, completed, failed
    configuration: Dict[str, any] = field(default_factory=dict)

    def add_test_dataset(self, dataset: Dataset) -> None:
        """Add a test dataset to the experiment."""
        if dataset.split != DataSplit.TEST:
            raise ValueError(f"Dataset must be TEST split, got {dataset.split}")
        self.test_datasets.append(dataset)

    def add_trained_model(self, model: TrainedModel) -> None:
        """Add a trained model to the experiment."""
        self.trained_models.append(model)

    def mark_completed(self) -> None:
        """Mark experiment as completed."""
        self.status = "completed"
        self.completed_at = datetime.utcnow()

    def mark_failed(self) -> None:
        """Mark experiment as failed."""
        self.status = "failed"
        self.completed_at = datetime.utcnow()

    @property
    def is_completed(self) -> bool:
        """Check if experiment is completed."""
        return self.status == "completed"

    @property
    def duration(self) -> Optional[float]:
        """Get experiment duration in seconds."""
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()


@dataclass
class PredictionBatch:
    """
    Aggregate representing a batch of predictions on a dataset.

    Contains predictions from a specific model on a specific dataset.
    """

    batch_id: UUID
    model: TrainedModel
    dataset: Dataset
    predictions: List[Tuple[TextSample, PredictionResult]] = field(default_factory=list)
    evaluation_metrics: Optional[EvaluationMetrics] = None
    predicted_at: datetime = field(default_factory=datetime.utcnow)

    def add_prediction(self, sample: TextSample, prediction: PredictionResult) -> None:
        """Add a prediction for a sample."""
        self.predictions.append((sample, prediction))

    @property
    def size(self) -> int:
        """Get number of predictions in batch."""
        return len(self.predictions)

    def compute_metrics(self) -> EvaluationMetrics:
        """Compute evaluation metrics for this batch."""
        # This would call domain services to compute metrics
        # Placeholder implementation
        correct = sum(
            1
            for sample, pred in self.predictions
            if sample.true_label and pred.is_correct(sample.true_label.level)
        )
        accuracy = correct / len(self.predictions) if self.predictions else 0.0

        adjacent_correct = sum(
            1
            for sample, pred in self.predictions
            if sample.true_label and pred.is_adjacent_correct(sample.true_label.level)
        )
        adjacent_accuracy = (
            adjacent_correct / len(self.predictions) if self.predictions else 0.0
        )

        metrics = EvaluationMetrics(
            accuracy=accuracy,
            adjacent_accuracy=adjacent_accuracy,
            macro_f1=0.0,  # Would compute properly
            weighted_f1=0.0,  # Would compute properly
        )

        self.evaluation_metrics = metrics
        return metrics


# ============================================================================
# Domain Events - Significant occurrences in the domain
# ============================================================================


@dataclass(frozen=True)
class DomainEvent:
    """Base class for all domain events."""

    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=datetime.utcnow)
    aggregate_id: UUID = field(default=uuid4())


@dataclass(frozen=True)
class ModelTrainedEvent(DomainEvent):
    """Event fired when a model is successfully trained."""

    model_id: UUID = field(default=uuid4())
    model_name: str = ""
    classifier_type: ClassifierType = ClassifierType.XGBOOST
    training_accuracy: float = 0.0


@dataclass(frozen=True)
class PredictionMadeEvent(DomainEvent):
    """Event fired when predictions are made on a dataset."""

    batch_id: UUID = field(default=uuid4())
    model_id: UUID = field(default=uuid4())
    dataset_id: UUID = field(default=uuid4())
    num_predictions: int = 0


@dataclass(frozen=True)
class DatasetCreatedEvent(DomainEvent):
    """Event fired when a new dataset is created."""

    dataset_id: UUID = field(default=uuid4())
    dataset_name: str = ""
    split: DataSplit = DataSplit.TRAIN
    num_samples: int = 0


@dataclass(frozen=True)
class FeaturesExtractedEvent(DomainEvent):
    """Event fired when features are extracted for a dataset."""

    dataset_id: UUID = field(default=uuid4())
    feature_type: FeatureType = FeatureType.TFIDF
    num_samples: int = 0
    feature_dimension: int = 0


@dataclass(frozen=True)
class ExperimentCompletedEvent(DomainEvent):
    """Event fired when an experiment run completes."""

    experiment_id: UUID = field(default=uuid4())
    num_models_trained: int = 0
    num_datasets_evaluated: int = 0
    duration_seconds: float = 0.0


# ============================================================================
# Protocols - Interfaces for domain services
# ============================================================================


class FeatureExtractorProtocol(Protocol):
    """Protocol for feature extraction services."""

    def extract(self, text: TextContent) -> FeatureVector:
        """Extract features from text."""
        ...

    def extract_batch(self, texts: List[TextContent]) -> List[FeatureVector]:
        """Extract features from multiple texts."""
        ...


class ClassifierProtocol(Protocol):
    """Protocol for classifier services."""

    def train(
        self, features: List[FeatureVector], labels: List[CEFRLabel]
    ) -> TrainedModel:
        """Train a classifier on features and labels."""
        ...

    def predict(self, model: TrainedModel, features: FeatureVector) -> PredictionResult:
        """Make a prediction using a trained model."""
        ...

    def predict_batch(
        self, model: TrainedModel, features: List[FeatureVector]
    ) -> List[PredictionResult]:
        """Make predictions on a batch of features."""
        ...


class EvaluatorProtocol(Protocol):
    """Protocol for evaluation services."""

    def evaluate(
        self, predictions: List[PredictionResult], true_labels: List[CEFRLabel]
    ) -> EvaluationMetrics:
        """Evaluate predictions against true labels."""
        ...

    def compute_confusion_matrix(
        self, predictions: List[CEFRLevel], true_labels: List[CEFRLevel]
    ) -> ConfusionMatrix:
        """Compute confusion matrix."""
        ...


class ExperimentOrchestratorProtocol(Protocol):
    """Protocol for experiment orchestration services."""

    def run_experiment(self, experiment: ExperimentRun) -> ExperimentRun:
        """Execute a complete experiment run."""
        ...

    def save_experiment(self, experiment: ExperimentRun) -> Path:
        """Persist experiment to disk."""
        ...

    def load_experiment(self, experiment_dir: Path) -> ExperimentRun:
        """Load experiment from disk."""
        ...


# ============================================================================
# Repository Interfaces
# ============================================================================


class DatasetRepository(Protocol):
    """Repository for managing dataset persistence."""

    def save(self, dataset: Dataset) -> None:
        """Save a dataset."""
        ...

    def find_by_id(self, dataset_id: UUID) -> Optional[Dataset]:
        """Find dataset by ID."""
        ...

    def find_by_name(self, name: str) -> Optional[Dataset]:
        """Find dataset by name."""
        ...

    def list_all(self) -> List[Dataset]:
        """List all datasets."""
        ...


class ModelRepository(Protocol):
    """Repository for managing trained model persistence."""

    def save(self, model: TrainedModel) -> None:
        """Save a trained model."""
        ...

    def find_by_id(self, model_id: UUID) -> Optional[TrainedModel]:
        """Find model by ID."""
        ...

    def find_by_name(self, name: str) -> Optional[TrainedModel]:
        """Find model by name."""
        ...

    def list_all(self) -> List[TrainedModel]:
        """List all trained models."""
        ...


class ExperimentRepository(Protocol):
    """Repository for managing experiment persistence."""

    def save(self, experiment: ExperimentRun) -> None:
        """Save an experiment."""
        ...

    def find_by_id(self, experiment_id: UUID) -> Optional[ExperimentRun]:
        """Find experiment by ID."""
        ...

    def find_by_name(self, name: str) -> Optional[ExperimentRun]:
        """Find experiment by name."""
        ...

    def list_all(self) -> List[ExperimentRun]:
        """List all experiments."""
        ...


# ============================================================================
# Factory Functions
# ============================================================================


def create_text_sample(
    text: str,
    label: Optional[CEFRLevel] = None,
    confidence: float = 1.0,
    metadata: Optional[Dict[str, str]] = None,
) -> TextSample:
    """
    Factory function to create a TextSample entity.

    Args:
        text: The text content
        label: Optional CEFR level
        confidence: Confidence in the label
        metadata: Optional metadata dictionary

    Returns:
        TextSample entity
    """
    cefr_label = CEFRLabel(level=label, confidence=confidence) if label else None
    return TextSample(
        sample_id=uuid4(),
        text=TextContent(text),
        true_label=cefr_label,
        metadata=metadata or {},
    )


def create_dataset(
    name: str, split: DataSplit, samples: Optional[List[TextSample]] = None
) -> Dataset:
    """
    Factory function to create a Dataset aggregate.

    Args:
        name: Dataset name
        split: Data split type
        samples: Optional list of samples

    Returns:
        Dataset aggregate
    """
    return Dataset(dataset_id=uuid4(), name=name, split=split, samples=samples or [])


def create_experiment(
    name: str, experiment_dir: Path, training_dataset: Dataset
) -> ExperimentRun:
    """
    Factory function to create an ExperimentRun aggregate.

    Args:
        name: Experiment name
        experiment_dir: Directory to store experiment artifacts
        training_dataset: Training dataset

    Returns:
        ExperimentRun aggregate
    """
    return ExperimentRun(
        experiment_id=uuid4(),
        name=name,
        experiment_dir=experiment_dir,
        training_dataset=training_dataset,
    )


# ============================================================================
# Utility Functions
# ============================================================================


def parse_cefr_level(level_str: str) -> CEFRLevel:
    """
    Parse a string into a CEFR level.

    Args:
        level_str: String representation (e.g., "A1", "C2")

    Returns:
        CEFRLevel enum

    Raises:
        ValueError: If string doesn't match a valid CEFR level
    """
    try:
        return CEFRLevel(level_str.upper())
    except ValueError:
        valid_levels = [level.value for level in CEFRLevel]
        raise ValueError(
            f"Invalid CEFR level: {level_str}. Valid levels: {valid_levels}"
        )


def get_all_cefr_levels() -> List[CEFRLevel]:
    """Get all CEFR levels in order."""
    return list(CEFRLevel)


def is_valid_ordinal_value(value: int) -> bool:
    """Check if an integer is a valid CEFR ordinal value (0-5)."""
    return 0 <= value < len(CEFRLevel)
