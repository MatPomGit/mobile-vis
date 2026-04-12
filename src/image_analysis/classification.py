"""Image classification utilities.

Provides a thin wrapper around a classification model so that the rest of
the codebase does not depend on a specific deep-learning framework.
Replace the stub implementations with your actual model.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .backends import ClassifierBackend, InferenceResult
from .types import Image
from .utils import validate_image

logger = logging.getLogger(__name__)

# Minimum confidence required before returning a prediction.
CLASSIFICATION_CONFIDENCE_THRESHOLD: float = 0.5

# Nazwa domyślnego backendu klasyfikacji obrazu.
DEFAULT_CLASSIFIER_BACKEND: str = "stub"

# Ujednolicony alias typu wyniku klasyfikacji.
ClassificationResult = InferenceResult


class StubClassifierBackend:
    """No-op classifier backend used as safe default fallback."""

    def classify(
        self,
        image: Image,
        confidence_threshold: float,
    ) -> ClassificationResult:
        """Return unknown class when no real model backend is configured."""
        _ = image
        _ = confidence_threshold
        return ClassificationResult(
            bbox=None,
            label="unknown",
            score=0.0,
            metadata={"backend": DEFAULT_CLASSIFIER_BACKEND},
        )


# Rejestr backendów klasyfikatora: klucz tekstowy -> fabryka backendu.
CLASSIFIER_BACKENDS: dict[str, Callable[[], ClassifierBackend]] = {
    DEFAULT_CLASSIFIER_BACKEND: StubClassifierBackend,
}


def register_classifier_backend(name: str, factory: Callable[[], ClassifierBackend]) -> None:
    """Register a classifier backend factory under a unique key."""
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError("Backend name must not be empty")

    # Walidujemy fabrykę przez próbę utworzenia instancji backendu.
    backend = factory()
    if not hasattr(backend, "classify"):
        raise TypeError("Classifier backend must implement classify(image, confidence_threshold)")

    CLASSIFIER_BACKENDS[normalized_name] = factory


def create_classifier_backend(name: str | None = None) -> ClassifierBackend:
    """Create classifier backend instance with fallback to default backend."""
    requested_name = (name or DEFAULT_CLASSIFIER_BACKEND).strip().lower()
    factory = CLASSIFIER_BACKENDS.get(requested_name)

    if factory is None:
        logger.warning(
            "Unknown classifier backend '%s'. Falling back to '%s'.",
            requested_name,
            DEFAULT_CLASSIFIER_BACKEND,
        )
        factory = CLASSIFIER_BACKENDS[DEFAULT_CLASSIFIER_BACKEND]

    return factory()


def classify_image(
    image: Image,
    model: object | None = None,
    confidence_threshold: float = CLASSIFICATION_CONFIDENCE_THRESHOLD,
    backend: str | ClassifierBackend | None = None,
) -> ClassificationResult:
    """Classify *image* and return unified prediction result.

    Args:
        image: Input image with shape ``(H, W, C)`` (typically ``C=3`` BGR)
            or ``(H, W)`` grayscale. Dtype must be ``uint8`` in ``[0, 255]``
            or ``float32`` in ``[0.0, 1.0]``.
        model: Optional pre-loaded classifier (reserved for custom backends).
        confidence_threshold: Minimum confidence required for a valid
            prediction. If the top prediction is below this value,
            ``"unknown"`` is returned.
        backend: Backend name, backend instance, or ``None`` (default backend).

    Returns:
        :class:`ClassificationResult` with fields ``bbox``, ``label``,
        ``score`` and ``metadata``.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *confidence_threshold* is outside ``[0.0, 1.0]``.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image).__name__}")
    validate_image(image)
    if not (0.0 <= confidence_threshold <= 1.0):
        raise ValueError(f"confidence_threshold must be in [0.0, 1.0], got {confidence_threshold}")

    # Parametr model pozostawiamy dla zgodności z wcześniejszym API funkcji.
    _ = model

    # Obsługujemy zarówno nazwę backendu, jak i gotową instancję strategii.
    selected_backend = create_classifier_backend(backend) if isinstance(backend, str) else backend
    if selected_backend is None:
        selected_backend = create_classifier_backend()

    prediction = selected_backend.classify(image, confidence_threshold)
    if prediction.score < confidence_threshold:
        logger.debug(
            "Top prediction score %.4f is below threshold %.4f; returning 'unknown'",
            prediction.score,
            confidence_threshold,
        )
        return ClassificationResult(
            bbox=prediction.bbox,
            label="unknown",
            score=0.0,
            metadata=prediction.metadata,
        )

    logger.debug("Classified image as '%s' with score %.4f", prediction.label, prediction.score)
    return prediction


def load_classifier(model_path: str | Path) -> object:
    """Load a classifier model from *model_path*.

    This is a **stub** implementation. Replace with framework-specific
    loading logic (e.g. ``torch.load``, ``tf.keras.models.load_model``).

    Args:
        model_path: Path to the serialised model file.

    Returns:
        Loaded model object.

    Raises:
        FileNotFoundError: If *model_path* does not exist.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # TODO(#issue-number): Implement model loading.
    logger.info("Loading classifier from '%s'", model_path)
    raise NotImplementedError("load_classifier is not yet implemented")


def evaluate_classifier(
    predictions: list[tuple[str, float]],
    ground_truth: list[str],
) -> dict[str, float]:
    """Compute accuracy metrics for a batch of predictions.

    Args:
        predictions: List of ``(label, confidence)`` tuples.
        ground_truth: Corresponding ground-truth class labels.

    Returns:
        Dictionary with metric names as keys and float values, e.g.
        ``{"accuracy": 0.95, "avg_confidence": 0.87}``.

    Raises:
        ValueError: If *predictions* and *ground_truth* have different lengths.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"predictions and ground_truth must have the same length, "
            f"got {len(predictions)} and {len(ground_truth)}"
        )
    for label, confidence in predictions:
        if not isinstance(label, str) or not label:
            raise ValueError("Prediction label must not be empty")
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"Prediction confidence must be in [0.0, 1.0], got {confidence}")

    for label in ground_truth:
        if not isinstance(label, str) or not label:
            raise ValueError("Ground-truth labels must be non-empty strings")

    if not predictions:
        return {"accuracy": 0.0, "avg_confidence": 0.0}

    correct = sum(
        pred_label == truth
        for (pred_label, _), truth in zip(predictions, ground_truth, strict=True)
    )
    accuracy = correct / len(predictions)
    avg_confidence = sum(conf for _, conf in predictions) / len(predictions)

    return {"accuracy": accuracy, "avg_confidence": avg_confidence}


# Rejestr publicznych symboli modułu używany przez image_analysis.__init__.
PUBLIC_EXPORTS: dict[str, str] = {
    "CLASSIFICATION_CONFIDENCE_THRESHOLD": "CLASSIFICATION_CONFIDENCE_THRESHOLD",
    "CLASSIFIER_BACKENDS": "CLASSIFIER_BACKENDS",
    "ClassificationResult": "ClassificationResult",
    "DEFAULT_CLASSIFIER_BACKEND": "DEFAULT_CLASSIFIER_BACKEND",
    "classify_image": "classify_image",
    "create_classifier_backend": "create_classifier_backend",
    "evaluate_classifier": "evaluate_classifier",
    "load_classifier": "load_classifier",
    "register_classifier_backend": "register_classifier_backend",
}
