"""Image classification utilities.

Provides a thin wrapper around a classification model so that the rest of
the codebase does not depend on a specific deep-learning framework.
Replace the stub implementations with your actual model.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Minimum confidence required before returning a prediction.
CLASSIFICATION_CONFIDENCE_THRESHOLD: float = 0.5


def classify_image(
    image: np.ndarray,
    model: object | None = None,
    confidence_threshold: float = CLASSIFICATION_CONFIDENCE_THRESHOLD,
) -> tuple[str, float]:
    """Classify *image* and return the top predicted label and confidence.

    This is a **stub** implementation that always returns ``("unknown", 0.0)``.
    Replace the body with your model's inference call.

    Args:
        image: BGR or normalised float32 image array.
        model: Optional pre-loaded classifier.  If ``None``, a default
            model should be loaded inside the implementation.
        confidence_threshold: Minimum confidence required for a valid
            prediction.  If the top prediction is below this value,
            ``"unknown"`` is returned.

    Returns:
        Tuple of ``(label, confidence)`` where *confidence* is in
        ``[0.0, 1.0]``.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *confidence_threshold* is outside ``[0.0, 1.0]``.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image).__name__}")
    if not (0.0 <= confidence_threshold <= 1.0):
        raise ValueError(
            f"confidence_threshold must be in [0.0, 1.0], got {confidence_threshold}"
        )

    # TODO(#issue-number): Replace stub with actual model inference.
    label: str = "unknown"
    confidence: float = 0.0

    if confidence < confidence_threshold:
        logger.debug(
            "Top prediction confidence %.4f is below threshold %.4f; returning 'unknown'",
            confidence,
            confidence_threshold,
        )
        return "unknown", 0.0

    logger.debug("Classified image as '%s' with confidence %.4f", label, confidence)
    return label, confidence


def load_classifier(model_path: str | Path) -> object:
    """Load a classifier model from *model_path*.

    This is a **stub** implementation.  Replace with framework-specific
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
    if not predictions:
        return {"accuracy": 0.0, "avg_confidence": 0.0}

    correct = sum(
        pred_label == truth
        for (pred_label, _), truth in zip(predictions, ground_truth, strict=True)
    )
    accuracy = correct / len(predictions)
    avg_confidence = sum(conf for _, conf in predictions) / len(predictions)

    return {"accuracy": accuracy, "avg_confidence": avg_confidence}
