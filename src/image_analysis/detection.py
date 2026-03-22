"""Object detection utilities.

Provides basic object-detection helpers that wrap common OpenCV and
third-party model interfaces.  Replace the stub implementations with
your actual model integration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Minimum confidence score for a detection to be kept.
DETECTION_CONFIDENCE_THRESHOLD: float = 0.5

# IoU threshold used by Non-Maximum Suppression.
NMS_IOU_THRESHOLD: float = 0.45


@dataclass(frozen=True)
class Detection:
    """A single object detection result.

    Attributes:
        label: Predicted class name.
        confidence: Prediction confidence in ``[0.0, 1.0]``.
        bbox: Bounding box as ``(x1, y1, x2, y2)`` in pixel coordinates.
    """

    label: str
    confidence: float
    bbox: tuple[int, int, int, int]


def detect_objects(
    image: NDArray[np.uint8],
    confidence_threshold: float = DETECTION_CONFIDENCE_THRESHOLD,
) -> list[Detection]:
    """Detect objects in *image* and return filtered detections.

    This is a **stub** implementation that returns an empty list.
    Replace the body with your model's inference call.

    Args:
        image: BGR image array with shape ``(H, W, 3)``, dtype ``uint8``.
        confidence_threshold: Minimum confidence score. Detections below
            this value are discarded.

    Returns:
        List of :class:`Detection` objects sorted by descending confidence.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* is not a 3-channel BGR array.
        ValueError: If *confidence_threshold* is outside ``[0.0, 1.0]``.
    """
    _validate_bgr_image(image)
    if not (0.0 <= confidence_threshold <= 1.0):
        raise ValueError(f"confidence_threshold must be in [0.0, 1.0], got {confidence_threshold}")

    # TODO(#issue-number): Replace stub with actual model inference.
    raw_detections: list[Detection] = []

    detections = [d for d in raw_detections if d.confidence >= confidence_threshold]
    detections.sort(key=lambda d: d.confidence, reverse=True)

    logger.debug("Detected %d objects above threshold %.2f", len(detections), confidence_threshold)
    return detections


def apply_nms(
    detections: list[Detection],
    iou_threshold: float = NMS_IOU_THRESHOLD,
) -> list[Detection]:
    """Apply Non-Maximum Suppression to remove overlapping bounding boxes.

    Args:
        detections: List of detections to filter.
        iou_threshold: Maximum allowed IoU between kept boxes.

    Returns:
        Filtered list of :class:`Detection` objects.

    Raises:
        ValueError: If *iou_threshold* is outside ``[0.0, 1.0]``.
    """
    if not (0.0 <= iou_threshold <= 1.0):
        raise ValueError(f"iou_threshold must be in [0.0, 1.0], got {iou_threshold}")

    if not detections:
        return []

    _validate_detections(detections)

    boxes = np.array([[d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]] for d in detections])
    scores = np.array([d.confidence for d in detections])

    # cv2.dnn.NMSBoxes expects (x, y, w, h) format
    boxes_xywh = boxes.copy().tolist()
    for i, (x1, y1, x2, y2) in enumerate(boxes.tolist()):
        boxes_xywh[i] = [x1, y1, int(x2 - x1), int(y2 - y1)]

    indices = cv2.dnn.NMSBoxes(
        boxes_xywh,
        scores.tolist(),
        score_threshold=0.0,
        nms_threshold=iou_threshold,
    )

    kept_indices = _normalize_nms_indices(indices)
    return [detections[i] for i in kept_indices]


def draw_bounding_boxes(
    image: NDArray[np.uint8],
    detections: list[Detection],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> NDArray[np.uint8]:
    """Draw bounding boxes and labels onto a copy of *image*.

    Args:
        image: BGR image array with shape ``(H, W, 3)``, dtype ``uint8``.
        detections: Detections to draw.
        color: BGR colour for the bounding box rectangle.
        thickness: Line thickness in pixels.

    Returns:
        Copy of *image* with bounding boxes and labels drawn.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* is not a 3-channel BGR array.
    """
    _validate_bgr_image(image)
    if thickness <= 0:
        raise ValueError(f"thickness must be positive, got {thickness}")

    _validate_detections(detections)
    output = image.copy()

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        label_text = f"{det.label}: {det.confidence:.2f}"
        cv2.putText(
            output,
            label_text,
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            thickness,
        )

    return output


def _validate_bgr_image(image: object) -> None:
    """Validate that *image* is a 3-channel BGR uint8 array.

    Args:
        image: Value to validate.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* does not have shape ``(H, W, 3)``.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image).__name__}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected 3-channel BGR image with shape (H, W, 3), got shape {image.shape}"
        )
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 BGR image, got {image.dtype}")


def _validate_detections(detections: list[Detection]) -> None:
    """Validate detection bounding boxes before drawing or applying NMS."""
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Detection bbox must satisfy x2 > x1 and y2 > y1, got {detection.bbox}"
            )
        if not (0.0 <= detection.confidence <= 1.0):
            raise ValueError(
                f"Detection confidence must be in [0.0, 1.0], got {detection.confidence}"
            )


def _normalize_nms_indices(indices: object) -> list[int]:
    """Convert OpenCV NMSBoxes output to a plain list of integer indices."""
    if indices is None:
        return []
    if isinstance(indices, np.ndarray):
        return [int(index) for index in indices.reshape(-1).tolist()]
    if isinstance(indices, tuple):
        return [int(index) for index in indices]
    if isinstance(indices, list):
        normalized: list[int] = []
        for item in indices:
            if isinstance(item, (list, tuple, np.ndarray)):
                normalized.extend(int(index) for index in np.asarray(item).reshape(-1).tolist())
            else:
                normalized.append(int(item))
        return normalized
    return []
