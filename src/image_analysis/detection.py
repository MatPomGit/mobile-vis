"""Object detection utilities.

Provides basic object-detection helpers that wrap common OpenCV and
third-party model interfaces. Replace the stub implementations with
your actual model integration.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from .backends import DetectorBackend, InferenceResult
from .types import BgrImageU8
from .utils import validate_bbox_xyxy, validate_bgr_image

logger = logging.getLogger(__name__)

# Minimum confidence score for a detection to be kept.
DETECTION_CONFIDENCE_THRESHOLD: float = 0.5

# IoU threshold used by Non-Maximum Suppression.
NMS_IOU_THRESHOLD: float = 0.45

# Nazwa domyślnego backendu wykrywania obiektów.
DEFAULT_DETECTOR_BACKEND: str = "stub"

# Ujednolicony alias typu wyniku wykrywania dla stabilności API.
Detection = InferenceResult


class StubDetectorBackend:
    """No-op detector backend used as safe default fallback."""

    def detect(
        self,
        image: BgrImageU8,
        confidence_threshold: float,
    ) -> list[Detection]:
        """Return an empty list to emulate unavailable model inference."""
        _ = image
        _ = confidence_threshold
        return []


# Rejestr backendów detektora: klucz tekstowy -> fabryka backendu.
DETECTOR_BACKENDS: dict[str, Callable[[], DetectorBackend]] = {
    DEFAULT_DETECTOR_BACKEND: StubDetectorBackend,
}


def register_detector_backend(name: str, factory: Callable[[], DetectorBackend]) -> None:
    """Register a detector backend factory under a unique key."""
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError("Backend name must not be empty")

    # Walidujemy fabrykę przez próbę utworzenia instancji backendu.
    backend = factory()
    if not hasattr(backend, "detect"):
        raise TypeError("Detector backend must implement detect(image, confidence_threshold)")

    DETECTOR_BACKENDS[normalized_name] = factory


def create_detector_backend(name: str | None = None) -> DetectorBackend:
    """Create detector backend instance with fallback to default backend."""
    requested_name = (name or DEFAULT_DETECTOR_BACKEND).strip().lower()
    factory = DETECTOR_BACKENDS.get(requested_name)

    if factory is None:
        logger.warning(
            "Unknown detector backend '%s'. Falling back to '%s'.",
            requested_name,
            DEFAULT_DETECTOR_BACKEND,
        )
        factory = DETECTOR_BACKENDS[DEFAULT_DETECTOR_BACKEND]

    return factory()


def detect_objects(
    image: BgrImageU8,
    confidence_threshold: float = DETECTION_CONFIDENCE_THRESHOLD,
    backend: str | DetectorBackend | None = None,
) -> list[Detection]:
    """Detect objects in *image* and return filtered detections.

    Args:
        image: BGR image array with shape ``(H, W, 3)``, dtype ``uint8``.
        confidence_threshold: Minimum confidence score. Detections below
            this value are discarded.
        backend: Backend name, backend instance, or ``None`` (default backend).

    Returns:
        List of :class:`Detection` objects sorted by descending confidence.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* is not a 3-channel BGR array.
        ValueError: If *confidence_threshold* is outside ``[0.0, 1.0]``.
    """
    validate_bgr_image(image, allowed_dtypes=(np.uint8,))
    if not (0.0 <= confidence_threshold <= 1.0):
        raise ValueError(f"confidence_threshold must be in [0.0, 1.0], got {confidence_threshold}")

    # Obsługujemy zarówno nazwę backendu, jak i gotową instancję strategii.
    selected_backend = create_detector_backend(backend) if isinstance(backend, str) else backend
    if selected_backend is None:
        selected_backend = create_detector_backend()

    raw_detections = selected_backend.detect(image, confidence_threshold)
    detections = [d for d in raw_detections if d.score >= confidence_threshold]
    detections.sort(key=lambda d: d.score, reverse=True)

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

    # Implementacja NMS bez zależności od OpenCV, aby testy działały w lekkim środowisku.
    ordered_indices = sorted(
        range(len(detections)),
        key=lambda i: detections[i].score,
        reverse=True,
    )
    kept_indices: list[int] = []

    while ordered_indices:
        current = ordered_indices.pop(0)
        kept_indices.append(current)
        current_bbox = detections[current].bbox
        assert current_bbox is not None

        remaining: list[int] = []
        for candidate in ordered_indices:
            candidate_bbox = detections[candidate].bbox
            assert candidate_bbox is not None
            iou = _compute_iou(current_bbox, candidate_bbox)
            if iou <= iou_threshold:
                remaining.append(candidate)
        ordered_indices = remaining

    return [detections[i] for i in kept_indices]


def draw_bounding_boxes(
    image: BgrImageU8,
    detections: list[Detection],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> BgrImageU8:
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
    validate_bgr_image(image, allowed_dtypes=(np.uint8,))
    if thickness <= 0:
        raise ValueError(f"thickness must be positive, got {thickness}")

    _validate_detections(detections)
    output = image.copy()

    # Rysujemy obramowanie NumPy slicingiem, bez zależności od OpenCV.
    for det in detections:
        assert det.bbox is not None
        x1, y1, x2, y2 = det.bbox
        x1 = max(0, min(x1, output.shape[1] - 1))
        x2 = max(0, min(x2, output.shape[1] - 1))
        y1 = max(0, min(y1, output.shape[0] - 1))
        y2 = max(0, min(y2, output.shape[0] - 1))

        output[y1 : y1 + thickness, x1 : x2 + 1] = color
        output[y2 - thickness + 1 : y2 + 1, x1 : x2 + 1] = color
        output[y1 : y2 + 1, x1 : x1 + thickness] = color
        output[y1 : y2 + 1, x2 - thickness + 1 : x2 + 1] = color

    return output


def _compute_iou(bbox_a: tuple[int, int, int, int], bbox_b: tuple[int, int, int, int]) -> float:
    """Compute IoU score for two bounding boxes in XYXY format."""
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def _validate_detections(detections: list[Detection]) -> None:
    """Validate detection bounding boxes before drawing or applying NMS."""
    for detection in detections:
        if detection.bbox is None:
            raise ValueError("Detection bbox must not be None")
        validate_bbox_xyxy(detection.bbox)
        if not (0.0 <= detection.score <= 1.0):
            raise ValueError(f"Detection score must be in [0.0, 1.0], got {detection.score}")


class ObjectDetectionService:
    """Service class encapsulating object-detection workflow steps.

    The service keeps backend resolution, inference, NMS and visualisation
    in one place, while preserving function-based API compatibility.
    """

    def detect_objects(
        self,
        image: BgrImageU8,
        confidence_threshold: float = DETECTION_CONFIDENCE_THRESHOLD,
        backend: str | DetectorBackend | None = None,
    ) -> list[Detection]:
        """Uruchamia detekcję obiektów przez wybrany backend."""
        return detect_objects(image, confidence_threshold=confidence_threshold, backend=backend)

    def apply_nms(
        self,
        detections: list[Detection],
        iou_threshold: float = NMS_IOU_THRESHOLD,
    ) -> list[Detection]:
        """Filtruje listę detekcji algorytmem NMS."""
        return apply_nms(detections, iou_threshold=iou_threshold)

    def draw_bounding_boxes(
        self,
        image: BgrImageU8,
        detections: list[Detection],
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> BgrImageU8:
        """Rysuje obwiednie detekcji na kopii obrazu wejściowego."""
        return draw_bounding_boxes(image, detections, color=color, thickness=thickness)


# Domyślna instancja serwisu do użycia w prostych skryptach.
detection_service = ObjectDetectionService()

# Rejestr publicznych symboli modułu używany przez image_analysis.__init__.
PUBLIC_EXPORTS: dict[str, str] = {
    "DEFAULT_DETECTOR_BACKEND": "DEFAULT_DETECTOR_BACKEND",
    "DETECTION_CONFIDENCE_THRESHOLD": "DETECTION_CONFIDENCE_THRESHOLD",
    "DETECTOR_BACKENDS": "DETECTOR_BACKENDS",
    "Detection": "Detection",
    "NMS_IOU_THRESHOLD": "NMS_IOU_THRESHOLD",
    "apply_nms": "apply_nms",
    "create_detector_backend": "create_detector_backend",
    "detect_objects": "detect_objects",
    "draw_bounding_boxes": "draw_bounding_boxes",
    "ObjectDetectionService": "ObjectDetectionService",
    "detection_service": "detection_service",
    "register_detector_backend": "register_detector_backend",
}
