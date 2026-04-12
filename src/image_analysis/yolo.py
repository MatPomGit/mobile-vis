"""YOLO helpers for detection, drawing and model export.

Public API conventions:
- input image: ``np.ndarray`` with shape ``(H, W, C)``, dtype ``uint8`` (BGR),
- bounding box: ``(x1, y1, x2, y2)``,
- drawing functions always return a copy (source image is not mutated).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

from .detector_common import load_with_retry, validate_bgr_uint8_image

if TYPE_CHECKING:  # pragma: no cover
    from ultralytics import YOLO as _UltralyticsYOLO  # noqa: N811

logger = logging.getLogger(__name__)

YOLO_CONFIDENCE_THRESHOLD: float = 0.5
YOLO_NMS_IOU_THRESHOLD: float = 0.45
DEFAULT_YOLO_MODEL: str = "yolov8n.pt"
YOLO_DOWNLOAD_MAX_RETRIES: int = 3
YOLO_DOWNLOAD_RETRY_DELAY_SECONDS: float = 2.0
YOLO_MODELS_DIR: Path = Path(
    os.environ.get(
        "IMAGE_ANALYSIS_MODELS_DIR",
        str(Path.home() / ".cache" / "image_analysis" / "yolo"),
    )
)


@dataclass(frozen=True)
class YoloDetection:
    """Single YOLO detection result.

    Attributes:
        label: Human-readable class label.
        class_id: Numeric class identifier from model vocabulary.
        confidence: Confidence in range ``[0.0, 1.0]``.
        bbox: Pixel coordinates in ``(x1, y1, x2, y2)`` format.
    """

    label: str
    class_id: int
    confidence: float
    bbox: tuple[int, int, int, int]


class YoloDetector:
    """Stateful YOLO detector for repeated inference calls."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_YOLO_MODEL,
        confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
        iou_threshold: float = YOLO_NMS_IOU_THRESHOLD,
    ) -> None:
        _require_ultralytics()
        self._model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self._model: _UltralyticsYOLO | None = None

    def initialize(self) -> None:
        """Load YOLO model if needed.

        Bare names (e.g. ``yolov8n.pt``) are resolved against ``YOLO_MODELS_DIR``.
        Explicit paths must exist on disk.
        """
        if self._model is not None:
            return

        from ultralytics import YOLO

        is_bare_name = self._model_path.parent == Path(".")
        if is_bare_name:
            cached_path = _get_models_dir() / self._model_path.name
            if cached_path.exists():
                logger.info("Loading YOLO model from cache: %s", cached_path)
                self._model = YOLO(str(cached_path))
            else:
                logger.info("Downloading YOLO model to cache: %s", cached_path)
                self._model = load_with_retry(
                    YOLO,
                    str(cached_path),
                    max_retries=YOLO_DOWNLOAD_MAX_RETRIES,
                    retry_delay=YOLO_DOWNLOAD_RETRY_DELAY_SECONDS,
                    logger=logger,
                    retry_message=(
                        "YOLO load failed for '{target}' ({attempt}/{max_retries}): "
                        "{error}; retry in {delay:.1f}s"
                    ),
                    failure_message=(
                        "Failed to download YOLO model '{target}' "
                        "after {max_retries} attempts."
                    ),
                )
            return

        if not self._model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {self._model_path}")

        logger.info("Loading YOLO model from path: %s", self._model_path)
        self._model = YOLO(str(self._model_path))

    def detect(
        self,
        image: NDArray[np.uint8],
        confidence_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> list[YoloDetection]:
        """Run detection on BGR image with optional threshold overrides."""
        validate_bgr_uint8_image(image)
        conf = self.confidence_threshold if confidence_threshold is None else confidence_threshold
        iou = self.iou_threshold if iou_threshold is None else iou_threshold

        self.initialize()
        assert self._model is not None
        return detect_yolo(
            image=image,
            model=self._model,
            confidence_threshold=conf,
            iou_threshold=iou,
        )

    def close(self) -> None:
        """Release model reference."""
        self._model = None

    def __enter__(self) -> YoloDetector:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def detect_yolo(
    image: NDArray[np.uint8],
    model: _UltralyticsYOLO,
    confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
    iou_threshold: float = YOLO_NMS_IOU_THRESHOLD,
) -> list[YoloDetection]:
    """Run YOLO inference and convert results to :class:`YoloDetection`."""
    validate_bgr_uint8_image(image)
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError(
            f"confidence_threshold must be in [0.0, 1.0], got {confidence_threshold}"
        )
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in [0.0, 1.0], got {iou_threshold}")

    # Przekazujemy gotowy obraz BGR; konwersję i preprocessing wykona backend modelu.
    results = model.predict(
        image,
        conf=confidence_threshold,
        iou=iou_threshold,
        verbose=False,
    )

    detections: list[YoloDetection] = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        names: dict[int, str] = result.names  # type: ignore[assignment]
        xyxy_values = boxes.xyxy.cpu().numpy()
        conf_values = boxes.conf.cpu().numpy()
        class_values = boxes.cls.cpu().numpy()

        for index in range(len(boxes)):
            x1, y1, x2, y2 = (int(v) for v in xyxy_values[index])
            class_id = int(class_values[index])
            confidence = float(conf_values[index])
            label = names.get(class_id, str(class_id))
            detections.append(
                YoloDetection(
                    label=label,
                    class_id=class_id,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                )
            )

    detections.sort(key=lambda det: det.confidence, reverse=True)
    return detections


def draw_yolo_detections(
    image: NDArray[np.uint8],
    detections: list[YoloDetection],
    color: tuple[int, int, int] | None = None,
    thickness: int = 2,
    font_scale: float = 0.5,
) -> NDArray[np.uint8]:
    """Draw detections on an image copy and return it."""
    validate_bgr_uint8_image(image)
    if thickness <= 0:
        raise ValueError(f"thickness must be positive, got {thickness}")

    rendered = image.copy()

    for detection in detections:
        box_color = color if color is not None else _class_color(detection.class_id)
        x1, y1, x2, y2 = detection.bbox

        cv2.rectangle(rendered, (x1, y1), (x2, y2), box_color, thickness)

        caption = f"{detection.label} {detection.confidence:.0%}"
        (text_width, text_height), baseline = cv2.getTextSize(
            caption,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        label_top = max(y1 - text_height - baseline, 0)
        cv2.rectangle(
            rendered,
            (x1, label_top),
            (x1 + text_width, label_top + text_height + baseline),
            box_color,
            cv2.FILLED,
        )

        text_color = (255, 255, 255) if _is_dark_color(box_color) else (0, 0, 0)
        cv2.putText(
            rendered,
            caption,
            (x1, label_top + text_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
        )

    return rendered


def export_yolo_to_onnx(
    model_path: str | Path,
    output_path: str | Path | None = None,
    img_size: int = 640,
) -> Path:
    """Export YOLO model to ONNX."""
    _require_ultralytics()
    source_path = Path(model_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Model file not found: {source_path}")
    if img_size <= 0 or img_size % 32 != 0:
        raise ValueError(f"img_size must be a positive multiple of 32, got {img_size}")

    from ultralytics import YOLO

    destination = source_path.with_suffix(".onnx") if output_path is None else Path(output_path)
    model = YOLO(str(source_path))
    exported_path = Path(str(model.export(format="onnx", imgsz=img_size, simplify=True)))

    if exported_path != destination:
        os.replace(exported_path, destination)

    return destination.resolve()


def export_yolo_to_torchscript(
    model_path: str | Path,
    output_path: str | Path | None = None,
    img_size: int = 640,
    optimize: bool = False,
) -> Path:
    """Export YOLO model to TorchScript."""
    _require_ultralytics()
    source_path = Path(model_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Model file not found: {source_path}")
    if img_size <= 0 or img_size % 32 != 0:
        raise ValueError(f"img_size must be a positive multiple of 32, got {img_size}")

    from ultralytics import YOLO

    destination = (
        source_path.with_suffix(".torchscript") if output_path is None else Path(output_path)
    )
    model = YOLO(str(source_path))
    exported_path = Path(
        str(model.export(format="torchscript", imgsz=img_size, optimize=optimize))
    )

    if exported_path != destination:
        os.replace(exported_path, destination)

    return destination.resolve()


@contextmanager
def yolo_detector(
    model_path: str | Path = DEFAULT_YOLO_MODEL,
    confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
    iou_threshold: float = YOLO_NMS_IOU_THRESHOLD,
) -> Generator[YoloDetector, None, None]:
    """Context manager yielding :class:`YoloDetector`."""
    detector = YoloDetector(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
    )
    try:
        yield detector
    finally:
        detector.close()


_PALETTE: list[tuple[int, int, int]] = [
    (56, 56, 255),
    (151, 157, 255),
    (31, 112, 255),
    (29, 178, 255),
    (49, 210, 207),
    (10, 249, 72),
    (23, 204, 146),
    (134, 219, 61),
    (52, 147, 26),
    (187, 212, 0),
    (168, 153, 44),
    (255, 194, 0),
    (255, 152, 0),
    (255, 87, 34),
    (244, 54, 45),
    (255, 48, 112),
    (221, 0, 255),
    (124, 77, 255),
    (157, 148, 255),
    (189, 151, 255),
]


def _get_models_dir() -> Path:
    """Return model cache path and ensure directory exists."""
    YOLO_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return YOLO_MODELS_DIR


def _class_color(class_id: int) -> tuple[int, int, int]:
    """Return deterministic class color."""
    return _PALETTE[class_id % len(_PALETTE)]


def _is_dark_color(bgr: tuple[int, int, int]) -> bool:
    """Return ``True`` when color is perceptually dark."""
    blue, green, red = bgr
    luminance = (0.114 * blue) + (0.587 * green) + (0.299 * red)
    return luminance < 128


def _require_ultralytics() -> None:
    """Validate optional dependency import."""
    try:
        import ultralytics  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'ultralytics' package is required for YOLO detection. "
            "Install it with: pip install ultralytics>=8.0 "
            "or: pip install 'image-analysis[yolo]'"
        ) from exc


# Rejestr publicznych symboli modułu używany przez image_analysis.__init__.
PUBLIC_EXPORTS: dict[str, str] = {
    "DEFAULT_YOLO_MODEL": "DEFAULT_YOLO_MODEL",
    "YOLO_CONFIDENCE_THRESHOLD": "YOLO_CONFIDENCE_THRESHOLD",
    "YOLO_DOWNLOAD_MAX_RETRIES": "YOLO_DOWNLOAD_MAX_RETRIES",
    "YOLO_DOWNLOAD_RETRY_DELAY_SECONDS": "YOLO_DOWNLOAD_RETRY_DELAY_SECONDS",
    "YOLO_MODELS_DIR": "YOLO_MODELS_DIR",
    "YOLO_NMS_IOU_THRESHOLD": "YOLO_NMS_IOU_THRESHOLD",
    "YoloDetection": "YoloDetection",
    "YoloDetector": "YoloDetector",
    "detect_yolo": "detect_yolo",
    "draw_yolo_detections": "draw_yolo_detections",
    "export_yolo_to_onnx": "export_yolo_to_onnx",
    "export_yolo_to_torchscript": "export_yolo_to_torchscript",
    "yolo_detector": "yolo_detector",
}
