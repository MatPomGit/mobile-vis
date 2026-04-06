"""YOLO-based object detection, instance segmentation and pose estimation.

Wraps the ``ultralytics`` library to provide real-time inference helpers that
integrate with the rest of the :mod:`image_analysis` package.  All public
functions follow the same input/output conventions as the rest of the package:

* Images are NumPy ``uint8`` BGR arrays with shape ``(H, W, 3)``.
* Bounding boxes use ``(x1, y1, x2, y2)`` pixel coordinates (top-left /
  bottom-right corners).
* Functions that modify the image always return a **copy**; the original array
  is never mutated.

Optional dependency
-------------------
This module requires the ``ultralytics`` package.  Install it with::

    pip install "image-analysis[yolo]"

or directly::

    pip install ultralytics>=8.0

If ``ultralytics`` is not installed, importing any symbol from this module
raises :class:`ImportError` with a descriptive message.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

from .utils import validate_image

if TYPE_CHECKING:  # pragma: no cover
    # Imported only for type-checking to avoid hard runtime dependency.
    from ultralytics import YOLO as _UltralyticsYOLO  # noqa: N811

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default minimum confidence score for a YOLO detection to be kept.
YOLO_CONFIDENCE_THRESHOLD: float = 0.5

#: IoU threshold used by Non-Maximum Suppression to merge overlapping boxes.
YOLO_NMS_IOU_THRESHOLD: float = 0.45

#: Default pretrained model weights downloaded by *ultralytics* on first use.
DEFAULT_YOLO_MODEL: str = "yolov8n.pt"

#: Maximum number of attempts when downloading YOLO model weights fails transiently.
YOLO_DOWNLOAD_MAX_RETRIES: int = 3

#: Seconds to wait before the first retry attempt.  Each subsequent retry waits twice
#: as long (exponential back-off: 2 s → 4 s → 8 s).
YOLO_DOWNLOAD_RETRY_DELAY_SECONDS: float = 2.0

#: Directory where downloaded YOLO model weights are cached between sessions.
#: On first use the directory is created automatically.  Set the environment variable
#: ``IMAGE_ANALYSIS_MODELS_DIR`` to override the default location.
YOLO_MODELS_DIR: Path = Path(
    os.environ.get(
        "IMAGE_ANALYSIS_MODELS_DIR",
        str(Path.home() / ".cache" / "image_analysis" / "yolo"),
    )
)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class YoloDetection:
    """A single YOLO object detection result.

    Attributes:
        label: Predicted class name from the COCO vocabulary.
        class_id: Numeric class index (0-based) in the model's vocabulary.
        confidence: Detection confidence in ``[0.0, 1.0]``.
        bbox: Bounding box as ``(x1, y1, x2, y2)`` pixel coordinates.
    """

    label: str
    class_id: int
    confidence: float
    bbox: tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# YoloDetector - lifecycle manager
# ---------------------------------------------------------------------------


class YoloDetector:
    """Manages the lifecycle of a YOLO model for repeated inference calls.

    The model is loaded lazily on the first call to :meth:`detect` and is
    released when :meth:`close` is called or the context manager exits.

    Args:
        model_path: Path to the model weights file (``*.pt`` or ``*.onnx``).
            Defaults to :data:`DEFAULT_YOLO_MODEL`, which the ``ultralytics``
            package will download automatically on first use.
        confidence_threshold: Minimum confidence for a detection to be kept.
        iou_threshold: IoU threshold for Non-Maximum Suppression.

    Raises:
        ImportError: If the ``ultralytics`` package is not installed.

    Example:
        >>> with YoloDetector() as detector:
        ...     detections = detector.detect(image)
    """

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Load the model into memory.

        Called automatically by :meth:`detect` if not yet loaded.
        Calling this method explicitly is useful to measure load time
        separately from inference time.

        Resolution order for bare model names (e.g. ``"yolov8n.pt"``):

        1. :data:`YOLO_MODELS_DIR` - the application's persistent model cache.
           If the weights file is already present there, it is loaded directly
           without any network access.
        2. Download from *Ultralytics* into :data:`YOLO_MODELS_DIR` so that
           subsequent calls find the file without re-downloading.  If the
           download fails transiently, it is retried up to
           :data:`YOLO_DOWNLOAD_MAX_RETRIES` times with exponential back-off.

        For explicit paths (absolute or containing path separators) the file
        must exist on disk; no download is attempted.

        Raises:
            FileNotFoundError: If *model_path* is an explicit path that does
                not exist on disk.
            RuntimeError: If the model cannot be downloaded after all retry
                attempts have been exhausted.
        """
        if self._model is not None:
            return
        from ultralytics import YOLO  # deferred import

        str_path = str(self._model_path)
        is_bare_name = self._model_path.parent == Path(".")

        if is_bare_name:
            # Bare model name - check the application models cache first so that
            # previously downloaded weights are reused without network access.
            cached_path = _get_models_dir() / self._model_path.name
            if cached_path.exists():
                logger.info("Loading YOLO model from cache: %s", cached_path)
                self._model = YOLO(str(cached_path))
            else:
                # Not in cache; pass the full target path to ultralytics so it
                # downloads the weights directly into the application models
                # directory and they are available on the next run.
                logger.info(
                    "YOLO model '%s' not found in cache, downloading to %s",
                    str_path,
                    cached_path,
                )
                self._model = _load_with_retry(YOLO, str(cached_path))
        elif not self._model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {self._model_path}")
        else:
            logger.info("Loading YOLO model from %s", self._model_path)
            self._model = YOLO(str_path)

    def detect(
        self,
        image: NDArray[np.uint8],
        confidence_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> list[YoloDetection]:
        """Run YOLO object detection on *image*.

        Args:
            image: BGR image array with shape ``(H, W, 3)``, dtype ``uint8``.
            confidence_threshold: Override for the instance-level threshold.
                Defaults to :attr:`confidence_threshold`.
            iou_threshold: Override for the NMS IoU threshold.
                Defaults to :attr:`iou_threshold`.

        Returns:
            Detections sorted by descending confidence.

        Raises:
            TypeError: If *image* is not a ``np.ndarray``.
            ValueError: If *image* is not a 3-channel BGR ``uint8`` array.
        """
        _validate_bgr_image(image)
        conf = (
            confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        )
        iou = iou_threshold if iou_threshold is not None else self.iou_threshold
        self.initialize()
        assert self._model is not None  # guaranteed by initialize()
        return detect_yolo(
            image,
            self._model,
            confidence_threshold=conf,
            iou_threshold=iou,
        )

    def close(self) -> None:
        """Release the model and free associated memory."""
        self._model = None
        logger.debug("YoloDetector closed")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> YoloDetector:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def detect_yolo(
    image: NDArray[np.uint8],
    model: _UltralyticsYOLO,
    confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
    iou_threshold: float = YOLO_NMS_IOU_THRESHOLD,
) -> list[YoloDetection]:
    """Detect objects in *image* using a loaded YOLO model.

    Args:
        image: BGR image array with shape ``(H, W, 3)``, dtype ``uint8``.
        model: A loaded :class:`ultralytics.YOLO` instance.
        confidence_threshold: Minimum confidence score in ``[0.0, 1.0]``.
            Detections below this value are discarded.
        iou_threshold: IoU threshold for Non-Maximum Suppression in
            ``[0.0, 1.0]``.

    Returns:
        List of :class:`YoloDetection` objects sorted by descending
        confidence.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* is not a 3-channel BGR ``uint8`` array.
        ValueError: If *confidence_threshold* or *iou_threshold* is outside
            ``[0.0, 1.0]``.
    """
    _validate_bgr_image(image)
    if not (0.0 <= confidence_threshold <= 1.0):
        raise ValueError(
            f"confidence_threshold must be in [0.0, 1.0], got {confidence_threshold}"
        )
    if not (0.0 <= iou_threshold <= 1.0):
        raise ValueError(f"iou_threshold must be in [0.0, 1.0], got {iou_threshold}")

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
        # Batch-transfer all tensors to CPU in one operation each to avoid
        # repeated individual GPU→CPU round-trips inside the loop.
        xyxy_all = boxes.xyxy.cpu().numpy()
        conf_all = boxes.conf.cpu().numpy()
        cls_all = boxes.cls.cpu().numpy()
        for i in range(len(boxes)):
            x1, y1, x2, y2 = (int(v) for v in xyxy_all[i])
            conf_val = float(conf_all[i])
            cls_id = int(cls_all[i])
            label = names.get(cls_id, str(cls_id))
            detections.append(
                YoloDetection(
                    label=label,
                    class_id=cls_id,
                    confidence=conf_val,
                    bbox=(x1, y1, x2, y2),
                )
            )

    detections.sort(key=lambda d: d.confidence, reverse=True)
    logger.debug(
        "YOLO detected %d objects above threshold %.2f",
        len(detections),
        confidence_threshold,
    )
    return detections


def draw_yolo_detections(
    image: NDArray[np.uint8],
    detections: list[YoloDetection],
    color: tuple[int, int, int] | None = None,
    thickness: int = 2,
    font_scale: float = 0.5,
) -> NDArray[np.uint8]:
    """Draw bounding boxes and labels for YOLO detections onto a copy of *image*.

    Each detection is drawn with a bounding-box rectangle, a filled label
    background and a text label showing the class name and confidence
    percentage.  When *color* is ``None``, a deterministic colour is derived
    from the detection's :attr:`~YoloDetection.class_id` so that different
    classes are visually distinct.

    Args:
        image: BGR image array with shape ``(H, W, 3)``, dtype ``uint8``.
        detections: YOLO detections to draw.
        color: Fixed BGR colour for all boxes.  When ``None``, colours are
            chosen per class.
        thickness: Line thickness in pixels.  Must be positive.
        font_scale: OpenCV font scale for label text.

    Returns:
        Copy of *image* with all detections drawn.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* is not a 3-channel BGR ``uint8`` array.
        ValueError: If *thickness* is not positive.
    """
    _validate_bgr_image(image)
    if thickness <= 0:
        raise ValueError(f"thickness must be positive, got {thickness}")

    output = image.copy()

    for det in detections:
        box_color = color if color is not None else _class_color(det.class_id)
        x1, y1, x2, y2 = det.bbox

        cv2.rectangle(output, (x1, y1), (x2, y2), box_color, thickness)

        label_text = f"{det.label} {det.confidence:.0%}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_y = max(y1 - text_h - baseline, 0)
        cv2.rectangle(
            output,
            (x1, label_y),
            (x1 + text_w, label_y + text_h + baseline),
            box_color,
            cv2.FILLED,
        )
        text_color = (255, 255, 255) if _is_dark_color(box_color) else (0, 0, 0)
        cv2.putText(
            output,
            label_text,
            (x1, label_y + text_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
        )

    return output


def export_yolo_to_onnx(
    model_path: str | Path,
    output_path: str | Path | None = None,
    img_size: int = 640,
) -> Path:
    """Export a YOLO model to ONNX format.

    .. deprecated::
        Prefer :func:`export_yolo_to_torchscript` for new Android deployments.
        ONNX export is retained for backward compatibility with OpenCV DNN
        pipelines.

    Args:
        model_path: Path to the source YOLO weights file (``*.pt``).
        output_path: Destination for the ``.onnx`` file.  When ``None``,
            the file is placed alongside *model_path* with the ``.onnx``
            extension replacing ``.pt``.
        img_size: Input image size (single int → square).  Must be a
            multiple of 32.

    Returns:
        Absolute path to the exported ``.onnx`` file.

    Raises:
        ImportError: If ``ultralytics`` is not installed.
        FileNotFoundError: If *model_path* does not exist.
        ValueError: If *img_size* is not a positive multiple of 32.
    """
    _require_ultralytics()
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if img_size <= 0 or img_size % 32 != 0:
        raise ValueError(f"img_size must be a positive multiple of 32, got {img_size}")

    from ultralytics import YOLO  # deferred import

    output_path = model_path.with_suffix(".onnx") if output_path is None else Path(output_path)

    logger.info("Exporting %s to ONNX at %s (imgsz=%d)", model_path.name, output_path, img_size)
    model = YOLO(str(model_path))
    exported = model.export(format="onnx", imgsz=img_size, simplify=True)
    exported_path = Path(str(exported))

    if exported_path != output_path:
        os.replace(exported_path, output_path)

    logger.info("ONNX export complete: %s", output_path)
    return output_path.resolve()


def export_yolo_to_torchscript(
    model_path: str | Path,
    output_path: str | Path | None = None,
    img_size: int = 640,
    optimize: bool = False,
) -> Path:
    """Export a YOLO model to TorchScript format for Android deployment.

    The exported TorchScript model can be loaded on Android using the PyTorch
    Mobile library (``Module.load(path)``).  Download the base ``*.pt``
    weights directly from the Ultralytics GitHub releases and pass the local
    path as *model_path*.

    Example::

        >>> from image_analysis.yolo import export_yolo_to_torchscript
        >>> path = export_yolo_to_torchscript("yolov8n.pt", img_size=640)
        >>> print(path)  # .../yolov8n.torchscript

    Args:
        model_path: Path to the source YOLO weights file (``*.pt``).
        output_path: Destination for the ``.torchscript`` file.  When
            ``None``, the file is placed alongside *model_path* with the
            ``.torchscript`` extension replacing ``.pt``.
        img_size: Input image size (single int → square).  Must be a
            multiple of 32.
        optimize: When ``True``, enables TorchScript mobile optimizations
            (``torch.utils.mobile_optimizer``).  Reduces model size and
            improves CPU inference speed on mobile devices.

    Returns:
        Absolute path to the exported ``.torchscript`` file.

    Raises:
        ImportError: If ``ultralytics`` is not installed.
        FileNotFoundError: If *model_path* does not exist.
        ValueError: If *img_size* is not a positive multiple of 32.
    """
    _require_ultralytics()
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if img_size <= 0 or img_size % 32 != 0:
        raise ValueError(f"img_size must be a positive multiple of 32, got {img_size}")

    from ultralytics import YOLO  # deferred import

    output_path = (
        model_path.with_suffix(".torchscript") if output_path is None else Path(output_path)
    )

    logger.info(
        "Exporting %s to TorchScript at %s (imgsz=%d, optimize=%s)",
        model_path.name,
        output_path,
        img_size,
        optimize,
    )
    model = YOLO(str(model_path))
    exported = model.export(format="torchscript", imgsz=img_size, optimize=optimize)
    exported_path = Path(str(exported))

    if exported_path != output_path:
        os.replace(exported_path, output_path)

    logger.info("TorchScript export complete: %s", output_path)
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Context manager helpers
# ---------------------------------------------------------------------------


@contextmanager
def yolo_detector(
    model_path: str | Path = DEFAULT_YOLO_MODEL,
    confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
    iou_threshold: float = YOLO_NMS_IOU_THRESHOLD,
) -> Generator[YoloDetector, None, None]:
    """Context manager that creates, yields and closes a :class:`YoloDetector`.

    Args:
        model_path: Path to the YOLO weights file.
        confidence_threshold: Minimum confidence threshold.
        iou_threshold: NMS IoU threshold.

    Yields:
        An initialised :class:`YoloDetector` instance.

    Example:
        >>> with yolo_detector("yolov8n.pt") as det:
        ...     results = det.detect(image)
    """
    detector = YoloDetector(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
    )
    try:
        yield detector
    finally:
        detector.close()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# Deterministic colour palette for 80 COCO classes (BGR).
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
    """Return the application YOLO models cache directory, creating it if needed.

    Returns:
        Absolute path to :data:`YOLO_MODELS_DIR` after ensuring it exists.
    """
    YOLO_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return YOLO_MODELS_DIR


def _class_color(class_id: int) -> tuple[int, int, int]:
    """Return a deterministic BGR colour for *class_id*."""
    return _PALETTE[class_id % len(_PALETTE)]


def _is_dark_color(bgr: tuple[int, int, int]) -> bool:
    """Return ``True`` when *bgr* has low perceptual brightness."""
    b, g, r = bgr
    luminance = 0.114 * b + 0.587 * g + 0.299 * r
    return luminance < 128


def _load_with_retry(
    yolo_cls: type,
    model_name: str,
    max_retries: int = YOLO_DOWNLOAD_MAX_RETRIES,
    retry_delay: float = YOLO_DOWNLOAD_RETRY_DELAY_SECONDS,
) -> "_UltralyticsYOLO":
    """Attempt to instantiate *yolo_cls(model_name)*, retrying on transient failures.

    Retries are spaced using exponential back-off: the first retry waits
    *retry_delay* seconds, the second waits twice as long, and so on.

    Args:
        yolo_cls: The ``ultralytics.YOLO`` class (passed in to keep the import
            deferred and make the function easy to test with a mock).
        model_name: Model name or path forwarded to *yolo_cls*.
        max_retries: Total number of attempts (including the initial one).
        retry_delay: Wait time in seconds before the first retry.

    Returns:
        A loaded ``ultralytics.YOLO`` instance.

    Raises:
        RuntimeError: If all attempts fail.
    """
    last_exc: BaseException | None = None
    delay = retry_delay
    for attempt in range(1, max_retries + 1):
        try:
            return yolo_cls(model_name)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_retries:
                logger.warning(
                    "Failed to load/download YOLO model '%s' (attempt %d/%d): %s"
                    " – retrying in %.1f s",
                    model_name,
                    attempt,
                    max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
                delay *= 2
            else:
                logger.error(
                    "Failed to load/download YOLO model '%s' after %d attempts: %s",
                    model_name,
                    max_retries,
                    exc,
                )
    raise RuntimeError(
        f"Failed to download YOLO model '{model_name}' after {max_retries} attempts."
    ) from last_exc


def _require_ultralytics() -> None:
    """Raise :class:`ImportError` if ``ultralytics`` is not installed."""
    try:
        import ultralytics  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'ultralytics' package is required for YOLO detection. "
            "Install it with: pip install ultralytics>=8.0  "
            "or: pip install 'image-analysis[yolo]'"
        ) from exc


def _validate_bgr_image(image: object) -> None:
    """Validate that *image* is a 3-channel BGR uint8 NumPy array.

    Args:
        image: Value to validate.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If the array is not a 3-channel uint8 BGR image.
    """
    validate_image(image)
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
    ):
        raise ValueError(
            "Expected uint8 3-channel BGR image with shape (H, W, 3), "
            f"got shape {getattr(image, 'shape', 'N/A')} and "
            f"dtype {getattr(image, 'dtype', 'N/A')}"
        )
