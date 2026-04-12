"""RTMDet-based object detection and rotated object detection.

Wraps the ``mmdet`` library (OpenMMLab MMDetection ≥ 3.0) to provide
real-time RTMDet inference helpers that integrate with the rest of the
:mod:`image_analysis` package.  All public functions follow the same
input/output conventions as the rest of the package:

* Images are NumPy ``uint8`` BGR arrays with shape ``(H, W, 3)``.
* Axis-aligned bounding boxes use ``(x1, y1, x2, y2)`` pixel coordinates
  (top-left / bottom-right corners).
* Rotated bounding boxes use ``(cx, cy, w, h, angle_deg)`` format, where
  the angle is measured counter-clockwise from the positive x-axis.
* Functions that modify the image always return a **copy**; the original
  array is never mutated.

Optional dependency
-------------------
This module requires the ``mmdet`` package (MMDetection ≥ 3.0).
Install it with::

    pip install "image-analysis[rtmdet]"

or directly::

    pip install mmdet>=3.0

If ``mmdet`` is not installed, importing any symbol from this module
raises :class:`ImportError` with a descriptive message.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

from .detector_common import load_with_retry, validate_bgr_uint8_image

if TYPE_CHECKING:  # pragma: no cover
    # Imported only for type-checking to avoid a hard runtime dependency.
    from mmdet.apis import DetInferencer as _MMDetInferencer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default minimum confidence score for an RTMDet detection to be kept.
RTMDET_CONFIDENCE_THRESHOLD: float = 0.3

#: IoU threshold used by Non-Maximum Suppression to merge overlapping boxes.
RTMDET_NMS_IOU_THRESHOLD: float = 0.45

#: Default pretrained RTMDet model identifier recognised by the mmdet
#: ``DetInferencer`` high-level API.
DEFAULT_RTMDET_MODEL: str = "rtmdet-nano"

#: Maximum number of attempts when initialising the mmdet inferencer fails.
RTMDET_DOWNLOAD_MAX_RETRIES: int = 3

#: Seconds to wait before the first retry attempt.  Each subsequent retry
#: waits twice as long (exponential back-off: 2 s → 4 s → 8 s).
RTMDET_DOWNLOAD_RETRY_DELAY_SECONDS: float = 2.0

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RtmDetDetection:
    """A single RTMDet object detection result.

    Attributes:
        label: Predicted class name from the COCO vocabulary.
        class_id: Numeric class index (0-based) in the model's vocabulary.
        confidence: Detection confidence in ``[0.0, 1.0]``.
        bbox: Axis-aligned bounding box as ``(x1, y1, x2, y2)`` pixel
            coordinates (top-left / bottom-right corners).
        angle_deg: Rotation angle in degrees (counter-clockwise from the
            positive x-axis).  ``None`` for axis-aligned detections.
    """

    label: str
    class_id: int
    confidence: float
    bbox: tuple[int, int, int, int]
    angle_deg: float | None = None


# ---------------------------------------------------------------------------
# RtmDetDetector - lifecycle manager
# ---------------------------------------------------------------------------


class RtmDetDetector:
    """Manages the lifecycle of an RTMDet model for repeated inference calls.

    The model is loaded lazily on the first call to :meth:`detect` and is
    released when :meth:`close` is called or the context manager exits.

    Args:
        model: RTMDet model identifier (e.g. ``"rtmdet-nano"``,
            ``"rtmdet-tiny"``) or path to a local config file.  The
            ``mmdet`` package will download weights automatically on first
            use.
        confidence_threshold: Minimum confidence for a detection to be kept.
        iou_threshold: IoU threshold for Non-Maximum Suppression.
        device: Torch device string (``"cpu"``, ``"cuda"``, etc.).

    Raises:
        ImportError: If the ``mmdet`` package is not installed.

    Example:
        >>> with RtmDetDetector("rtmdet-nano") as detector:
        ...     detections = detector.detect(image)
    """

    def __init__(
        self,
        model: str | Path = DEFAULT_RTMDET_MODEL,
        confidence_threshold: float = RTMDET_CONFIDENCE_THRESHOLD,
        iou_threshold: float = RTMDET_NMS_IOU_THRESHOLD,
        device: str = "cpu",
    ) -> None:
        _require_mmdet()
        self._model = str(model)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self._inferencer: _MMDetInferencer | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Load the model into memory.

        Called automatically by :meth:`detect` if not yet loaded.
        Calling this method explicitly is useful to measure load time
        separately from inference time.

        Raises:
            ImportError: If ``mmdet`` is not installed.
            RuntimeError: If the model cannot be initialised after all retry
                attempts have been exhausted.
        """
        if self._inferencer is not None:
            return
        from mmdet.apis import DetInferencer  # deferred import

        logger.info("Loading RTMDet model '%s' on device '%s'", self._model, self.device)
        self._inferencer = load_with_retry(
            DetInferencer,
            model=self._model,
            device=self.device,
            max_retries=RTMDET_DOWNLOAD_MAX_RETRIES,
            retry_delay=RTMDET_DOWNLOAD_RETRY_DELAY_SECONDS,
            logger=logger,
            retry_message=(
                "Failed to load RTMDet model '{target}' (attempt {attempt}/{max_retries}): "
                "{error} - retrying in {delay:.1f} s"
            ),
            failure_message="Failed to load RTMDet model '{target}' after {max_retries} attempts.",
        )

    def detect(
        self,
        image: NDArray[np.uint8],
        confidence_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> list[RtmDetDetection]:
        """Run RTMDet object detection on *image*.

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
        validate_bgr_uint8_image(image)
        conf = (
            confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        )
        iou = iou_threshold if iou_threshold is not None else self.iou_threshold
        self.initialize()
        assert self._inferencer is not None  # guaranteed by initialize()
        return detect_rtmdet(
            image,
            self._inferencer,
            confidence_threshold=conf,
            iou_threshold=iou,
        )

    def close(self) -> None:
        """Release the model and free associated memory."""
        self._inferencer = None
        logger.debug("RtmDetDetector closed")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> RtmDetDetector:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def detect_rtmdet(
    image: NDArray[np.uint8],
    inferencer: _MMDetInferencer,
    confidence_threshold: float = RTMDET_CONFIDENCE_THRESHOLD,
    iou_threshold: float = RTMDET_NMS_IOU_THRESHOLD,
) -> list[RtmDetDetection]:
    """Detect objects in *image* using a loaded RTMDet model.

    Args:
        image: BGR image array with shape ``(H, W, 3)``, dtype ``uint8``.
        inferencer: A loaded :class:`mmdet.apis.DetInferencer` instance.
        confidence_threshold: Minimum confidence score in ``[0.0, 1.0]``.
            Detections below this value are discarded.
        iou_threshold: IoU threshold for Non-Maximum Suppression in
            ``[0.0, 1.0]``.

    Returns:
        List of :class:`RtmDetDetection` objects sorted by descending
        confidence.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* is not a 3-channel BGR ``uint8`` array.
        ValueError: If *confidence_threshold* or *iou_threshold* is outside
            ``[0.0, 1.0]``.
    """
    validate_bgr_uint8_image(image)
    if not (0.0 <= confidence_threshold <= 1.0):
        raise ValueError(
            f"confidence_threshold must be in [0.0, 1.0], got {confidence_threshold}"
        )
    if not (0.0 <= iou_threshold <= 1.0):
        raise ValueError(f"iou_threshold must be in [0.0, 1.0], got {iou_threshold}")

    # mmdet expects RGB; convert from BGR
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = inferencer(
        image_rgb,
        pred_score_thr=confidence_threshold,
        return_datasamples=False,
    )

    predictions = result.get("predictions", [])
    if not predictions:
        return []

    pred = predictions[0]
    scores: list[float] = pred.get("scores", [])
    labels: list[int] = pred.get("labels", [])
    bboxes: list[list[float]] = pred.get("bboxes", [])

    # Attempt to retrieve COCO class names from the inferencer
    class_names: list[str] = []
    with suppress(AttributeError):
        class_names = inferencer.model.dataset_meta.get("classes", [])  # type: ignore[union-attr]

    detections: list[RtmDetDetection] = []
    for score, label_id, bbox in zip(scores, labels, bboxes, strict=False):
        if score < confidence_threshold:
            continue
        label_name = class_names[label_id] if label_id < len(class_names) else str(label_id)
        angle_deg: float | None = None
        if len(bbox) == 5:
            # Rotated bounding box: (cx, cy, w, h, angle_deg)
            cx, cy, w, h, a = bbox
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            angle_deg = float(a)
        else:
            x1, y1, x2, y2 = (int(v) for v in bbox[:4])

        detections.append(
            RtmDetDetection(
                label=label_name,
                class_id=int(label_id),
                confidence=float(score),
                bbox=(x1, y1, x2, y2),
                angle_deg=angle_deg,
            )
        )

    detections.sort(key=lambda d: d.confidence, reverse=True)
    logger.debug(
        "RTMDet detected %d objects above threshold %.2f",
        len(detections),
        confidence_threshold,
    )
    return detections


def draw_rtmdet_detections(
    image: NDArray[np.uint8],
    detections: list[RtmDetDetection],
    color: tuple[int, int, int] | None = None,
    thickness: int = 2,
    font_scale: float = 0.5,
) -> NDArray[np.uint8]:
    """Draw bounding boxes and labels for RTMDet detections onto a copy of *image*.

    Each axis-aligned detection is drawn with a bounding-box rectangle, a
    filled label background and a text label showing the class name and
    confidence percentage.  Rotated detections include the rotation angle
    in the label text.  When *color* is ``None``, a deterministic colour is
    derived from the detection's :attr:`~RtmDetDetection.class_id` so that
    different classes are visually distinct.

    Args:
        image: BGR image array with shape ``(H, W, 3)``, dtype ``uint8``.
        detections: RTMDet detections to draw.
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
    validate_bgr_uint8_image(image)
    if thickness <= 0:
        raise ValueError(f"thickness must be positive, got {thickness}")

    output = image.copy()

    for det in detections:
        box_color = color if color is not None else _class_color(det.class_id)
        x1, y1, x2, y2 = det.bbox

        if det.angle_deg is not None:
            # Rotated bounding box - draw as a rotated rectangle using polylines
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            rect = ((cx, cy), (w, h), det.angle_deg)
            box_pts = cv2.boxPoints(rect).astype(np.int32)
            cv2.polylines(output, [box_pts], isClosed=True, color=box_color, thickness=thickness)
            label_text = f"{det.label} {det.confidence:.0%} {det.angle_deg:.1f}°"
        else:
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


def export_rtmdet_to_onnx(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    img_size: int = 640,
) -> Path:
    """Export an RTMDet model to ONNX format for deployment on Android.

    The exported ONNX model can be loaded on Android using the OpenCV DNN
    module (``Dnn.readNetFromONNX``).  This function wraps the MMDetection
    ``pytorch2onnx`` utility.

    Args:
        config_path: Path to the MMDetection model configuration file (``.py``).
        checkpoint_path: Path to the trained model checkpoint (``.pth``).
        output_path: Destination for the ``.onnx`` file.  When ``None``,
            the file is placed alongside *checkpoint_path* with the
            ``.onnx`` extension replacing ``.pth``.
        img_size: Input image size (single int → square).  Must be a
            positive multiple of 32.

    Returns:
        Absolute path to the exported ``.onnx`` file.

    Raises:
        ImportError: If ``mmdet`` is not installed.
        FileNotFoundError: If *config_path* or *checkpoint_path* does not
            exist.
        ValueError: If *img_size* is not a positive multiple of 32.
    """
    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if img_size <= 0 or img_size % 32 != 0:
        raise ValueError(f"img_size must be a positive multiple of 32, got {img_size}")

    _require_mmdet()

    output_path = (
        checkpoint_path.with_suffix(".onnx") if output_path is None else Path(output_path)
    )

    logger.info(
        "Exporting RTMDet %s to ONNX at %s (imgsz=%d)",
        config_path.name,
        output_path,
        img_size,
    )

    try:
        from mmdeploy.apis import torch2onnx  # deferred import

        deploy_cfg = "mmdeploy/configs/mmdet/detection/detection_onnxruntime_static.py"
        torch2onnx(
            img=np.zeros((img_size, img_size, 3), dtype=np.uint8),
            work_dir=str(output_path.parent),
            save_file=output_path.name,
            deploy_cfg=deploy_cfg,
            model_cfg=str(config_path),
            model_checkpoint=str(checkpoint_path),
            device="cpu",
        )
    except ImportError:
        # Fall back to a basic torch.onnx export when mmdeploy is unavailable
        import torch
        from mmdet.apis import init_detector

        logger.warning(
            "mmdeploy not found; using basic torch.onnx export (post-processing not included)"
        )
        model = init_detector(str(config_path), str(checkpoint_path), device="cpu")
        model.eval()
        dummy_input = torch.zeros(1, 3, img_size, img_size)
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            opset_version=11,
            do_constant_folding=True,
        )

    exported_path = output_path
    if not exported_path.exists():
        # mmdeploy may write to a subdirectory
        candidate = output_path.parent / output_path.name
        if candidate.exists():
            os.replace(candidate, output_path)
        else:
            raise RuntimeError(f"ONNX export did not produce expected file: {output_path}")

    logger.info("RTMDet ONNX export complete: %s", output_path)
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Context manager helpers
# ---------------------------------------------------------------------------


@contextmanager
def rtmdet_detector(
    model: str | Path = DEFAULT_RTMDET_MODEL,
    confidence_threshold: float = RTMDET_CONFIDENCE_THRESHOLD,
    iou_threshold: float = RTMDET_NMS_IOU_THRESHOLD,
    device: str = "cpu",
) -> Generator[RtmDetDetector, None, None]:
    """Context manager that creates, yields and closes an :class:`RtmDetDetector`.

    Args:
        model: RTMDet model identifier or path to a local config file.
        confidence_threshold: Minimum confidence threshold.
        iou_threshold: NMS IoU threshold.
        device: Torch device string.

    Yields:
        An initialised :class:`RtmDetDetector` instance.

    Example:
        >>> with rtmdet_detector("rtmdet-nano") as det:
        ...     results = det.detect(image)
    """
    detector = RtmDetDetector(
        model=model,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        device=device,
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


def _class_color(class_id: int) -> tuple[int, int, int]:
    """Return a deterministic BGR colour for *class_id*."""
    return _PALETTE[class_id % len(_PALETTE)]


def _is_dark_color(bgr: tuple[int, int, int]) -> bool:
    """Return ``True`` when *bgr* has low perceptual brightness."""
    b, g, r = bgr
    # ITU-R BT.601 luma coefficients for perceptual brightness
    luminance = 0.114 * b + 0.587 * g + 0.299 * r
    return luminance < 128


def _require_mmdet() -> None:
    """Raise :class:`ImportError` if ``mmdet`` is not installed."""
    try:
        import mmdet  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'mmdet' package (MMDetection ≥ 3.0) is required for RTMDet detection. "
            "Install it with: pip install mmdet>=3.0  "
            "or: pip install 'image-analysis[rtmdet]'"
        ) from exc


# Rejestr publicznych symboli modułu używany przez image_analysis.__init__.
PUBLIC_EXPORTS: dict[str, str] = {
    "DEFAULT_RTMDET_MODEL": "DEFAULT_RTMDET_MODEL",
    "RTMDET_CONFIDENCE_THRESHOLD": "RTMDET_CONFIDENCE_THRESHOLD",
    "RTMDET_DOWNLOAD_MAX_RETRIES": "RTMDET_DOWNLOAD_MAX_RETRIES",
    "RTMDET_DOWNLOAD_RETRY_DELAY_SECONDS": "RTMDET_DOWNLOAD_RETRY_DELAY_SECONDS",
    "RTMDET_NMS_IOU_THRESHOLD": "RTMDET_NMS_IOU_THRESHOLD",
    "RtmDetDetection": "RtmDetDetection",
    "RtmDetDetector": "RtmDetDetector",
    "detect_rtmdet": "detect_rtmdet",
    "draw_rtmdet_detections": "draw_rtmdet_detections",
    "export_rtmdet_to_onnx": "export_rtmdet_to_onnx",
    "rtmdet_detector": "rtmdet_detector",
}
