"""QR code detection utilities.

Provides functions for detecting and decoding QR codes in images using
OpenCV's built-in :class:`cv2.QRCodeDetector`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Pixel offset for QR code label text above the bounding box top edge.
TEXT_OFFSET_PIXELS: int = 5

# Font scale used when drawing QR code labels onto images.
DEFAULT_FONT_SCALE: float = 0.5


@dataclass(frozen=True)
class QRCode:
    """A single QR code detection result.

    Attributes:
        data: Decoded text content of the QR code.
        bbox: Axis-aligned bounding box as ``(x1, y1, x2, y2)`` in pixel
            coordinates, where ``(x1, y1)`` is the top-left corner and
            ``(x2, y2)`` is the bottom-right corner.
        polygon: List of ``(x, y)`` corner points of the QR code polygon
            (usually four points).
    """

    data: str
    bbox: tuple[int, int, int, int]
    polygon: list[tuple[int, int]] = field(default_factory=list)


def detect_qr_codes(image: NDArray[np.uint8]) -> list[QRCode]:
    """Detect and decode all QR codes present in *image*.

    Uses OpenCV's :class:`cv2.QRCodeDetector` to locate and decode QR codes.
    Grayscale conversion is applied internally when needed.

    Args:
        image: BGR or grayscale image array with dtype ``uint8``.
            Accepted shapes: ``(H, W, 3)`` for BGR or ``(H, W)`` for
            grayscale.

    Returns:
        List of :class:`QRCode` objects, one per detected QR code.
        Returns an empty list if no QR codes are found.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* has an unsupported shape or dtype.
    """
    _validate_image(image)

    detector = cv2.QRCodeDetector()

    # detectAndDecodeMulti returns (retval, decoded_info, points, straight_code)
    _, decoded_texts, points_array, _ = detector.detectAndDecodeMulti(image)

    if points_array is None or len(decoded_texts) == 0:
        logger.debug("No QR codes detected in image")
        return []

    results: list[QRCode] = []
    for text, corners in zip(decoded_texts, points_array, strict=True):
        if not text:
            # QR code was detected but could not be decoded - skip.
            continue

        corners_int = corners.reshape(-1, 2).astype(np.int32)

        xs = corners_int[:, 0]
        ys = corners_int[:, 1]
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())

        polygon = [(int(pt[0]), int(pt[1])) for pt in corners_int]

        results.append(QRCode(data=text, bbox=(x1, y1, x2, y2), polygon=polygon))

    logger.debug("Detected %d QR code(s) in image", len(results))
    return results


def draw_qr_codes(
    image: NDArray[np.uint8],
    qr_codes: list[QRCode],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> NDArray[np.uint8]:
    """Draw QR code outlines and decoded text onto a copy of *image*.

    Args:
        image: BGR image array with shape ``(H, W, 3)``, dtype ``uint8``.
        qr_codes: QR codes to draw, as returned by :func:`detect_qr_codes`.
        color: BGR colour used for the polygon outline and label text.
        thickness: Line thickness in pixels (must be positive).

    Returns:
        Copy of *image* with QR code outlines and decoded text drawn.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* is not a 3-channel BGR array or *thickness*
            is not positive.
    """
    _validate_bgr_image(image)
    if thickness <= 0:
        raise ValueError(f"thickness must be positive, got {thickness}")

    output = image.copy()

    for qr in qr_codes:
        if qr.polygon:
            pts = np.array(qr.polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                output,
                [pts],
                isClosed=True,
                color=color,
                thickness=thickness,
            )
        else:
            x1, y1, x2, y2 = qr.bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

        x1, y1 = qr.bbox[0], qr.bbox[1]
        cv2.putText(
            output,
            qr.data,
            (x1, max(y1 - TEXT_OFFSET_PIXELS, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            DEFAULT_FONT_SCALE,
            color,
            thickness,
        )

    return output


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_image(image: object) -> None:
    """Validate that *image* is a uint8 array with an accepted shape.

    Args:
        image: Value to validate.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* has an unsupported shape or dtype.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image).__name__}")
    if image.ndim not in (2, 3):
        raise ValueError(
            f"Expected 2-D or 3-D array, got {image.ndim}-D with shape {image.shape}"
        )
    if image.ndim == 3 and image.shape[2] not in (1, 3):
        raise ValueError(
            f"Expected grayscale (H, W) or BGR (H, W, 3) image, got shape {image.shape}"
        )
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got dtype {image.dtype}")


def _validate_bgr_image(image: object) -> None:
    """Validate that *image* is a 3-channel BGR uint8 array.

    Args:
        image: Value to validate.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* does not have shape ``(H, W, 3)`` or dtype
            ``uint8``.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image).__name__}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected 3-channel BGR image with shape (H, W, 3), got shape {image.shape}"
        )
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 BGR image, got dtype {image.dtype}")
