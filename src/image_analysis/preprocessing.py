"""Image preprocessing utilities.

Provides functions for loading, resizing, and normalising images.
All functions expect and return BGR NumPy arrays unless stated otherwise.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk as a BGR NumPy array.

    Args:
        path: Path to the image file (JPEG, PNG, BMP, TIFF, …).

    Returns:
        BGR image array with shape ``(H, W, 3)``, dtype ``uint8``.

    Raises:
        FileNotFoundError: If the file does not exist at *path*.
        ValueError: If the file cannot be decoded as an image.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")

    logger.debug("Loaded image '%s' with shape %s", path.name, image.shape)
    return image


def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an image to the specified dimensions.

    Uses ``cv2.INTER_AREA`` when downscaling and ``cv2.INTER_LINEAR``
    when upscaling for best quality.

    Args:
        image: Input image array with shape ``(H, W)`` or ``(H, W, C)``.
        width: Target width in pixels. Must be positive.
        height: Target height in pixels. Must be positive.

    Returns:
        Resized image array with shape ``(height, width[, C])``.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *width* or *height* is not a positive integer.
    """
    _validate_image(image)
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got width={width}, height={height}")

    original_h, original_w = image.shape[:2]
    downscaling = width < original_w or height < original_h
    interpolation = cv2.INTER_AREA if downscaling else cv2.INTER_LINEAR

    resized = cv2.resize(image, (width, height), interpolation=interpolation)
    logger.debug("Resized image from (%d, %d) to (%d, %d)", original_h, original_w, height, width)
    return resized


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalise a ``uint8`` image to ``float32`` in the range ``[0.0, 1.0]``.

    Args:
        image: Input image array with dtype ``uint8``.

    Returns:
        Normalised image array with the same shape, dtype ``float32``.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* dtype is not ``uint8``.
    """
    _validate_image(image)
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image.dtype}")

    return (image / 255.0).astype(np.float32)


def _validate_image(image: np.ndarray) -> None:
    """Raise an error if *image* is not a valid NumPy image array.

    Args:
        image: Value to validate.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* has fewer than 2 dimensions.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image).__name__}")
    if image.ndim < 2:
        raise ValueError(f"Image must have at least 2 dimensions, got {image.ndim}")
