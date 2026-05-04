"""Image preprocessing utilities.

Provides functions for loading, resizing, and normalising images.
All functions expect and return BGR NumPy arrays unless stated otherwise.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .types import Image, ImageF32, ImageU8
from .utils import validate_image

logger = logging.getLogger(__name__)


def load_image(path: str | Path) -> ImageU8:
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
    return cast(NDArray[np.uint8], image)


def resize_image(
    image: Image,
    width: int,
    height: int,
) -> Image:
    """Resize an image to the specified dimensions.

    Uses ``cv2.INTER_AREA`` when downscaling and ``cv2.INTER_LINEAR``
    when upscaling for best quality.

    Args:
        image: Input image with shape ``(H, W)`` or ``(H, W, C)``, dtype
            ``uint8 [0, 255]`` or ``float32 [0.0, 1.0]``.
        width: Target width in pixels. Must be positive.
        height: Target height in pixels. Must be positive.

    Returns:
        Resized image array with shape ``(height, width[, C])``.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *width* or *height* is not a positive integer.
    """
    validate_image(image)
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got width={width}, height={height}")

    original_h, original_w = image.shape[:2]
    downscaling = width < original_w or height < original_h
    interpolation = cv2.INTER_AREA if downscaling else cv2.INTER_LINEAR

    resized = cv2.resize(image, (width, height), interpolation=interpolation)
    logger.debug("Resized image from (%d, %d) to (%d, %d)", original_h, original_w, height, width)
    return cast(Image, resized)


def normalize_image(image: ImageU8) -> ImageF32:
    """Normalise a ``uint8`` image to ``float32`` in the range ``[0.0, 1.0]``.

    Args:
        image: Input image array with dtype ``uint8``.

    Returns:
        Normalised image array with the same shape, dtype ``float32``.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* dtype is not ``uint8``.
    """
    validate_image(image)
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image.dtype}")

    return (image / 255.0).astype(np.float32)


def resize_with_aspect_ratio(
    image: Image,
    max_width: int,
    max_height: int,
) -> Image:
    """Resize an image while preserving aspect ratio.

    The resulting image fits inside ``(max_height, max_width)`` and never exceeds
    these dimensions. If the original image already fits, the original size is kept.

    Args:
        image: Input image with shape ``(H, W)`` or ``(H, W, C)``, dtype
            ``uint8 [0, 255]`` or ``float32 [0.0, 1.0]``.
        max_width: Maximum output width in pixels. Must be positive.
        max_height: Maximum output height in pixels. Must be positive.

    Returns:
        Resized image with preserved aspect ratio.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If target dimensions are not positive.
    """
    validate_image(image)
    if max_width <= 0 or max_height <= 0:
        raise ValueError(
            "max_width and max_height must be positive, "
            f"got max_width={max_width}, max_height={max_height}"
        )

    original_h, original_w = image.shape[:2]
    width_scale = max_width / original_w
    height_scale = max_height / original_h
    scale = min(width_scale, height_scale, 1.0)

    # Komentarz: Zaokrąglamy rozmiar i pilnujemy minimum 1 piksela.
    target_w = max(1, round(original_w * scale))
    target_h = max(1, round(original_h * scale))

    # TODO: Rozważyć opcję "letterbox" z paddingiem, aby zachować stały canvas modelu.
    return resize_image(image, width=target_w, height=target_h)


def center_crop(
    image: Image,
    crop_width: int,
    crop_height: int,
) -> Image:
    """Crop the image around its center.

    Args:
        image: Input image with shape ``(H, W)`` or ``(H, W, C)``, dtype
            ``uint8 [0, 255]`` or ``float32 [0.0, 1.0]``.
        crop_width: Output crop width in pixels. Must be positive and <= input width.
        crop_height: Output crop height in pixels. Must be positive and <= input height.

    Returns:
        Cropped image with shape ``(crop_height, crop_width[, C])``.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If crop dimensions are invalid.
    """
    validate_image(image)
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError(
            "crop_width and crop_height must be positive, "
            f"got crop_width={crop_width}, crop_height={crop_height}"
        )

    input_h, input_w = image.shape[:2]
    if crop_width > input_w or crop_height > input_h:
        raise ValueError(
            "Crop size must not exceed input size, "
            f"got crop=({crop_width}, {crop_height}), image=({input_w}, {input_h})"
        )

    # Komentarz: Wyznaczamy okno wycinka symetrycznie względem środka obrazu.
    left = (input_w - crop_width) // 2
    top = (input_h - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height

    # TODO: Dodać tryb cropowania oparty o ROI z detekcji obiektu.
    return image[top:bottom, left:right].copy()


class ImagePreprocessor:
    """Service class grouping preprocessing operations for easier composition.

    This class exposes stateless methods so that callers can inject one
    object into pipelines instead of importing multiple free functions.
    """

    def load_image(self, path: str | Path) -> ImageU8:
        """Deleguje do funkcji modułowej `load_image`."""
        return load_image(path)

    def resize_image(self, image: Image, width: int, height: int) -> Image:
        """Deleguje do funkcji modułowej `resize_image`."""
        return resize_image(image=image, width=width, height=height)

    def normalize_image(self, image: ImageU8) -> ImageF32:
        """Deleguje do funkcji modułowej `normalize_image`."""
        return normalize_image(image)

    def resize_with_aspect_ratio(self, image: Image, max_width: int, max_height: int) -> Image:
        """Deleguje do funkcji modułowej `resize_with_aspect_ratio`."""
        return resize_with_aspect_ratio(image=image, max_width=max_width, max_height=max_height)

    def center_crop(self, image: Image, crop_width: int, crop_height: int) -> Image:
        """Deleguje do funkcji modułowej `center_crop`."""
        return center_crop(image=image, crop_width=crop_width, crop_height=crop_height)


# Wspólna instancja serwisu ułatwia korzystanie z API obiektowego bez konfiguracji.
preprocessor = ImagePreprocessor()

# Rejestr publicznych symboli modułu używany przez image_analysis.__init__.
PUBLIC_EXPORTS: dict[str, str] = {
    "center_crop": "center_crop",
    "load_image": "load_image",
    "normalize_image": "normalize_image",
    "resize_image": "resize_image",
    "resize_with_aspect_ratio": "resize_with_aspect_ratio",
    "ImagePreprocessor": "ImagePreprocessor",
    "preprocessor": "preprocessor",
}
