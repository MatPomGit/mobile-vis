"""General utility functions for the image_analysis package.

Covers logging setup, path helpers, and common image validation.
"""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import numpy as np

from .types import BboxXYXY

_PROJECT_ROOT: Path | None = None


def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    The root is resolved as the parent of the ``src/`` directory that
    contains this package.  The result is cached after the first call.

    Returns:
        Absolute ``Path`` to the project root.
    """
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    return _PROJECT_ROOT


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a simple console handler.

    Args:
        level: Logging level (e.g. ``logging.DEBUG``, ``logging.INFO``).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def validate_image(image: object) -> None:
    """Raise an error if *image* is not a valid NumPy image array.

    Acceptable shapes:
    - ``(H, W)``        - grayscale
    - ``(H, W, 1)``     - single-channel
    - ``(H, W, 3)``     - BGR or RGB
    - ``(H, W, 4)``     - BGRA or RGBA

    Acceptable dtypes: ``uint8``, ``float32``.

    Args:
        image: Array to validate.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If the array shape or dtype is not acceptable.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image).__name__}")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(f"3-D image must have 1, 3, or 4 channels, got {image.shape[2]}")
    if image.ndim not in (2, 3):
        raise ValueError(f"Image must be 2-D or 3-D, got {image.ndim}-D")
    if image.dtype not in (np.uint8, np.float32):
        raise ValueError(f"Expected dtype uint8 or float32, got {image.dtype}")


def validate_bgr_image(
    image: object,
    *,
    allowed_dtypes: tuple[type[np.uint8] | type[np.float32], ...] = (np.uint8, np.float32),
) -> None:
    """Validate a BGR image contract.

    Args:
        image: Value expected to be a BGR image with shape ``(H, W, 3)``.
        allowed_dtypes: Allowed dtypes for input image values.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If shape, dtype, or float32 value range is invalid.
    """
    validate_image(image)
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Expected BGR image shape (H, W, 3), "
            f"got {getattr(image, 'shape', 'N/A')}"
        )
    if image.dtype.type not in allowed_dtypes:
        allowed = ", ".join(dtype.__name__ for dtype in allowed_dtypes)
        raise ValueError(f"Expected dtype in ({allowed}), got {image.dtype}")
    if image.dtype == np.float32 and (image.min() < 0.0 or image.max() > 1.0):
        raise ValueError("Expected float32 BGR image values in range [0.0, 1.0]")


def validate_gray_image(
    image: object,
    *,
    allowed_dtypes: tuple[type[np.uint8] | type[np.float32], ...] = (np.uint8, np.float32),
) -> None:
    """Validate a grayscale image contract.

    Args:
        image: Value expected to be grayscale with shape ``(H, W)`` or ``(H, W, 1)``.
        allowed_dtypes: Allowed dtypes for input image values.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If shape, dtype, or float32 value range is invalid.
    """
    validate_image(image)
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image).__name__}")

    valid_shape = image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1)
    if not valid_shape:
        raise ValueError(
            "Expected grayscale image with shape (H, W) or (H, W, 1), "
            f"got {image.shape}"
        )
    if image.dtype.type not in allowed_dtypes:
        allowed = ", ".join(dtype.__name__ for dtype in allowed_dtypes)
        raise ValueError(f"Expected dtype in ({allowed}), got {image.dtype}")
    if image.dtype == np.float32 and (image.min() < 0.0 or image.max() > 1.0):
        raise ValueError("Expected float32 grayscale values in range [0.0, 1.0]")


def validate_bbox_xyxy(bbox: object) -> BboxXYXY:
    """Validate a bounding box in ``(x1, y1, x2, y2)`` format.

    Args:
        bbox: Candidate bounding box coordinates.

    Returns:
        Bounding box normalized to integer tuple.

    Raises:
        TypeError: If *bbox* is not a 4-item sequence of numeric values.
        ValueError: If coordinates are invalid (non-finite or inverted box).
    """
    if not isinstance(bbox, (tuple, list, np.ndarray)):
        raise TypeError(f"bbox must be tuple/list/ndarray, got {type(bbox).__name__}")
    if len(bbox) != 4:
        raise ValueError(f"bbox must have exactly 4 values, got {len(bbox)}")

    coordinates = tuple(float(value) for value in bbox)
    if not all(np.isfinite(value) for value in coordinates):
        raise ValueError(f"bbox coordinates must be finite numbers, got {bbox}")

    x1, y1, x2, y2 = coordinates
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bbox must satisfy x2 > x1 and y2 > y1, got {bbox}")
    return int(x1), int(y1), int(x2), int(y2)


def safe_makedirs(directory: str | Path) -> Path:
    """Create *directory* (and any parents) if it does not already exist.

    Args:
        directory: Directory path to create.

    Returns:
        Resolved absolute ``Path`` to the directory.
    """
    path = Path(directory).resolve()
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"Path exists and is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_images(
    directory: str | Path,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
) -> list[Path]:
    """Return a sorted list of image file paths in *directory*.

    Args:
        directory: Directory to search (non-recursive).
        extensions: Tuple of lower-case file extensions to include.

    Returns:
        Sorted list of ``Path`` objects for matching files.

    Raises:
        NotADirectoryError: If *directory* does not exist or is not a directory.
    """
    path = Path(directory)
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    return sorted(p for p in path.iterdir() if p.suffix.lower() in extensions)


# Rejestr publicznych symboli modułu używany przez image_analysis.__init__.
PUBLIC_EXPORTS: dict[str, str] = {
    "get_project_root": "get_project_root",
    "list_images": "list_images",
    "safe_makedirs": "safe_makedirs",
    "setup_logging": "setup_logging",
    "validate_bbox_xyxy": "validate_bbox_xyxy",
    "validate_bgr_image": "validate_bgr_image",
    "validate_gray_image": "validate_gray_image",
    "validate_image": "validate_image",
}
