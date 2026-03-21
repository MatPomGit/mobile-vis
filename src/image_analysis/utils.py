"""General utility functions for the image_analysis package.

Covers logging setup, path helpers, and common image validation.
"""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import numpy as np

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


def validate_image(image: np.ndarray) -> None:
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
        raise ValueError(
            f"3-D image must have 1, 3, or 4 channels, got {image.shape[2]}"
        )
    if image.ndim not in (2, 3):
        raise ValueError(f"Image must be 2-D or 3-D, got {image.ndim}-D")
    if image.dtype not in (np.uint8, np.float32):
        raise ValueError(f"Expected dtype uint8 or float32, got {image.dtype}")


def safe_makedirs(directory: str | Path) -> Path:
    """Create *directory* (and any parents) if it does not already exist.

    Args:
        directory: Directory path to create.

    Returns:
        Resolved absolute ``Path`` to the directory.
    """
    path = Path(directory).resolve()
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
