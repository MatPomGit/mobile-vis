"""Camera calibration utilities using chessboard patterns.

Provides functions for detecting chessboard inner corners, computing camera
intrinsic parameters from multiple calibration images, and undistorting frames.
All functions work with BGR ``uint8`` NumPy arrays unless stated otherwise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .utils import validate_image

logger = logging.getLogger(__name__)

# Default inner-corner dimensions of the calibration chessboard (width x height).
DEFAULT_BOARD_WIDTH: int = 9
DEFAULT_BOARD_HEIGHT: int = 6

# Termination criteria used for sub-pixel corner refinement.
_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Minimum number of valid frames required to compute a calibration.
MIN_CALIBRATION_FRAMES: int = 3


@dataclass
class CalibrationResult:
    """Result of a camera calibration procedure.

    Attributes:
        camera_matrix: 3x3 intrinsic camera matrix (focal lengths and principal
            point) with dtype ``float64``.
        dist_coefficients: Distortion coefficients ``(k1, k2, p1, p2[, k3…])``
            as a column vector with dtype ``float64``.
        rms_error: Root-mean-square reprojection error in pixels.  Lower is
            better; values below 1.0 are considered good for most applications.
        image_size: ``(width, height)`` of images used during calibration.
        frame_count: Number of frames that contributed to the calibration.
    """

    camera_matrix: NDArray[np.float64]
    dist_coefficients: NDArray[np.float64]
    rms_error: float
    image_size: tuple[int, int]
    frame_count: int


def find_chessboard_corners(
    image: NDArray[np.uint8] | NDArray[np.float32],
    board_width: int = DEFAULT_BOARD_WIDTH,
    board_height: int = DEFAULT_BOARD_HEIGHT,
) -> NDArray[np.float32] | None:
    """Detect inner chessboard corners in *image* and refine them to sub-pixel accuracy.

    Args:
        image: Grayscale ``(H, W)`` or BGR ``(H, W, 3)`` image.
            Accepted dtypes: ``uint8 [0, 255]`` or ``float32 [0.0, 1.0]``.
        board_width: Number of inner corners along the horizontal axis.
        board_height: Number of inner corners along the vertical axis.

    Returns:
        Refined corner positions as a ``(board_width * board_height, 1, 2)``
        array with dtype ``float32``, or ``None`` if the pattern was not found.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* has an unsupported shape or dtype.
        ValueError: If *board_width* or *board_height* is not positive.
    """
    validate_image(image)
    if board_width <= 0 or board_height <= 0:
        raise ValueError(
            f"board_width and board_height must be positive, got {board_width}x{board_height}"
        )

    gray = _to_grayscale_uint8(image)
    pattern_size = (board_width, board_height)

    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )

    if not found or corners is None:
        logger.debug("Chessboard %dx%d not found", board_width, board_height)
        return None

    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _SUBPIX_CRITERIA)
    logger.debug("Found chessboard %dx%d corners", board_width, board_height)
    return np.asarray(refined, dtype=np.float32)


def calibrate_camera(
    images: list[NDArray[np.uint8]],
    board_width: int = DEFAULT_BOARD_WIDTH,
    board_height: int = DEFAULT_BOARD_HEIGHT,
    square_size: float = 1.0,
) -> CalibrationResult:
    """Compute camera intrinsic parameters from multiple chessboard images.

    All input images must share the same resolution.  Images in which the
    chessboard cannot be detected are silently skipped.

    Args:
        images: List of BGR or grayscale ``uint8`` images, each showing the
            complete calibration board.
        board_width: Number of inner corners along the horizontal axis.
        board_height: Number of inner corners along the vertical axis.
        square_size: Physical size of one square in arbitrary units.  Affects
            the scale of translation vectors; the focal lengths in the camera
            matrix are expressed in the same unit.

    Returns:
        :class:`CalibrationResult` with the camera matrix, distortion
        coefficients, RMS reprojection error, image size, and frame count.

    Raises:
        ValueError: If fewer than :data:`MIN_CALIBRATION_FRAMES` images are
            provided or contain a detectable chessboard.
        ValueError: If *board_width*, *board_height*, or *square_size* is
            not positive.
    """
    if len(images) < MIN_CALIBRATION_FRAMES:
        raise ValueError(
            f"At least {MIN_CALIBRATION_FRAMES} calibration images required, got {len(images)}"
        )
    if board_width <= 0 or board_height <= 0:
        raise ValueError(
            f"board_width and board_height must be positive, got {board_width}x{board_height}"
        )
    if square_size <= 0:
        raise ValueError(f"square_size must be positive, got {square_size}")

    # Object points template: (0,0,0), (1,0,0), …, (W-1, H-1, 0)
    obj_pts_template = np.zeros((board_width * board_height, 3), dtype=np.float32)
    obj_pts_template[:, :2] = (
        np.mgrid[0:board_width, 0:board_height].T.reshape(-1, 2) * square_size
    )

    object_points: list[NDArray[np.float32]] = []
    image_points: list[NDArray[np.float32]] = []
    img_size: tuple[int, int] | None = None

    for idx, image in enumerate(images):
        corners = find_chessboard_corners(image, board_width, board_height)
        if corners is None:
            logger.debug("Skipping image %d - chessboard not found", idx)
            continue

        h, w = image.shape[:2]
        if img_size is None:
            img_size = (w, h)
        elif img_size != (w, h):
            logger.warning(
                "Image %d has size %dx%d, expected %dx%d - skipped",
                idx,
                w,
                h,
                *img_size,
            )
            continue

        object_points.append(obj_pts_template.copy())
        image_points.append(corners)

    if len(object_points) < MIN_CALIBRATION_FRAMES:
        raise ValueError(
            f"At least {MIN_CALIBRATION_FRAMES} images with a detectable chessboard "
            f"are required, got {len(object_points)}"
        )

    assert img_size is not None  # guaranteed by the check above

    initial_camera_matrix = np.eye(3, dtype=np.float64)
    initial_dist_coeffs = np.zeros((8, 1), dtype=np.float64)
    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points,
        image_points,
        img_size,
        initial_camera_matrix,
        initial_dist_coeffs,
    )

    logger.info(
        "Calibration complete: RMS=%.4f using %d/%d images",
        rms,
        len(object_points),
        len(images),
    )
    return CalibrationResult(
        camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
        dist_coefficients=np.asarray(dist_coeffs, dtype=np.float64),
        rms_error=float(rms),
        image_size=img_size,
        frame_count=len(object_points),
    )


def undistort_image(
    image: NDArray[np.uint8],
    calibration: CalibrationResult,
) -> NDArray[np.uint8]:
    """Remove lens distortion from *image* using calibration parameters.

    Args:
        image: BGR image array with dtype ``uint8``.
        calibration: Calibration result from :func:`calibrate_camera`.

    Returns:
        Undistorted copy of *image* with the same shape and dtype.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* dtype is not ``uint8``.
    """
    validate_image(image)
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image.dtype}")

    result = cv2.undistort(
        image,
        calibration.camera_matrix,
        calibration.dist_coefficients,
    )
    return np.asarray(result, dtype=np.uint8)


def draw_chessboard_corners(
    image: NDArray[np.uint8],
    corners: NDArray[np.float32],
    board_width: int = DEFAULT_BOARD_WIDTH,
    board_height: int = DEFAULT_BOARD_HEIGHT,
    pattern_found: bool = True,
) -> NDArray[np.uint8]:
    """Draw detected chessboard corners onto a copy of *image*.

    Args:
        image: BGR image array with shape ``(H, W, 3)`` and dtype ``uint8``.
        corners: Corner array as returned by :func:`find_chessboard_corners`.
        board_width: Number of inner corners along the horizontal axis.
        board_height: Number of inner corners along the vertical axis.
        pattern_found: Whether the full pattern was detected.  Passed to
            ``cv2.drawChessboardCorners`` to control colour coding.

    Returns:
        Copy of *image* with corner overlay rendered by OpenCV.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* is not a 3-channel BGR ``uint8`` array.
        ValueError: If *board_width* or *board_height* is not positive.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image).__name__}")
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be a BGR uint8 array with shape (H, W, 3)")
    if board_width <= 0 or board_height <= 0:
        raise ValueError(
            f"board_width and board_height must be positive, got {board_width}x{board_height}"
        )

    output = image.copy()
    cv2.drawChessboardCorners(output, (board_width, board_height), corners, pattern_found)
    return output


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_grayscale_uint8(
    image: NDArray[np.uint8] | NDArray[np.float32],
) -> NDArray[np.uint8]:
    """Convert an image to a single-channel ``uint8`` grayscale array."""
    if image.dtype == np.float32:
        float_image = np.asarray(image, dtype=np.float32)
        scaled = np.round(np.clip(float_image, 0.0, 1.0) * 255)
        arr_uint8: NDArray[np.uint8] = np.asarray(scaled, dtype=np.uint8)
    else:
        arr_uint8 = np.asarray(image, dtype=np.uint8)

    if arr_uint8.ndim == 2:
        return arr_uint8
    if arr_uint8.ndim == 3 and arr_uint8.shape[2] == 1:
        return arr_uint8[:, :, 0]
    if arr_uint8.ndim == 3 and arr_uint8.shape[2] == 3:
        return cast(NDArray[np.uint8], cv2.cvtColor(arr_uint8, cv2.COLOR_BGR2GRAY))

    raise ValueError(f"Unsupported image shape for grayscale conversion: {arr_uint8.shape}")
