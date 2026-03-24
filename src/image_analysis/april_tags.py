"""AprilTag detection utilities.

Provides helpers for detecting AprilTag fiducial markers in grayscale or
BGR images using OpenCV's ArUco-based AprilTag dictionaries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .utils import validate_image

logger = logging.getLogger(__name__)

APRILTAG_FAMILY_TO_DICTIONARY: dict[str, int] = {
    "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "tag36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
}
DEFAULT_APRILTAG_FAMILY = "tag36h11"


@dataclass(frozen=True)
class AprilTagDetection:
    """A single detected AprilTag marker.

    Attributes:
        tag_id: Integer identifier decoded from the tag.
        family: AprilTag family name, e.g. ``"tag36h11"``.
        corners: Tag corners as ``(top-left, top-right, bottom-right, bottom-left)``
            in pixel coordinates with shape ``(4, 2)`` and dtype ``float32``.
        center: Tag center point as ``(x, y)`` in pixel coordinates.
        bbox: Axis-aligned bounding box as ``(x1, y1, x2, y2)`` in pixel coordinates.
    """

    tag_id: int
    family: str
    corners: NDArray[np.float32]
    center: tuple[float, float]
    bbox: tuple[int, int, int, int]


def detect_april_tags(
    image: NDArray[np.uint8] | NDArray[np.float32],
    family: str = DEFAULT_APRILTAG_FAMILY,
) -> list[AprilTagDetection]:
    """Detect AprilTag markers in *image*.

    Args:
        image: Grayscale ``(H, W)``, BGR ``(H, W, 3)`` or BGRA ``(H, W, 4)`` image with dtype
            ``uint8 [0, 255]`` or ``float32 [0.0, 1.0]``.
        family: AprilTag family name. Supported values are listed in
            :data:`APRILTAG_FAMILY_TO_DICTIONARY`.

    Returns:
        List of detected AprilTags sorted by ascending ``tag_id``.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* has an unsupported shape or dtype.
        ValueError: If *family* is not supported.
    """
    validate_image(image)
    normalized_family = family.strip().lower()
    if normalized_family not in APRILTAG_FAMILY_TO_DICTIONARY:
        supported = ", ".join(sorted(APRILTAG_FAMILY_TO_DICTIONARY))
        raise ValueError(f"Unsupported AprilTag family '{family}'. Supported values: {supported}")

    grayscale_image = _to_grayscale_uint8(image)
    dictionary = cv2.aruco.getPredefinedDictionary(
        APRILTAG_FAMILY_TO_DICTIONARY[normalized_family]
    )
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(grayscale_image)

    if ids is None or len(corners) == 0:
        logger.debug("No AprilTags detected for family '%s'", normalized_family)
        return []

    detections = [
        _build_detection(cast(NDArray[np.float32], tag_corners), int(tag_id), normalized_family)
        for tag_corners, tag_id in zip(corners, ids.reshape(-1), strict=True)
    ]
    detections.sort(key=lambda detection: detection.tag_id)

    logger.debug("Detected %d AprilTags for family '%s'", len(detections), normalized_family)
    return detections


def draw_april_tags(
    image: NDArray[np.uint8],
    detections: list[AprilTagDetection],
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> NDArray[np.uint8]:
    """Draw AprilTag outlines, IDs, and center points on a copy of *image*.

    Args:
        image: BGR image array with shape ``(H, W, 3)`` and dtype ``uint8``.
        detections: AprilTag detections to render.
        color: BGR colour used to draw outlines and labels.
        thickness: Line thickness in pixels.

    Returns:
        Copy of *image* with rendered AprilTag annotations.

    Raises:
        ValueError: If *image* is not a BGR ``uint8`` array.
        ValueError: If *thickness* is not positive.
    """
    validate_image(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be a BGR uint8 array with shape (H, W, 3)")
    if thickness <= 0:
        raise ValueError(f"thickness must be positive, got {thickness}")

    output = image.copy()
    for detection in detections:
        polygon = np.round(detection.corners).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(output, [polygon], isClosed=True, color=color, thickness=thickness)
        center_x, center_y = detection.center
        center_point = (round(center_x), round(center_y))
        cv2.circle(output, center_point, radius=max(thickness, 2), color=color, thickness=-1)
        label_position = (int(polygon[0, 0, 0]), max(int(polygon[0, 0, 1]) - 8, 0))
        cv2.putText(
            output,
            f"id={detection.tag_id}",
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            thickness,
        )

    return output


def _to_grayscale_uint8(
    image: NDArray[np.uint8] | NDArray[np.float32],
) -> NDArray[np.uint8]:
    """Convert an image to grayscale uint8 for AprilTag detection."""
    image_array = np.asarray(image)
    if image_array.dtype == np.float32:
        scaled_image = np.round(np.clip(image_array, 0.0, 1.0) * 255.0)
        image_uint8 = np.asarray(scaled_image, dtype=np.uint8)
    else:
        image_uint8 = np.asarray(image_array, dtype=np.uint8)

    if image_uint8.ndim == 2:
        return image_uint8
    if image_uint8.ndim == 3:
        if image_uint8.shape[2] == 1:
            return image_uint8[:, :, 0]
        if image_uint8.shape[2] == 3:
            grayscale = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2GRAY)
            return cast(NDArray[np.uint8], grayscale)
        if image_uint8.shape[2] == 4:
            grayscale = cv2.cvtColor(image_uint8, cv2.COLOR_BGRA2GRAY)
            return cast(NDArray[np.uint8], grayscale)

    raise ValueError(f"Unsupported image shape for AprilTag detection: {image_array.shape}")


def _build_detection(
    corners: NDArray[Any],
    tag_id: int,
    family: str,
) -> AprilTagDetection:
    """Create an :class:`AprilTagDetection` from OpenCV detector output."""
    normalized_corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    x_coordinates = normalized_corners[:, 0]
    y_coordinates = normalized_corners[:, 1]
    bbox = (
        int(np.floor(x_coordinates.min())),
        int(np.floor(y_coordinates.min())),
        int(np.ceil(x_coordinates.max())),
        int(np.ceil(y_coordinates.max())),
    )
    center = (
        float(np.mean(x_coordinates)),
        float(np.mean(y_coordinates)),
    )
    return AprilTagDetection(
        tag_id=tag_id,
        family=family,
        corners=normalized_corners,
        center=center,
        bbox=bbox,
    )
