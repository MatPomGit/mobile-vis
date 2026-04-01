"""CCTag (Circular Concentric Tag) detection utilities.

Provides helpers for detecting CCTag fiducial markers in grayscale or BGR
images using OpenCV contour analysis.

CCTag markers consist of concentric black-and-white rings.  The number of
concentric ring boundaries detected in a group determines the tag identifier.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .utils import validate_image

logger = logging.getLogger(__name__)

# Private internal type alias: (cx, cy, radius, circularity)
_Circle = tuple[float, float, float, float]

# Minimum area (pixels²) for a contour to be considered a ring candidate.
MIN_CONTOUR_AREA: float = 50.0

# Minimum circularity score [0, 1] for a contour to be treated as a ring.
# Circularity = 4π * area / perimeter².  A perfect circle scores 1.0.
MIN_CIRCULARITY: float = 0.5

# Maximum distance between two ring centres (as a fraction of the outer ring
# radius) for them to be considered concentric.
MAX_CENTRE_OFFSET_FRACTION: float = 0.25

# Minimum and maximum number of concentric ring boundaries that constitute a
# valid CCTag detection.
MIN_CCTAG_RINGS: int = 2
MAX_CCTAG_RINGS: int = 5

# Kernel size for Gaussian blur applied before binary thresholding.
BLUR_KERNEL_SIZE: tuple[int, int] = (5, 5)

# Pixel offset used to position the tag-ID label above the bounding circle.
LABEL_VERTICAL_OFFSET: int = 8

# Minimum radius (pixels) of the filled centre point drawn by draw_cc_tags.
MIN_CENTER_POINT_RADIUS: int = 2

# Font and scale used for tag-ID labels drawn by draw_cc_tags.
LABEL_FONT: int = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE: float = 0.5


@dataclass(frozen=True)
class CCTagDetection:
    """A single detected CCTag marker.

    Attributes:
        tag_id: Integer identifier equal to the number of concentric ring
            boundaries found in the marker (``MIN_CCTAG_RINGS``-``MAX_CCTAG_RINGS``).
        center: Tag centre as ``(x, y)`` in pixel coordinates.
        radius: Radius of the outermost detected ring in pixels.
        bbox: Axis-aligned bounding box as ``(x1, y1, x2, y2)`` in pixel
            coordinates, where ``(x1, y1)`` is the top-left corner and
            ``(x2, y2)`` is the bottom-right corner.
        rings_count: Number of concentric ring boundaries detected.
        confidence: Detection quality score in ``[0, 1]`` computed as the mean
            circularity of the concentric ring contours.  A higher value
            indicates more circle-like contours and a more reliable detection.
            Defaults to ``0.0`` when constructed directly without a computed
            value.
    """

    tag_id: int
    center: tuple[float, float]
    radius: float
    bbox: tuple[int, int, int, int]
    rings_count: int
    confidence: float = field(default=0.0)


def detect_cc_tags(
    image: NDArray[np.uint8] | NDArray[np.float32],
    min_circularity: float = MIN_CIRCULARITY,
    min_area: float = MIN_CONTOUR_AREA,
    use_adaptive: bool = False,
) -> list[CCTagDetection]:
    """Detect CCTag markers in *image*.

    CCTag markers are identified by finding groups of concentric circular
    contours in a Canny edge image.  The number of concentric ring boundaries
    in each group is used as the tag identifier.

    Args:
        image: Grayscale ``(H, W)`` or BGR ``(H, W, 3)`` image with dtype
            ``uint8 [0, 255]`` or ``float32 [0.0, 1.0]``.
        min_circularity: Minimum circularity score in ``[0, 1]`` for a
            contour to be treated as a ring candidate.  Higher values
            require more circle-like shapes.
        min_area: Minimum enclosed area in pixels² for a contour to be
            considered.
        use_adaptive: When ``True``, use adaptive Gaussian thresholding
            instead of global Otsu thresholding.  Adaptive thresholding is
            more robust under non-uniform illumination at the cost of
            increased computation.

    Returns:
        List of :class:`CCTagDetection` objects sorted by ascending
        ``tag_id``.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* has an unsupported shape or dtype.
        ValueError: If *min_circularity* is not in ``[0, 1]``.
        ValueError: If *min_area* is not positive.
    """
    validate_image(image)
    if not 0.0 <= min_circularity <= 1.0:
        raise ValueError(f"min_circularity must be in [0, 1], got {min_circularity}")
    if min_area <= 0.0:
        raise ValueError(f"min_area must be positive, got {min_area}")

    gray = _to_grayscale_uint8(image)
    circles = _find_circle_contours(gray, min_circularity, min_area, use_adaptive)
    groups = _group_concentric_circles(circles)
    detections = _build_detections(groups)
    detections.sort(key=lambda d: d.tag_id)

    logger.debug("Detected %d CCTag(s)", len(detections))
    return detections


def draw_cc_tags(
    image: NDArray[np.uint8],
    detections: list[CCTagDetection],
    color: tuple[int, int, int] = (0, 165, 255),
    thickness: int = 2,
) -> NDArray[np.uint8]:
    """Draw CCTag outlines, IDs, and centre points on a copy of *image*.

    Args:
        image: BGR image array with shape ``(H, W, 3)`` and dtype ``uint8``.
        detections: CCTag detections to render.
        color: BGR colour used to draw circles and labels.
        thickness: Line thickness in pixels (must be positive).

    Returns:
        Copy of *image* with rendered CCTag annotations.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* is not a BGR ``uint8`` array with shape ``(H, W, 3)``.
        ValueError: If *thickness* is not positive.
    """
    validate_image(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be a BGR uint8 array with shape (H, W, 3)")
    if thickness <= 0:
        raise ValueError(f"thickness must be positive, got {thickness}")

    output = image.copy()
    for detection in detections:
        cx = round(detection.center[0])
        cy = round(detection.center[1])
        radius = max(1, round(detection.radius))
        cv2.circle(output, (cx, cy), radius, color, thickness)
        cv2.circle(output, (cx, cy), max(thickness, MIN_CENTER_POINT_RADIUS), color, -1)
        label_position = (max(cx - radius, 0), max(cy - radius - LABEL_VERTICAL_OFFSET, 0))
        cv2.putText(
            output,
            f"id={detection.tag_id}",
            label_position,
            LABEL_FONT,
            LABEL_FONT_SCALE,
            color,
            thickness,
        )

    return output


def estimate_cctag_pose(
    detection: CCTagDetection,
    camera_matrix: NDArray[np.float64],
    dist_coeffs: NDArray[np.float64],
    tag_physical_radius_m: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estimate the 6-DoF pose of a CCTag marker via OpenCV PnP.

    Four cardinal points on the detected circle boundary are matched to
    the corresponding physical positions on the marker plane (``z = 0``)
    to compute rotation and translation vectors.

    Args:
        detection: A :class:`CCTagDetection` produced by
            :func:`detect_cc_tags`.
        camera_matrix: ``(3, 3)`` intrinsic camera matrix of dtype
            ``float64`` as returned by :func:`calibrate_camera`.
        dist_coeffs: Distortion coefficients array (length 4, 5, or 8)
            of dtype ``float64`` as returned by :func:`calibrate_camera`.
        tag_physical_radius_m: Physical radius of the outermost ring in
            **metres** (must be positive).

    Returns:
        ``(rvec, tvec)`` – rotation vector ``(3, 1)`` and translation
        vector ``(3, 1)`` in the camera coordinate frame, both as
        ``float64`` NumPy arrays.

    Raises:
        ValueError: If *tag_physical_radius_m* is not positive.
        ValueError: If *camera_matrix* is not a ``(3, 3)`` array.
        RuntimeError: If :func:`cv2.solvePnP` fails to converge.
    """
    if tag_physical_radius_m <= 0.0:
        raise ValueError(
            f"tag_physical_radius_m must be positive, got {tag_physical_radius_m}"
        )
    cm = np.asarray(camera_matrix, dtype=np.float64)
    if cm.shape != (3, 3):
        raise ValueError(f"camera_matrix must have shape (3, 3), got {cm.shape}")

    r = tag_physical_radius_m
    # Cardinal points on the physical marker plane at z = 0 (right, top, left, bottom).
    object_points = np.array(
        [[r, 0.0, 0.0], [0.0, r, 0.0], [-r, 0.0, 0.0], [0.0, -r, 0.0]],
        dtype=np.float64,
    )

    cx, cy = detection.center
    pr = detection.radius
    # Corresponding pixel positions on the detected circle boundary.
    image_points = np.array(
        [[cx + pr, cy], [cx, cy + pr], [cx - pr, cy], [cx, cy - pr]],
        dtype=np.float64,
    )

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        cm,
        np.asarray(dist_coeffs, dtype=np.float64),
    )
    if not success:
        raise RuntimeError("cv2.solvePnP failed to converge for the given CCTag detection.")

    return cast(NDArray[np.float64], rvec), cast(NDArray[np.float64], tvec)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_grayscale_uint8(
    image: NDArray[np.uint8] | NDArray[np.float32],
) -> NDArray[np.uint8]:
    """Convert an image to a grayscale uint8 array for CCTag detection."""
    image_array = np.asarray(image)
    if image_array.dtype == np.float32:
        scaled = np.round(np.clip(image_array, 0.0, 1.0) * 255.0)
        image_uint8 = cast(NDArray[np.uint8], np.asarray(scaled, dtype=np.uint8))
    else:
        image_uint8 = cast(NDArray[np.uint8], np.asarray(image_array, dtype=np.uint8))

    if image_uint8.ndim == 2:
        return image_uint8
    if image_uint8.ndim == 3 and image_uint8.shape[2] == 1:
        return image_uint8[:, :, 0]
    if image_uint8.ndim == 3 and image_uint8.shape[2] == 3:
        return cast(NDArray[np.uint8], cv2.cvtColor(image_uint8, cv2.COLOR_BGR2GRAY))
    raise ValueError(f"Unsupported image shape for CCTag detection: {image.shape}")


def _circularity(area: float, perimeter: float) -> float:
    """Compute circularity as 4π·area / perimeter².

    Args:
        area: Enclosed area of the contour.
        perimeter: Perimeter length of the contour.

    Returns:
        Circularity score in ``[0, 1]``; 1.0 for a perfect circle.
    """
    if perimeter <= 0.0:
        return 0.0
    return 4.0 * float(np.pi) * area / (perimeter * perimeter)


def _find_circle_contours(
    gray: NDArray[np.uint8],
    min_circularity: float,
    min_area: float,
    use_adaptive: bool = False,
) -> list[_Circle]:
    """Return ``(cx, cy, radius, circularity)`` for each circle-like contour in *gray*.

    A binary threshold is applied before contour extraction: Otsu's global
    method by default, or adaptive Gaussian thresholding when *use_adaptive*
    is ``True``.  Top-level contours (those with no parent in the hierarchy,
    i.e. image background regions) are excluded so that the frame boundary
    does not interfere with CCTag grouping.

    Args:
        gray: Grayscale uint8 image.
        min_circularity: Minimum circularity score for a contour to be kept.
        min_area: Minimum enclosed area for a contour to be kept.
        use_adaptive: When ``True``, use adaptive Gaussian thresholding
            instead of Otsu global thresholding.

    Returns:
        List of ``(cx, cy, radius, circularity)`` tuples for candidate ring
        contours.
    """
    blurred = cv2.GaussianBlur(gray, BLUR_KERNEL_SIZE, 0)
    if use_adaptive:
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    else:
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None or len(contours) == 0:
        return []

    circles: list[_Circle] = []
    for i, contour in enumerate(contours):
        # Skip top-level (parentless) contours - these are background regions
        # spanning the full image, not CCTag ring boundaries.
        if hierarchy[0][i][3] == -1:
            continue
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        circ = _circularity(area, perimeter)
        if circ < min_circularity:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        circles.append((float(cx), float(cy), float(radius), circ))

    return circles


def _group_concentric_circles(
    circles: list[_Circle],
) -> list[list[_Circle]]:
    """Group circles with the same centre into concentric groups.

    Circles are considered concentric if the distance between their centres
    is at most :data:`MAX_CENTRE_OFFSET_FRACTION` times the outer radius.
    Only groups with at least :data:`MIN_CCTAG_RINGS` circles are returned.

    Args:
        circles: List of ``(cx, cy, radius, circularity)`` tuples.

    Returns:
        List of concentric-circle groups, each ordered by descending radius.
    """
    if not circles:
        return []

    sorted_circles = sorted(circles, key=lambda c: c[2], reverse=True)
    groups: list[list[_Circle]] = []
    used = [False] * len(sorted_circles)

    for i, outer in enumerate(sorted_circles):
        if used[i]:
            continue
        outer_cx, outer_cy, outer_r, _ = outer
        group: list[_Circle] = [outer]

        for j, inner in enumerate(sorted_circles):
            if i == j or used[j]:
                continue
            inner_cx, inner_cy, inner_r, _ = inner
            if inner_r >= outer_r:
                continue
            dist = float(np.hypot(inner_cx - outer_cx, inner_cy - outer_cy))
            if dist <= MAX_CENTRE_OFFSET_FRACTION * outer_r:
                group.append(inner)
                used[j] = True

        if len(group) >= MIN_CCTAG_RINGS:
            used[i] = True
            groups.append(group)

    return groups


def _build_detections(
    groups: list[list[_Circle]],
) -> list[CCTagDetection]:
    """Build :class:`CCTagDetection` objects from concentric circle groups.

    The ``confidence`` of each detection is computed as the mean circularity
    of the ring contours in its group.

    Args:
        groups: List of concentric-circle groups as returned by
            :func:`_group_concentric_circles`.

    Returns:
        List of :class:`CCTagDetection` instances, one per valid group.
    """
    detections: list[CCTagDetection] = []
    for group in groups:
        rings_count = len(group)
        if rings_count > MAX_CCTAG_RINGS:
            continue
        cx, cy, outer_radius, _ = group[0]
        confidence = float(np.mean([c[3] for c in group]))
        r = int(np.ceil(outer_radius))
        bbox = (
            max(0, int(cx) - r),
            max(0, int(cy) - r),
            int(cx) + r,
            int(cy) + r,
        )
        detections.append(
            CCTagDetection(
                tag_id=rings_count,
                center=(cx, cy),
                radius=outer_radius,
                bbox=bbox,
                rings_count=rings_count,
                confidence=confidence,
            )
        )
    return detections
