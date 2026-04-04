"""Plane detection utilities from monocular RGB images.

Provides helpers for detecting planar surfaces in grayscale or BGR images
using vanishing-point analysis and RANSAC-based line clustering.

The algorithm works as follows:
1. Apply Canny edge detection and extract line segments with HoughLinesP.
2. Cluster line segments by angle to find groups of parallel lines.
3. Compute vanishing points as the intersections of parallel-line clusters.
4. Derive plane normals from pairs of vanishing points.
5. Use RANSAC on 3-D point clouds (when provided) to fit plane equations.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .utils import validate_image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Minimum line-segment length accepted by HoughLinesP (pixels).
MIN_LINE_LENGTH: int = 30

# Maximum allowed gap between collinear line-segment pixels (pixels).
MAX_LINE_GAP: int = 10

# Minimum accumulator votes in the Hough transform to keep a line.
HOUGH_THRESHOLD: int = 50

# Maximum angular difference (degrees) between two lines to treat them as
# parallel (same direction cluster).
ANGLE_CLUSTER_TOLERANCE: float = 5.0

# Multiplier applied to RANSAC_THRESHOLD when testing whether a vanishing
# point lies close enough to a line to count that line as an inlier.
# A wider tolerance is used here than for plane fitting because vanishing
# points can lie far outside the image bounds.
VP_INLIER_DISTANCE_MULTIPLIER: float = 5.0

# Minimum pixel distance from a line to treat a point as an inlier when
# computing a RANSAC-based vanishing point.
RANSAC_THRESHOLD: float = 3.0
RANSAC_MAX_ITER: int = 1000

# Minimum fraction of inliers required to accept a RANSAC-fit plane.
MIN_INLIER_FRACTION: float = 0.3

# Default maximum number of planes returned by detect_planes.
MAX_PLANES: int = 3

# BGR colours used to draw successive detected planes.
PLANE_COLORS: list[tuple[int, int, int]] = [
    (0, 255, 0),    # green
    (0, 0, 255),    # red
    (255, 165, 0),  # orange
]

# Thickness (pixels) of the normal-vector arrow drawn on each plane.
NORMAL_ARROW_THICKNESS: int = 2

# Length (pixels) of the normal-vector arrow drawn on each plane.
NORMAL_ARROW_LENGTH: int = 60

# Transparency (alpha) for the plane-mask overlay drawn on top of the image.
DEFAULT_OVERLAY_ALPHA: float = 0.35

# Font used for confidence labels.
LABEL_FONT: int = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE: float = 0.5
LABEL_THICKNESS: int = 1

# Epsilon used to guard against division by near-zero segment lengths.
MIN_SEGMENT_LENGTH_EPSILON: float = 1e-10

# Hard cap on the number of lines used per vanishing point when fitting a
# plane.  Once this many lines have been accumulated for a VP the algorithm
# stops adding more, regardless of the current precision, to bound CPU usage.
MAX_LINES_PER_VP: int = 50

# Precision threshold [0, 1] for early stopping in VP line selection.
# When the mean perpendicular VP-to-line distance already yields a precision
# at or above this value, no further lines are added to the VP even if more
# inliers are available.  Higher values demand a tighter geometric fit before
# stopping; lower values accept a rougher estimate and stop sooner.
VP_FIT_PRECISION_THRESHOLD: float = 0.85


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VanishingPoint:
    """A vanishing point detected from groups of parallel line segments.

    Attributes:
        point: Image coordinates ``(x, y)`` of the vanishing point.
        lines: Line-segment endpoints ``[(x1, y1, x2, y2), ...]`` that
            contributed to this vanishing point.
        confidence: Quality score in ``[0, 1]``.  Higher values indicate
            more consistent angular agreement among the contributing lines.
    """

    point: tuple[float, float]
    lines: list[tuple[int, int, int, int]]
    confidence: float = field(default=0.0)


@dataclass(frozen=True)
class PlaneDetection:
    """A single planar surface detected in an image.

    Attributes:
        normal: Unit normal vector ``(nx, ny, nz)`` of the detected plane
            in camera space.  ``(0, 0, 1)`` points directly away from the
            camera along the optical axis.
        centroid: Image-space centroid ``(cx, cy)`` of the plane region.
        confidence: Quality score in ``[0, 1]``.  For vanishing-point
            planes this is the mean confidence of the constituent vanishing
            points; for RANSAC planes it equals the inlier fraction.
        mask: Optional binary mask with shape ``(H, W)`` and dtype
            ``uint8`` where non-zero pixels belong to this plane.
        bbox: Axis-aligned bounding box ``(x1, y1, x2, y2)`` in pixels.
        inlier_count: Number of points (lines or 3-D points) used as
            inliers when fitting this plane.
        precision: Geometric fit quality in ``[0, 1]``.  Measures how
            tightly the inlier lines converge to the plane's vanishing
            points.  A value of ``1.0`` indicates perfect convergence
            (all inlier lines pass exactly through their respective VP);
            ``0.0`` indicates the worst-case alignment.  Defaults to
            ``0.0`` for planes constructed without an explicit precision
            estimate.
    """

    normal: tuple[float, float, float]
    centroid: tuple[float, float]
    confidence: float
    mask: NDArray[np.uint8] | None
    bbox: tuple[int, int, int, int]
    inlier_count: int
    precision: float = field(default=0.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_vanishing_points(
    image: NDArray[np.uint8] | NDArray[np.float32],
    n_points: int = 2,
) -> list[VanishingPoint]:
    """Detect vanishing points in *image* from groups of parallel lines.

    Uses ``HoughLinesP`` to extract line segments, clusters them by
    orientation, and then computes a RANSAC-estimated intersection for each
    cluster as the vanishing point.

    Args:
        image: Grayscale ``(H, W)`` or BGR ``(H, W, 3)`` image with dtype
            ``uint8 [0, 255]`` or ``float32 [0.0, 1.0]``.
        n_points: Maximum number of vanishing points to return.  Must be
            at least 1.

    Returns:
        List of :class:`VanishingPoint` objects (at most *n_points*)
        sorted by descending confidence.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* has an unsupported shape or dtype.
        ValueError: If *n_points* is less than 1.
    """
    validate_image(image)
    if n_points < 1:
        raise ValueError(f"n_points must be at least 1, got {n_points}")

    gray = _to_grayscale_uint8(image)
    lines = _detect_lines(gray)

    if lines is None or len(lines) == 0:
        logger.debug("No lines found; returning empty vanishing-point list")
        return []

    clusters = _cluster_lines_by_direction(lines, ANGLE_CLUSTER_TOLERANCE)
    vanishing_points: list[VanishingPoint] = []

    for cluster in sorted(clusters, key=len, reverse=True):
        if len(vanishing_points) >= n_points:
            break
        vp = _intersect_lines(cluster)
        if vp is None:
            continue
        vx, vy = vp
        vp_threshold = RANSAC_THRESHOLD * VP_INLIER_DISTANCE_MULTIPLIER
        inlier_lines = _inlier_lines_for_vp(cluster, vx, vy, vp_threshold)
        if not inlier_lines:
            inlier_lines = cluster
        confidence = min(1.0, len(inlier_lines) / max(1, len(lines)))
        vanishing_points.append(
            VanishingPoint(
                point=(vx, vy),
                lines=inlier_lines,
                confidence=confidence,
            )
        )

    vanishing_points.sort(key=lambda vp: vp.confidence, reverse=True)
    logger.debug("Detected %d vanishing point(s)", len(vanishing_points))
    return vanishing_points


def detect_planes(
    image: NDArray[np.uint8] | NDArray[np.float32],
    camera_matrix: NDArray[np.float64] | None = None,
    max_planes: int = MAX_PLANES,
    min_inliers: int = 5,
) -> list[PlaneDetection]:
    """Detect planar surfaces in *image* via vanishing-point analysis.

    The algorithm detects up to *max_planes* planes by:

    1. Detecting vanishing points from groups of parallel line segments.
    2. For each pair of vanishing points, computing a plane normal via the
       cross product of their directions in camera space (if
       *camera_matrix* is provided) or falling back to a 2-D image-space
       normal.
    3. Building a binary mask by marking pixels whose local gradient
       direction is consistent with the plane normal.

    Args:
        image: Grayscale ``(H, W)`` or BGR ``(H, W, 3)`` image with dtype
            ``uint8 [0, 255]`` or ``float32 [0.0, 1.0]``.
        camera_matrix: Optional ``(3, 3)`` intrinsic camera matrix of
            dtype ``float64``.  When provided, plane normals are expressed
            in 3-D camera space; otherwise they are approximated in 2-D
            image space with ``nz = 1``.
        max_planes: Maximum number of :class:`PlaneDetection` objects to
            return.  Must be at least 1.
        min_inliers: Minimum number of inlier line segments required to
            accept a detected plane.  Must be at least 1.

    Returns:
        List of :class:`PlaneDetection` objects (at most *max_planes*)
        sorted by descending confidence.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* has an unsupported shape or dtype.
        ValueError: If *camera_matrix* is not a ``(3, 3)`` array.
        ValueError: If *max_planes* is less than 1.
        ValueError: If *min_inliers* is less than 1.
    """
    validate_image(image)
    if max_planes < 1:
        raise ValueError(f"max_planes must be at least 1, got {max_planes}")
    if min_inliers < 1:
        raise ValueError(f"min_inliers must be at least 1, got {min_inliers}")

    cm: NDArray[np.float64] | None = None
    if camera_matrix is not None:
        cm = np.asarray(camera_matrix, dtype=np.float64)
        if cm.shape != (3, 3):
            raise ValueError(f"camera_matrix must have shape (3, 3), got {cm.shape}")

    height, width = image.shape[:2]
    gray = _to_grayscale_uint8(image)
    lines = _detect_lines(gray)

    if lines is None or len(lines) == 0:
        logger.debug("No lines found; returning empty plane list")
        return []

    clusters = _cluster_lines_by_direction(lines, ANGLE_CLUSTER_TOLERANCE)
    # Need at least 2 clusters to form a vanishing-point pair.
    clusters = [c for c in clusters if len(c) >= 2]

    planes: list[PlaneDetection] = []
    used_clusters: set[int] = set()
    vp_threshold = RANSAC_THRESHOLD * VP_INLIER_DISTANCE_MULTIPLIER

    for i in range(len(clusters)):
        if len(planes) >= max_planes:
            break
        for j in range(i + 1, len(clusters)):
            if len(planes) >= max_planes:
                break
            if i in used_clusters or j in used_clusters:
                continue

            vp1 = _intersect_lines(clusters[i])
            vp2 = _intersect_lines(clusters[j])
            if vp1 is None or vp2 is None:
                continue

            # Select the tightest-fitting subset of inlier lines for each VP,
            # with early stopping when precision is already high enough and a
            # hard cap to avoid accumulating too many lines unnecessarily.
            inliers_i, prec_i = _select_lines_for_vp(
                clusters[i], vp1[0], vp1[1], vp_threshold
            )
            inliers_j, prec_j = _select_lines_for_vp(
                clusters[j], vp2[0], vp2[1], vp_threshold
            )
            if not inliers_i:
                logger.debug(
                    "No inliers found for cluster %d VP; falling back to full cluster", i
                )
                inliers_i = clusters[i]
                prec_i = 0.0
            if not inliers_j:
                logger.debug(
                    "No inliers found for cluster %d VP; falling back to full cluster", j
                )
                inliers_j = clusters[j]
                prec_j = 0.0

            inlier_lines = inliers_i + inliers_j
            if len(inlier_lines) < min_inliers:
                continue

            normal = _compute_plane_normal(vp1, vp2, cm)
            mask = _build_line_mask(inlier_lines, height, width)
            bbox = _mask_to_bbox(mask)
            if bbox is None:
                bbox = (0, 0, width, height)
            cx = float((bbox[0] + bbox[2]) / 2)
            cy = float((bbox[1] + bbox[3]) / 2)
            confidence = min(1.0, len(inlier_lines) / max(1, len(lines)))
            precision = (prec_i + prec_j) / 2.0

            planes.append(
                PlaneDetection(
                    normal=normal,
                    centroid=(cx, cy),
                    confidence=confidence,
                    mask=mask,
                    bbox=bbox,
                    inlier_count=len(inlier_lines),
                    precision=precision,
                )
            )
            used_clusters.add(i)
            used_clusters.add(j)

    planes.sort(key=lambda p: p.confidence, reverse=True)
    planes = _resolve_mask_overlaps(planes)
    logger.debug("Detected %d plane(s)", len(planes))
    return planes[: max_planes]


def fit_plane_ransac(
    points: NDArray[np.float64],
    threshold: float = RANSAC_THRESHOLD,
    max_iter: int = RANSAC_MAX_ITER,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Fit a plane to a 3-D point cloud using RANSAC.

    The plane is parameterised by its unit normal vector ``n`` such that
    ``n · p = d`` for all inlier points ``p``.

    Args:
        points: Array of 3-D points with shape ``(N, 3)`` and dtype
            compatible with ``float64``.  Requires at least 3 points.
        threshold: Maximum orthogonal distance (in the same units as
            *points*) for a point to be counted as an inlier.  Must be
            positive.
        max_iter: Maximum number of RANSAC iterations.  Must be positive.

    Returns:
        ``(normal, inliers_mask)`` where *normal* is a unit vector of
        shape ``(3,)`` and *inliers_mask* is a boolean array of shape
        ``(N,)`` that is ``True`` for inlier points.

    Raises:
        TypeError: If *points* is not a ``np.ndarray``.
        ValueError: If *points* does not have shape ``(N, 3)`` with
            ``N >= 3``.
        ValueError: If *threshold* is not positive.
        ValueError: If *max_iter* is not positive.
    """
    if not isinstance(points, np.ndarray):
        raise TypeError(f"Expected np.ndarray for points, got {type(points).__name__}")
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    if pts.shape[0] < 3:
        raise ValueError(f"At least 3 points required, got {pts.shape[0]}")
    if threshold <= 0.0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}")

    n_pts = pts.shape[0]
    rng = np.random.default_rng(seed=0)

    best_normal: NDArray[np.float64] = np.array([0.0, 0.0, 1.0])
    best_inliers: NDArray[np.bool_] = np.zeros(n_pts, dtype=bool)
    best_count = 0

    for _ in range(max_iter):
        sample_idx = rng.choice(n_pts, size=3, replace=False)
        sample = pts[sample_idx]

        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]
        normal_raw = np.cross(v1, v2)
        norm = float(np.linalg.norm(normal_raw))
        if norm < 1e-10:
            continue
        normal = cast(NDArray[np.float64], normal_raw / norm)

        distances = np.abs(pts @ normal - float(np.dot(sample[0], normal)))
        inliers: NDArray[np.bool_] = cast(NDArray[np.bool_], distances < threshold)
        count = int(np.sum(inliers))

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_normal = normal

    # Refine with all inliers via SVD
    if best_count >= 3:
        inlier_pts = pts[best_inliers]
        centroid = inlier_pts.mean(axis=0)
        _, _, vh = np.linalg.svd(inlier_pts - centroid)
        best_normal = cast(NDArray[np.float64], vh[-1])

    logger.debug("RANSAC plane fit: %d / %d inliers", best_count, n_pts)
    return best_normal, best_inliers


def draw_planes(
    image: NDArray[np.uint8],
    planes: list[PlaneDetection],
    alpha: float = DEFAULT_OVERLAY_ALPHA,
) -> NDArray[np.uint8]:
    """Draw plane masks, normals, bounding boxes, and confidence labels.

    Renders each detected plane as a semi-transparent colour overlay, an
    arrow showing the projected normal direction, the bounding rectangle,
    and a confidence label.

    Args:
        image: BGR image array with shape ``(H, W, 3)`` and dtype ``uint8``.
        planes: List of :class:`PlaneDetection` objects to render.
        alpha: Blending factor for the plane-mask overlay.  ``0.0`` means
            fully transparent (no mask drawn); ``1.0`` means fully opaque.
            Must be in ``[0, 1]``.

    Returns:
        Copy of *image* with plane annotations rendered on top.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* is not a BGR ``uint8`` array with shape
            ``(H, W, 3)``.
        ValueError: If *alpha* is not in ``[0, 1]``.
    """
    validate_image(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be a BGR uint8 array with shape (H, W, 3)")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    output = image.copy()
    h, w = image.shape[:2]

    for idx, plane in enumerate(planes):
        color = PLANE_COLORS[idx % len(PLANE_COLORS)]

        # Draw semi-transparent mask overlay
        if plane.mask is not None and alpha > 0.0:
            overlay = output.copy()
            pmask = plane.mask
            if pmask.shape[:2] != (h, w):
                pmask = cv2.resize(pmask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_bool = pmask > 0
            overlay[mask_bool] = color
            cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0.0, output)

        # Draw bounding box
        x1, y1, x2, y2 = plane.bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        # Draw projected normal arrow from centroid
        cx = round(plane.centroid[0])
        cy = round(plane.centroid[1])
        nx, ny, _ = plane.normal
        norm_xy = math.hypot(nx, ny)
        if norm_xy > 1e-6:
            dx = round(nx / norm_xy * NORMAL_ARROW_LENGTH)
            dy = round(ny / norm_xy * NORMAL_ARROW_LENGTH)
        else:
            dx, dy = 0, -NORMAL_ARROW_LENGTH
        cv2.arrowedLine(
            output,
            (cx, cy),
            (cx + dx, cy + dy),
            color,
            NORMAL_ARROW_THICKNESS,
            tipLength=0.3,
        )

        # Draw confidence and precision label
        label = f"P{idx + 1} conf:{plane.confidence:.2f} precision:{plane.precision:.2f}"
        label_pos = (max(x1, 0), max(y1 - 6, 12))
        cv2.putText(
            output,
            label,
            label_pos,
            LABEL_FONT,
            LABEL_FONT_SCALE,
            color,
            LABEL_THICKNESS,
        )

    return output


def estimate_plane_pose(
    plane: PlaneDetection,
    camera_matrix: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estimate a rotation and translation for a detected plane.

    Projects four corner points of the plane bounding box onto a unit
    square in 3-D space (at ``z = 0`` depth), then solves the PnP problem
    to recover the 6-DoF pose.

    Args:
        plane: A :class:`PlaneDetection` from :func:`detect_planes`.
        camera_matrix: ``(3, 3)`` intrinsic camera matrix of dtype
            ``float64``.

    Returns:
        ``(rvec, tvec)`` - rotation vector ``(3, 1)`` and translation
        vector ``(3, 1)`` in camera coordinates, both as ``float64``
        NumPy arrays.

    Raises:
        TypeError: If *plane* is not a :class:`PlaneDetection`.
        ValueError: If *camera_matrix* is not a ``(3, 3)`` float64 array.
        RuntimeError: If ``cv2.solvePnP`` fails to converge.
    """
    if not isinstance(plane, PlaneDetection):
        raise TypeError(f"Expected PlaneDetection, got {type(plane).__name__}")
    cm = np.asarray(camera_matrix, dtype=np.float64)
    if cm.shape != (3, 3):
        raise ValueError(f"camera_matrix must have shape (3, 3), got {cm.shape}")

    x1, y1, x2, y2 = plane.bbox
    half_w = (x2 - x1) / 2.0
    half_h = (y2 - y1) / 2.0

    # Physical 3-D corners of a unit rectangle on the z = 0 plane.
    object_points = np.array(
        [
            [-half_w, -half_h, 0.0],
            [half_w, -half_h, 0.0],
            [half_w, half_h, 0.0],
            [-half_w, half_h, 0.0],
        ],
        dtype=np.float64,
    )

    # Corresponding image-space corners (bbox corners).
    image_points = np.array(
        [
            [float(x1), float(y1)],
            [float(x2), float(y1)],
            [float(x2), float(y2)],
            [float(x1), float(y2)],
        ],
        dtype=np.float64,
    )

    dist_coeffs = np.zeros(5, dtype=np.float64)
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, cm, dist_coeffs)
    if not success:
        raise RuntimeError("cv2.solvePnP failed to converge for the given PlaneDetection.")

    return (
        cast(NDArray[np.float64], rvec),
        cast(NDArray[np.float64], tvec),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_grayscale_uint8(
    image: NDArray[np.uint8] | NDArray[np.float32],
) -> NDArray[np.uint8]:
    """Convert any supported image format to a grayscale uint8 array."""
    arr = np.asarray(image)
    if arr.dtype == np.float32:
        arr_uint8 = cast(
            NDArray[np.uint8],
            np.asarray(np.round(np.clip(arr, 0.0, 1.0) * 255.0), dtype=np.uint8),
        )
    else:
        arr_uint8 = cast(NDArray[np.uint8], np.asarray(arr, dtype=np.uint8))

    if arr_uint8.ndim == 2:
        return arr_uint8
    if arr_uint8.ndim == 3 and arr_uint8.shape[2] == 1:
        return arr_uint8[:, :, 0]
    if arr_uint8.ndim == 3 and arr_uint8.shape[2] in (3, 4):
        code = cv2.COLOR_BGRA2GRAY if arr_uint8.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        return cast(NDArray[np.uint8], cv2.cvtColor(arr_uint8, code))
    raise ValueError(f"Unsupported image shape for plane detection: {image.shape}")


def _detect_lines(gray: NDArray[np.uint8]) -> NDArray[np.int32] | None:
    """Return HoughLinesP line segments detected in a grayscale image.

    Args:
        gray: Grayscale uint8 image ``(H, W)``.

    Returns:
        Array of shape ``(N, 1, 4)`` with ``(x1, y1, x2, y2)`` per row,
        or ``None`` if no lines were found.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines: NDArray[np.int32] | None = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=float(np.pi) / 180.0,
        threshold=HOUGH_THRESHOLD,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP,
    )
    return lines


def _line_angle_deg(x1: int, y1: int, x2: int, y2: int) -> float:
    """Return the orientation angle of a line segment in degrees ``[0, 180)``.

    Args:
        x1: Start x-coordinate.
        y1: Start y-coordinate.
        x2: End x-coordinate.
        y2: End y-coordinate.

    Returns:
        Angle in degrees in the range ``[0, 180)``.
    """
    angle = math.degrees(math.atan2(float(y2 - y1), float(x2 - x1))) % 180.0
    return angle


def _cluster_lines_by_direction(
    lines: NDArray[np.int32],
    angle_tol_deg: float = ANGLE_CLUSTER_TOLERANCE,
) -> list[list[tuple[int, int, int, int]]]:
    """Cluster line segments by their orientation.

    Lines whose orientation angles differ by at most *angle_tol_deg* are
    placed in the same cluster.  Clustering is done greedily: each line is
    assigned to the first existing cluster whose mean angle is within
    tolerance, or a new cluster is started.

    Args:
        lines: Array of shape ``(N, 1, 4)`` with ``(x1, y1, x2, y2)``.
        angle_tol_deg: Maximum angular difference (degrees) to consider
            two lines as parallel.

    Returns:
        List of clusters, where each cluster is a list of
        ``(x1, y1, x2, y2)`` tuples.
    """
    clusters: list[list[tuple[int, int, int, int]]] = []
    cluster_angles: list[float] = []

    for line in lines:
        x1, y1, x2, y2 = int(line[0][0]), int(line[0][1]), int(line[0][2]), int(line[0][3])
        angle = _line_angle_deg(x1, y1, x2, y2)

        assigned = False
        for k, mean_angle in enumerate(cluster_angles):
            diff = abs(angle - mean_angle)
            diff = min(diff, 180.0 - diff)  # account for 0/180 wrap
            if diff <= angle_tol_deg:
                clusters[k].append((x1, y1, x2, y2))
                # Update mean angle (circular mean simplified for small ranges)
                n = len(clusters[k])
                cluster_angles[k] = mean_angle + (angle - mean_angle) / n
                assigned = True
                break

        if not assigned:
            clusters.append([(x1, y1, x2, y2)])
            cluster_angles.append(angle)

    return clusters


def _intersect_lines(
    line_group: list[tuple[int, int, int, int]],
) -> tuple[float, float] | None:
    """Estimate the vanishing point for a group of lines.

    The intersection is computed as the least-squares solution to the
    over-determined system formed by the line equations.  Each line
    ``ax + by = c`` contributes one equation.

    Args:
        line_group: List of ``(x1, y1, x2, y2)`` tuples.

    Returns:
        ``(vx, vy)`` coordinates of the estimated vanishing point, or
        ``None`` if the system cannot be solved.
    """
    if len(line_group) < 2:
        return None

    lhs_rows: list[list[float]] = []
    b_rows: list[float] = []

    for x1, y1, x2, y2 in line_group:
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        # Line equation: dy·x - dx·y = dy·x1 - dx·y1
        lhs_rows.append([dy, -dx])
        b_rows.append(dy * x1 - dx * y1)

    lhs = np.array(lhs_rows, dtype=np.float64)
    b = np.array(b_rows, dtype=np.float64)

    result, _residuals, rank, _ = np.linalg.lstsq(lhs, b, rcond=None)
    if rank < 2:
        return None

    return float(result[0]), float(result[1])


def _resolve_mask_overlaps(
    planes: list[PlaneDetection],
) -> list[PlaneDetection]:
    """Remove overlapping pixels from plane masks.

    Planes are assumed to be ordered by descending confidence.  Each
    pixel is assigned to the first (most confident) plane that claims
    it; later planes have those pixels cleared from their masks.
    Bounding boxes and centroids are recomputed after the update.
    Planes whose mask becomes entirely empty after conflict resolution
    are dropped from the result.

    Args:
        planes: List of :class:`PlaneDetection` objects sorted by
            descending confidence.  Objects with ``mask=None`` are
            passed through unchanged.

    Returns:
        New list of :class:`PlaneDetection` objects with non-overlapping
        masks.
    """
    if not planes:
        return planes

    # Find first mask that is not None to determine image dimensions.
    ref_mask = next((p.mask for p in planes if p.mask is not None), None)
    if ref_mask is None:
        return planes

    h, w = ref_mask.shape[:2]
    claimed = np.zeros((h, w), dtype=bool)
    result: list[PlaneDetection] = []

    for plane in planes:
        if plane.mask is None:
            result.append(plane)
            continue

        pmask = plane.mask
        if pmask.shape[:2] != (h, w):
            pmask = cv2.resize(pmask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Remove pixels already owned by a more confident plane.
        new_mask = pmask.copy()
        new_mask[claimed] = 0

        # Register this plane's remaining pixels as claimed.
        claimed |= new_mask > 0

        bbox = _mask_to_bbox(new_mask)
        if bbox is None:
            # Mask is empty after conflict resolution; drop this plane so
            # that stale bbox/centroid data does not mislead callers.
            logger.debug("Dropping plane with empty mask after overlap resolution")
            continue

        cx = float((bbox[0] + bbox[2]) / 2)
        cy = float((bbox[1] + bbox[3]) / 2)
        result.append(
            PlaneDetection(
                normal=plane.normal,
                centroid=(cx, cy),
                confidence=plane.confidence,
                mask=new_mask,
                bbox=bbox,
                inlier_count=plane.inlier_count,
                precision=plane.precision,
            )
        )

    return result


def _select_lines_for_vp(
    candidate_lines: list[tuple[int, int, int, int]],
    vx: float,
    vy: float,
    vp_threshold: float,
    *,
    max_lines: int = MAX_LINES_PER_VP,
    precision_threshold: float = VP_FIT_PRECISION_THRESHOLD,
) -> tuple[list[tuple[int, int, int, int]], float]:
    """Select the tightest-fitting subset of inlier lines for a VP.

    Lines whose perpendicular distance to the vanishing point exceeds
    *vp_threshold* are discarded.  The remaining inliers are sorted by
    ascending distance and added one by one.  The loop stops early when
    either:

    * the running precision score reaches *precision_threshold* (the VP is
      already well-determined, extra lines are unnecessary), or
    * *max_lines* lines have been accumulated (hard cap to bound CPU cost).

    Precision is defined as ``max(0, 1 - mean_dist / vp_threshold)`` where
    *mean_dist* is the mean perpendicular VP-to-line distance of the
    selected lines.  A value of ``1.0`` means all selected lines pass
    exactly through the VP; ``0.0`` means the mean distance equals the
    threshold.

    Args:
        candidate_lines: List of ``(x1, y1, x2, y2)`` tuples to consider.
        vx: Vanishing point x-coordinate.
        vy: Vanishing point y-coordinate.
        vp_threshold: Maximum perpendicular distance (pixels) from the VP
            to a line for that line to be considered an inlier.  Must be
            positive.
        max_lines: Hard cap on the number of lines to accept.
        precision_threshold: Early-stopping precision.  When the running
            precision reaches this value, no more lines are added.

    Returns:
        ``(selected_lines, precision)`` where *selected_lines* is the
        accepted subset and *precision* is in ``[0, 1]``.  Returns
        ``([], 0.0)`` when no inliers exist.
    """
    scored: list[tuple[float, tuple[int, int, int, int]]] = []
    for seg in candidate_lines:
        x1, y1, x2, y2 = seg
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = math.hypot(dx, dy)
        if length < MIN_SEGMENT_LENGTH_EPSILON:
            continue
        dist = abs(dy * (vx - x1) - dx * (vy - y1)) / length
        if dist < vp_threshold:
            scored.append((dist, seg))

    if not scored:
        return [], 0.0

    # Process closest lines first so precision improves as fast as possible.
    scored.sort(key=lambda t: t[0])

    selected: list[tuple[int, int, int, int]] = []
    cumulative_dist = 0.0
    precision = 0.0

    for dist, seg in scored:
        selected.append(seg)
        cumulative_dist += dist
        mean_dist = cumulative_dist / len(selected)
        precision = max(0.0, 1.0 - mean_dist / vp_threshold)

        if len(selected) >= max_lines or precision >= precision_threshold:
            break

    logger.debug(
        "VP (%.1f, %.1f): selected %d / %d lines, precision=%.3f",
        vx,
        vy,
        len(selected),
        len(candidate_lines),
        precision,
    )
    return selected, precision


def _inlier_lines_for_vp(
    line_group: list[tuple[int, int, int, int]],
    vx: float,
    vy: float,
    threshold: float,
) -> list[tuple[int, int, int, int]]:
    """Return lines that converge toward the vanishing point *vp*.

    A line is considered an inlier when the perpendicular distance from
    the vanishing point to the infinite line defined by the segment is
    less than *threshold* pixels.  This is the geometrically correct
    criterion: a line that truly converges to a VP passes through (or
    very close to) that VP, so the VP-to-line distance is small.

    Args:
        line_group: List of ``(x1, y1, x2, y2)`` tuples.
        vx: Vanishing point x-coordinate.
        vy: Vanishing point y-coordinate.
        threshold: Maximum perpendicular distance (pixels) from the VP
            to the line for the line to be accepted as an inlier.

    Returns:
        Subset of *line_group* that are inliers.
    """
    inliers: list[tuple[int, int, int, int]] = []
    for seg in line_group:
        x1, y1, x2, y2 = seg
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = math.hypot(dx, dy)
        if length < MIN_SEGMENT_LENGTH_EPSILON:
            continue
        # Perpendicular distance from (vx, vy) to the infinite line
        # through (x1, y1) in direction (dx, dy):
        #   dist = |dy*(vx - x1) - dx*(vy - y1)| / length
        dist = abs(dy * (vx - x1) - dx * (vy - y1)) / length
        if dist < threshold:
            inliers.append(seg)
    return inliers


def _compute_plane_normal(
    vp1: tuple[float, float],
    vp2: tuple[float, float],
    camera_matrix: NDArray[np.float64] | None,
) -> tuple[float, float, float]:
    """Compute a plane normal from two vanishing points.

    If *camera_matrix* is provided the vanishing points are back-projected
    to 3-D directions and their cross product gives the plane normal.
    Without a camera matrix a 2-D approximation is used with ``nz = 1``.

    Args:
        vp1: First vanishing point ``(x, y)`` in image coordinates.
        vp2: Second vanishing point ``(x, y)`` in image coordinates.
        camera_matrix: Optional ``(3, 3)`` intrinsic camera matrix.

    Returns:
        Unit normal vector ``(nx, ny, nz)``.
    """
    if camera_matrix is not None:
        fx = float(camera_matrix[0, 0])
        fy = float(camera_matrix[1, 1])
        cx = float(camera_matrix[0, 2])
        cy = float(camera_matrix[1, 2])
        d1 = np.array([(vp1[0] - cx) / fx, (vp1[1] - cy) / fy, 1.0])
        d2 = np.array([(vp2[0] - cx) / fx, (vp2[1] - cy) / fy, 1.0])
    else:
        d1 = np.array([vp1[0], vp1[1], 1.0])
        d2 = np.array([vp2[0], vp2[1], 1.0])

    normal_raw = np.cross(d1, d2)
    norm = float(np.linalg.norm(normal_raw))
    if norm < 1e-10:
        return (0.0, 0.0, 1.0)
    normal = normal_raw / norm
    return float(normal[0]), float(normal[1]), float(normal[2])


def _build_line_mask(
    lines: list[tuple[int, int, int, int]],
    height: int,
    width: int,
    dilation_radius: int = 5,
) -> NDArray[np.uint8]:
    """Rasterise line segments into a binary mask.

    Each line segment is drawn as a thick white stroke (thickness
    ``2 * dilation_radius + 1``) on a black background.

    Args:
        lines: List of ``(x1, y1, x2, y2)`` tuples.
        height: Mask height in pixels.
        width: Mask width in pixels.
        dilation_radius: Half-thickness of the drawn lines in pixels.

    Returns:
        Binary mask array of shape ``(H, W)`` and dtype ``uint8``.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    thickness = max(1, 2 * dilation_radius + 1)
    for x1, y1, x2, y2 in lines:
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
    return mask


def _mask_to_bbox(
    mask: NDArray[np.uint8],
) -> tuple[int, int, int, int] | None:
    """Return the bounding box of non-zero pixels in *mask*.

    Args:
        mask: Binary mask array of shape ``(H, W)`` and dtype ``uint8``.

    Returns:
        ``(x1, y1, x2, y2)`` bounding box, or ``None`` if the mask is
        empty.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
