"""MediaPipe Iris tracking utilities.

Provides detection and visualisation of iris and eye landmarks using the
MediaPipe Face Mesh solution with landmark refinement enabled.

When ``refine_landmarks=True`` is passed to the Face Mesh model, MediaPipe
returns 478 landmarks instead of the base 468.  The additional 10 landmarks
(indices 468-477) describe the iris centres and contours:

* **Left iris** - landmarks 468-472 (centre + 4 contour points).
* **Right iris** - landmarks 473-477 (centre + 4 contour points).

These landmarks enable estimating the pupil position, iris radius, and gaze
direction.

Example::

    import cv2
    from image_analysis.iris import create_face_mesh_iris, process_iris, draw_iris_results

    with create_face_mesh_iris() as face_mesh:
        image = cv2.imread("face.jpg")
        result = process_iris(image, face_mesh)
        output = draw_iris_results(image, result)
        cv2.imwrite("output.jpg", output)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import mediapipe as mp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Iris landmark index ranges
# ---------------------------------------------------------------------------

#: Index of the left iris centre landmark.
LEFT_IRIS_CENTER_IDX: int = 468

#: Slice of left iris contour landmarks (centre + 4 contour points).
LEFT_IRIS_INDICES: tuple[int, ...] = (468, 469, 470, 471, 472)

#: Index of the right iris centre landmark.
RIGHT_IRIS_CENTER_IDX: int = 473

#: Slice of right iris contour landmarks (centre + 4 contour points).
RIGHT_IRIS_INDICES: tuple[int, ...] = (473, 474, 475, 476, 477)

#: Total number of landmarks when ``refine_landmarks=True``.
REFINED_LANDMARK_COUNT: int = 478

# ---------------------------------------------------------------------------
# Drawing constants
# ---------------------------------------------------------------------------

#: BGR colour for left-iris circle.
LEFT_IRIS_COLOUR: tuple[int, int, int] = (0, 200, 0)

#: BGR colour for right-iris circle.
RIGHT_IRIS_COLOUR: tuple[int, int, int] = (200, 0, 0)

#: BGR colour for eye-contour landmarks.
EYE_COLOUR: tuple[int, int, int] = (200, 200, 0)

#: Thickness of the iris circle outline (pixels).
IRIS_CIRCLE_THICKNESS: int = 2

#: Radius of the iris centre dot (pixels).
IRIS_CENTER_DOT_RADIUS: int = 3


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class IrisLandmark:
    """A single normalised 2-D iris landmark.

    Attributes:
        x: Horizontal position normalised to ``[0.0, 1.0]``.
        y: Vertical position normalised to ``[0.0, 1.0]``.
        z: Relative depth (smaller = closer to camera).
    """

    x: float
    y: float
    z: float


@dataclass
class IrisResult:
    """Result of MediaPipe Iris processing.

    Attributes:
        left_iris: 5 iris landmarks for the left eye (center + 4 contour points),
            or ``None`` if no face detected.
        right_iris: 5 iris landmarks for the right eye (center + 4 contour points),
            or ``None`` if no face detected.
        face_landmarks: All 478 refined face landmarks, or ``None`` if no face
            detected.
    """

    left_iris: list[IrisLandmark] | None = None
    right_iris: list[IrisLandmark] | None = None
    face_landmarks: list[IrisLandmark] | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_face_mesh_iris(
    max_num_faces: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> mp.solutions.face_mesh.FaceMesh:
    """Create a MediaPipe FaceMesh instance with iris refinement enabled.

    The returned object is a context manager::

        with create_face_mesh_iris() as face_mesh:
            result = process_iris(image, face_mesh)

    Args:
        max_num_faces: Maximum number of faces to detect per frame.
        min_detection_confidence: Minimum detection confidence (``[0.0, 1.0]``).
        min_tracking_confidence: Minimum tracking confidence (``[0.0, 1.0]``).

    Returns:
        A ``mediapipe.solutions.face_mesh.FaceMesh`` instance with
        ``refine_landmarks=True``.

    Raises:
        ImportError: If the ``mediapipe`` package is not installed.
        ValueError: If *max_num_faces* is not a positive integer.
    """
    _check_mediapipe()

    if max_num_faces < 1:
        raise ValueError(
            f"max_num_faces must be a positive integer, got {max_num_faces}"
        )

    import mediapipe as mp

    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=max_num_faces,
        refine_landmarks=True,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def process_iris(
    image: NDArray[np.uint8],
    face_mesh: mp.solutions.face_mesh.FaceMesh,
) -> IrisResult:
    """Run MediaPipe Iris (FaceMesh with refinement) on a single BGR frame.

    Processes only the *first* detected face.  If no face is detected, all
    fields of the returned :class:`IrisResult` are ``None``.

    Args:
        image: BGR image array with shape ``(H, W, 3)`` and dtype ``uint8``.
        face_mesh: FaceMesh instance created with :func:`create_face_mesh_iris`.

    Returns:
        :class:`IrisResult` with left and right iris landmarks and full face
        landmark list.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* does not have shape ``(H, W, 3)`` or dtype
            ``uint8``.
    """
    _validate_bgr_image(image)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_result = face_mesh.process(rgb)

    if not mp_result.multi_face_landmarks:
        logger.debug("Iris: no face detected")
        return IrisResult()

    # Use only the first detected face.
    landmark_list = mp_result.multi_face_landmarks[0].landmark
    num_landmarks = len(landmark_list)

    if num_landmarks < REFINED_LANDMARK_COUNT:
        logger.warning(
            "Expected %d refined landmarks for iris tracking, got %d. "
            "Ensure refine_landmarks=True was set on the FaceMesh.",
            REFINED_LANDMARK_COUNT,
            num_landmarks,
        )

    all_landmarks = [
        IrisLandmark(x=float(lm.x), y=float(lm.y), z=float(lm.z))
        for lm in landmark_list
    ]

    left_iris = _extract_iris(all_landmarks, LEFT_IRIS_INDICES)
    right_iris = _extract_iris(all_landmarks, RIGHT_IRIS_INDICES)

    logger.debug(
        "Iris: left=%s right=%s total_landmarks=%d",
        "detected" if left_iris else "none",
        "detected" if right_iris else "none",
        num_landmarks,
    )

    return IrisResult(
        left_iris=left_iris,
        right_iris=right_iris,
        face_landmarks=all_landmarks,
    )


def draw_iris_results(
    image: NDArray[np.uint8],
    result: IrisResult,
    draw_face_mesh: bool = False,
) -> NDArray[np.uint8]:
    """Draw iris circles and optional face mesh onto a copy of *image*.

    Left iris is drawn in green, right iris in red.  If *draw_face_mesh* is
    ``True``, all 478 face landmarks are overlaid as small dots.

    Args:
        image: BGR image array with shape ``(H, W, 3)`` and dtype ``uint8``.
        result: Detection result from :func:`process_iris`.
        draw_face_mesh: Whether to draw the full 478-landmark face mesh.

    Returns:
        Copy of *image* with iris overlays.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* does not have shape ``(H, W, 3)`` or dtype
            ``uint8``.
    """
    _validate_bgr_image(image)

    output = image.copy()
    h, w = output.shape[:2]

    if draw_face_mesh and result.face_landmarks:
        for lm in result.face_landmarks:
            px, py = int(lm.x * w), int(lm.y * h)
            cv2.circle(output, (px, py), 1, EYE_COLOUR, -1)

    if result.left_iris:
        _draw_iris_circle(output, result.left_iris, w, h, LEFT_IRIS_COLOUR)

    if result.right_iris:
        _draw_iris_circle(output, result.right_iris, w, h, RIGHT_IRIS_COLOUR)

    return output


def estimate_gaze_offset(
    iris_result: IrisResult,
    image_width: int,
    image_height: int,
) -> tuple[float, float] | None:
    """Estimate normalised gaze offset from the eye centres.

    Computes the average horizontal and vertical displacement of both iris
    centres relative to their respective eye-region midpoints.  A positive
    x-offset means gaze is to the right; a positive y-offset means gaze is
    downward.

    Args:
        iris_result: Detection result from :func:`process_iris`.
        image_width: Width of the source image in pixels.
        image_height: Height of the source image in pixels.

    Returns:
        ``(offset_x, offset_y)`` in normalised image coordinates, or
        ``None`` if neither iris was detected.

    Raises:
        ValueError: If *image_width* or *image_height* is not positive.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            f"image_width and image_height must be positive, "
            f"got ({image_width}, {image_height})"
        )

    offsets: list[tuple[float, float]] = []

    for iris in (iris_result.left_iris, iris_result.right_iris):
        if iris is None or len(iris) == 0:
            continue
        center = iris[0]
        contour = iris[1:]
        if not contour:
            continue
        mean_x = sum(p.x for p in contour) / len(contour)
        mean_y = sum(p.y for p in contour) / len(contour)
        offsets.append((center.x - mean_x, center.y - mean_y))

    if not offsets:
        return None

    avg_x = sum(o[0] for o in offsets) / len(offsets)
    avg_y = sum(o[1] for o in offsets) / len(offsets)
    return (avg_x, avg_y)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_mediapipe() -> None:
    """Raise :exc:`ImportError` if ``mediapipe`` is not installed."""
    try:
        import mediapipe  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'mediapipe' package is required for iris tracking. "
            "Install it with: pip install mediapipe"
        ) from exc


def _extract_iris(
    all_landmarks: list[IrisLandmark],
    indices: tuple[int, ...],
) -> list[IrisLandmark] | None:
    """Extract iris landmarks by index.

    Args:
        all_landmarks: Full landmark list.
        indices: Tuple of landmark indices to extract.

    Returns:
        Subset of *all_landmarks* for the given indices, or ``None`` if any
        index is out of range.
    """
    if any(idx >= len(all_landmarks) for idx in indices):
        return None
    return [all_landmarks[idx] for idx in indices]


def _iris_radius_pixels(
    iris: list[IrisLandmark],
    width: int,
    height: int,
) -> int:
    """Compute approximate iris radius in pixels from contour landmarks.

    Uses the mean distance from the centre point (index 0) to each of the four
    contour points (indices 1-4).

    Args:
        iris: 5 iris landmarks (centre + 4 contour points).
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Estimated radius in pixels (minimum 1).
    """
    cx, cy = iris[0].x * width, iris[0].y * height
    radii: list[float] = []
    for contour_pt in iris[1:]:
        px, py = contour_pt.x * width, contour_pt.y * height
        radii.append(((px - cx) ** 2 + (py - cy) ** 2) ** 0.5)
    if not radii:
        return 1
    return max(1, int(sum(radii) / len(radii)))


def _draw_iris_circle(
    image: NDArray[np.uint8],
    iris: list[IrisLandmark],
    width: int,
    height: int,
    colour: tuple[int, int, int],
) -> None:
    """Draw an iris circle and centre dot onto *image* in place.

    Args:
        image: BGR image array (mutated in place).
        iris: 5 iris landmarks.
        width: Image width in pixels.
        height: Image height in pixels.
        colour: BGR colour.
    """
    center_lm = iris[0]
    cx = int(center_lm.x * width)
    cy = int(center_lm.y * height)

    radius = _iris_radius_pixels(iris, width, height)
    cv2.circle(image, (cx, cy), radius, colour, IRIS_CIRCLE_THICKNESS)
    cv2.circle(image, (cx, cy), IRIS_CENTER_DOT_RADIUS, colour, -1)


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


# Rejestr publicznych symboli modułu używany przez image_analysis.__init__.
PUBLIC_EXPORTS: dict[str, str] = {
    "IrisLandmark": "IrisLandmark",
    "IrisResult": "IrisResult",
    "create_face_mesh_iris": "create_face_mesh_iris",
    "draw_iris_results": "draw_iris_results",
    "estimate_gaze_offset": "estimate_gaze_offset",
    "process_iris": "process_iris",
}
