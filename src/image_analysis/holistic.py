"""MediaPipe Holistic tracking utilities.

Provides detection and visualisation of human body pose, head orientation,
hand landmarks and face mesh using the MediaPipe Holistic pipeline.

The MediaPipe ``Holistic`` solution processes a single RGB image and returns
normalised landmarks for the full body (33 pose landmarks), left hand, right
hand (21 landmarks each) and face mesh (468 landmarks).  Landmarks are
normalised to the range ``[0.0, 1.0]`` relative to image dimensions.

Example::

    import cv2
    from image_analysis.holistic import create_holistic, process_holistic, draw_holistic_results

    with create_holistic() as holistic:
        image = cv2.imread("person.jpg")
        result = process_holistic(image, holistic)
        output = draw_holistic_results(image, result)
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
# Drawing constants
# ---------------------------------------------------------------------------

#: Radius (pixels) of the landmark dot drawn on the image.
LANDMARK_RADIUS: int = 3

#: Line thickness (pixels) for skeleton connections.
CONNECTION_THICKNESS: int = 2

#: BGR colour for pose landmarks.
POSE_COLOUR: tuple[int, int, int] = (0, 255, 0)

#: BGR colour for left-hand landmarks.
LEFT_HAND_COLOUR: tuple[int, int, int] = (255, 128, 0)

#: BGR colour for right-hand landmarks.
RIGHT_HAND_COLOUR: tuple[int, int, int] = (0, 128, 255)

#: BGR colour for face-mesh landmarks.
FACE_COLOUR: tuple[int, int, int] = (200, 200, 200)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class HolisticLandmark:
    """A single normalised 3-D landmark.

    Attributes:
        x: Horizontal position normalised to ``[0.0, 1.0]``.
        y: Vertical position normalised to ``[0.0, 1.0]``.
        z: Depth estimate (relative scale); smaller value is closer.
        visibility: Confidence that the landmark is visible (``0.0``-``1.0``).
    """

    x: float
    y: float
    z: float
    visibility: float = 1.0


@dataclass
class HolisticResult:
    """Result of a MediaPipe Holistic inference pass.

    Attributes:
        pose_landmarks: 33 body-pose landmarks or ``None`` if no person detected.
        left_hand_landmarks: 21 left-hand landmarks or ``None`` if not detected.
        right_hand_landmarks: 21 right-hand landmarks or ``None`` if not detected.
        face_landmarks: 468 face-mesh landmarks or ``None`` if no face detected.
    """

    pose_landmarks: list[HolisticLandmark] | None = None
    left_hand_landmarks: list[HolisticLandmark] | None = None
    right_hand_landmarks: list[HolisticLandmark] | None = None
    face_landmarks: list[HolisticLandmark] | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_holistic(
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    smooth_landmarks: bool = True,
    enable_segmentation: bool = False,
) -> mp.solutions.holistic.Holistic:
    """Create a MediaPipe Holistic context manager instance.

    The returned object is a context manager; use it with ``with`` to ensure
    resources are released:

    .. code-block:: python

        with create_holistic() as holistic:
            result = process_holistic(image, holistic)

    Args:
        model_complexity: Complexity of the pose landmark model:
            ``0`` = Lite, ``1`` = Full, ``2`` = Heavy.
        min_detection_confidence: Minimum confidence value (``[0.0, 1.0]``)
            for the person-detection model to be considered successful.
        min_tracking_confidence: Minimum confidence value (``[0.0, 1.0]``)
            for the landmark-tracking model.
        smooth_landmarks: If ``True``, applies a low-pass filter to reduce
            jitter in pose landmarks across frames.
        enable_segmentation: If ``True``, generates a segmentation mask
            alongside landmarks.

    Returns:
        A ``mediapipe.solutions.holistic.Holistic`` instance.

    Raises:
        ImportError: If the ``mediapipe`` package is not installed.
        ValueError: If *model_complexity* is not 0, 1, or 2.
    """
    _check_mediapipe()

    if model_complexity not in (0, 1, 2):
        raise ValueError(
            f"model_complexity must be 0, 1, or 2, got {model_complexity}"
        )

    import mediapipe as mp

    return mp.solutions.holistic.Holistic(
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        smooth_landmarks=smooth_landmarks,
        enable_segmentation=enable_segmentation,
    )


def process_holistic(
    image: NDArray[np.uint8],
    holistic: mp.solutions.holistic.Holistic,
) -> HolisticResult:
    """Run the MediaPipe Holistic pipeline on a single BGR frame.

    The image is converted to RGB internally before passing to MediaPipe; the
    original array is **not** mutated.

    Args:
        image: BGR image array with shape ``(H, W, 3)`` and dtype ``uint8``.
        holistic: MediaPipe Holistic instance (from :func:`create_holistic`).

    Returns:
        :class:`HolisticResult` containing detected landmarks.  Any component
        that was not detected is represented as ``None``.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* does not have shape ``(H, W, 3)`` or dtype
            ``uint8``.
    """
    _validate_bgr_image(image)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_result = holistic.process(rgb)

    pose = _extract_landmark_list(mp_result.pose_landmarks, with_visibility=True)
    left_hand = _extract_landmark_list(mp_result.left_hand_landmarks)
    right_hand = _extract_landmark_list(mp_result.right_hand_landmarks)
    face = _extract_landmark_list(mp_result.face_landmarks)

    logger.debug(
        "Holistic: pose=%s, left_hand=%s, right_hand=%s, face=%s",
        "detected" if pose else "none",
        "detected" if left_hand else "none",
        "detected" if right_hand else "none",
        "detected" if face else "none",
    )

    return HolisticResult(
        pose_landmarks=pose,
        left_hand_landmarks=left_hand,
        right_hand_landmarks=right_hand,
        face_landmarks=face,
    )


def draw_holistic_results(
    image: NDArray[np.uint8],
    result: HolisticResult,
    draw_pose: bool = True,
    draw_hands: bool = True,
    draw_face: bool = True,
) -> NDArray[np.uint8]:
    """Draw holistic landmarks onto a copy of *image*.

    Pose landmarks are drawn in green, left-hand landmarks in orange, right-hand
    landmarks in blue, and face-mesh landmarks in light grey.  Connections between
    landmarks are rendered using the standard MediaPipe drawing utilities when
    available; otherwise individual dots are drawn.

    Args:
        image: BGR image array with shape ``(H, W, 3)`` and dtype ``uint8``.
        result: Detection result from :func:`process_holistic`.
        draw_pose: Whether to draw body-pose landmarks.
        draw_hands: Whether to draw hand landmarks.
        draw_face: Whether to draw face-mesh landmarks.

    Returns:
        Copy of *image* with landmarks overlaid.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* does not have shape ``(H, W, 3)`` or dtype
            ``uint8``.
    """
    _validate_bgr_image(image)

    output = image.copy()
    h, w = output.shape[:2]

    if draw_face and result.face_landmarks:
        _draw_landmarks_dots(output, result.face_landmarks, w, h, FACE_COLOUR, radius=1)

    if draw_pose and result.pose_landmarks:
        _draw_landmarks_dots(
            output, result.pose_landmarks, w, h, POSE_COLOUR, radius=LANDMARK_RADIUS
        )
        _draw_pose_connections(output, result.pose_landmarks, w, h)

    if draw_hands:
        if result.left_hand_landmarks:
            _draw_landmarks_dots(
                output,
                result.left_hand_landmarks,
                w,
                h,
                LEFT_HAND_COLOUR,
                radius=LANDMARK_RADIUS,
            )
            _draw_hand_connections(output, result.left_hand_landmarks, w, h, LEFT_HAND_COLOUR)
        if result.right_hand_landmarks:
            _draw_landmarks_dots(
                output,
                result.right_hand_landmarks,
                w,
                h,
                RIGHT_HAND_COLOUR,
                radius=LANDMARK_RADIUS,
            )
            _draw_hand_connections(output, result.right_hand_landmarks, w, h, RIGHT_HAND_COLOUR)

    return output


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# MediaPipe pose landmark connections (pairs of landmark indices).
_POSE_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

# MediaPipe hand landmark connections (finger bones in order).
_HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def _check_mediapipe() -> None:
    """Raise :exc:`ImportError` if ``mediapipe`` is not installed."""
    try:
        import mediapipe  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'mediapipe' package is required for holistic tracking. "
            "Install it with: pip install mediapipe"
        ) from exc


def _extract_landmark_list(
    mp_landmark_list: object,
    *,
    with_visibility: bool = False,
) -> list[HolisticLandmark] | None:
    """Convert a MediaPipe NormalizedLandmarkList to a list of :class:`HolisticLandmark`.

    Args:
        mp_landmark_list: A ``mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList``
            or ``None``.
        with_visibility: If ``True``, read the ``visibility`` attribute.

    Returns:
        List of :class:`HolisticLandmark` objects, or ``None`` if input is ``None``.
    """
    if mp_landmark_list is None:
        return None

    result: list[HolisticLandmark] = []
    for lm in mp_landmark_list.landmark:
        result.append(
            HolisticLandmark(
                x=float(lm.x),
                y=float(lm.y),
                z=float(lm.z),
                visibility=float(lm.visibility) if with_visibility else 1.0,
            )
        )
    return result


def _draw_landmarks_dots(
    image: NDArray[np.uint8],
    landmarks: list[HolisticLandmark],
    width: int,
    height: int,
    colour: tuple[int, int, int],
    radius: int = LANDMARK_RADIUS,
) -> None:
    """Draw filled circles for each landmark onto *image* in place.

    Args:
        image: BGR image array (mutated in place).
        landmarks: Landmarks with normalised coordinates.
        width: Image width in pixels.
        height: Image height in pixels.
        colour: BGR fill colour.
        radius: Circle radius in pixels.
    """
    for lm in landmarks:
        px = int(lm.x * width)
        py = int(lm.y * height)
        cv2.circle(image, (px, py), radius, colour, -1)


def _draw_pose_connections(
    image: NDArray[np.uint8],
    landmarks: list[HolisticLandmark],
    width: int,
    height: int,
) -> None:
    """Draw pose skeleton connections onto *image* in place."""
    for start_idx, end_idx in _POSE_CONNECTIONS:
        if start_idx >= len(landmarks) or end_idx >= len(landmarks):
            continue
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        pt1 = (int(start.x * width), int(start.y * height))
        pt2 = (int(end.x * width), int(end.y * height))
        cv2.line(image, pt1, pt2, POSE_COLOUR, CONNECTION_THICKNESS)


def _draw_hand_connections(
    image: NDArray[np.uint8],
    landmarks: list[HolisticLandmark],
    width: int,
    height: int,
    colour: tuple[int, int, int],
) -> None:
    """Draw hand finger-bone connections onto *image* in place."""
    for start_idx, end_idx in _HAND_CONNECTIONS:
        if start_idx >= len(landmarks) or end_idx >= len(landmarks):
            continue
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        pt1 = (int(start.x * width), int(start.y * height))
        pt2 = (int(end.x * width), int(end.y * height))
        cv2.line(image, pt1, pt2, colour, CONNECTION_THICKNESS)


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
