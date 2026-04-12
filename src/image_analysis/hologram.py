"""3D hologram rendering utilities using face position to control rotation.

Provides functions to compute face orientation from MediaPipe face landmarks
and render a rotating 3D wireframe object that responds to the viewer's
position relative to the screen centre.

The hologram object (a unit cube) is rotated around the Y axis (yaw) and
X axis (pitch) according to the viewer's nose-tip position relative to the
screen centre.  The further the face is from centre, the more the hologram
rotates, up to :data:`MAX_YAW_DEGREES` / :data:`MAX_PITCH_DEGREES`.

Example::

    import cv2
    from image_analysis.hologram import (
        create_face_mesh_hologram,
        process_hologram,
        draw_hologram_3d,
    )

    with create_face_mesh_hologram() as face_mesh:
        image = cv2.imread("face.jpg")
        result = process_hologram(image, face_mesh)
        output = draw_hologram_3d(image, result)
        cv2.imwrite("hologram.jpg", output)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import mediapipe as mp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Face landmark indices used for orientation estimation
# ---------------------------------------------------------------------------

#: MediaPipe face mesh nose tip landmark index.
NOSE_TIP_IDX: int = 4

#: MediaPipe face mesh left eye inner corner landmark index.
LEFT_EYE_IDX: int = 133

#: MediaPipe face mesh right eye inner corner landmark index.
RIGHT_EYE_IDX: int = 362

# ---------------------------------------------------------------------------
# Drawing constants
# ---------------------------------------------------------------------------

#: BGR colour for hologram wireframe edges.
HOLOGRAM_EDGE_COLOUR: tuple[int, int, int] = (0, 255, 200)

#: BGR colour for hologram face fill (used in semi-transparent overlay).
HOLOGRAM_FILL_COLOUR: tuple[int, int, int] = (0, 80, 60)

#: Hologram edge line thickness in pixels.
HOLOGRAM_LINE_THICKNESS: int = 2

#: Hologram object size as a fraction of the shorter image dimension.
HOLOGRAM_SIZE_FRACTION: float = 0.25

#: Maximum yaw rotation angle in degrees, reached when the face is at the
#: screen edge (normalised offset ±0.5 after :data:`ORIENTATION_SCALE`).
MAX_YAW_DEGREES: float = 60.0

#: Maximum pitch rotation angle in degrees.
MAX_PITCH_DEGREES: float = 45.0

#: Multiplier applied to the normalised face-centre offset before clamping.
#: A value of 2.0 means an offset of ±0.25 from screen centre already
#: produces the maximum rotation.
ORIENTATION_SCALE: float = 2.0

# ---------------------------------------------------------------------------
# 3-D cube geometry
# ---------------------------------------------------------------------------

#: 8 vertices of a unit cube centred at the origin, shape ``(8, 3)``.
_CUBE_VERTICES: NDArray[np.float64] = np.array(
    [
        [-1.0, -1.0, -1.0],
        [+1.0, -1.0, -1.0],
        [+1.0, +1.0, -1.0],
        [-1.0, +1.0, -1.0],
        [-1.0, -1.0, +1.0],
        [+1.0, -1.0, +1.0],
        [+1.0, +1.0, +1.0],
        [-1.0, +1.0, +1.0],
    ],
    dtype=np.float64,
)

#: Pairs of vertex indices forming the 12 edges of the cube.
_CUBE_EDGES: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # back face
    (4, 5), (5, 6), (6, 7), (7, 4),  # front face
    (0, 4), (1, 5), (2, 6), (3, 7),  # connecting edges
]

#: Groups of 4 vertex indices forming the 6 faces of the cube.
_CUBE_FACES: list[tuple[int, int, int, int]] = [
    (0, 1, 2, 3),  # back
    (4, 5, 6, 7),  # front
    (0, 1, 5, 4),  # bottom
    (2, 3, 7, 6),  # top
    (0, 3, 7, 4),  # left
    (1, 2, 6, 5),  # right
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FaceOrientation:
    """Estimated face orientation derived from face landmark positions.

    Attributes:
        yaw_deg: Horizontal rotation angle in degrees.  Positive values
            indicate the face has moved to the right of screen centre.
        pitch_deg: Vertical rotation angle in degrees.  Positive values
            indicate the face has moved below screen centre.
        face_detected: ``True`` if a face was found in the frame.
        face_center_x: Normalised horizontal face position (``[0.0, 1.0]``).
        face_center_y: Normalised vertical face position (``[0.0, 1.0]``).
    """

    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    face_detected: bool = False
    face_center_x: float = 0.5
    face_center_y: float = 0.5


@dataclass
class HologramResult:
    """Result of hologram processing containing orientation data.

    Attributes:
        orientation: Estimated face/gaze orientation used to drive rotation.
        face_landmarks: All detected face landmark triples ``(x, y, z)``
            in normalised image coordinates, or ``None`` if no face was found.
    """

    orientation: FaceOrientation = field(default_factory=FaceOrientation)
    face_landmarks: list[tuple[float, float, float]] | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_face_mesh_hologram(
    max_num_faces: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> mp.solutions.face_mesh.FaceMesh:
    """Create a MediaPipe FaceMesh instance for hologram orientation estimation.

    The returned object is a context manager::

        with create_face_mesh_hologram() as face_mesh:
            result = process_hologram(image, face_mesh)

    Args:
        max_num_faces: Maximum number of faces to detect per frame.
        min_detection_confidence: Minimum detection confidence (``[0.0, 1.0]``).
        min_tracking_confidence: Minimum tracking confidence (``[0.0, 1.0]``).

    Returns:
        A ``mediapipe.solutions.face_mesh.FaceMesh`` instance.

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
        refine_landmarks=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def process_hologram(
    image: NDArray[np.uint8],
    face_mesh: mp.solutions.face_mesh.FaceMesh,
) -> HologramResult:
    """Detect face landmarks and compute orientation for hologram control.

    Processes only the *first* detected face.  If no face is detected, a
    default :class:`HologramResult` with no rotation is returned.

    Args:
        image: BGR image array with shape ``(H, W, 3)`` and dtype ``uint8``.
        face_mesh: FaceMesh instance created with :func:`create_face_mesh_hologram`.

    Returns:
        :class:`HologramResult` containing orientation and landmark data.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* does not have shape ``(H, W, 3)`` or dtype
            ``uint8``.
    """
    _validate_bgr_image(image)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_result = face_mesh.process(rgb)

    if not mp_result.multi_face_landmarks:
        logger.debug("Hologram: no face detected")
        return HologramResult()

    landmark_list = mp_result.multi_face_landmarks[0].landmark
    landmarks = [
        (float(lm.x), float(lm.y), float(lm.z)) for lm in landmark_list
    ]

    orientation = compute_face_orientation(landmarks)

    logger.debug(
        "Hologram: yaw=%.1f pitch=%.1f face_center=(%.2f, %.2f)",
        orientation.yaw_deg,
        orientation.pitch_deg,
        orientation.face_center_x,
        orientation.face_center_y,
    )

    return HologramResult(orientation=orientation, face_landmarks=landmarks)


def compute_face_orientation(
    landmarks: list[tuple[float, float, float]],
) -> FaceOrientation:
    """Compute yaw and pitch from normalised face landmarks.

    Uses the nose tip position relative to the screen centre to estimate
    horizontal (yaw) and vertical (pitch) rotation.  The offset is scaled
    by :data:`ORIENTATION_SCALE` and clamped so that the maximum rotation
    equals :data:`MAX_YAW_DEGREES` / :data:`MAX_PITCH_DEGREES`.

    Args:
        landmarks: List of ``(x, y, z)`` triples in normalised image
            coordinates, as returned by MediaPipe FaceMesh.  Must contain
            at least ``max(NOSE_TIP_IDX, LEFT_EYE_IDX, RIGHT_EYE_IDX) + 1``
            entries.

    Returns:
        :class:`FaceOrientation` with computed angles.

    Raises:
        ValueError: If *landmarks* does not contain enough entries.
    """
    min_required = max(NOSE_TIP_IDX, LEFT_EYE_IDX, RIGHT_EYE_IDX) + 1
    if len(landmarks) < min_required:
        raise ValueError(
            f"Expected at least {min_required} landmarks, got {len(landmarks)}"
        )

    nose = landmarks[NOSE_TIP_IDX]
    left_eye = landmarks[LEFT_EYE_IDX]
    right_eye = landmarks[RIGHT_EYE_IDX]

    # Face centre as midpoint between the two eye inner corners.
    face_cx = (left_eye[0] + right_eye[0]) / 2.0
    face_cy = (left_eye[1] + right_eye[1]) / 2.0

    # Use nose tip for a stable single-point orientation signal.
    nose_x, nose_y = nose[0], nose[1]

    # Offset from screen centre [-0.5, +0.5] x ORIENTATION_SCALE.
    offset_x = (nose_x - 0.5) * ORIENTATION_SCALE
    offset_y = (nose_y - 0.5) * ORIENTATION_SCALE

    # Clamp to [-1, +1] and scale to angle range.
    yaw_deg = float(np.clip(offset_x, -1.0, 1.0)) * MAX_YAW_DEGREES
    pitch_deg = float(np.clip(offset_y, -1.0, 1.0)) * MAX_PITCH_DEGREES

    return FaceOrientation(
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        face_detected=True,
        face_center_x=face_cx,
        face_center_y=face_cy,
    )


def render_hologram_3d(
    image: NDArray[np.uint8],
    orientation: FaceOrientation,
    center: tuple[int, int] | None = None,
    size: int | None = None,
) -> NDArray[np.uint8]:
    """Render a rotating 3D hologram wireframe onto a copy of *image*.

    The hologram object (unit cube) is rotated according to *orientation*
    and projected onto the canvas using simple perspective projection.
    Semi-transparent face fills are drawn first, then wireframe edges.

    Args:
        image: BGR image array with shape ``(H, W, 3)`` and dtype ``uint8``.
        orientation: Face orientation from :func:`compute_face_orientation`.
        center: ``(cx, cy)`` pixel coordinates for the hologram centre.
            Defaults to the centre of the image.
        size: Half-edge length in pixels.  Defaults to
            :data:`HOLOGRAM_SIZE_FRACTION` x ``min(H, W)``.

    Returns:
        Copy of *image* with the 3D hologram overlay.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* does not have shape ``(H, W, 3)`` or dtype
            ``uint8``.
    """
    _validate_bgr_image(image)

    h, w = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    if size is None:
        size = max(1, int(min(h, w) * HOLOGRAM_SIZE_FRACTION))

    output = image.copy()

    yaw_rad = math.radians(orientation.yaw_deg)
    pitch_rad = math.radians(orientation.pitch_deg)

    # Build combined rotation matrix: first yaw (around Y), then pitch (around X).
    rot_y = _rotation_y(yaw_rad)
    rot_x = _rotation_x(pitch_rad)
    rot = rot_x @ rot_y

    # Scale and rotate cube vertices.
    vertices_3d = (_CUBE_VERTICES @ rot.T) * size

    # Perspective projection parameters.
    focal = float(w) * 1.5
    cam_z = float(size) * 4.0
    pts_2d = _project_vertices(vertices_3d, focal, cam_z, center)

    # Draw semi-transparent face fills.
    overlay = output.copy()
    for face_indices in _CUBE_FACES:
        poly = np.array([pts_2d[i] for i in face_indices], dtype=np.int32)
        cv2.fillPoly(overlay, [poly], HOLOGRAM_FILL_COLOUR)
    cv2.addWeighted(overlay, 0.3, output, 0.7, 0, output)

    # Draw wireframe edges.
    for start_idx, end_idx in _CUBE_EDGES:
        p1 = pts_2d[start_idx]
        p2 = pts_2d[end_idx]
        cv2.line(output, p1, p2, HOLOGRAM_EDGE_COLOUR, HOLOGRAM_LINE_THICKNESS, cv2.LINE_AA)

    # Draw vertex dots.
    for pt in pts_2d:
        cv2.circle(output, pt, 3, HOLOGRAM_EDGE_COLOUR, -1, cv2.LINE_AA)

    return output


def draw_hologram_3d(
    image: NDArray[np.uint8],
    result: HologramResult,
    center: tuple[int, int] | None = None,
    size: int | None = None,
    show_orientation: bool = True,
) -> NDArray[np.uint8]:
    """Draw the 3D hologram and optional orientation HUD onto a copy of *image*.

    If no face was detected (``result.orientation.face_detected`` is ``False``),
    a static (unrotated) hologram is drawn with a ``"Brak twarzy"`` indicator.

    Args:
        image: BGR image array with shape ``(H, W, 3)`` and dtype ``uint8``.
        result: Processing result from :func:`process_hologram`.
        center: Hologram centre in pixels; defaults to image centre.
        size: Hologram half-edge size in pixels.
        show_orientation: Whether to overlay yaw/pitch angle text.

    Returns:
        Copy of *image* with hologram and HUD overlays.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* does not have shape ``(H, W, 3)`` or dtype
            ``uint8``.
    """
    _validate_bgr_image(image)

    output = render_hologram_3d(image, result.orientation, center=center, size=size)

    if show_orientation:
        h = output.shape[0]
        if result.orientation.face_detected:
            label = (
                f"Yaw: {result.orientation.yaw_deg:+.1f}  "
                f"Pitch: {result.orientation.pitch_deg:+.1f}"
            )
            colour = HOLOGRAM_EDGE_COLOUR
        else:
            label = "Brak twarzy"
            colour = (0, 100, 255)

        # Draw text shadow then coloured text for legibility.
        cv2.putText(
            output, label, (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            output, label, (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1, cv2.LINE_AA,
        )

    return output


# ---------------------------------------------------------------------------
# Internal 3-D geometry helpers
# ---------------------------------------------------------------------------


def _rotation_y(angle: float) -> NDArray[np.float64]:
    """Build a 3x3 rotation matrix around the Y axis.

    Args:
        angle: Rotation angle in radians.

    Returns:
        3x3 rotation matrix as ``float64`` array.
    """
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float64,
    )


def _rotation_x(angle: float) -> NDArray[np.float64]:
    """Build a 3x3 rotation matrix around the X axis.

    Args:
        angle: Rotation angle in radians.

    Returns:
        3x3 rotation matrix as ``float64`` array.
    """
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float64,
    )


def _project_vertices(
    vertices_3d: NDArray[np.float64],
    focal: float,
    cam_z: float,
    center: tuple[int, int],
) -> list[tuple[int, int]]:
    """Project 3-D vertices onto a 2-D canvas with perspective division.

    Args:
        vertices_3d: Array of shape ``(N, 3)`` with 3-D vertex coordinates.
        focal: Focal length in pixels.
        cam_z: Camera-to-object Z distance (added to each vertex Z component).
        center: ``(cx, cy)`` canvas offset applied after projection.

    Returns:
        List of ``(x, y)`` integer pixel tuples.
    """
    projected: list[tuple[int, int]] = []
    for v in vertices_3d:
        z = cam_z + v[2]
        if abs(z) < 1e-6:
            z = 1e-6
        px = int(v[0] * focal / z) + center[0]
        py = int(v[1] * focal / z) + center[1]
        projected.append((px, py))
    return projected


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_mediapipe() -> None:
    """Raise :exc:`ImportError` if the ``mediapipe`` package is not installed."""
    try:
        import mediapipe  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'mediapipe' package is required for hologram rendering. "
            "Install it with: pip install mediapipe"
        ) from exc


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
    "FaceOrientation": "FaceOrientation",
    "HOLOGRAM_EDGE_COLOUR": "HOLOGRAM_EDGE_COLOUR",
    "HOLOGRAM_SIZE_FRACTION": "HOLOGRAM_SIZE_FRACTION",
    "HologramResult": "HologramResult",
    "MAX_PITCH_DEGREES": "MAX_PITCH_DEGREES",
    "MAX_YAW_DEGREES": "MAX_YAW_DEGREES",
    "compute_face_orientation": "compute_face_orientation",
    "create_face_mesh_hologram": "create_face_mesh_hologram",
    "draw_hologram_3d": "draw_hologram_3d",
    "process_hologram": "process_hologram",
    "render_hologram_3d": "render_hologram_3d",
}
