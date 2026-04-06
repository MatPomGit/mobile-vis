"""Tests for image_analysis.hologram module."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from image_analysis.hologram import (
    HOLOGRAM_EDGE_COLOUR,
    HOLOGRAM_SIZE_FRACTION,
    LEFT_EYE_IDX,
    MAX_PITCH_DEGREES,
    MAX_YAW_DEGREES,
    NOSE_TIP_IDX,
    ORIENTATION_SCALE,
    RIGHT_EYE_IDX,
    FaceOrientation,
    HologramResult,
    _project_vertices,
    _rotation_x,
    _rotation_y,
    _validate_bgr_image,
    compute_face_orientation,
    create_face_mesh_hologram,
    draw_hologram_3d,
    process_hologram,
    render_hologram_3d,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bgr_image() -> np.ndarray:
    """Return a synthetic 200x300 BGR uint8 test image."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)


@pytest.fixture
def small_bgr() -> np.ndarray:
    """Return a tiny 64x64 BGR uint8 image for quick rendering tests."""
    return np.zeros((64, 64, 3), dtype=np.uint8)


def _make_mp_landmark(x: float = 0.5, y: float = 0.5, z: float = 0.0) -> MagicMock:
    lm = MagicMock()
    lm.x = x
    lm.y = y
    lm.z = z
    return lm


def _make_mp_face_mesh_result(n_landmarks: int = 468) -> MagicMock:
    """Create a mock face_mesh.process() result with one face."""
    face_lms = MagicMock()
    face_lms.landmark = [
        _make_mp_landmark(x=0.5, y=0.5) for _ in range(n_landmarks)
    ]
    result = MagicMock()
    result.multi_face_landmarks = [face_lms]
    return result


def _make_landmarks(
    n: int = 468,
    nose_x: float = 0.5,
    nose_y: float = 0.5,
) -> list[tuple[float, float, float]]:
    """Build a synthetic landmark list with a controlled nose position."""
    lms: list[tuple[float, float, float]] = [(0.5, 0.5, 0.0)] * n
    # Set nose tip
    lms[NOSE_TIP_IDX] = (nose_x, nose_y, 0.0)
    # Eyes at centre by default
    lms[LEFT_EYE_IDX] = (0.45, 0.45, 0.0)
    lms[RIGHT_EYE_IDX] = (0.55, 0.45, 0.0)
    return lms


# ---------------------------------------------------------------------------
# FaceOrientation dataclass
# ---------------------------------------------------------------------------


class TestFaceOrientation:
    def test_defaults(self) -> None:
        fo = FaceOrientation()
        assert fo.yaw_deg == pytest.approx(0.0)
        assert fo.pitch_deg == pytest.approx(0.0)
        assert fo.face_detected is False
        assert fo.face_center_x == pytest.approx(0.5)
        assert fo.face_center_y == pytest.approx(0.5)

    def test_custom_values(self) -> None:
        fo = FaceOrientation(yaw_deg=30.0, pitch_deg=-15.0, face_detected=True)
        assert fo.yaw_deg == pytest.approx(30.0)
        assert fo.pitch_deg == pytest.approx(-15.0)
        assert fo.face_detected is True


# ---------------------------------------------------------------------------
# HologramResult dataclass
# ---------------------------------------------------------------------------


class TestHologramResult:
    def test_defaults(self) -> None:
        hr = HologramResult()
        assert hr.face_landmarks is None
        assert hr.orientation.face_detected is False

    def test_with_orientation(self) -> None:
        fo = FaceOrientation(yaw_deg=10.0, face_detected=True)
        hr = HologramResult(orientation=fo)
        assert hr.orientation.yaw_deg == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# compute_face_orientation
# ---------------------------------------------------------------------------


class TestComputeFaceOrientation:
    def test_face_at_centre_gives_zero_rotation(self) -> None:
        lms = _make_landmarks(nose_x=0.5, nose_y=0.5)
        result = compute_face_orientation(lms)
        assert result.face_detected is True
        assert result.yaw_deg == pytest.approx(0.0, abs=1e-6)
        assert result.pitch_deg == pytest.approx(0.0, abs=1e-6)

    def test_face_right_gives_positive_yaw(self) -> None:
        lms = _make_landmarks(nose_x=0.75, nose_y=0.5)
        result = compute_face_orientation(lms)
        assert result.yaw_deg > 0.0

    def test_face_left_gives_negative_yaw(self) -> None:
        lms = _make_landmarks(nose_x=0.25, nose_y=0.5)
        result = compute_face_orientation(lms)
        assert result.yaw_deg < 0.0

    def test_face_below_gives_positive_pitch(self) -> None:
        lms = _make_landmarks(nose_x=0.5, nose_y=0.75)
        result = compute_face_orientation(lms)
        assert result.pitch_deg > 0.0

    def test_face_above_gives_negative_pitch(self) -> None:
        lms = _make_landmarks(nose_x=0.5, nose_y=0.25)
        result = compute_face_orientation(lms)
        assert result.pitch_deg < 0.0

    def test_yaw_clamped_at_max(self) -> None:
        # Extreme offset should be clamped to MAX_YAW_DEGREES.
        lms = _make_landmarks(nose_x=1.5, nose_y=0.5)
        result = compute_face_orientation(lms)
        assert result.yaw_deg == pytest.approx(MAX_YAW_DEGREES)

    def test_pitch_clamped_at_max(self) -> None:
        lms = _make_landmarks(nose_x=0.5, nose_y=1.5)
        result = compute_face_orientation(lms)
        assert result.pitch_deg == pytest.approx(MAX_PITCH_DEGREES)

    def test_face_center_computed_from_eyes(self) -> None:
        lms = _make_landmarks()
        lms[LEFT_EYE_IDX] = (0.3, 0.4, 0.0)
        lms[RIGHT_EYE_IDX] = (0.7, 0.4, 0.0)
        result = compute_face_orientation(lms)
        assert result.face_center_x == pytest.approx(0.5)
        assert result.face_center_y == pytest.approx(0.4)

    def test_raises_for_too_few_landmarks(self) -> None:
        with pytest.raises(ValueError, match="Expected at least"):
            compute_face_orientation([(0.5, 0.5, 0.0)] * 2)

    def test_orientation_scale_applied(self) -> None:
        # offset_x = (0.5 + 0.25 - 0.5) * ORIENTATION_SCALE = 0.25 * ORIENTATION_SCALE
        lms = _make_landmarks(nose_x=0.75, nose_y=0.5)
        result = compute_face_orientation(lms)
        expected_raw = 0.25 * ORIENTATION_SCALE
        expected_deg = min(expected_raw, 1.0) * MAX_YAW_DEGREES
        assert result.yaw_deg == pytest.approx(expected_deg, rel=1e-3)


# ---------------------------------------------------------------------------
# _rotation_y / _rotation_x
# ---------------------------------------------------------------------------


class TestRotationMatrices:
    def test_rotation_y_zero_is_identity(self) -> None:
        rot = _rotation_y(0.0)
        assert pytest.approx(np.eye(3), abs=1e-10) == rot

    def test_rotation_x_zero_is_identity(self) -> None:
        rot = _rotation_x(0.0)
        assert pytest.approx(np.eye(3), abs=1e-10) == rot

    def test_rotation_y_90_maps_x_to_neg_z(self) -> None:
        rot = _rotation_y(math.pi / 2)
        x_rotated = rot @ np.array([1.0, 0.0, 0.0])
        # x -> -z after 90 deg around Y
        assert x_rotated[2] == pytest.approx(-1.0, abs=1e-10)

    def test_rotation_x_90_maps_y_to_z(self) -> None:
        rot = _rotation_x(math.pi / 2)
        y_rotated = rot @ np.array([0.0, 1.0, 0.0])
        # y -> z after 90 deg around X
        assert y_rotated[2] == pytest.approx(1.0, abs=1e-10)

    def test_rotation_y_is_orthogonal(self) -> None:
        rot = _rotation_y(0.7)
        assert pytest.approx(np.eye(3), abs=1e-10) == rot @ rot.T

    def test_rotation_x_is_orthogonal(self) -> None:
        rot = _rotation_x(1.2)
        assert pytest.approx(np.eye(3), abs=1e-10) == rot @ rot.T


# ---------------------------------------------------------------------------
# _project_vertices
# ---------------------------------------------------------------------------


class TestProjectVertices:
    def test_returns_correct_count(self) -> None:
        verts = np.zeros((8, 3), dtype=np.float64)
        result = _project_vertices(verts, focal=300.0, cam_z=100.0, center=(50, 50))
        assert len(result) == 8

    def test_returns_integer_tuples(self) -> None:
        verts = np.ones((4, 3), dtype=np.float64)
        result = _project_vertices(verts, focal=100.0, cam_z=50.0, center=(0, 0))
        for pt in result:
            assert isinstance(pt[0], int)
            assert isinstance(pt[1], int)

    def test_centre_vertex_maps_to_offset(self) -> None:
        # A vertex at (0, 0, 0) should map to exactly the centre.
        verts = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        result = _project_vertices(verts, focal=300.0, cam_z=100.0, center=(160, 120))
        assert result[0] == (160, 120)


# ---------------------------------------------------------------------------
# render_hologram_3d
# ---------------------------------------------------------------------------


class TestRenderHologram3d:
    def test_returns_copy_same_shape(self, bgr_image: np.ndarray) -> None:
        orientation = FaceOrientation(yaw_deg=0.0, pitch_deg=0.0)
        output = render_hologram_3d(bgr_image, orientation)
        assert output.shape == bgr_image.shape
        assert output.dtype == np.uint8
        assert output is not bgr_image

    def test_modifies_pixels(self, bgr_image: np.ndarray) -> None:
        orientation = FaceOrientation(yaw_deg=30.0, pitch_deg=20.0, face_detected=True)
        output = render_hologram_3d(bgr_image, orientation)
        # At least some pixels should differ from the input.
        assert not np.array_equal(output, bgr_image)

    def test_custom_centre_and_size(self, bgr_image: np.ndarray) -> None:
        orientation = FaceOrientation()
        output = render_hologram_3d(bgr_image, orientation, center=(50, 50), size=20)
        assert output.shape == bgr_image.shape

    def test_raises_for_non_array(self) -> None:
        with pytest.raises(TypeError):
            render_hologram_3d("not an image", FaceOrientation())  # type: ignore[arg-type]

    def test_raises_for_grayscale(self) -> None:
        gray = np.zeros((64, 64), dtype=np.uint8)
        with pytest.raises(ValueError):
            render_hologram_3d(gray, FaceOrientation())  # type: ignore[arg-type]

    def test_raises_for_wrong_dtype(self, bgr_image: np.ndarray) -> None:
        float_img = bgr_image.astype(np.float32)
        with pytest.raises(ValueError):
            render_hologram_3d(float_img, FaceOrientation())  # type: ignore[arg-type]

    @pytest.mark.parametrize("yaw", [-60.0, -30.0, 0.0, 30.0, 60.0])
    def test_various_yaw_angles(self, small_bgr: np.ndarray, yaw: float) -> None:
        orientation = FaceOrientation(yaw_deg=yaw, pitch_deg=0.0)
        output = render_hologram_3d(small_bgr, orientation)
        assert output.shape == small_bgr.shape

    @pytest.mark.parametrize("pitch", [-45.0, -20.0, 0.0, 20.0, 45.0])
    def test_various_pitch_angles(self, small_bgr: np.ndarray, pitch: float) -> None:
        orientation = FaceOrientation(yaw_deg=0.0, pitch_deg=pitch)
        output = render_hologram_3d(small_bgr, orientation)
        assert output.shape == small_bgr.shape

    def test_default_size_uses_fraction(self, bgr_image: np.ndarray) -> None:
        # render_hologram_3d should not raise when size is None.
        orientation = FaceOrientation()
        output = render_hologram_3d(bgr_image, orientation, size=None)
        assert output.shape == bgr_image.shape


# ---------------------------------------------------------------------------
# draw_hologram_3d
# ---------------------------------------------------------------------------


class TestDrawHologram3d:
    def test_returns_same_shape(self, bgr_image: np.ndarray) -> None:
        result = HologramResult()
        output = draw_hologram_3d(bgr_image, result)
        assert output.shape == bgr_image.shape

    def test_no_face_shows_static_hologram(self, bgr_image: np.ndarray) -> None:
        result = HologramResult()  # face_detected=False
        output = draw_hologram_3d(bgr_image, result)
        assert output.shape == bgr_image.shape

    def test_with_face_detected(self, bgr_image: np.ndarray) -> None:
        fo = FaceOrientation(yaw_deg=20.0, pitch_deg=10.0, face_detected=True)
        result = HologramResult(orientation=fo)
        output = draw_hologram_3d(bgr_image, result)
        assert output.shape == bgr_image.shape

    def test_show_orientation_false(self, bgr_image: np.ndarray) -> None:
        result = HologramResult()
        output = draw_hologram_3d(bgr_image, result, show_orientation=False)
        assert output.shape == bgr_image.shape


# ---------------------------------------------------------------------------
# process_hologram (with mocked MediaPipe)
# ---------------------------------------------------------------------------


class TestProcessHologram:
    def test_no_face_returns_default(self, bgr_image: np.ndarray) -> None:
        face_mesh = MagicMock()
        face_mesh.process.return_value = MagicMock(multi_face_landmarks=None)
        result = process_hologram(bgr_image, face_mesh)
        assert result.face_landmarks is None
        assert result.orientation.face_detected is False

    def test_face_detected_returns_orientation(self, bgr_image: np.ndarray) -> None:
        mp_result = _make_mp_face_mesh_result()
        face_mesh = MagicMock()
        face_mesh.process.return_value = mp_result
        result = process_hologram(bgr_image, face_mesh)
        assert result.face_landmarks is not None
        assert result.orientation.face_detected is True

    def test_face_landmarks_have_correct_length(self, bgr_image: np.ndarray) -> None:
        n = 468
        mp_result = _make_mp_face_mesh_result(n_landmarks=n)
        face_mesh = MagicMock()
        face_mesh.process.return_value = mp_result
        result = process_hologram(bgr_image, face_mesh)
        assert result.face_landmarks is not None
        assert len(result.face_landmarks) == n

    def test_raises_for_non_array(self) -> None:
        face_mesh = MagicMock()
        with pytest.raises(TypeError):
            process_hologram("not an image", face_mesh)  # type: ignore[arg-type]

    def test_raises_for_grayscale(self) -> None:
        face_mesh = MagicMock()
        gray = np.zeros((64, 64), dtype=np.uint8)
        with pytest.raises(ValueError):
            process_hologram(gray, face_mesh)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# create_face_mesh_hologram (with mocked MediaPipe)
# ---------------------------------------------------------------------------


class TestCreateFaceMeshHologram:
    def test_raises_import_error_without_mediapipe(self) -> None:
        with patch.dict("sys.modules", {"mediapipe": None}), pytest.raises(
            ImportError, match="mediapipe"
        ):
            create_face_mesh_hologram()

    def test_raises_for_zero_faces(self) -> None:
        import mediapipe as mp  # noqa: F401 - needed to ensure import succeeds

        with pytest.raises(ValueError, match="max_num_faces"):
            create_face_mesh_hologram(max_num_faces=0)

    def test_raises_for_negative_faces(self) -> None:
        import mediapipe as mp  # noqa: F401

        with pytest.raises(ValueError, match="max_num_faces"):
            create_face_mesh_hologram(max_num_faces=-1)


# ---------------------------------------------------------------------------
# _validate_bgr_image
# ---------------------------------------------------------------------------


class TestValidateBgrImage:
    def test_valid_image_passes(self, bgr_image: np.ndarray) -> None:
        _validate_bgr_image(bgr_image)  # should not raise

    def test_raises_for_non_ndarray(self) -> None:
        with pytest.raises(TypeError):
            _validate_bgr_image([1, 2, 3])  # type: ignore[arg-type]

    def test_raises_for_1d(self) -> None:
        with pytest.raises(ValueError):
            _validate_bgr_image(np.zeros((10,), dtype=np.uint8))

    def test_raises_for_2d(self) -> None:
        with pytest.raises(ValueError):
            _validate_bgr_image(np.zeros((10, 10), dtype=np.uint8))

    def test_raises_for_4_channel(self) -> None:
        with pytest.raises(ValueError):
            _validate_bgr_image(np.zeros((10, 10, 4), dtype=np.uint8))

    def test_raises_for_float32(self, bgr_image: np.ndarray) -> None:
        with pytest.raises(ValueError):
            _validate_bgr_image(bgr_image.astype(np.float32))


# ---------------------------------------------------------------------------
# Constants sanity checks
# ---------------------------------------------------------------------------


class TestConstants:
    def test_hologram_size_fraction_positive(self) -> None:
        assert 0.0 < HOLOGRAM_SIZE_FRACTION <= 1.0

    def test_max_yaw_positive(self) -> None:
        assert MAX_YAW_DEGREES > 0.0

    def test_max_pitch_positive(self) -> None:
        assert MAX_PITCH_DEGREES > 0.0

    def test_hologram_edge_colour_is_bgr(self) -> None:
        assert len(HOLOGRAM_EDGE_COLOUR) == 3
        for channel in HOLOGRAM_EDGE_COLOUR:
            assert 0 <= channel <= 255
