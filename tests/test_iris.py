"""Tests for image_analysis.iris module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from image_analysis.iris import (
    LEFT_IRIS_INDICES,
    REFINED_LANDMARK_COUNT,
    RIGHT_IRIS_INDICES,
    IrisLandmark,
    IrisResult,
    _draw_iris_circle,
    _extract_iris,
    _iris_radius_pixels,
    _validate_bgr_image,
    create_face_mesh_iris,
    draw_iris_results,
    estimate_gaze_offset,
    process_iris,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bgr_image() -> np.ndarray:
    """Return a synthetic 200x300 BGR uint8 image."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)


@pytest.fixture
def gray_image() -> np.ndarray:
    """Return a synthetic 200x300 grayscale uint8 image."""
    rng = np.random.default_rng(seed=1)
    return rng.integers(0, 255, (200, 300), dtype=np.uint8)


def _make_iris_landmark(x: float = 0.5, y: float = 0.5, z: float = 0.0) -> IrisLandmark:
    return IrisLandmark(x=x, y=y, z=z)


def _make_mp_landmark(x: float = 0.5, y: float = 0.5, z: float = 0.0) -> MagicMock:
    lm = MagicMock()
    lm.x = x
    lm.y = y
    lm.z = z
    return lm


def _make_mp_face_mesh_result(n_landmarks: int = REFINED_LANDMARK_COUNT) -> MagicMock:
    """Create a mock face_mesh.process() return value with one face."""
    face_lms = MagicMock()
    face_lms.landmark = [
        _make_mp_landmark(x=i / max(n_landmarks, 1), y=i / max(n_landmarks, 1))
        for i in range(n_landmarks)
    ]
    mp_result = MagicMock()
    mp_result.multi_face_landmarks = [face_lms]
    return mp_result


# ---------------------------------------------------------------------------
# IrisLandmark dataclass
# ---------------------------------------------------------------------------


class TestIrisLandmark:
    def test_fields_accessible(self) -> None:
        lm = IrisLandmark(x=0.1, y=0.2, z=0.3)
        assert lm.x == pytest.approx(0.1)
        assert lm.y == pytest.approx(0.2)
        assert lm.z == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# IrisResult dataclass
# ---------------------------------------------------------------------------


class TestIrisResult:
    def test_all_none_by_default(self) -> None:
        result = IrisResult()
        assert result.left_iris is None
        assert result.right_iris is None
        assert result.face_landmarks is None

    def test_can_set_fields(self) -> None:
        iris = [_make_iris_landmark()]
        result = IrisResult(left_iris=iris)
        assert result.left_iris is not None
        assert len(result.left_iris) == 1


# ---------------------------------------------------------------------------
# _validate_bgr_image
# ---------------------------------------------------------------------------


class TestValidateBgrImage:
    def test_raises_for_non_ndarray(self) -> None:
        with pytest.raises(TypeError, match=r"np\.ndarray"):
            _validate_bgr_image([[1, 2, 3]])

    def test_raises_for_grayscale(self, gray_image: np.ndarray) -> None:
        with pytest.raises(ValueError, match="3-channel"):
            _validate_bgr_image(gray_image)

    def test_raises_for_float_dtype(self, bgr_image: np.ndarray) -> None:
        with pytest.raises(ValueError, match="uint8"):
            _validate_bgr_image(bgr_image.astype(np.float32))

    def test_accepts_bgr_image(self, bgr_image: np.ndarray) -> None:
        _validate_bgr_image(bgr_image)


# ---------------------------------------------------------------------------
# _extract_iris
# ---------------------------------------------------------------------------


class TestExtractIris:
    def _make_landmarks(self, n: int) -> list[IrisLandmark]:
        return [IrisLandmark(x=i / max(n, 1), y=i / max(n, 1), z=0.0) for i in range(n)]

    def test_returns_none_when_index_out_of_range(self) -> None:
        landmarks = self._make_landmarks(5)
        result = _extract_iris(landmarks, (0, 1, 2, 3, 100))
        assert result is None

    def test_returns_subset_for_valid_indices(self) -> None:
        landmarks = self._make_landmarks(REFINED_LANDMARK_COUNT)
        result = _extract_iris(landmarks, LEFT_IRIS_INDICES)
        assert result is not None
        assert len(result) == len(LEFT_IRIS_INDICES)

    def test_extracts_correct_elements(self) -> None:
        landmarks = self._make_landmarks(REFINED_LANDMARK_COUNT)
        result = _extract_iris(landmarks, (468, 469))
        assert result is not None
        assert result[0] is landmarks[468]
        assert result[1] is landmarks[469]


# ---------------------------------------------------------------------------
# _iris_radius_pixels
# ---------------------------------------------------------------------------


class TestIrisRadiusPixels:
    def test_returns_positive_radius(self) -> None:
        center = IrisLandmark(x=0.5, y=0.5, z=0.0)
        contour = [
            IrisLandmark(x=0.5, y=0.4, z=0.0),
            IrisLandmark(x=0.6, y=0.5, z=0.0),
            IrisLandmark(x=0.5, y=0.6, z=0.0),
            IrisLandmark(x=0.4, y=0.5, z=0.0),
        ]
        radius = _iris_radius_pixels([center, *contour], width=100, height=100)
        assert radius >= 1

    def test_returns_minimum_1_for_coincident_points(self) -> None:
        lm = IrisLandmark(x=0.5, y=0.5, z=0.0)
        radius = _iris_radius_pixels([lm, lm, lm, lm, lm], width=100, height=100)
        assert radius == 1

    def test_handles_single_point(self) -> None:
        lm = IrisLandmark(x=0.5, y=0.5, z=0.0)
        radius = _iris_radius_pixels([lm], width=100, height=100)
        assert radius == 1


# ---------------------------------------------------------------------------
# create_face_mesh_iris
# ---------------------------------------------------------------------------


class TestCreateFaceMeshIris:
    def test_raises_import_error_without_mediapipe(self) -> None:
        with (
            patch.dict("sys.modules", {"mediapipe": None}),
            pytest.raises(ImportError, match="mediapipe"),
        ):
            create_face_mesh_iris()

    def test_raises_for_non_positive_max_faces(self) -> None:
        mock_mp = MagicMock()
        with (
            patch.dict("sys.modules", {"mediapipe": mock_mp}),
            pytest.raises(ValueError, match="max_num_faces"),
        ):
            create_face_mesh_iris(max_num_faces=0)

    def test_creates_face_mesh_with_refine_landmarks(self) -> None:
        mock_mp = MagicMock()
        with patch.dict("sys.modules", {"mediapipe": mock_mp}):
            _ = create_face_mesh_iris()
            call_kwargs = mock_mp.solutions.face_mesh.FaceMesh.call_args[1]
            assert call_kwargs["refine_landmarks"] is True

    def test_passes_parameters(self) -> None:
        mock_mp = MagicMock()
        with patch.dict("sys.modules", {"mediapipe": mock_mp}):
            _ = create_face_mesh_iris(max_num_faces=2, min_detection_confidence=0.8)
            call_kwargs = mock_mp.solutions.face_mesh.FaceMesh.call_args[1]
            assert call_kwargs["max_num_faces"] == 2
            assert call_kwargs["min_detection_confidence"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# process_iris
# ---------------------------------------------------------------------------


class TestProcessIris:
    def test_raises_for_non_ndarray(self) -> None:
        face_mesh = MagicMock()
        with pytest.raises(TypeError):
            process_iris("not an image", face_mesh)  # type: ignore[arg-type]

    def test_raises_for_grayscale(self, gray_image: np.ndarray) -> None:
        face_mesh = MagicMock()
        with pytest.raises(ValueError):
            process_iris(gray_image, face_mesh)

    def test_returns_empty_result_when_no_face(self, bgr_image: np.ndarray) -> None:
        face_mesh = MagicMock()
        mp_result = MagicMock()
        mp_result.multi_face_landmarks = None
        face_mesh.process.return_value = mp_result

        result = process_iris(bgr_image, face_mesh)

        assert result.left_iris is None
        assert result.right_iris is None
        assert result.face_landmarks is None

    def test_returns_iris_landmarks_for_refined_mesh(self, bgr_image: np.ndarray) -> None:
        face_mesh = MagicMock()
        face_mesh.process.return_value = _make_mp_face_mesh_result(REFINED_LANDMARK_COUNT)

        result = process_iris(bgr_image, face_mesh)

        assert result.face_landmarks is not None
        assert len(result.face_landmarks) == REFINED_LANDMARK_COUNT
        assert result.left_iris is not None
        assert len(result.left_iris) == len(LEFT_IRIS_INDICES)
        assert result.right_iris is not None
        assert len(result.right_iris) == len(RIGHT_IRIS_INDICES)

    def test_returns_none_iris_for_base_mesh(self, bgr_image: np.ndarray) -> None:
        """With only 468 landmarks, iris indices 468-477 are out of range."""
        face_mesh = MagicMock()
        face_mesh.process.return_value = _make_mp_face_mesh_result(468)

        result = process_iris(bgr_image, face_mesh)

        assert result.left_iris is None
        assert result.right_iris is None

    def test_converts_to_rgb_before_processing(self, bgr_image: np.ndarray) -> None:
        face_mesh = MagicMock()
        face_mesh.process.return_value = _make_mp_face_mesh_result(REFINED_LANDMARK_COUNT)

        process_iris(bgr_image, face_mesh)

        face_mesh.process.assert_called_once()
        called_arg = face_mesh.process.call_args[0][0]
        assert called_arg.shape == bgr_image.shape


# ---------------------------------------------------------------------------
# draw_iris_results
# ---------------------------------------------------------------------------


class TestDrawIrisResults:
    def _make_iris(self, cx: float = 0.5, cy: float = 0.5) -> list[IrisLandmark]:
        """Return 5 iris landmarks (centre + 4 contour points at radius ~0.05)."""
        return [
            IrisLandmark(x=cx, y=cy, z=0.0),
            IrisLandmark(x=cx, y=cy - 0.05, z=0.0),
            IrisLandmark(x=cx + 0.05, y=cy, z=0.0),
            IrisLandmark(x=cx, y=cy + 0.05, z=0.0),
            IrisLandmark(x=cx - 0.05, y=cy, z=0.0),
        ]

    def test_raises_for_non_ndarray(self) -> None:
        with pytest.raises(TypeError):
            draw_iris_results("not an image", IrisResult())  # type: ignore[arg-type]

    def test_raises_for_grayscale(self, gray_image: np.ndarray) -> None:
        with pytest.raises(ValueError):
            draw_iris_results(gray_image, IrisResult())

    def test_returns_copy_when_no_landmarks(self, bgr_image: np.ndarray) -> None:
        output = draw_iris_results(bgr_image, IrisResult())
        assert output is not bgr_image
        np.testing.assert_array_equal(output, bgr_image)

    def test_output_shape_matches_input(self, bgr_image: np.ndarray) -> None:
        iris = self._make_iris()
        result = IrisResult(left_iris=iris)
        output = draw_iris_results(bgr_image, result)
        assert output.shape == bgr_image.shape
        assert output.dtype == bgr_image.dtype

    def test_does_not_mutate_input(self, bgr_image: np.ndarray) -> None:
        original = bgr_image.copy()
        iris = self._make_iris()
        draw_iris_results(bgr_image, IrisResult(left_iris=iris))
        np.testing.assert_array_equal(bgr_image, original)

    def test_draws_left_iris(self, bgr_image: np.ndarray) -> None:
        iris = self._make_iris(cx=0.5, cy=0.5)
        result = IrisResult(left_iris=iris)
        output = draw_iris_results(bgr_image, result)
        assert not np.array_equal(output, bgr_image)

    def test_draws_right_iris(self, bgr_image: np.ndarray) -> None:
        iris = self._make_iris(cx=0.5, cy=0.5)
        result = IrisResult(right_iris=iris)
        output = draw_iris_results(bgr_image, result)
        assert not np.array_equal(output, bgr_image)

    def test_draw_face_mesh_flag(self, bgr_image: np.ndarray) -> None:
        n = REFINED_LANDMARK_COUNT
        all_lms = [IrisLandmark(x=i / n, y=0.5, z=0.0) for i in range(n)]
        result = IrisResult(face_landmarks=all_lms)
        output = draw_iris_results(bgr_image, result, draw_face_mesh=True)
        assert not np.array_equal(output, bgr_image)


# ---------------------------------------------------------------------------
# _draw_iris_circle  (internal helper)
# ---------------------------------------------------------------------------


class TestDrawIrisCircle:
    def _make_iris(self) -> list[IrisLandmark]:
        return [
            IrisLandmark(x=0.5, y=0.5, z=0.0),
            IrisLandmark(x=0.5, y=0.4, z=0.0),
            IrisLandmark(x=0.6, y=0.5, z=0.0),
            IrisLandmark(x=0.5, y=0.6, z=0.0),
            IrisLandmark(x=0.4, y=0.5, z=0.0),
        ]

    def test_draws_circle_on_image(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        _draw_iris_circle(image, self._make_iris(), 100, 100, (0, 200, 0))
        assert image.sum() > 0


# ---------------------------------------------------------------------------
# estimate_gaze_offset
# ---------------------------------------------------------------------------


class TestEstimateGazeOffset:
    def _make_centered_iris(self) -> list[IrisLandmark]:
        """Iris centre at (0.5, 0.5) with symmetric contour → offset near (0, 0)."""
        return [
            IrisLandmark(x=0.5, y=0.5, z=0.0),
            IrisLandmark(x=0.5, y=0.4, z=0.0),
            IrisLandmark(x=0.6, y=0.5, z=0.0),
            IrisLandmark(x=0.5, y=0.6, z=0.0),
            IrisLandmark(x=0.4, y=0.5, z=0.0),
        ]

    def test_returns_none_when_no_iris(self) -> None:
        result = IrisResult()
        assert estimate_gaze_offset(result, 100, 100) is None

    def test_returns_tuple_when_iris_present(self) -> None:
        result = IrisResult(left_iris=self._make_centered_iris())
        offset = estimate_gaze_offset(result, 100, 100)
        assert offset is not None
        assert isinstance(offset, tuple)
        assert len(offset) == 2

    def test_symmetric_iris_gives_near_zero_offset(self) -> None:
        result = IrisResult(left_iris=self._make_centered_iris())
        offset = estimate_gaze_offset(result, 100, 100)
        assert offset is not None
        assert abs(offset[0]) < 1e-9
        assert abs(offset[1]) < 1e-9

    def test_raises_for_non_positive_dimensions(self) -> None:
        result = IrisResult(left_iris=self._make_centered_iris())
        with pytest.raises(ValueError):
            estimate_gaze_offset(result, 0, 100)
        with pytest.raises(ValueError):
            estimate_gaze_offset(result, 100, -1)

    def test_uses_both_irises_when_present(self) -> None:
        result = IrisResult(
            left_iris=self._make_centered_iris(),
            right_iris=self._make_centered_iris(),
        )
        offset = estimate_gaze_offset(result, 100, 100)
        assert offset is not None
