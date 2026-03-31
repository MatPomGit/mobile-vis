"""Tests for image_analysis.holistic module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from image_analysis.holistic import (
    HolisticLandmark,
    HolisticResult,
    _draw_hand_connections,
    _draw_landmarks_dots,
    _draw_pose_connections,
    _extract_landmark_list,
    _validate_bgr_image,
    create_holistic,
    draw_holistic_results,
    process_holistic,
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


def _make_mp_landmark(x: float = 0.5, y: float = 0.5, z: float = 0.0) -> MagicMock:
    """Create a mock MediaPipe landmark with x, y, z, visibility attributes."""
    lm = MagicMock()
    lm.x = x
    lm.y = y
    lm.z = z
    lm.visibility = 1.0
    return lm


def _make_mp_landmark_list(n: int) -> MagicMock:
    """Create a mock NormalizedLandmarkList with *n* landmarks."""
    landmark_list = MagicMock()
    landmark_list.landmark = [
        _make_mp_landmark(x=i / max(n, 1), y=i / max(n, 1)) for i in range(n)
    ]
    return landmark_list


# ---------------------------------------------------------------------------
# HolisticLandmark dataclass
# ---------------------------------------------------------------------------


class TestHolisticLandmark:
    def test_fields_accessible(self) -> None:
        lm = HolisticLandmark(x=0.1, y=0.2, z=0.3, visibility=0.9)
        assert lm.x == pytest.approx(0.1)
        assert lm.y == pytest.approx(0.2)
        assert lm.z == pytest.approx(0.3)
        assert lm.visibility == pytest.approx(0.9)

    def test_default_visibility(self) -> None:
        lm = HolisticLandmark(x=0.0, y=0.0, z=0.0)
        assert lm.visibility == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# HolisticResult dataclass
# ---------------------------------------------------------------------------


class TestHolisticResult:
    def test_all_none_by_default(self) -> None:
        result = HolisticResult()
        assert result.pose_landmarks is None
        assert result.left_hand_landmarks is None
        assert result.right_hand_landmarks is None
        assert result.face_landmarks is None

    def test_can_set_landmarks(self) -> None:
        landmarks = [HolisticLandmark(0.5, 0.5, 0.0)]
        result = HolisticResult(pose_landmarks=landmarks)
        assert result.pose_landmarks is not None
        assert len(result.pose_landmarks) == 1


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
        # No exception expected
        _validate_bgr_image(bgr_image)


# ---------------------------------------------------------------------------
# _extract_landmark_list
# ---------------------------------------------------------------------------


class TestExtractLandmarkList:
    def test_returns_none_for_none_input(self) -> None:
        assert _extract_landmark_list(None) is None

    def test_converts_landmark_list(self) -> None:
        mp_list = _make_mp_landmark_list(5)
        result = _extract_landmark_list(mp_list)
        assert result is not None
        assert len(result) == 5

    def test_extracts_coordinates(self) -> None:
        mp_list = MagicMock()
        mp_list.landmark = [_make_mp_landmark(x=0.3, y=0.7, z=0.1)]
        result = _extract_landmark_list(mp_list)
        assert result is not None
        assert result[0].x == pytest.approx(0.3)
        assert result[0].y == pytest.approx(0.7)
        assert result[0].z == pytest.approx(0.1)

    def test_visibility_with_flag(self) -> None:
        mp_list = MagicMock()
        mp_list.landmark = [_make_mp_landmark()]
        result = _extract_landmark_list(mp_list, with_visibility=True)
        assert result is not None
        assert result[0].visibility == pytest.approx(1.0)

    def test_visibility_defaults_without_flag(self) -> None:
        mp_list = MagicMock()
        lm = _make_mp_landmark()
        lm.visibility = 0.3
        mp_list.landmark = [lm]
        result = _extract_landmark_list(mp_list, with_visibility=False)
        assert result is not None
        # Without the flag, visibility should default to 1.0
        assert result[0].visibility == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# create_holistic
# ---------------------------------------------------------------------------


class TestCreateHolistic:
    def test_raises_import_error_without_mediapipe(self) -> None:
        with (
            patch.dict("sys.modules", {"mediapipe": None}),
            pytest.raises(ImportError, match="mediapipe"),
        ):
            create_holistic()

    def test_raises_for_invalid_complexity(self) -> None:
        mock_mp = MagicMock()
        with (
            patch.dict("sys.modules", {"mediapipe": mock_mp}),
            pytest.raises(ValueError, match="model_complexity"),
        ):
            create_holistic(model_complexity=3)

    def test_creates_holistic_with_defaults(self) -> None:
        mock_mp = MagicMock()
        with patch.dict("sys.modules", {"mediapipe": mock_mp}):
            _ = create_holistic()
            mock_mp.solutions.holistic.Holistic.assert_called_once()

    def test_passes_parameters(self) -> None:
        mock_mp = MagicMock()
        with patch.dict("sys.modules", {"mediapipe": mock_mp}):
            _ = create_holistic(model_complexity=0, min_detection_confidence=0.7)
            mock_mp.solutions.holistic.Holistic.assert_called_once_with(
                model_complexity=0,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
                smooth_landmarks=True,
                enable_segmentation=False,
            )


# ---------------------------------------------------------------------------
# process_holistic
# ---------------------------------------------------------------------------


class TestProcessHolistic:
    def _make_mock_holistic(
        self,
        n_pose: int = 33,
        n_left_hand: int = 0,
        n_right_hand: int = 0,
        n_face: int = 0,
    ) -> MagicMock:
        holistic = MagicMock()
        mp_result = MagicMock()
        mp_result.pose_landmarks = (
            _make_mp_landmark_list(n_pose) if n_pose > 0 else None
        )
        mp_result.left_hand_landmarks = (
            _make_mp_landmark_list(n_left_hand) if n_left_hand > 0 else None
        )
        mp_result.right_hand_landmarks = (
            _make_mp_landmark_list(n_right_hand) if n_right_hand > 0 else None
        )
        mp_result.face_landmarks = (
            _make_mp_landmark_list(n_face) if n_face > 0 else None
        )
        holistic.process.return_value = mp_result
        return holistic

    def test_raises_for_non_ndarray(self) -> None:
        holistic = MagicMock()
        with pytest.raises(TypeError):
            process_holistic("not an image", holistic)  # type: ignore[arg-type]

    def test_raises_for_grayscale(self, gray_image: np.ndarray) -> None:
        holistic = MagicMock()
        with pytest.raises(ValueError):
            process_holistic(gray_image, holistic)

    def test_returns_none_when_nothing_detected(self, bgr_image: np.ndarray) -> None:
        holistic = self._make_mock_holistic(n_pose=0)
        result = process_holistic(bgr_image, holistic)
        assert result.pose_landmarks is None

    def test_returns_pose_landmarks(self, bgr_image: np.ndarray) -> None:
        holistic = self._make_mock_holistic(n_pose=33)
        result = process_holistic(bgr_image, holistic)
        assert result.pose_landmarks is not None
        assert len(result.pose_landmarks) == 33

    def test_returns_hand_landmarks(self, bgr_image: np.ndarray) -> None:
        holistic = self._make_mock_holistic(n_pose=0, n_left_hand=21, n_right_hand=21)
        result = process_holistic(bgr_image, holistic)
        assert result.left_hand_landmarks is not None
        assert len(result.left_hand_landmarks) == 21
        assert result.right_hand_landmarks is not None
        assert len(result.right_hand_landmarks) == 21

    def test_returns_face_landmarks(self, bgr_image: np.ndarray) -> None:
        holistic = self._make_mock_holistic(n_face=468)
        result = process_holistic(bgr_image, holistic)
        assert result.face_landmarks is not None
        assert len(result.face_landmarks) == 468

    def test_converts_to_rgb_before_processing(self, bgr_image: np.ndarray) -> None:
        holistic = self._make_mock_holistic()
        process_holistic(bgr_image, holistic)
        # process() should be called once with RGB array
        holistic.process.assert_called_once()
        called_arg = holistic.process.call_args[0][0]
        assert called_arg.shape == bgr_image.shape
        assert called_arg.dtype == np.uint8


# ---------------------------------------------------------------------------
# draw_holistic_results
# ---------------------------------------------------------------------------


class TestDrawHolisticResults:
    def _make_landmark_list(self, n: int) -> list[HolisticLandmark]:
        return [HolisticLandmark(x=i / max(n, 1), y=i / max(n, 1), z=0.0) for i in range(n)]

    def test_raises_for_non_ndarray(self) -> None:
        with pytest.raises(TypeError):
            draw_holistic_results("not an image", HolisticResult())  # type: ignore[arg-type]

    def test_raises_for_grayscale(self, gray_image: np.ndarray) -> None:
        with pytest.raises(ValueError):
            draw_holistic_results(gray_image, HolisticResult())

    def test_returns_copy_when_no_landmarks(self, bgr_image: np.ndarray) -> None:
        result = draw_holistic_results(bgr_image, HolisticResult())
        assert result is not bgr_image
        np.testing.assert_array_equal(result, bgr_image)

    def test_output_shape_matches_input(self, bgr_image: np.ndarray) -> None:
        landmarks = self._make_landmark_list(33)
        holistic_result = HolisticResult(pose_landmarks=landmarks)
        output = draw_holistic_results(bgr_image, holistic_result)
        assert output.shape == bgr_image.shape
        assert output.dtype == bgr_image.dtype

    def test_does_not_mutate_input(self, bgr_image: np.ndarray) -> None:
        original = bgr_image.copy()
        landmarks = self._make_landmark_list(33)
        draw_holistic_results(bgr_image, HolisticResult(pose_landmarks=landmarks))
        np.testing.assert_array_equal(bgr_image, original)

    def test_draws_pose_landmarks(self, bgr_image: np.ndarray) -> None:
        # Place a pose landmark in the centre so it's definitely drawn.
        landmark = HolisticLandmark(x=0.5, y=0.5, z=0.0)
        result_data = HolisticResult(pose_landmarks=[landmark])
        output = draw_holistic_results(bgr_image, result_data, draw_hands=False, draw_face=False)
        # Output should differ from input after drawing.
        assert not np.array_equal(output, bgr_image)

    def test_flag_draw_pose_false_skips_pose(self, bgr_image: np.ndarray) -> None:
        landmark = HolisticLandmark(x=0.5, y=0.5, z=0.0)
        result_data = HolisticResult(pose_landmarks=[landmark])
        output = draw_holistic_results(
            bgr_image, result_data, draw_pose=False, draw_hands=False, draw_face=False
        )
        np.testing.assert_array_equal(output, bgr_image)


# ---------------------------------------------------------------------------
# _draw_landmarks_dots  (internal helper)
# ---------------------------------------------------------------------------


class TestDrawLandmarksDots:
    def test_draws_dot_on_image(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        landmarks = [HolisticLandmark(x=0.5, y=0.5, z=0.0)]
        _draw_landmarks_dots(image, landmarks, 100, 100, (0, 255, 0))
        # The centre region should now be non-zero.
        assert image[50, 50].sum() > 0

    def test_empty_landmarks_no_change(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        _draw_landmarks_dots(image, [], 100, 100, (0, 255, 0))
        assert image.sum() == 0


# ---------------------------------------------------------------------------
# _draw_pose_connections  (internal helper)
# ---------------------------------------------------------------------------


class TestDrawPoseConnections:
    def test_draws_lines_for_valid_connections(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # 33 landmarks spread across the image
        landmarks = [
            HolisticLandmark(x=i / 32, y=0.5, z=0.0) for i in range(33)
        ]
        _draw_pose_connections(image, landmarks, 100, 100)
        assert image.sum() > 0

    def test_skips_connection_if_index_out_of_range(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Only 2 landmarks - most connections will be skipped
        landmarks = [HolisticLandmark(x=0.1, y=0.5, z=0.0)] * 2
        # Should not raise; just skips out-of-range connections
        _draw_pose_connections(image, landmarks, 100, 100)


# ---------------------------------------------------------------------------
# _draw_hand_connections  (internal helper)
# ---------------------------------------------------------------------------


class TestDrawHandConnections:
    def test_draws_lines_for_hand(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        landmarks = [HolisticLandmark(x=i / 20, y=0.5, z=0.0) for i in range(21)]
        _draw_hand_connections(image, landmarks, 100, 100, (255, 0, 0))
        assert image.sum() > 0
