"""Tests for image_analysis.calibration module."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from image_analysis.calibration import (
    MIN_CALIBRATION_FRAMES,
    CalibrationResult,
    calibrate_camera,
    draw_chessboard_corners,
    find_chessboard_corners,
    undistort_image,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

BOARD_WIDTH = 5
BOARD_HEIGHT = 4


def _make_calibration_board(
    board_width: int = BOARD_WIDTH,
    board_height: int = BOARD_HEIGHT,
    square_size: int = 50,
) -> np.ndarray:
    """Generate a synthetic BGR chessboard image that OpenCV can detect.

    Creates a board with ``(board_width + 1)`` squares horizontally and
    ``(board_height + 1)`` squares vertically, plus one extra border square
    on every side for reliable detection.
    """
    # One extra square per side beyond the inner-corner count.
    total_cols = board_width + 1 + 2
    total_rows = board_height + 1 + 2
    img_h = total_rows * square_size
    img_w = total_cols * square_size

    img = np.full((img_h, img_w), 255, dtype=np.uint8)
    for r in range(total_rows):
        for c in range(total_cols):
            if (r + c) % 2 == 1:
                y0 = r * square_size
                x0 = c * square_size
                img[y0 : y0 + square_size, x0 : x0 + square_size] = 0

    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


@pytest.fixture
def chessboard_image() -> np.ndarray:
    """Synthetic BGR chessboard image used in multiple tests."""
    return _make_calibration_board()


@pytest.fixture
def blank_image() -> np.ndarray:
    """Plain white BGR image with no chessboard pattern."""
    return np.full((200, 200, 3), 255, dtype=np.uint8)


# ---------------------------------------------------------------------------
# find_chessboard_corners
# ---------------------------------------------------------------------------


class TestFindChessboardCorners:
    def test_detects_corners_in_synthetic_board(self, chessboard_image: np.ndarray) -> None:
        corners = find_chessboard_corners(chessboard_image, BOARD_WIDTH, BOARD_HEIGHT)

        assert corners is not None
        assert corners.dtype == np.float32
        assert corners.shape == (BOARD_WIDTH * BOARD_HEIGHT, 1, 2)

    def test_returns_none_when_board_not_present(self, blank_image: np.ndarray) -> None:
        result = find_chessboard_corners(blank_image, BOARD_WIDTH, BOARD_HEIGHT)
        assert result is None

    def test_accepts_grayscale_image(self, chessboard_image: np.ndarray) -> None:
        gray = cv2.cvtColor(chessboard_image, cv2.COLOR_BGR2GRAY)
        corners = find_chessboard_corners(gray, BOARD_WIDTH, BOARD_HEIGHT)
        assert corners is not None

    def test_accepts_float32_image(self, chessboard_image: np.ndarray) -> None:
        float_image = (chessboard_image / 255.0).astype(np.float32)
        corners = find_chessboard_corners(float_image, BOARD_WIDTH, BOARD_HEIGHT)
        assert corners is not None

    def test_raises_for_non_array(self) -> None:
        with pytest.raises(TypeError, match=r"np\.ndarray"):
            find_chessboard_corners([[0, 0], [0, 0]])  # type: ignore[arg-type]

    def test_raises_for_zero_board_width(self, chessboard_image: np.ndarray) -> None:
        with pytest.raises(ValueError, match="positive"):
            find_chessboard_corners(chessboard_image, board_width=0)

    def test_raises_for_negative_board_height(self, chessboard_image: np.ndarray) -> None:
        with pytest.raises(ValueError, match="positive"):
            find_chessboard_corners(chessboard_image, board_height=-1)


# ---------------------------------------------------------------------------
# calibrate_camera
# ---------------------------------------------------------------------------


class TestCalibrateCamera:
    def test_raises_when_fewer_than_min_images_provided(self) -> None:
        images = [np.zeros((100, 100, 3), dtype=np.uint8)] * (MIN_CALIBRATION_FRAMES - 1)
        with pytest.raises(ValueError, match=str(MIN_CALIBRATION_FRAMES)):
            calibrate_camera(images, BOARD_WIDTH, BOARD_HEIGHT)

    def test_raises_when_no_chessboard_detected_in_any_image(
        self, blank_image: np.ndarray
    ) -> None:
        images = [blank_image.copy() for _ in range(MIN_CALIBRATION_FRAMES + 2)]
        with pytest.raises(ValueError, match="detectable chessboard"):
            calibrate_camera(images, BOARD_WIDTH, BOARD_HEIGHT)

    def test_raises_for_non_positive_square_size(self, blank_image: np.ndarray) -> None:
        images = [blank_image] * MIN_CALIBRATION_FRAMES
        with pytest.raises(ValueError, match="square_size must be positive"):
            calibrate_camera(images, BOARD_WIDTH, BOARD_HEIGHT, square_size=0.0)

    def test_raises_for_zero_board_dimensions(self, blank_image: np.ndarray) -> None:
        images = [blank_image] * MIN_CALIBRATION_FRAMES
        with pytest.raises(ValueError, match="positive"):
            calibrate_camera(images, board_width=0, board_height=BOARD_HEIGHT)

    def test_returns_calibration_result_with_correct_types(self) -> None:
        board = _make_calibration_board()
        images = [board.copy() for _ in range(MIN_CALIBRATION_FRAMES)]
        result = calibrate_camera(images, BOARD_WIDTH, BOARD_HEIGHT)

        assert isinstance(result, CalibrationResult)
        assert result.camera_matrix.shape == (3, 3)
        assert result.camera_matrix.dtype == np.float64
        assert result.dist_coefficients.dtype == np.float64
        assert isinstance(result.rms_error, float)
        assert isinstance(result.frame_count, int)
        assert result.frame_count >= MIN_CALIBRATION_FRAMES
        assert result.image_size == (board.shape[1], board.shape[0])


# ---------------------------------------------------------------------------
# undistort_image
# ---------------------------------------------------------------------------


class TestUndistortImage:
    @pytest.fixture
    def dummy_calibration(self) -> CalibrationResult:
        """A CalibrationResult with near-zero distortion (identity-like)."""
        w, h = 200, 200
        focal = 200.0
        camera_matrix = np.array(
            [[focal, 0.0, w / 2], [0.0, focal, h / 2], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coefficients = np.zeros((1, 5), dtype=np.float64)
        return CalibrationResult(
            camera_matrix=camera_matrix,
            dist_coefficients=dist_coefficients,
            rms_error=0.1,
            image_size=(w, h),
            frame_count=MIN_CALIBRATION_FRAMES,
        )

    def test_returns_array_with_same_shape_and_dtype(
        self, dummy_calibration: CalibrationResult
    ) -> None:
        image = np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8)
        result = undistort_image(image, dummy_calibration)

        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_near_identity_calibration_preserves_image(
        self, dummy_calibration: CalibrationResult
    ) -> None:
        image = np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8)
        result = undistort_image(image, dummy_calibration)

        # With near-zero distortion the output should be very close to the input.
        assert float(np.mean(np.abs(result.astype(np.int32) - image.astype(np.int32)))) < 5.0

    def test_raises_for_float32_image(self, dummy_calibration: CalibrationResult) -> None:
        float_img = np.zeros((100, 100, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="uint8"):
            undistort_image(float_img, dummy_calibration)

    def test_raises_for_non_array_input(self, dummy_calibration: CalibrationResult) -> None:
        with pytest.raises(TypeError):
            undistort_image("not_an_image", dummy_calibration)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# draw_chessboard_corners
# ---------------------------------------------------------------------------


class TestDrawChessboardCorners:
    def test_returns_copy_with_same_shape(self, chessboard_image: np.ndarray) -> None:
        corners = find_chessboard_corners(chessboard_image, BOARD_WIDTH, BOARD_HEIGHT)
        assert corners is not None

        result = draw_chessboard_corners(chessboard_image, corners, BOARD_WIDTH, BOARD_HEIGHT)

        assert result is not chessboard_image
        assert result.shape == chessboard_image.shape
        assert result.dtype == np.uint8

    def test_modifies_output_when_drawing_corners(self, chessboard_image: np.ndarray) -> None:
        corners = find_chessboard_corners(chessboard_image, BOARD_WIDTH, BOARD_HEIGHT)
        assert corners is not None

        result = draw_chessboard_corners(chessboard_image, corners, BOARD_WIDTH, BOARD_HEIGHT)
        assert np.any(result != chessboard_image)

    def test_does_not_modify_original_image(self, chessboard_image: np.ndarray) -> None:
        original = chessboard_image.copy()
        corners = find_chessboard_corners(chessboard_image, BOARD_WIDTH, BOARD_HEIGHT)
        assert corners is not None

        draw_chessboard_corners(chessboard_image, corners, BOARD_WIDTH, BOARD_HEIGHT)
        np.testing.assert_array_equal(chessboard_image, original)

    def test_raises_for_non_bgr_image(self, chessboard_image: np.ndarray) -> None:
        corners = find_chessboard_corners(chessboard_image, BOARD_WIDTH, BOARD_HEIGHT)
        assert corners is not None

        gray = cv2.cvtColor(chessboard_image, cv2.COLOR_BGR2GRAY)
        with pytest.raises(ValueError, match="BGR uint8"):
            draw_chessboard_corners(gray, corners, BOARD_WIDTH, BOARD_HEIGHT)

    def test_raises_for_non_positive_board_dimensions(self, chessboard_image: np.ndarray) -> None:
        dummy_corners = np.zeros((BOARD_WIDTH * BOARD_HEIGHT, 1, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="positive"):
            draw_chessboard_corners(
                chessboard_image, dummy_corners, board_width=0, board_height=BOARD_HEIGHT
            )

    def test_raises_for_non_array_image(self) -> None:
        corners = np.zeros((BOARD_WIDTH * BOARD_HEIGHT, 1, 2), dtype=np.float32)
        with pytest.raises(TypeError, match=r"np\.ndarray"):
            draw_chessboard_corners("not_an_image", corners)  # type: ignore[arg-type]
