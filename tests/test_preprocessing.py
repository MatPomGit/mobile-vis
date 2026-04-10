"""Tests for image_analysis.preprocessing module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bgr_image() -> np.ndarray:
    """Return a synthetic 100x200 BGR uint8 image."""
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, 255, (100, 200, 3), dtype=np.uint8)


@pytest.fixture
def gray_image() -> np.ndarray:
    """Return a synthetic 100x200 grayscale uint8 image."""
    rng = np.random.default_rng(seed=1)
    return rng.integers(0, 255, (100, 200), dtype=np.uint8)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------


class TestLoadImage:
    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        from image_analysis.preprocessing import load_image

        with pytest.raises(FileNotFoundError):
            load_image(tmp_path / "nonexistent.jpg")

    def test_raises_value_error_for_non_image(self, tmp_path: Path) -> None:
        from image_analysis.preprocessing import load_image

        corrupt = tmp_path / "bad.jpg"
        corrupt.write_bytes(b"not an image")
        with pytest.raises(ValueError):
            load_image(corrupt)

    def test_loads_valid_image(self, tmp_path: Path, bgr_image: np.ndarray) -> None:
        import cv2

        from image_analysis.preprocessing import load_image

        img_path = tmp_path / "test.png"
        cv2.imwrite(str(img_path), bgr_image)
        loaded = load_image(img_path)
        assert loaded.shape == bgr_image.shape
        assert loaded.dtype == np.uint8


# ---------------------------------------------------------------------------
# resize_image
# ---------------------------------------------------------------------------


class TestResizeImage:
    def test_returns_correct_shape(self, bgr_image: np.ndarray) -> None:
        from image_analysis.preprocessing import resize_image

        result = resize_image(bgr_image, width=50, height=40)
        assert result.shape == (40, 50, 3)

    def test_grayscale_image(self, gray_image: np.ndarray) -> None:
        from image_analysis.preprocessing import resize_image

        result = resize_image(gray_image, width=30, height=20)
        assert result.shape == (20, 30)

    @pytest.mark.parametrize("width,height", [(0, 10), (10, 0), (-1, 10), (10, -5)])
    def test_raises_for_non_positive_dimensions(
        self, bgr_image: np.ndarray, width: int, height: int
    ) -> None:
        from image_analysis.preprocessing import resize_image

        with pytest.raises(ValueError):
            resize_image(bgr_image, width=width, height=height)

    def test_raises_for_non_ndarray(self) -> None:
        from image_analysis.preprocessing import resize_image

        with pytest.raises(TypeError):
            resize_image([1, 2, 3], width=10, height=10)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# normalize_image
# ---------------------------------------------------------------------------


class TestNormalizeImage:
    def test_output_dtype_is_float32(self, bgr_image: np.ndarray) -> None:
        from image_analysis.preprocessing import normalize_image

        result = normalize_image(bgr_image)
        assert result.dtype == np.float32

    def test_output_range_is_zero_to_one(self, bgr_image: np.ndarray) -> None:
        from image_analysis.preprocessing import normalize_image

        result = normalize_image(bgr_image)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_shape_preserved(self, bgr_image: np.ndarray) -> None:
        from image_analysis.preprocessing import normalize_image

        result = normalize_image(bgr_image)
        assert result.shape == bgr_image.shape

    def test_raises_for_float_input(self, bgr_image: np.ndarray) -> None:
        from image_analysis.preprocessing import normalize_image

        float_img = bgr_image.astype(np.float32)
        with pytest.raises(ValueError):
            normalize_image(float_img)


class TestResizeWithAspectRatio:
    def test_respects_max_size_and_keeps_aspect_ratio(self, bgr_image: np.ndarray) -> None:
        from image_analysis.preprocessing import resize_with_aspect_ratio

        resized = resize_with_aspect_ratio(bgr_image, max_width=80, max_height=80)
        assert resized.shape == (40, 80, 3)

    def test_does_not_upscale_smaller_image(self) -> None:
        from image_analysis.preprocessing import resize_with_aspect_ratio

        image = np.zeros((20, 30, 3), dtype=np.uint8)
        resized = resize_with_aspect_ratio(image, max_width=200, max_height=200)
        assert resized.shape == image.shape

    @pytest.mark.parametrize("max_width,max_height", [(0, 10), (10, 0), (-1, 10)])
    def test_raises_for_invalid_target_size(
        self,
        bgr_image: np.ndarray,
        max_width: int,
        max_height: int,
    ) -> None:
        from image_analysis.preprocessing import resize_with_aspect_ratio

        with pytest.raises(ValueError):
            resize_with_aspect_ratio(bgr_image, max_width=max_width, max_height=max_height)


class TestCenterCrop:
    def test_returns_center_crop_with_expected_size(self, bgr_image: np.ndarray) -> None:
        from image_analysis.preprocessing import center_crop

        cropped = center_crop(bgr_image, crop_width=120, crop_height=60)
        assert cropped.shape == (60, 120, 3)

    def test_returns_copy_not_view(self, bgr_image: np.ndarray) -> None:
        from image_analysis.preprocessing import center_crop

        cropped = center_crop(bgr_image, crop_width=50, crop_height=50)
        cropped[:, :, :] = 0
        assert not np.array_equal(cropped, bgr_image[25:75, 75:125, :])

    @pytest.mark.parametrize("crop_width,crop_height", [(0, 10), (10, 0), (-1, 10)])
    def test_raises_for_non_positive_crop_dimensions(
        self, bgr_image: np.ndarray, crop_width: int, crop_height: int
    ) -> None:
        from image_analysis.preprocessing import center_crop

        with pytest.raises(ValueError):
            center_crop(bgr_image, crop_width=crop_width, crop_height=crop_height)

    def test_raises_when_crop_larger_than_input(self, bgr_image: np.ndarray) -> None:
        from image_analysis.preprocessing import center_crop

        with pytest.raises(ValueError):
            center_crop(bgr_image, crop_width=500, crop_height=500)
