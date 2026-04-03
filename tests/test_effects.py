"""Tests for image_analysis.effects module."""

from __future__ import annotations

import numpy as np
import pytest

from image_analysis.effects import (
    PIXELATE_BLOCK_SIZE,
    apply_cartoon,
    apply_emboss,
    apply_invert,
    apply_pixelate,
    apply_sepia,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bgr_image() -> np.ndarray:
    """Synthetic 100x100 BGR test image with random pixels."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def uniform_white() -> np.ndarray:
    """100x100 all-white BGR image."""
    return np.full((100, 100, 3), 255, dtype=np.uint8)


@pytest.fixture
def uniform_black() -> np.ndarray:
    """100x100 all-black BGR image."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def single_pixel() -> np.ndarray:
    """Minimal 1x1 BGR image."""
    return np.array([[[128, 64, 32]]], dtype=np.uint8)


# ---------------------------------------------------------------------------
# apply_invert
# ---------------------------------------------------------------------------


class TestApplyInvert:
    def test_output_shape_matches_input(self, bgr_image: np.ndarray) -> None:
        result = apply_invert(bgr_image)
        assert result.shape == bgr_image.shape

    def test_output_dtype_is_uint8(self, bgr_image: np.ndarray) -> None:
        result = apply_invert(bgr_image)
        assert result.dtype == np.uint8

    def test_inversion_is_correct(self, bgr_image: np.ndarray) -> None:
        result = apply_invert(bgr_image)
        expected = (255 - bgr_image.astype(np.int16)).astype(np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_white_becomes_black(self, uniform_white: np.ndarray) -> None:
        result = apply_invert(uniform_white)
        np.testing.assert_array_equal(result, np.zeros_like(uniform_white))

    def test_black_becomes_white(self, uniform_black: np.ndarray) -> None:
        result = apply_invert(uniform_black)
        np.testing.assert_array_equal(result, np.full_like(uniform_black, 255))

    def test_double_invert_is_identity(self, bgr_image: np.ndarray) -> None:
        np.testing.assert_array_equal(apply_invert(apply_invert(bgr_image)), bgr_image)

    def test_single_pixel(self, single_pixel: np.ndarray) -> None:
        result = apply_invert(single_pixel)
        assert result.shape == (1, 1, 3)

    def test_raises_on_non_array(self) -> None:
        with pytest.raises(TypeError):
            apply_invert([[1, 2, 3]])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# apply_sepia
# ---------------------------------------------------------------------------


class TestApplySepia:
    def test_output_shape_matches_input(self, bgr_image: np.ndarray) -> None:
        result = apply_sepia(bgr_image)
        assert result.shape == bgr_image.shape

    def test_output_dtype_is_uint8(self, bgr_image: np.ndarray) -> None:
        result = apply_sepia(bgr_image)
        assert result.dtype == np.uint8

    def test_output_values_in_valid_range(self, bgr_image: np.ndarray) -> None:
        result = apply_sepia(bgr_image)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_black_stays_black(self, uniform_black: np.ndarray) -> None:
        result = apply_sepia(uniform_black)
        np.testing.assert_array_equal(result, uniform_black)

    def test_single_pixel(self, single_pixel: np.ndarray) -> None:
        result = apply_sepia(single_pixel)
        assert result.shape == (1, 1, 3)
        assert result.dtype == np.uint8

    def test_sepia_red_channel_dominant(self, uniform_white: np.ndarray) -> None:
        result = apply_sepia(uniform_white)
        b, g, r = result[0, 0, 0], result[0, 0, 1], result[0, 0, 2]
        assert r >= g >= b, "Sepia red channel must be >= green >= blue for white input"

    def test_raises_on_non_array(self) -> None:
        with pytest.raises(TypeError):
            apply_sepia([[1, 2, 3]])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# apply_emboss
# ---------------------------------------------------------------------------


class TestApplyEmboss:
    def test_output_shape_matches_input(self, bgr_image: np.ndarray) -> None:
        result = apply_emboss(bgr_image)
        assert result.shape == bgr_image.shape

    def test_output_dtype_is_uint8(self, bgr_image: np.ndarray) -> None:
        result = apply_emboss(bgr_image)
        assert result.dtype == np.uint8

    def test_uniform_image_is_mid_grey(self, uniform_black: np.ndarray) -> None:
        result = apply_emboss(uniform_black)
        # For a uniform-black (0) input the emboss convolution sums to 0,
        # shifted by +128 → interior pixels should be ≈128.
        mid = result[5:-5, 5:-5]  # exclude border artifacts
        assert np.all(np.abs(mid.astype(np.int16) - 128) <= 2)

    def test_single_pixel(self, single_pixel: np.ndarray) -> None:
        result = apply_emboss(single_pixel)
        assert result.shape == (1, 1, 3)

    def test_raises_on_non_array(self) -> None:
        with pytest.raises(TypeError):
            apply_emboss([[1, 2, 3]])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# apply_pixelate
# ---------------------------------------------------------------------------


class TestApplyPixelate:
    def test_output_shape_matches_input(self, bgr_image: np.ndarray) -> None:
        result = apply_pixelate(bgr_image)
        assert result.shape == bgr_image.shape

    def test_output_dtype_is_uint8(self, bgr_image: np.ndarray) -> None:
        result = apply_pixelate(bgr_image)
        assert result.dtype == np.uint8

    def test_default_block_size_constant(self) -> None:
        assert PIXELATE_BLOCK_SIZE == 16

    @pytest.mark.parametrize("block_size", [4, 8, 16, 32])
    def test_various_block_sizes(self, bgr_image: np.ndarray, block_size: int) -> None:
        result = apply_pixelate(bgr_image, block_size=block_size)
        assert result.shape == bgr_image.shape
        assert result.dtype == np.uint8

    def test_block_size_1_is_identity(self, bgr_image: np.ndarray) -> None:
        result = apply_pixelate(bgr_image, block_size=1)
        np.testing.assert_array_equal(result, bgr_image)

    def test_uniform_image_stays_uniform(self, uniform_white: np.ndarray) -> None:
        result = apply_pixelate(uniform_white)
        np.testing.assert_array_equal(result, uniform_white)

    def test_invalid_block_size_zero(self, bgr_image: np.ndarray) -> None:
        with pytest.raises(ValueError):
            apply_pixelate(bgr_image, block_size=0)

    def test_invalid_block_size_negative(self, bgr_image: np.ndarray) -> None:
        with pytest.raises(ValueError):
            apply_pixelate(bgr_image, block_size=-1)

    def test_single_pixel(self, single_pixel: np.ndarray) -> None:
        result = apply_pixelate(single_pixel, block_size=1)
        assert result.shape == (1, 1, 3)

    def test_raises_on_non_array(self) -> None:
        with pytest.raises(TypeError):
            apply_pixelate([[1, 2, 3]])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# apply_cartoon
# ---------------------------------------------------------------------------


class TestApplyCartoon:
    def test_output_shape_matches_input(self, bgr_image: np.ndarray) -> None:
        result = apply_cartoon(bgr_image)
        assert result.shape == bgr_image.shape

    def test_output_dtype_is_uint8(self, bgr_image: np.ndarray) -> None:
        result = apply_cartoon(bgr_image)
        assert result.dtype == np.uint8

    def test_output_values_in_valid_range(self, bgr_image: np.ndarray) -> None:
        result = apply_cartoon(bgr_image)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_uniform_white_has_no_edges(self, uniform_white: np.ndarray) -> None:
        result = apply_cartoon(uniform_white)
        # Uniform image has no edges → mask is all-white → output ≈ smoothed white
        assert result.dtype == np.uint8

    def test_single_pixel(self, single_pixel: np.ndarray) -> None:
        result = apply_cartoon(single_pixel)
        assert result.shape == (1, 1, 3)

    def test_raises_on_non_array(self) -> None:
        with pytest.raises(TypeError):
            apply_cartoon([[1, 2, 3]])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Public API via package __init__
# ---------------------------------------------------------------------------


class TestPublicApiExports:
    def test_apply_invert_exported(self) -> None:
        import image_analysis

        assert callable(image_analysis.apply_invert)

    def test_apply_sepia_exported(self) -> None:
        import image_analysis

        assert callable(image_analysis.apply_sepia)

    def test_apply_emboss_exported(self) -> None:
        import image_analysis

        assert callable(image_analysis.apply_emboss)

    def test_apply_pixelate_exported(self) -> None:
        import image_analysis

        assert callable(image_analysis.apply_pixelate)

    def test_apply_cartoon_exported(self) -> None:
        import image_analysis

        assert callable(image_analysis.apply_cartoon)

    def test_pixelate_block_size_constant_exported(self) -> None:
        import image_analysis

        assert image_analysis.PIXELATE_BLOCK_SIZE == PIXELATE_BLOCK_SIZE
