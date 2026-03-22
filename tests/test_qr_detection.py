"""Tests for image_analysis.qr_detection module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from image_analysis.qr_detection import (
    QRCode,
    detect_qr_codes,
    draw_qr_codes,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bgr_image() -> np.ndarray:
    """Return a synthetic 200x300 BGR uint8 image."""
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)


@pytest.fixture
def gray_image() -> np.ndarray:
    """Return a synthetic 200x300 grayscale uint8 image."""
    rng = np.random.default_rng(seed=1)
    return rng.integers(0, 255, (200, 300), dtype=np.uint8)


@pytest.fixture
def sample_qr_code() -> QRCode:
    """Return a sample QRCode with known values."""
    return QRCode(
        data="https://example.com",
        bbox=(10, 10, 110, 110),
        polygon=[(10, 10), (110, 10), (110, 110), (10, 110)],
    )


# ---------------------------------------------------------------------------
# QRCode dataclass
# ---------------------------------------------------------------------------


class TestQRCodeDataclass:
    def test_fields_accessible(self, sample_qr_code: QRCode) -> None:
        assert sample_qr_code.data == "https://example.com"
        assert sample_qr_code.bbox == (10, 10, 110, 110)
        assert len(sample_qr_code.polygon) == 4

    def test_is_frozen(self, sample_qr_code: QRCode) -> None:
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            sample_qr_code.data = "changed"  # type: ignore[misc]

    def test_default_polygon_is_empty(self) -> None:
        qr = QRCode(data="test", bbox=(0, 0, 10, 10))
        assert qr.polygon == []


# ---------------------------------------------------------------------------
# detect_qr_codes - input validation
# ---------------------------------------------------------------------------


class TestDetectQrCodesValidation:
    def test_raises_for_non_ndarray(self) -> None:
        with pytest.raises(TypeError, match=r"np\.ndarray"):
            detect_qr_codes([[1, 2, 3]])  # type: ignore[arg-type]

    def test_raises_for_float_dtype(self, bgr_image: np.ndarray) -> None:
        float_img = bgr_image.astype(np.float32)
        with pytest.raises(ValueError, match="uint8"):
            detect_qr_codes(float_img)

    def test_raises_for_4d_array(self) -> None:
        bad = np.zeros((10, 10, 3, 1), dtype=np.uint8)
        with pytest.raises(ValueError):
            detect_qr_codes(bad)

    def test_raises_for_4_channel_image(self) -> None:
        rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        with pytest.raises(ValueError):
            detect_qr_codes(rgba)

    def test_accepts_bgr_image(self, bgr_image: np.ndarray) -> None:
        # No exception - result can be empty list (no real QR in synthetic image)
        result = detect_qr_codes(bgr_image)
        assert isinstance(result, list)

    def test_accepts_grayscale_image(self, gray_image: np.ndarray) -> None:
        result = detect_qr_codes(gray_image)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# detect_qr_codes - detection logic (mocked detector)
# ---------------------------------------------------------------------------


class TestDetectQrCodesDetection:
    def _make_corners(self) -> np.ndarray:
        """Return a (1, 4, 2) float32 corners array for a single QR code."""
        corners = np.array(
            [[[10.0, 10.0], [110.0, 10.0], [110.0, 110.0], [10.0, 110.0]]],
            dtype=np.float32,
        )
        return corners

    def test_returns_empty_list_when_no_qr_found(self, bgr_image: np.ndarray) -> None:
        with patch("image_analysis.qr_detection.cv2.QRCodeDetector") as mock_cls:
            mock_detector = MagicMock()
            mock_detector.detectAndDecodeMulti.return_value = (False, [], None, None)
            mock_cls.return_value = mock_detector

            result = detect_qr_codes(bgr_image)

        assert result == []

    def test_returns_empty_list_when_points_none(self, bgr_image: np.ndarray) -> None:
        with patch("image_analysis.qr_detection.cv2.QRCodeDetector") as mock_cls:
            mock_detector = MagicMock()
            mock_detector.detectAndDecodeMulti.return_value = (False, ["text"], None, None)
            mock_cls.return_value = mock_detector

            result = detect_qr_codes(bgr_image)

        assert result == []

    def test_skips_empty_decoded_text(self, bgr_image: np.ndarray) -> None:
        corners = self._make_corners()
        with patch("image_analysis.qr_detection.cv2.QRCodeDetector") as mock_cls:
            mock_detector = MagicMock()
            mock_detector.detectAndDecodeMulti.return_value = (True, [""], corners, None)
            mock_cls.return_value = mock_detector

            result = detect_qr_codes(bgr_image)

        assert result == []

    def test_returns_qr_code_with_correct_data(self, bgr_image: np.ndarray) -> None:
        corners = self._make_corners()
        with patch("image_analysis.qr_detection.cv2.QRCodeDetector") as mock_cls:
            mock_detector = MagicMock()
            mock_detector.detectAndDecodeMulti.return_value = (
                True,
                ["https://example.com"],
                corners,
                None,
            )
            mock_cls.return_value = mock_detector

            result = detect_qr_codes(bgr_image)

        assert len(result) == 1
        assert result[0].data == "https://example.com"

    def test_bbox_matches_corner_extents(self, bgr_image: np.ndarray) -> None:
        corners = self._make_corners()
        with patch("image_analysis.qr_detection.cv2.QRCodeDetector") as mock_cls:
            mock_detector = MagicMock()
            mock_detector.detectAndDecodeMulti.return_value = (
                True,
                ["data"],
                corners,
                None,
            )
            mock_cls.return_value = mock_detector

            result = detect_qr_codes(bgr_image)

        assert result[0].bbox == (10, 10, 110, 110)

    def test_polygon_has_four_points(self, bgr_image: np.ndarray) -> None:
        corners = self._make_corners()
        with patch("image_analysis.qr_detection.cv2.QRCodeDetector") as mock_cls:
            mock_detector = MagicMock()
            mock_detector.detectAndDecodeMulti.return_value = (
                True,
                ["data"],
                corners,
                None,
            )
            mock_cls.return_value = mock_detector

            result = detect_qr_codes(bgr_image)

        assert len(result[0].polygon) == 4

    def test_multiple_qr_codes_returned(self, bgr_image: np.ndarray) -> None:
        corners = np.array(
            [
                [[10.0, 10.0], [60.0, 10.0], [60.0, 60.0], [10.0, 60.0]],
                [[70.0, 70.0], [120.0, 70.0], [120.0, 120.0], [70.0, 120.0]],
            ],
            dtype=np.float32,
        )
        with patch("image_analysis.qr_detection.cv2.QRCodeDetector") as mock_cls:
            mock_detector = MagicMock()
            mock_detector.detectAndDecodeMulti.return_value = (
                True,
                ["first", "second"],
                corners,
                None,
            )
            mock_cls.return_value = mock_detector

            result = detect_qr_codes(bgr_image)

        assert len(result) == 2
        assert result[0].data == "first"
        assert result[1].data == "second"


# ---------------------------------------------------------------------------
# draw_qr_codes - input validation
# ---------------------------------------------------------------------------


class TestDrawQrCodesValidation:
    def test_raises_for_non_ndarray(self, sample_qr_code: QRCode) -> None:
        with pytest.raises(TypeError):
            draw_qr_codes("not an image", [sample_qr_code])  # type: ignore[arg-type]

    def test_raises_for_grayscale_image(
        self, gray_image: np.ndarray, sample_qr_code: QRCode
    ) -> None:
        with pytest.raises(ValueError):
            draw_qr_codes(gray_image, [sample_qr_code])

    def test_raises_for_float_image(self, sample_qr_code: QRCode) -> None:
        img = np.zeros((100, 100, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="uint8"):
            draw_qr_codes(img, [sample_qr_code])

    def test_raises_for_non_positive_thickness(
        self, bgr_image: np.ndarray, sample_qr_code: QRCode
    ) -> None:
        with pytest.raises(ValueError, match="thickness"):
            draw_qr_codes(bgr_image, [sample_qr_code], thickness=0)


# ---------------------------------------------------------------------------
# draw_qr_codes - drawing behaviour
# ---------------------------------------------------------------------------


class TestDrawQrCodes:
    def test_returns_copy_not_inplace(
        self, bgr_image: np.ndarray, sample_qr_code: QRCode
    ) -> None:
        original = bgr_image.copy()
        result = draw_qr_codes(bgr_image, [sample_qr_code])
        np.testing.assert_array_equal(bgr_image, original)
        assert result is not bgr_image

    def test_output_shape_matches_input(
        self, bgr_image: np.ndarray, sample_qr_code: QRCode
    ) -> None:
        result = draw_qr_codes(bgr_image, [sample_qr_code])
        assert result.shape == bgr_image.shape
        assert result.dtype == bgr_image.dtype

    def test_empty_qr_codes_returns_copy(self, bgr_image: np.ndarray) -> None:
        result = draw_qr_codes(bgr_image, [])
        np.testing.assert_array_equal(result, bgr_image)

    def test_draws_with_polygon(self, bgr_image: np.ndarray) -> None:
        qr = QRCode(
            data="test",
            bbox=(10, 10, 60, 60),
            polygon=[(10, 10), (60, 10), (60, 60), (10, 60)],
        )
        result = draw_qr_codes(bgr_image, [qr])
        assert result.shape == bgr_image.shape

    def test_draws_with_empty_polygon_uses_bbox(self, bgr_image: np.ndarray) -> None:
        qr = QRCode(data="test", bbox=(10, 10, 60, 60))
        result = draw_qr_codes(bgr_image, [qr])
        assert result.shape == bgr_image.shape

    def test_output_differs_from_input_when_qr_drawn(
        self, bgr_image: np.ndarray, sample_qr_code: QRCode
    ) -> None:
        result = draw_qr_codes(bgr_image, [sample_qr_code])
        assert not np.array_equal(result, bgr_image)
