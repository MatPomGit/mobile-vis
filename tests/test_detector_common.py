"""Tests for shared detector helpers."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from image_analysis.detector_common import load_with_retry, validate_bgr_uint8_image
from image_analysis.utils import validate_bbox_xyxy


def test_validate_bgr_uint8_image_accepts_valid_input() -> None:
    """Should accept a valid BGR uint8 image with shape (H, W, 3)."""
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    validate_bgr_uint8_image(image)


def test_validate_bgr_uint8_image_accepts_1x1_image() -> None:
    """Should accept the smallest possible BGR image with one pixel."""
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    validate_bgr_uint8_image(image)


@pytest.mark.parametrize(
    ("image", "expected_error"),
    [
        (np.zeros((16, 16, 3), dtype=np.float32), ValueError),
        (np.zeros((16, 16), dtype=np.uint8), ValueError),
        (np.zeros((16, 16, 1), dtype=np.uint8), ValueError),
    ],
)
def test_validate_bgr_uint8_image_rejects_invalid_dtype_or_shape(
    image: np.ndarray,
    expected_error: type[Exception],
) -> None:
    """Should reject invalid dtype and shape for image validation."""
    with pytest.raises(expected_error):
        validate_bgr_uint8_image(image)


@pytest.mark.parametrize("bbox", [(0, 0, 1, 1), [2, 3, 10, 12]])
def test_validate_bbox_xyxy_accepts_valid_boxes(bbox: object) -> None:
    """Should accept valid XYXY boxes represented as tuple or list."""
    validated = validate_bbox_xyxy(bbox)
    assert validated[2] > validated[0]
    assert validated[3] > validated[1]


@pytest.mark.parametrize("bbox", [(0, 0, 0, 1), (3, 2, 1, 5), (1, 2, 3), "bad"])
def test_validate_bbox_xyxy_rejects_invalid_boxes(bbox: object) -> None:
    """Should reject malformed or geometrically invalid XYXY boxes."""
    with pytest.raises((TypeError, ValueError)):
        validate_bbox_xyxy(bbox)


def test_load_with_retry_succeeds_after_retry() -> None:
    """Should retry once and return a created object on second attempt."""
    created_object = MagicMock()
    factory = MagicMock(side_effect=[RuntimeError("network"), created_object])
    test_logger = logging.getLogger("tests.detector_common")

    with patch("image_analysis.detector_common.time.sleep") as sleep_mock:
        result = load_with_retry(
            factory,
            "yolov8n.pt",
            max_retries=3,
            retry_delay=1.0,
            logger=test_logger,
            retry_message=(
                "YOLO load failed for '{target}' ({attempt}/{max_retries}): "
                "{error}; retry in {delay:.1f}s"
            ),
            failure_message=(
                "Failed to download YOLO model '{target}' "
                "after {max_retries} attempts."
            ),
        )

    assert result is created_object
    assert factory.call_count == 2
    sleep_mock.assert_called_once_with(1.0)


def test_load_with_retry_raises_runtime_error_after_retries() -> None:
    """Should raise RuntimeError and preserve cause after all retries fail."""
    original_error = OSError("permanent failure")
    factory = MagicMock(side_effect=original_error)
    test_logger = logging.getLogger("tests.detector_common")

    with (
        patch("image_analysis.detector_common.time.sleep") as sleep_mock,
        pytest.raises(RuntimeError, match="Failed to load RTMDet model") as exc_info,
    ):
        load_with_retry(
            factory,
            "rtmdet-nano",
            max_retries=2,
            retry_delay=2.0,
            logger=test_logger,
            retry_message=(
                "Failed to load RTMDet model '{target}' "
                "(attempt {attempt}/{max_retries}): {error} - retrying in {delay:.1f} s"
            ),
            failure_message=(
                "Failed to load RTMDet model '{target}' "
                "after {max_retries} attempts."
            ),
        )

    assert exc_info.value.__cause__ is original_error
    assert factory.call_count == 2
    assert sleep_mock.call_args_list == [call(2.0)]
