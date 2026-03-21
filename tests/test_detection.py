"""Tests for image_analysis.detection module."""

from __future__ import annotations

import numpy as np
import pytest

from image_analysis.detection import (
    Detection,
    apply_nms,
    detect_objects,
    draw_bounding_boxes,
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
def sample_detections() -> list[Detection]:
    """Return two overlapping sample detections."""
    return [
        Detection(label="cat", confidence=0.9, bbox=(10, 20, 110, 120)),
        Detection(label="cat", confidence=0.7, bbox=(15, 25, 115, 125)),
        Detection(label="dog", confidence=0.6, bbox=(200, 50, 280, 150)),
    ]


# ---------------------------------------------------------------------------
# detect_objects
# ---------------------------------------------------------------------------


class TestDetectObjects:
    def test_returns_empty_list_for_stub(self, bgr_image: np.ndarray) -> None:
        result = detect_objects(bgr_image)
        assert result == []

    def test_raises_for_non_ndarray(self) -> None:
        with pytest.raises(TypeError):
            detect_objects([[1, 2, 3]])  # type: ignore[arg-type]

    def test_raises_for_grayscale_image(self) -> None:
        gray = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError):
            detect_objects(gray)

    @pytest.mark.parametrize("threshold", [-0.1, 1.1, 2.0])
    def test_raises_for_invalid_threshold(
        self, bgr_image: np.ndarray, threshold: float
    ) -> None:
        with pytest.raises(ValueError):
            detect_objects(bgr_image, confidence_threshold=threshold)


# ---------------------------------------------------------------------------
# apply_nms
# ---------------------------------------------------------------------------


class TestApplyNms:
    def test_empty_input_returns_empty(self) -> None:
        assert apply_nms([]) == []

    def test_single_detection_returned_unchanged(self) -> None:
        det = Detection(label="cat", confidence=0.9, bbox=(0, 0, 50, 50))
        result = apply_nms([det])
        assert len(result) == 1
        assert result[0] is det

    @pytest.mark.parametrize("threshold", [-0.1, 1.1])
    def test_raises_for_invalid_iou_threshold(
        self, sample_detections: list[Detection], threshold: float
    ) -> None:
        with pytest.raises(ValueError):
            apply_nms(sample_detections, iou_threshold=threshold)


# ---------------------------------------------------------------------------
# draw_bounding_boxes
# ---------------------------------------------------------------------------


class TestDrawBoundingBoxes:
    def test_returns_copy_not_inplace(
        self, bgr_image: np.ndarray, sample_detections: list[Detection]
    ) -> None:
        original = bgr_image.copy()
        result = draw_bounding_boxes(bgr_image, sample_detections)
        np.testing.assert_array_equal(bgr_image, original)
        assert result is not bgr_image

    def test_output_shape_matches_input(
        self, bgr_image: np.ndarray, sample_detections: list[Detection]
    ) -> None:
        result = draw_bounding_boxes(bgr_image, sample_detections)
        assert result.shape == bgr_image.shape

    def test_empty_detections_returns_copy(self, bgr_image: np.ndarray) -> None:
        result = draw_bounding_boxes(bgr_image, [])
        np.testing.assert_array_equal(result, bgr_image)

    def test_raises_for_non_ndarray(self, sample_detections: list[Detection]) -> None:
        with pytest.raises(TypeError):
            draw_bounding_boxes("not an image", sample_detections)  # type: ignore[arg-type]
