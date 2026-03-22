"""Tests for image_analysis.april_tags module."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from image_analysis.april_tags import (
    DEFAULT_APRILTAG_FAMILY,
    AprilTagDetection,
    detect_april_tags,
    draw_april_tags,
)


@pytest.fixture
def april_tag_canvas() -> np.ndarray:
    """Return a synthetic BGR image containing two AprilTags."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    canvas = np.full((420, 420), 255, dtype=np.uint8)
    tag_zero = cv2.aruco.generateImageMarker(dictionary, 0, 120)
    tag_one = cv2.aruco.generateImageMarker(dictionary, 1, 120)
    canvas[40:160, 40:160] = tag_zero
    canvas[220:340, 220:340] = tag_one
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


class TestDetectAprilTags:
    def test_detects_april_tags_in_bgr_image(self, april_tag_canvas: np.ndarray) -> None:
        detections = detect_april_tags(april_tag_canvas)

        assert [detection.tag_id for detection in detections] == [0, 1]
        assert all(detection.family == DEFAULT_APRILTAG_FAMILY for detection in detections)
        assert all(detection.corners.shape == (4, 2) for detection in detections)
        assert detections[0].bbox[0] < detections[0].bbox[2]
        assert detections[0].bbox[1] < detections[0].bbox[3]

    def test_detects_april_tags_in_float32_grayscale(self, april_tag_canvas: np.ndarray) -> None:
        grayscale = cv2.cvtColor(april_tag_canvas, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        detections = detect_april_tags(grayscale)

        assert len(detections) == 2

    def test_returns_empty_list_when_no_tags_present(self) -> None:
        image = np.full((128, 128, 3), 255, dtype=np.uint8)
        assert detect_april_tags(image) == []

    def test_raises_for_unsupported_family(self, april_tag_canvas: np.ndarray) -> None:
        with pytest.raises(ValueError, match='Unsupported AprilTag family'):
            detect_april_tags(april_tag_canvas, family='tag99h99')

    def test_raises_for_invalid_channel_count(self) -> None:
        image = np.zeros((32, 32, 4), dtype=np.uint8)
        with pytest.raises(ValueError):
            detect_april_tags(image)


class TestDrawAprilTags:
    def test_draws_annotations_on_copy(self, april_tag_canvas: np.ndarray) -> None:
        detections = detect_april_tags(april_tag_canvas)

        rendered = draw_april_tags(april_tag_canvas, detections)

        assert rendered is not april_tag_canvas
        assert rendered.shape == april_tag_canvas.shape
        assert np.any(rendered != april_tag_canvas)

    def test_returns_identical_copy_for_empty_detections(
        self, april_tag_canvas: np.ndarray
    ) -> None:
        rendered = draw_april_tags(april_tag_canvas, [])
        np.testing.assert_array_equal(rendered, april_tag_canvas)

    def test_raises_for_non_positive_thickness(self, april_tag_canvas: np.ndarray) -> None:
        with pytest.raises(ValueError, match='thickness must be positive'):
            draw_april_tags(april_tag_canvas, [], thickness=0)

    def test_raises_for_non_bgr_input(self) -> None:
        detection = AprilTagDetection(
            tag_id=7,
            family=DEFAULT_APRILTAG_FAMILY,
            corners=np.array(
                [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                dtype=np.float32,
            ),
            center=(5.0, 5.0),
            bbox=(0, 0, 10, 10),
        )
        with pytest.raises(ValueError, match='image must be a BGR uint8 array'):
            draw_april_tags(np.zeros((32, 32), dtype=np.uint8), [detection])
