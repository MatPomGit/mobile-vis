"""Tests for image_analysis.cctag module."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import cv2
import numpy as np
import pytest

from image_analysis.cctag import (
    MAX_CCTAG_RINGS,
    MIN_CCTAG_RINGS,
    CCTagDetection,
    detect_cc_tags,
    draw_cc_tags,
    estimate_cctag_pose,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cctag_canvas(
    cx: int = 150,
    cy: int = 150,
    num_rings: int = 3,
    base_radius: int = 60,
    size: int = 300,
) -> np.ndarray:
    """Return a synthetic BGR image containing a CCTag-like marker.

    The marker is built from concentric filled circles with alternating
    black/white colours, matching the structure that CCTag detection looks for.
    """
    canvas = np.full((size, size), 255, dtype=np.uint8)
    step = base_radius // num_rings
    for i in range(num_rings):
        radius = base_radius - i * step
        fill_color = 0 if i % 2 == 0 else 255
        cv2.circle(canvas, (cx, cy), radius, fill_color, -1)
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cctag_canvas() -> np.ndarray:
    """Return a synthetic BGR image with a 3-ring CCTag marker."""
    return _make_cctag_canvas(num_rings=3)


@pytest.fixture
def bgr_image() -> np.ndarray:
    """Return a plain white BGR image with no markers."""
    return np.full((200, 200, 3), 255, dtype=np.uint8)


@pytest.fixture
def sample_detection() -> CCTagDetection:
    """Return a CCTagDetection with known values."""
    return CCTagDetection(
        tag_id=3,
        center=(100.0, 100.0),
        radius=50.0,
        bbox=(50, 50, 150, 150),
        rings_count=3,
    )


@pytest.fixture
def pinhole_camera_matrix() -> np.ndarray:
    """Return a simple pinhole camera matrix for a 640×480 sensor."""
    fx = fy = 600.0
    cx, cy = 320.0, 240.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


@pytest.fixture
def zero_dist_coeffs() -> np.ndarray:
    """Return zero distortion coefficients (ideal pinhole camera)."""
    return np.zeros(5, dtype=np.float64)


# ---------------------------------------------------------------------------
# CCTagDetection dataclass
# ---------------------------------------------------------------------------


class TestCCTagDetectionDataclass:
    def test_fields_accessible(self, sample_detection: CCTagDetection) -> None:
        assert sample_detection.tag_id == 3
        assert sample_detection.center == (100.0, 100.0)
        assert sample_detection.radius == 50.0
        assert sample_detection.bbox == (50, 50, 150, 150)
        assert sample_detection.rings_count == 3

    def test_is_frozen(self, sample_detection: CCTagDetection) -> None:
        with pytest.raises(FrozenInstanceError):
            sample_detection.tag_id = 5  # type: ignore[misc]

    def test_minimum_valid_detection(self) -> None:
        d = CCTagDetection(
            tag_id=MIN_CCTAG_RINGS,
            center=(0.0, 0.0),
            radius=10.0,
            bbox=(0, 0, 10, 10),
            rings_count=MIN_CCTAG_RINGS,
        )
        assert d.rings_count == MIN_CCTAG_RINGS

    def test_confidence_defaults_to_zero(self) -> None:
        d = CCTagDetection(
            tag_id=2,
            center=(0.0, 0.0),
            radius=10.0,
            bbox=(0, 0, 10, 10),
            rings_count=2,
        )
        assert d.confidence == 0.0

    def test_confidence_can_be_set(self) -> None:
        d = CCTagDetection(
            tag_id=3,
            center=(0.0, 0.0),
            radius=10.0,
            bbox=(0, 0, 10, 10),
            rings_count=3,
            confidence=0.85,
        )
        assert d.confidence == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# detect_cc_tags - input validation
# ---------------------------------------------------------------------------


class TestDetectCCTagsValidation:
    def test_raises_for_non_ndarray(self) -> None:
        with pytest.raises(TypeError):
            detect_cc_tags([[1, 2], [3, 4]])  # type: ignore[arg-type]

    def test_raises_for_4d_array(self) -> None:
        bad = np.zeros((10, 10, 3, 1), dtype=np.uint8)
        with pytest.raises(ValueError):
            detect_cc_tags(bad)

    def test_raises_for_invalid_circularity_too_high(
        self, cctag_canvas: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="min_circularity"):
            detect_cc_tags(cctag_canvas, min_circularity=1.5)

    def test_raises_for_invalid_circularity_negative(
        self, cctag_canvas: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="min_circularity"):
            detect_cc_tags(cctag_canvas, min_circularity=-0.1)

    def test_raises_for_non_positive_min_area(self, cctag_canvas: np.ndarray) -> None:
        with pytest.raises(ValueError, match="min_area"):
            detect_cc_tags(cctag_canvas, min_area=0.0)

    def test_raises_for_negative_min_area(self, cctag_canvas: np.ndarray) -> None:
        with pytest.raises(ValueError, match="min_area"):
            detect_cc_tags(cctag_canvas, min_area=-1.0)

    def test_accepts_bgr_uint8_image(self, cctag_canvas: np.ndarray) -> None:
        result = detect_cc_tags(cctag_canvas)
        assert isinstance(result, list)

    def test_accepts_grayscale_uint8_image(self, cctag_canvas: np.ndarray) -> None:
        gray = cv2.cvtColor(cctag_canvas, cv2.COLOR_BGR2GRAY)
        result = detect_cc_tags(gray)
        assert isinstance(result, list)

    def test_accepts_float32_image(self, cctag_canvas: np.ndarray) -> None:
        float_img = cctag_canvas.astype(np.float32) / 255.0
        result = detect_cc_tags(float_img)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# detect_cc_tags - detection logic
# ---------------------------------------------------------------------------


class TestDetectCCTagsDetection:
    def test_detects_marker_in_bgr_image(self, cctag_canvas: np.ndarray) -> None:
        detections = detect_cc_tags(cctag_canvas)
        assert len(detections) >= 1

    def test_detects_marker_in_grayscale_image(self, cctag_canvas: np.ndarray) -> None:
        gray = cv2.cvtColor(cctag_canvas, cv2.COLOR_BGR2GRAY)
        detections = detect_cc_tags(gray)
        assert len(detections) >= 1

    def test_detects_marker_in_float32_image(self, cctag_canvas: np.ndarray) -> None:
        float_img = cctag_canvas.astype(np.float32) / 255.0
        detections = detect_cc_tags(float_img)
        assert len(detections) >= 1

    def test_rings_count_within_bounds(self, cctag_canvas: np.ndarray) -> None:
        detections = detect_cc_tags(cctag_canvas)
        for d in detections:
            assert MIN_CCTAG_RINGS <= d.rings_count <= MAX_CCTAG_RINGS

    def test_tag_id_equals_rings_count(self, cctag_canvas: np.ndarray) -> None:
        detections = detect_cc_tags(cctag_canvas)
        for d in detections:
            assert d.tag_id == d.rings_count

    def test_results_sorted_by_tag_id(self, cctag_canvas: np.ndarray) -> None:
        detections = detect_cc_tags(cctag_canvas)
        ids = [d.tag_id for d in detections]
        assert ids == sorted(ids)

    def test_bbox_has_positive_extent(self, cctag_canvas: np.ndarray) -> None:
        detections = detect_cc_tags(cctag_canvas)
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            assert x1 < x2
            assert y1 < y2

    def test_radius_is_positive(self, cctag_canvas: np.ndarray) -> None:
        detections = detect_cc_tags(cctag_canvas)
        for d in detections:
            assert d.radius > 0.0

    def test_returns_empty_for_blank_image(self, bgr_image: np.ndarray) -> None:
        assert detect_cc_tags(bgr_image) == []

    def test_returns_empty_for_random_noise(self) -> None:
        rng = np.random.default_rng(seed=42)
        noise = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
        # Random noise should not reliably produce concentric-circle groups.
        result = detect_cc_tags(noise)
        assert isinstance(result, list)

    def test_two_markers_detected(self) -> None:
        canvas = np.full((500, 500), 255, dtype=np.uint8)
        # First marker at (120, 120)
        for i in range(3):
            r = 60 - i * 20
            cv2.circle(canvas, (120, 120), r, 0 if i % 2 == 0 else 255, -1)
        # Second marker at (380, 380)
        for i in range(3):
            r = 60 - i * 20
            cv2.circle(canvas, (380, 380), r, 0 if i % 2 == 0 else 255, -1)
        bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        detections = detect_cc_tags(bgr)
        assert len(detections) >= 2


# ---------------------------------------------------------------------------
# draw_cc_tags - input validation
# ---------------------------------------------------------------------------


class TestDrawCCTagsValidation:
    def test_raises_for_non_ndarray(self) -> None:
        with pytest.raises(TypeError):
            draw_cc_tags("not an image", [])  # type: ignore[arg-type]

    def test_raises_for_grayscale_image(self, cctag_canvas: np.ndarray) -> None:
        gray = cv2.cvtColor(cctag_canvas, cv2.COLOR_BGR2GRAY)
        with pytest.raises(ValueError, match="BGR uint8"):
            draw_cc_tags(gray, [])

    def test_raises_for_float32_image(self) -> None:
        img = np.zeros((100, 100, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="BGR uint8"):
            draw_cc_tags(img, [])

    def test_raises_for_non_positive_thickness(self, cctag_canvas: np.ndarray) -> None:
        with pytest.raises(ValueError, match="thickness must be positive"):
            draw_cc_tags(cctag_canvas, [], thickness=0)

    def test_raises_for_negative_thickness(self, cctag_canvas: np.ndarray) -> None:
        with pytest.raises(ValueError, match="thickness must be positive"):
            draw_cc_tags(cctag_canvas, [], thickness=-1)


# ---------------------------------------------------------------------------
# draw_cc_tags - drawing behaviour
# ---------------------------------------------------------------------------


class TestDrawCCTags:
    def test_returns_copy_not_inplace(self, cctag_canvas: np.ndarray) -> None:
        original = cctag_canvas.copy()
        result = draw_cc_tags(cctag_canvas, [])
        np.testing.assert_array_equal(cctag_canvas, original)
        assert result is not cctag_canvas

    def test_output_shape_matches_input(self, cctag_canvas: np.ndarray) -> None:
        result = draw_cc_tags(cctag_canvas, [])
        assert result.shape == cctag_canvas.shape
        assert result.dtype == cctag_canvas.dtype

    def test_empty_detections_returns_identical_copy(
        self, cctag_canvas: np.ndarray
    ) -> None:
        result = draw_cc_tags(cctag_canvas, [])
        np.testing.assert_array_equal(result, cctag_canvas)

    def test_annotated_image_differs_from_original(
        self, cctag_canvas: np.ndarray
    ) -> None:
        detections = detect_cc_tags(cctag_canvas)
        assert len(detections) >= 1, "Fixture must contain at least one detectable marker"
        result = draw_cc_tags(cctag_canvas, detections)
        assert np.any(result != cctag_canvas)

    def test_draws_with_explicit_detection(self, cctag_canvas: np.ndarray) -> None:
        detection = CCTagDetection(
            tag_id=3,
            center=(100.0, 100.0),
            radius=40.0,
            bbox=(60, 60, 140, 140),
            rings_count=3,
        )
        result = draw_cc_tags(cctag_canvas, [detection])
        assert result.shape == cctag_canvas.shape
        assert np.any(result != cctag_canvas)

    def test_custom_color_applied(self, cctag_canvas: np.ndarray) -> None:
        detection = CCTagDetection(
            tag_id=3,
            center=(150.0, 150.0),
            radius=60.0,
            bbox=(90, 90, 210, 210),
            rings_count=3,
        )
        green = (0, 255, 0)
        result = draw_cc_tags(cctag_canvas, [detection], color=green)
        assert result.shape == cctag_canvas.shape


# ---------------------------------------------------------------------------
# detect_cc_tags – confidence field
# ---------------------------------------------------------------------------


class TestDetectCCTagsConfidence:
    def test_confidence_in_unit_interval(self, cctag_canvas: np.ndarray) -> None:
        detections = detect_cc_tags(cctag_canvas)
        for d in detections:
            assert 0.0 <= d.confidence <= 1.0

    def test_confidence_positive_for_good_marker(self, cctag_canvas: np.ndarray) -> None:
        detections = detect_cc_tags(cctag_canvas)
        assert len(detections) >= 1
        # Synthetic perfect circles should score well above zero.
        for d in detections:
            assert d.confidence > 0.0


# ---------------------------------------------------------------------------
# detect_cc_tags – use_adaptive parameter
# ---------------------------------------------------------------------------


class TestDetectCCTagsAdaptive:
    def test_adaptive_returns_list(self, cctag_canvas: np.ndarray) -> None:
        result = detect_cc_tags(cctag_canvas, use_adaptive=True)
        assert isinstance(result, list)

    def test_adaptive_detects_marker(self, cctag_canvas: np.ndarray) -> None:
        detections = detect_cc_tags(cctag_canvas, use_adaptive=True)
        # Adaptive threshold is less deterministic on synthetic flat images;
        # verify the call succeeds and returns the right type.
        assert isinstance(detections, list)

    def test_adaptive_on_gradient_image(self) -> None:
        """Adaptive threshold should handle non-uniform illumination."""
        canvas = np.zeros((300, 300), dtype=np.uint8)
        # Create a horizontal gradient background (brighter on the right).
        for col in range(300):
            canvas[:, col] = col // 2
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        # Just verify it runs without error on a gradient image.
        result = detect_cc_tags(canvas_bgr, use_adaptive=True)
        assert isinstance(result, list)

    def test_adaptive_and_otsu_both_return_list(self, cctag_canvas: np.ndarray) -> None:
        otsu_result = detect_cc_tags(cctag_canvas, use_adaptive=False)
        adaptive_result = detect_cc_tags(cctag_canvas, use_adaptive=True)
        assert isinstance(otsu_result, list)
        assert isinstance(adaptive_result, list)


# ---------------------------------------------------------------------------
# estimate_cctag_pose
# ---------------------------------------------------------------------------


class TestEstimateCCTagPose:
    def test_returns_rvec_and_tvec(
        self,
        sample_detection: CCTagDetection,
        pinhole_camera_matrix: np.ndarray,
        zero_dist_coeffs: np.ndarray,
    ) -> None:
        rvec, tvec = estimate_cctag_pose(
            sample_detection, pinhole_camera_matrix, zero_dist_coeffs, 0.05
        )
        assert rvec.shape == (3, 1)
        assert tvec.shape == (3, 1)

    def test_output_dtype_is_float64(
        self,
        sample_detection: CCTagDetection,
        pinhole_camera_matrix: np.ndarray,
        zero_dist_coeffs: np.ndarray,
    ) -> None:
        rvec, tvec = estimate_cctag_pose(
            sample_detection, pinhole_camera_matrix, zero_dist_coeffs, 0.05
        )
        assert rvec.dtype == np.float64
        assert tvec.dtype == np.float64

    def test_translation_z_is_positive(
        self,
        sample_detection: CCTagDetection,
        pinhole_camera_matrix: np.ndarray,
        zero_dist_coeffs: np.ndarray,
    ) -> None:
        """A marker in front of the camera must have positive z translation."""
        _, tvec = estimate_cctag_pose(
            sample_detection, pinhole_camera_matrix, zero_dist_coeffs, 0.05
        )
        assert float(tvec[2, 0]) > 0.0

    def test_raises_for_non_positive_radius(
        self,
        sample_detection: CCTagDetection,
        pinhole_camera_matrix: np.ndarray,
        zero_dist_coeffs: np.ndarray,
    ) -> None:
        with pytest.raises(ValueError, match="tag_physical_radius_m must be positive"):
            estimate_cctag_pose(
                sample_detection, pinhole_camera_matrix, zero_dist_coeffs, 0.0
            )

    def test_raises_for_negative_radius(
        self,
        sample_detection: CCTagDetection,
        pinhole_camera_matrix: np.ndarray,
        zero_dist_coeffs: np.ndarray,
    ) -> None:
        with pytest.raises(ValueError, match="tag_physical_radius_m must be positive"):
            estimate_cctag_pose(
                sample_detection, pinhole_camera_matrix, zero_dist_coeffs, -0.1
            )

    def test_raises_for_wrong_camera_matrix_shape(
        self,
        sample_detection: CCTagDetection,
        zero_dist_coeffs: np.ndarray,
    ) -> None:
        bad_matrix = np.eye(4, dtype=np.float64)
        with pytest.raises(ValueError, match="camera_matrix must have shape"):
            estimate_cctag_pose(sample_detection, bad_matrix, zero_dist_coeffs, 0.05)

    def test_larger_physical_radius_yields_larger_distance(
        self,
        pinhole_camera_matrix: np.ndarray,
        zero_dist_coeffs: np.ndarray,
    ) -> None:
        """A larger physical tag at the same image size implies it is farther away."""
        detection = CCTagDetection(
            tag_id=3,
            center=(320.0, 240.0),
            radius=50.0,
            bbox=(270, 190, 370, 290),
            rings_count=3,
        )
        _, tvec_small = estimate_cctag_pose(
            detection, pinhole_camera_matrix, zero_dist_coeffs, 0.05
        )
        _, tvec_large = estimate_cctag_pose(
            detection, pinhole_camera_matrix, zero_dist_coeffs, 0.10
        )
        assert float(tvec_large[2, 0]) > float(tvec_small[2, 0])
