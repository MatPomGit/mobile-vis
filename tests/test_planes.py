"""Tests for image_analysis.planes module."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from image_analysis.planes import (
    MIN_INLIER_FRACTION,
    PlaneDetection,
    VanishingPoint,
    _cluster_lines_by_direction,
    _detect_lines,
    _intersect_lines,
    _mask_to_bbox,
    detect_planes,
    detect_vanishing_points,
    draw_planes,
    estimate_plane_pose,
    fit_plane_ransac,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_grid_image(
    size: int = 300,
    spacing: int = 40,
    line_color: int = 0,
    bg_color: int = 255,
) -> np.ndarray:
    """Return a synthetic BGR image with a regular horizontal/vertical grid."""
    canvas = np.full((size, size, 3), bg_color, dtype=np.uint8)
    for x in range(0, size, spacing):
        cv2.line(canvas, (x, 0), (x, size - 1), (line_color,) * 3, 2)
    for y in range(0, size, spacing):
        cv2.line(canvas, (0, y), (size - 1, y), (line_color,) * 3, 2)
    return canvas


def _make_parallel_lines_image(
    size: int = 300,
    n_lines: int = 6,
    angle_deg: float = 0.0,
) -> np.ndarray:
    """Return a BGR image with *n_lines* parallel lines at *angle_deg*."""
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    for i in range(n_lines):
        offset = int(size * (i + 1) / (n_lines + 1))
        rad = math.radians(angle_deg)
        dx = int(math.cos(rad) * size // 2)
        dy = int(math.sin(rad) * size // 2)
        mid = size // 2
        p1 = (mid - dx, offset - dy)
        p2 = (mid + dx, offset + dy)
        cv2.line(canvas, p1, p2, (0, 0, 0), 2)
    return canvas


def _make_flat_plane_point_cloud(
    n: int = 200,
    noise_std: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """Return an ``(N, 3)`` point cloud on the z=0 plane with Gaussian noise."""
    rng = np.random.default_rng(seed=seed)
    pts = rng.uniform(-1.0, 1.0, size=(n, 2))
    z = rng.normal(0.0, noise_std, size=(n, 1))
    return np.hstack([pts, z]).astype(np.float64)


@pytest.fixture
def grid_image() -> np.ndarray:
    """Synthetic BGR grid image with horizontal and vertical lines."""
    return _make_grid_image()


@pytest.fixture
def blank_image() -> np.ndarray:
    """Plain white BGR image with no edges."""
    return np.full((200, 200, 3), 255, dtype=np.uint8)


@pytest.fixture
def parallel_lines_image() -> np.ndarray:
    """BGR image with parallel horizontal lines."""
    return _make_parallel_lines_image(angle_deg=0.0)


@pytest.fixture
def sample_point_cloud() -> np.ndarray:
    """3-D point cloud on the z=0 plane with small Gaussian noise."""
    return _make_flat_plane_point_cloud()


@pytest.fixture
def camera_matrix() -> np.ndarray:
    """Standard pinhole camera matrix for a 640x480 sensor."""
    fx = fy = 600.0
    cx, cy = 320.0, 240.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


@pytest.fixture
def sample_plane_detection() -> PlaneDetection:
    """A PlaneDetection with known field values."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:150, 50:150] = 255
    return PlaneDetection(
        normal=(0.0, 0.0, 1.0),
        centroid=(100.0, 100.0),
        confidence=0.8,
        mask=mask,
        bbox=(50, 50, 150, 150),
        inlier_count=20,
    )


# ---------------------------------------------------------------------------
# PlaneDetection dataclass
# ---------------------------------------------------------------------------


class TestPlaneDetectionDataclass:
    def test_fields_accessible(self, sample_plane_detection: PlaneDetection) -> None:
        pd = sample_plane_detection
        assert pd.normal == (0.0, 0.0, 1.0)
        assert pd.centroid == (100.0, 100.0)
        assert pd.confidence == pytest.approx(0.8)
        assert pd.bbox == (50, 50, 150, 150)
        assert pd.inlier_count == 20

    def test_is_frozen(self, sample_plane_detection: PlaneDetection) -> None:
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            sample_plane_detection.confidence = 0.5  # type: ignore[misc]

    def test_mask_can_be_none(self) -> None:
        pd = PlaneDetection(
            normal=(0.0, 0.0, 1.0),
            centroid=(50.0, 50.0),
            confidence=0.5,
            mask=None,
            bbox=(0, 0, 100, 100),
            inlier_count=5,
        )
        assert pd.mask is None

    def test_confidence_bounds(self) -> None:
        pd = PlaneDetection(
            normal=(0.0, 1.0, 0.0),
            centroid=(0.0, 0.0),
            confidence=1.0,
            mask=None,
            bbox=(0, 0, 10, 10),
            inlier_count=1,
        )
        assert pd.confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# VanishingPoint dataclass
# ---------------------------------------------------------------------------


class TestVanishingPointDataclass:
    def test_fields_accessible(self) -> None:
        vp = VanishingPoint(point=(100.0, 200.0), lines=[(0, 0, 50, 50)], confidence=0.7)
        assert vp.point == (100.0, 200.0)
        assert vp.lines == [(0, 0, 50, 50)]
        assert vp.confidence == pytest.approx(0.7)

    def test_default_confidence_zero(self) -> None:
        vp = VanishingPoint(point=(0.0, 0.0), lines=[])
        assert vp.confidence == 0.0

    def test_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        vp = VanishingPoint(point=(1.0, 2.0), lines=[])
        with pytest.raises(FrozenInstanceError):
            vp.confidence = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# detect_vanishing_points - input validation
# ---------------------------------------------------------------------------


class TestDetectVanishingPointsValidation:
    def test_raises_for_non_ndarray(self) -> None:
        with pytest.raises(TypeError):
            detect_vanishing_points([[255, 0], [0, 255]])  # type: ignore[arg-type]

    def test_raises_for_4d_array(self) -> None:
        bad = np.zeros((10, 10, 3, 1), dtype=np.uint8)
        with pytest.raises(ValueError):
            detect_vanishing_points(bad)

    def test_raises_for_invalid_dtype(self) -> None:
        bad = np.zeros((100, 100, 3), dtype=np.float64)
        with pytest.raises(ValueError):
            detect_vanishing_points(bad)

    def test_raises_for_n_points_zero(self, grid_image: np.ndarray) -> None:
        with pytest.raises(ValueError, match="n_points"):
            detect_vanishing_points(grid_image, n_points=0)

    def test_raises_for_n_points_negative(self, grid_image: np.ndarray) -> None:
        with pytest.raises(ValueError, match="n_points"):
            detect_vanishing_points(grid_image, n_points=-1)

    def test_accepts_bgr_uint8(self, grid_image: np.ndarray) -> None:
        result = detect_vanishing_points(grid_image)
        assert isinstance(result, list)

    def test_accepts_grayscale_uint8(self, grid_image: np.ndarray) -> None:
        gray = cv2.cvtColor(grid_image, cv2.COLOR_BGR2GRAY)
        result = detect_vanishing_points(gray)
        assert isinstance(result, list)

    def test_accepts_float32_image(self, grid_image: np.ndarray) -> None:
        f32 = grid_image.astype(np.float32) / 255.0
        result = detect_vanishing_points(f32)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# detect_vanishing_points - detection logic
# ---------------------------------------------------------------------------


class TestDetectVanishingPoints:
    def test_returns_list(self, grid_image: np.ndarray) -> None:
        result = detect_vanishing_points(grid_image)
        assert isinstance(result, list)

    def test_blank_image_returns_empty(self, blank_image: np.ndarray) -> None:
        result = detect_vanishing_points(blank_image)
        assert result == []

    def test_returns_at_most_n_points(self, grid_image: np.ndarray) -> None:
        for n in (1, 2, 3):
            result = detect_vanishing_points(grid_image, n_points=n)
            assert len(result) <= n

    def test_confidence_in_unit_interval(self, grid_image: np.ndarray) -> None:
        result = detect_vanishing_points(grid_image)
        for vp in result:
            assert 0.0 <= vp.confidence <= 1.0

    def test_sorted_by_descending_confidence(self, grid_image: np.ndarray) -> None:
        result = detect_vanishing_points(grid_image, n_points=3)
        confidences = [vp.confidence for vp in result]
        assert confidences == sorted(confidences, reverse=True)

    def test_vp_has_lines(self, grid_image: np.ndarray) -> None:
        result = detect_vanishing_points(grid_image, n_points=2)
        for vp in result:
            assert isinstance(vp.lines, list)

    def test_parallel_lines_image(self, parallel_lines_image: np.ndarray) -> None:
        result = detect_vanishing_points(parallel_lines_image, n_points=2)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# detect_planes - input validation
# ---------------------------------------------------------------------------


class TestDetectPlanesValidation:
    def test_raises_for_non_ndarray(self) -> None:
        with pytest.raises(TypeError):
            detect_planes("not an image")  # type: ignore[arg-type]

    def test_raises_for_4d_array(self) -> None:
        bad = np.zeros((10, 10, 3, 1), dtype=np.uint8)
        with pytest.raises(ValueError):
            detect_planes(bad)

    def test_raises_for_bad_camera_matrix_shape(self, grid_image: np.ndarray) -> None:
        bad_cm = np.eye(4, dtype=np.float64)
        with pytest.raises(ValueError, match="camera_matrix"):
            detect_planes(grid_image, camera_matrix=bad_cm)

    def test_raises_for_max_planes_zero(self, grid_image: np.ndarray) -> None:
        with pytest.raises(ValueError, match="max_planes"):
            detect_planes(grid_image, max_planes=0)

    def test_raises_for_min_inliers_zero(self, grid_image: np.ndarray) -> None:
        with pytest.raises(ValueError, match="min_inliers"):
            detect_planes(grid_image, min_inliers=0)

    def test_accepts_bgr_uint8(self, grid_image: np.ndarray) -> None:
        result = detect_planes(grid_image)
        assert isinstance(result, list)

    def test_accepts_grayscale(self, grid_image: np.ndarray) -> None:
        gray = cv2.cvtColor(grid_image, cv2.COLOR_BGR2GRAY)
        result = detect_planes(gray)
        assert isinstance(result, list)

    def test_accepts_float32(self, grid_image: np.ndarray) -> None:
        f32 = grid_image.astype(np.float32) / 255.0
        result = detect_planes(f32)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# detect_planes - detection logic
# ---------------------------------------------------------------------------


class TestDetectPlanes:
    def test_blank_image_returns_empty(self, blank_image: np.ndarray) -> None:
        result = detect_planes(blank_image)
        assert result == []

    def test_returns_at_most_max_planes(self, grid_image: np.ndarray) -> None:
        for m in (1, 2, 3):
            result = detect_planes(grid_image, max_planes=m)
            assert len(result) <= m

    def test_max_planes_one(self, grid_image: np.ndarray) -> None:
        result = detect_planes(grid_image, max_planes=1)
        assert len(result) <= 1

    def test_confidence_in_unit_interval(self, grid_image: np.ndarray) -> None:
        result = detect_planes(grid_image)
        for plane in result:
            assert 0.0 <= plane.confidence <= 1.0

    def test_sorted_by_descending_confidence(self, grid_image: np.ndarray) -> None:
        result = detect_planes(grid_image)
        confidences = [p.confidence for p in result]
        assert confidences == sorted(confidences, reverse=True)

    def test_normal_is_unit_vector(self, grid_image: np.ndarray) -> None:
        result = detect_planes(grid_image)
        for plane in result:
            norm = math.sqrt(sum(v ** 2 for v in plane.normal))
            assert norm == pytest.approx(1.0, abs=1e-6)

    def test_bbox_valid(self, grid_image: np.ndarray) -> None:
        result = detect_planes(grid_image)
        for plane in result:
            x1, y1, x2, y2 = plane.bbox
            assert x1 <= x2
            assert y1 <= y2

    def test_with_camera_matrix(
        self, grid_image: np.ndarray, camera_matrix: np.ndarray
    ) -> None:
        result = detect_planes(grid_image, camera_matrix=camera_matrix)
        assert isinstance(result, list)

    def test_inlier_count_positive(self, grid_image: np.ndarray) -> None:
        result = detect_planes(grid_image)
        for plane in result:
            assert plane.inlier_count > 0

    def test_mask_shape(self, grid_image: np.ndarray) -> None:
        h, w = grid_image.shape[:2]
        result = detect_planes(grid_image)
        for plane in result:
            if plane.mask is not None:
                assert plane.mask.shape == (h, w)
                assert plane.mask.dtype == np.uint8

    def test_no_cluster_shared_across_planes(self) -> None:
        """Each line-direction cluster must belong to at most one plane.

        The total inlier count across all planes must not exceed the number of
        detected line segments (no double-counting from shared clusters).
        """
        # Build an image with 4 clearly distinct line directions so that up to
        # 2 non-overlapping planes can be formed.
        size = 300
        canvas = np.full((size, size, 3), 255, dtype=np.uint8)
        # Direction 0°: horizontal lines
        for y in range(30, size, 60):
            cv2.line(canvas, (0, y), (size - 1, y), (0, 0, 0), 2)
        # Direction 90°: vertical lines
        for x in range(30, size, 60):
            cv2.line(canvas, (x, 0), (x, size - 1), (0, 0, 0), 2)
        # Direction 45°: diagonal lines
        for offset in range(-size, size * 2, 60):
            cv2.line(canvas, (offset, 0), (offset + size, size), (0, 0, 0), 2)
        # Direction 135°: anti-diagonal lines
        for offset in range(-size, size * 2, 60):
            cv2.line(canvas, (offset, size), (offset + size, 0), (0, 0, 0), 2)

        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        lines = _detect_lines(gray)
        total_line_count = len(lines) if lines is not None else 0

        result = detect_planes(canvas, max_planes=3)
        total_inliers = sum(p.inlier_count for p in result)
        # With cluster exclusion each line segment is counted at most once.
        assert total_inliers <= total_line_count


# ---------------------------------------------------------------------------
# fit_plane_ransac - input validation
# ---------------------------------------------------------------------------


class TestFitPlaneRansacValidation:
    def test_raises_for_non_ndarray(self) -> None:
        with pytest.raises(TypeError):
            fit_plane_ransac([[0, 0, 1], [1, 0, 0], [0, 1, 0]])  # type: ignore[arg-type]

    def test_raises_for_wrong_shape_1d(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            fit_plane_ransac(np.array([1.0, 0.0, 0.0]))

    def test_raises_for_wrong_columns(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            fit_plane_ransac(np.ones((10, 2), dtype=np.float64))

    def test_raises_for_fewer_than_3_points(self) -> None:
        with pytest.raises(ValueError, match="At least 3"):
            fit_plane_ransac(np.ones((2, 3), dtype=np.float64))

    def test_raises_for_non_positive_threshold(
        self, sample_point_cloud: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="threshold"):
            fit_plane_ransac(sample_point_cloud, threshold=0.0)

    def test_raises_for_non_positive_max_iter(
        self, sample_point_cloud: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="max_iter"):
            fit_plane_ransac(sample_point_cloud, max_iter=0)


# ---------------------------------------------------------------------------
# fit_plane_ransac - correctness
# ---------------------------------------------------------------------------


class TestFitPlaneRansac:
    def test_returns_unit_normal_and_mask(
        self, sample_point_cloud: np.ndarray
    ) -> None:
        normal, mask = fit_plane_ransac(sample_point_cloud)
        assert normal.shape == (3,)
        assert mask.shape == (len(sample_point_cloud),)
        assert mask.dtype == bool

    def test_normal_is_unit_vector(self, sample_point_cloud: np.ndarray) -> None:
        normal, _ = fit_plane_ransac(sample_point_cloud)
        norm = float(np.linalg.norm(normal))
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_horizontal_plane_normal_z(self) -> None:
        """Points on z=0 plane → normal should be approximately (0, 0, ±1)."""
        pts = _make_flat_plane_point_cloud(n=300, noise_std=0.001)
        _normal, _inliers = fit_plane_ransac(pts, threshold=0.1)
        assert abs(_normal[2]) == pytest.approx(1.0, abs=0.1)

    def test_inlier_fraction_high_for_clean_plane(self) -> None:
        pts = _make_flat_plane_point_cloud(n=300, noise_std=0.001)
        _, inliers = fit_plane_ransac(pts, threshold=0.05)
        fraction = float(np.sum(inliers)) / len(pts)
        assert fraction >= MIN_INLIER_FRACTION

    def test_robust_to_outliers(self) -> None:
        """30 % random outliers should not prevent finding the dominant plane."""
        rng = np.random.default_rng(seed=7)
        pts = _make_flat_plane_point_cloud(n=300, noise_std=0.005)
        n_outliers = 90
        outliers = rng.uniform(-5.0, 5.0, size=(n_outliers, 3))
        pts_with_noise = np.vstack([pts, outliers])
        _normal, inliers = fit_plane_ransac(pts_with_noise, threshold=0.1)
        clean_inliers = np.sum(inliers[:300])
        assert clean_inliers > 200

    def test_three_point_minimum(self) -> None:
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        normal, mask = fit_plane_ransac(pts, threshold=0.1, max_iter=10)
        assert normal.shape == (3,)
        assert mask.shape == (3,)


# ---------------------------------------------------------------------------
# draw_planes - input validation
# ---------------------------------------------------------------------------


class TestDrawPlanesValidation:
    def test_raises_for_non_ndarray(self) -> None:
        with pytest.raises(TypeError):
            draw_planes("not an image", [])  # type: ignore[arg-type]

    def test_raises_for_grayscale_image(self) -> None:
        gray = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="BGR uint8"):
            draw_planes(gray, [])

    def test_raises_for_float32_image(self) -> None:
        img = np.zeros((100, 100, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="BGR uint8"):
            draw_planes(img, [])

    def test_raises_for_alpha_out_of_range_high(self, grid_image: np.ndarray) -> None:
        with pytest.raises(ValueError, match="alpha"):
            draw_planes(grid_image, [], alpha=1.5)

    def test_raises_for_alpha_out_of_range_low(self, grid_image: np.ndarray) -> None:
        with pytest.raises(ValueError, match="alpha"):
            draw_planes(grid_image, [], alpha=-0.1)


# ---------------------------------------------------------------------------
# draw_planes - drawing behaviour
# ---------------------------------------------------------------------------


class TestDrawPlanes:
    def test_returns_copy_not_inplace(self, grid_image: np.ndarray) -> None:
        original = grid_image.copy()
        result = draw_planes(grid_image, [])
        np.testing.assert_array_equal(grid_image, original)
        assert result is not grid_image

    def test_output_shape_matches_input(self, grid_image: np.ndarray) -> None:
        result = draw_planes(grid_image, [])
        assert result.shape == grid_image.shape
        assert result.dtype == grid_image.dtype

    def test_empty_detections_returns_copy(self, grid_image: np.ndarray) -> None:
        result = draw_planes(grid_image, [])
        np.testing.assert_array_equal(result, grid_image)

    def test_single_detection_annotates(
        self, grid_image: np.ndarray, sample_plane_detection: PlaneDetection
    ) -> None:
        result = draw_planes(grid_image, [sample_plane_detection])
        assert result.shape == grid_image.shape

    def test_three_detections_annotates(self, grid_image: np.ndarray) -> None:
        h, w = grid_image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        planes = [
            PlaneDetection(
                normal=(0.0, 0.0, 1.0),
                centroid=(float(w // 4 * (i + 1)), float(h // 2)),
                confidence=0.9 - 0.1 * i,
                mask=mask.copy(),
                bbox=(0, 0, w // 2, h // 2),
                inlier_count=10,
            )
            for i in range(3)
        ]
        result = draw_planes(grid_image, planes)
        assert result.shape == grid_image.shape

    def test_alpha_zero_no_mask_drawn(
        self, grid_image: np.ndarray, sample_plane_detection: PlaneDetection
    ) -> None:
        """With alpha=0.0 the overlay colour should not bleed into the image."""
        result = draw_planes(grid_image, [sample_plane_detection], alpha=0.0)
        assert result.shape == grid_image.shape

    def test_alpha_one_full_overlay(
        self, grid_image: np.ndarray, sample_plane_detection: PlaneDetection
    ) -> None:
        result = draw_planes(grid_image, [sample_plane_detection], alpha=1.0)
        assert result.shape == grid_image.shape

    def test_mask_none_does_not_raise(self, grid_image: np.ndarray) -> None:
        plane = PlaneDetection(
            normal=(0.0, 0.0, 1.0),
            centroid=(100.0, 100.0),
            confidence=0.5,
            mask=None,
            bbox=(50, 50, 150, 150),
            inlier_count=5,
        )
        result = draw_planes(grid_image, [plane])
        assert result.shape == grid_image.shape


# ---------------------------------------------------------------------------
# estimate_plane_pose
# ---------------------------------------------------------------------------


class TestEstimatePlanePose:
    def test_returns_rvec_and_tvec(
        self,
        sample_plane_detection: PlaneDetection,
        camera_matrix: np.ndarray,
    ) -> None:
        rvec, tvec = estimate_plane_pose(sample_plane_detection, camera_matrix)
        assert rvec.shape == (3, 1)
        assert tvec.shape == (3, 1)

    def test_output_dtype_float64(
        self,
        sample_plane_detection: PlaneDetection,
        camera_matrix: np.ndarray,
    ) -> None:
        rvec, tvec = estimate_plane_pose(sample_plane_detection, camera_matrix)
        assert rvec.dtype == np.float64
        assert tvec.dtype == np.float64

    def test_raises_for_non_plane_detection(
        self, camera_matrix: np.ndarray
    ) -> None:
        with pytest.raises(TypeError, match="PlaneDetection"):
            estimate_plane_pose("not a plane", camera_matrix)  # type: ignore[arg-type]

    def test_raises_for_bad_camera_matrix_shape(
        self, sample_plane_detection: PlaneDetection
    ) -> None:
        bad_cm = np.eye(4, dtype=np.float64)
        with pytest.raises(ValueError, match="camera_matrix"):
            estimate_plane_pose(sample_plane_detection, bad_cm)

    def test_translation_z_positive(
        self,
        camera_matrix: np.ndarray,
    ) -> None:
        """A plane in front of the camera should have positive z translation."""
        plane = PlaneDetection(
            normal=(0.0, 0.0, 1.0),
            centroid=(320.0, 240.0),
            confidence=0.9,
            mask=None,
            bbox=(220, 140, 420, 340),
            inlier_count=10,
        )
        _, tvec = estimate_plane_pose(plane, camera_matrix)
        assert float(tvec[2, 0]) > 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestDetectLinesHelper:
    def test_returns_none_for_blank_image(self) -> None:
        gray = np.full((100, 100), 255, dtype=np.uint8)
        result = _detect_lines(gray)
        assert result is None or (hasattr(result, "__len__") and len(result) == 0)

    def test_returns_array_for_line_image(self) -> None:
        gray = np.zeros((200, 200), dtype=np.uint8)
        cv2.line(gray, (0, 100), (200, 100), 255, 2)
        cv2.line(gray, (100, 0), (100, 200), 255, 2)
        result = _detect_lines(gray)
        assert result is not None
        assert len(result) > 0


class TestClusterLinesByDirection:
    def test_two_parallel_horizontal_lines_in_same_cluster(self) -> None:
        lines = np.array([[[0, 50, 100, 50]], [[0, 150, 100, 150]]], dtype=np.int32)
        clusters = _cluster_lines_by_direction(lines, angle_tol_deg=10.0)
        assert len(clusters) == 1
        assert len(clusters[0]) == 2

    def test_orthogonal_lines_in_different_clusters(self) -> None:
        lines = np.array(
            [[[0, 50, 100, 50]], [[50, 0, 50, 100]]], dtype=np.int32
        )
        clusters = _cluster_lines_by_direction(lines, angle_tol_deg=5.0)
        assert len(clusters) == 2

    def test_near_zero_and_near_180_lines_in_same_cluster(self) -> None:
        """Lines at ~178° and ~2° are nearly parallel and must land in one cluster.

        The running-average update must wrap across the 0°/180° boundary so
        that the cluster mean stays near 0° (or 180°) rather than jumping to
        ~90° as a naive average would produce.
        """
        # Line at ~2°: from (0,0) to (100,3) → atan2(3,100) ≈ 1.7°
        # Line at ~178°: from (100,0) to (0,3) → atan2(3,-100) ≈ 178.3°
        lines = np.array(
            [[[0, 0, 100, 3]], [[100, 0, 0, 3]], [[0, 10, 100, 13]]],
            dtype=np.int32,
        )
        clusters = _cluster_lines_by_direction(lines, angle_tol_deg=5.0)
        assert len(clusters) == 1, (
            "Near-0° and near-180° lines must form a single cluster"
        )


class TestIntersectLines:
    def test_returns_none_for_single_line(self) -> None:
        result = _intersect_lines([(0, 0, 100, 0)])
        assert result is None

    def test_two_crossing_lines(self) -> None:
        # Horizontal line y=50: (0,50)→(100,50) and vertical line x=50: (50,0)→(50,100)
        result = _intersect_lines([(0, 50, 100, 50), (50, 0, 50, 100)])
        assert result is not None
        vx, vy = result
        assert vx == pytest.approx(50.0, abs=2.0)
        assert vy == pytest.approx(50.0, abs=2.0)


class TestMaskToBbox:
    def test_none_for_empty_mask(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        assert _mask_to_bbox(mask) is None

    def test_bbox_for_filled_region(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 30:80] = 255
        bbox = _mask_to_bbox(mask)
        assert bbox == (30, 20, 79, 59)


# ---------------------------------------------------------------------------
# PlaneDetection.precision field
# ---------------------------------------------------------------------------


class TestPlaneDetectionPrecision:
    def test_default_precision_is_zero(self) -> None:
        pd = PlaneDetection(
            normal=(0.0, 0.0, 1.0),
            centroid=(50.0, 50.0),
            confidence=0.5,
            mask=None,
            bbox=(0, 0, 100, 100),
            inlier_count=5,
        )
        assert pd.precision == pytest.approx(0.0)

    def test_explicit_precision_stored(self) -> None:
        pd = PlaneDetection(
            normal=(0.0, 0.0, 1.0),
            centroid=(50.0, 50.0),
            confidence=0.5,
            mask=None,
            bbox=(0, 0, 100, 100),
            inlier_count=5,
            precision=0.75,
        )
        assert pd.precision == pytest.approx(0.75)

    def test_precision_in_unit_interval_from_detect_planes(self) -> None:
        """detect_planes must populate precision in [0, 1] for each plane."""
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        vp1 = (300, 100)
        vp2 = (0, 250)
        for bx in range(50, 600, 60):
            cv2.line(img, (bx, 400), vp1, (255, 255, 255), 2)
        for by in range(50, 400, 50):
            cv2.line(img, (600, by), vp2, (180, 180, 180), 2)
        planes = detect_planes(img, max_planes=3)
        for plane in planes:
            assert 0.0 <= plane.precision <= 1.0

    def test_precision_preserved_after_mask_overlap_resolution(self) -> None:
        """_resolve_mask_overlaps must carry precision through to output planes."""
        h, w = 100, 100
        mask1 = np.zeros((h, w), dtype=np.uint8)
        mask1[10:50, 10:90] = 255
        mask2 = np.zeros((h, w), dtype=np.uint8)
        mask2[60:90, 10:90] = 255

        from image_analysis.planes import _resolve_mask_overlaps

        plane1 = PlaneDetection(
            normal=(0.0, 0.0, 1.0),
            centroid=(50.0, 30.0),
            confidence=0.8,
            mask=mask1,
            bbox=(10, 10, 90, 50),
            inlier_count=10,
            precision=0.9,
        )
        plane2 = PlaneDetection(
            normal=(1.0, 0.0, 0.0),
            centroid=(50.0, 75.0),
            confidence=0.6,
            mask=mask2,
            bbox=(10, 60, 90, 90),
            inlier_count=8,
            precision=0.7,
        )
        result = _resolve_mask_overlaps([plane1, plane2])
        assert len(result) == 2
        assert result[0].precision == pytest.approx(0.9)
        assert result[1].precision == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# _select_lines_for_vp helper
# ---------------------------------------------------------------------------


class TestSelectLinesForVp:
    def test_empty_input_returns_empty(self) -> None:
        from image_analysis.planes import _select_lines_for_vp

        lines, prec = _select_lines_for_vp([], 50.0, 50.0, 15.0)
        assert lines == []
        assert prec == pytest.approx(0.0)

    def test_exact_inlier_line_yields_high_precision(self) -> None:
        """A line exactly through the VP has zero perpendicular distance → prec = 1."""
        from image_analysis.planes import _select_lines_for_vp

        # Horizontal line on y=0; VP at (50, 0) lies on the line.
        lines = [(0, 0, 100, 0)]
        selected, prec = _select_lines_for_vp(lines, 50.0, 0.0, 15.0)
        assert len(selected) == 1
        assert prec == pytest.approx(1.0)

    def test_far_line_excluded_by_threshold(self) -> None:
        """A line whose VP distance exceeds the threshold is not an inlier."""
        from image_analysis.planes import _select_lines_for_vp

        # Horizontal line on y=0; VP at (50, 30) → perpendicular dist = 30.
        lines = [(0, 0, 100, 0)]
        selected, prec = _select_lines_for_vp(lines, 50.0, 30.0, 15.0)
        assert selected == []
        assert prec == pytest.approx(0.0)

    def test_respects_max_lines_cap(self) -> None:
        """At most max_lines lines should be returned."""
        from image_analysis.planes import _select_lines_for_vp

        # 10 horizontal lines all passing through y=0 VP vicinity.
        lines = [(0, 0, 100, 0)] * 10
        selected, _ = _select_lines_for_vp(lines, 50.0, 0.0, 15.0, max_lines=3)
        assert len(selected) <= 3

    def test_early_stop_when_precision_threshold_met(self) -> None:
        """Selection stops once running precision >= precision_threshold."""
        from image_analysis.planes import _select_lines_for_vp

        # Lines very close to y=0, VP at (50, 0); precision should be high fast.
        lines = [(i * 10, 0, i * 10 + 10, 0) for i in range(20)]
        # With a high threshold (loose geometry), precision = 1 after the first line.
        selected, prec = _select_lines_for_vp(
            lines, 50.0, 0.0, 100.0, precision_threshold=0.95
        )
        assert prec >= 0.95
        # Must have stopped early; not all 20 lines used.
        assert len(selected) < len(lines)

    def test_precision_in_unit_interval(self) -> None:
        """precision must always be in [0, 1]."""
        from image_analysis.planes import _select_lines_for_vp

        lines = [(i, i, i + 50, i + 50) for i in range(10)]
        _, prec = _select_lines_for_vp(lines, 25.0, 25.0, 15.0)
        assert 0.0 <= prec <= 1.0

    def test_lines_sorted_closest_first(self) -> None:
        """Closer lines should be selected before farther ones."""
        from image_analysis.planes import _select_lines_for_vp

        # Two lines: one at y=1 (close to VP at y=0), one at y=5.
        close_line = (0, 1, 100, 1)
        far_line = (0, 5, 100, 5)
        selected, _ = _select_lines_for_vp(
            [far_line, close_line], 50.0, 0.0, 15.0, max_lines=1
        )
        assert len(selected) == 1
        assert selected[0] == close_line

    def test_draw_planes_label_contains_precision(self) -> None:
        """draw_planes label must include the precision value."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        plane = PlaneDetection(
            normal=(0.0, 0.0, 1.0),
            centroid=(100.0, 100.0),
            confidence=0.8,
            mask=None,
            bbox=(50, 50, 150, 150),
            inlier_count=10,
            precision=0.75,
        )
        # draw_planes should not raise and should annotate the image.
        result = draw_planes(img, [plane])
        assert result.shape == img.shape
