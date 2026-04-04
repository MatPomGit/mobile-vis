"""Tests for image_analysis.rtmdet module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from image_analysis.rtmdet import (
    DEFAULT_RTMDET_MODEL,
    RTMDET_CONFIDENCE_THRESHOLD,
    RTMDET_DOWNLOAD_MAX_RETRIES,
    RTMDET_DOWNLOAD_RETRY_DELAY_SECONDS,
    RTMDET_NMS_IOU_THRESHOLD,
    RtmDetDetection,
    RtmDetDetector,
    _class_color,
    _is_dark_color,
    _load_inferencer_with_retry,
    detect_rtmdet,
    draw_rtmdet_detections,
    export_rtmdet_to_onnx,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bgr_image() -> np.ndarray:
    """Return a synthetic 100x100 BGR uint8 image."""
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_detections() -> list[RtmDetDetection]:
    """Return two axis-aligned sample RTMDet detections."""
    return [
        RtmDetDetection(label="person", class_id=0, confidence=0.90, bbox=(10, 10, 60, 80)),
        RtmDetDetection(label="car", class_id=2, confidence=0.75, bbox=(40, 20, 90, 70)),
    ]


@pytest.fixture
def sample_rotated_detections() -> list[RtmDetDetection]:
    """Return one rotated bounding-box RTMDet detection."""
    return [
        RtmDetDetection(
            label="airplane",
            class_id=4,
            confidence=0.85,
            bbox=(20, 20, 80, 60),
            angle_deg=30.0,
        ),
    ]


@pytest.fixture
def mock_mmdet() -> MagicMock:
    """Patch mmdet.apis.DetInferencer and return the mock class."""
    with patch("image_analysis.rtmdet.DetInferencer", create=True) as mock_cls:
        yield mock_cls


# ---------------------------------------------------------------------------
# RtmDetDetection dataclass
# ---------------------------------------------------------------------------


class TestRtmDetDetection:
    def test_fields_stored_correctly(self) -> None:
        det = RtmDetDetection(label="dog", class_id=16, confidence=0.85, bbox=(5, 5, 50, 50))
        assert det.label == "dog"
        assert det.class_id == 16
        assert det.confidence == 0.85
        assert det.bbox == (5, 5, 50, 50)
        assert det.angle_deg is None

    def test_rotated_detection_stores_angle(self) -> None:
        det = RtmDetDetection(
            label="plane",
            class_id=4,
            confidence=0.7,
            bbox=(0, 0, 100, 50),
            angle_deg=45.0,
        )
        assert det.angle_deg == 45.0

    def test_frozen_raises_on_mutation(self) -> None:
        det = RtmDetDetection(label="cat", class_id=15, confidence=0.7, bbox=(0, 0, 10, 10))
        with pytest.raises(AttributeError):
            det.label = "dog"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = RtmDetDetection(label="car", class_id=2, confidence=0.9, bbox=(0, 0, 100, 100))
        b = RtmDetDetection(label="car", class_id=2, confidence=0.9, bbox=(0, 0, 100, 100))
        assert a == b

    def test_inequality_different_confidence(self) -> None:
        a = RtmDetDetection(label="car", class_id=2, confidence=0.9, bbox=(0, 0, 100, 100))
        b = RtmDetDetection(label="car", class_id=2, confidence=0.5, bbox=(0, 0, 100, 100))
        assert a != b

    def test_default_angle_is_none(self) -> None:
        det = RtmDetDetection(label="x", class_id=0, confidence=0.5, bbox=(0, 0, 10, 10))
        assert det.angle_deg is None


# ---------------------------------------------------------------------------
# detect_rtmdet - validation
# ---------------------------------------------------------------------------


class TestDetectRtmDetValidation:
    def test_raises_type_error_for_list(self) -> None:
        fake_inferencer = MagicMock()
        with pytest.raises(TypeError):
            detect_rtmdet([[1, 2, 3]], fake_inferencer)  # type: ignore[arg-type]

    def test_raises_value_error_for_grayscale(self) -> None:
        gray = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError):
            detect_rtmdet(gray, MagicMock())  # type: ignore[arg-type]

    def test_raises_value_error_for_float_image(self) -> None:
        img = np.zeros((100, 100, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            detect_rtmdet(img, MagicMock())  # type: ignore[arg-type]

    def test_raises_value_error_for_invalid_confidence(self, bgr_image: np.ndarray) -> None:
        with pytest.raises(ValueError):
            detect_rtmdet(bgr_image, MagicMock(), confidence_threshold=1.5)

    def test_raises_value_error_for_negative_confidence(self, bgr_image: np.ndarray) -> None:
        with pytest.raises(ValueError):
            detect_rtmdet(bgr_image, MagicMock(), confidence_threshold=-0.1)

    def test_raises_value_error_for_invalid_iou(self, bgr_image: np.ndarray) -> None:
        with pytest.raises(ValueError):
            detect_rtmdet(bgr_image, MagicMock(), iou_threshold=2.0)


# ---------------------------------------------------------------------------
# detect_rtmdet - result parsing
# ---------------------------------------------------------------------------


class TestDetectRtmDetResults:
    def _make_inferencer(
        self,
        scores: list[float],
        labels: list[int],
        bboxes: list[list[float]],
        class_names: list[str] | None = None,
    ) -> MagicMock:
        """Build a mock DetInferencer that returns the given predictions."""
        mock = MagicMock()
        mock.return_value = {
            "predictions": [
                {"scores": scores, "labels": labels, "bboxes": bboxes}
            ]
        }
        if class_names is not None:
            mock.model.dataset_meta = {"classes": class_names}
        else:
            del mock.model.dataset_meta
            mock.model.dataset_meta = {}
        return mock

    def test_returns_empty_list_when_no_predictions(self, bgr_image: np.ndarray) -> None:
        mock = MagicMock()
        mock.return_value = {"predictions": []}
        result = detect_rtmdet(bgr_image, mock)
        assert result == []

    def test_returns_detections_sorted_by_confidence(self, bgr_image: np.ndarray) -> None:
        inferencer = self._make_inferencer(
            scores=[0.6, 0.9, 0.4],
            labels=[0, 1, 2],
            bboxes=[[0, 0, 50, 50], [10, 10, 80, 80], [5, 5, 30, 30]],
        )
        results = detect_rtmdet(bgr_image, inferencer, confidence_threshold=0.3)
        confidences = [d.confidence for d in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_filters_below_threshold(self, bgr_image: np.ndarray) -> None:
        inferencer = self._make_inferencer(
            scores=[0.8, 0.1],
            labels=[0, 1],
            bboxes=[[0, 0, 50, 50], [10, 10, 80, 80]],
        )
        results = detect_rtmdet(bgr_image, inferencer, confidence_threshold=0.5)
        assert len(results) == 1
        assert results[0].confidence == pytest.approx(0.8)

    def test_uses_class_names_from_model(self, bgr_image: np.ndarray) -> None:
        inferencer = self._make_inferencer(
            scores=[0.9],
            labels=[0],
            bboxes=[[0, 0, 50, 50]],
            class_names=["person", "bicycle"],
        )
        results = detect_rtmdet(bgr_image, inferencer)
        assert results[0].label == "person"

    def test_falls_back_to_class_id_string_when_no_names(
        self, bgr_image: np.ndarray
    ) -> None:
        inferencer = self._make_inferencer(
            scores=[0.9],
            labels=[5],
            bboxes=[[0, 0, 50, 50]],
            class_names=[],
        )
        results = detect_rtmdet(bgr_image, inferencer)
        assert results[0].label == "5"

    def test_parses_rotated_bboxes(self, bgr_image: np.ndarray) -> None:
        inferencer = self._make_inferencer(
            scores=[0.85],
            labels=[4],
            bboxes=[[50.0, 50.0, 40.0, 20.0, 30.0]],  # cx, cy, w, h, angle_deg
        )
        results = detect_rtmdet(bgr_image, inferencer)
        assert len(results) == 1
        assert results[0].angle_deg == pytest.approx(30.0)

    def test_axis_aligned_bbox_has_no_angle(self, bgr_image: np.ndarray) -> None:
        inferencer = self._make_inferencer(
            scores=[0.9],
            labels=[0],
            bboxes=[[10.0, 20.0, 60.0, 80.0]],
        )
        results = detect_rtmdet(bgr_image, inferencer)
        assert results[0].angle_deg is None


# ---------------------------------------------------------------------------
# RtmDetDetector lifecycle
# ---------------------------------------------------------------------------


class TestRtmDetDetector:
    def test_requires_mmdet(self) -> None:
        with patch.dict("sys.modules", {"mmdet": None}), pytest.raises(ImportError, match="mmdet"):
            RtmDetDetector()

    def test_initialize_loads_inferencer(self) -> None:
        mock_inferencer_cls = MagicMock()
        mock_instance = MagicMock()
        mock_inferencer_cls.return_value = mock_instance

        mock_mmdet = MagicMock()
        mock_mmdet_apis = MagicMock()
        mock_mmdet_apis.DetInferencer = mock_inferencer_cls

        with (
            patch.dict("sys.modules", {"mmdet": mock_mmdet, "mmdet.apis": mock_mmdet_apis}),
            patch("image_analysis.rtmdet._require_mmdet"),
            patch("image_analysis.rtmdet._load_inferencer_with_retry") as mock_loader,
        ):
            mock_loader.return_value = mock_instance
            detector = RtmDetDetector("rtmdet-nano")
            detector.initialize()
            assert detector._inferencer is mock_instance

    def test_initialize_is_idempotent(self) -> None:
        mock_mmdet = MagicMock()
        mock_mmdet_apis = MagicMock()

        with (
            patch.dict("sys.modules", {"mmdet": mock_mmdet, "mmdet.apis": mock_mmdet_apis}),
            patch("image_analysis.rtmdet._require_mmdet"),
            patch("image_analysis.rtmdet._load_inferencer_with_retry") as mock_loader,
        ):
            mock_loader.return_value = MagicMock()
            detector = RtmDetDetector("rtmdet-nano")
            detector.initialize()
            detector.initialize()
            assert mock_loader.call_count == 1

    def test_close_releases_inferencer(self) -> None:
        mock_mmdet = MagicMock()
        mock_mmdet_apis = MagicMock()

        with (
            patch.dict("sys.modules", {"mmdet": mock_mmdet, "mmdet.apis": mock_mmdet_apis}),
            patch("image_analysis.rtmdet._require_mmdet"),
            patch("image_analysis.rtmdet._load_inferencer_with_retry") as mock_loader,
        ):
            mock_loader.return_value = MagicMock()
            detector = RtmDetDetector()
            detector.initialize()
            detector.close()
            assert detector._inferencer is None

    def test_context_manager_calls_close(self) -> None:
        mock_mmdet = MagicMock()
        mock_mmdet_apis = MagicMock()

        with (
            patch.dict("sys.modules", {"mmdet": mock_mmdet, "mmdet.apis": mock_mmdet_apis}),
            patch("image_analysis.rtmdet._require_mmdet"),
            patch("image_analysis.rtmdet._load_inferencer_with_retry") as mock_loader,
        ):
            mock_loader.return_value = MagicMock()
            with RtmDetDetector() as det:
                det.initialize()
            assert det._inferencer is None

    def test_default_constants_are_correct(self) -> None:
        assert RTMDET_CONFIDENCE_THRESHOLD == 0.3
        assert RTMDET_NMS_IOU_THRESHOLD == 0.45
        assert DEFAULT_RTMDET_MODEL == "rtmdet-nano"
        assert RTMDET_DOWNLOAD_MAX_RETRIES == 3
        assert RTMDET_DOWNLOAD_RETRY_DELAY_SECONDS == 2.0


# ---------------------------------------------------------------------------
# draw_rtmdet_detections
# ---------------------------------------------------------------------------


class TestDrawRtmDetDetections:
    def test_returns_copy_not_original(
        self, bgr_image: np.ndarray, sample_detections: list[RtmDetDetection]
    ) -> None:
        result = draw_rtmdet_detections(bgr_image, sample_detections)
        assert result is not bgr_image

    def test_output_same_shape_as_input(
        self, bgr_image: np.ndarray, sample_detections: list[RtmDetDetection]
    ) -> None:
        result = draw_rtmdet_detections(bgr_image, sample_detections)
        assert result.shape == bgr_image.shape

    def test_handles_empty_detections(self, bgr_image: np.ndarray) -> None:
        result = draw_rtmdet_detections(bgr_image, [])
        assert result.shape == bgr_image.shape

    def test_raises_for_non_positive_thickness(
        self, bgr_image: np.ndarray, sample_detections: list[RtmDetDetection]
    ) -> None:
        with pytest.raises(ValueError):
            draw_rtmdet_detections(bgr_image, sample_detections, thickness=0)

    def test_raises_for_list_input(
        self, sample_detections: list[RtmDetDetection]
    ) -> None:
        with pytest.raises(TypeError):
            draw_rtmdet_detections([[1, 2, 3]], sample_detections)  # type: ignore[arg-type]

    def test_draws_rotated_detection(
        self,
        bgr_image: np.ndarray,
        sample_rotated_detections: list[RtmDetDetection],
    ) -> None:
        result = draw_rtmdet_detections(bgr_image, sample_rotated_detections)
        assert result.shape == bgr_image.shape
        # Ensure the output was modified (at least one pixel changed)
        assert not np.array_equal(result, bgr_image)

    def test_custom_color_applied(self, bgr_image: np.ndarray) -> None:
        detections = [
            RtmDetDetection(label="person", class_id=0, confidence=0.9, bbox=(5, 5, 50, 50))
        ]
        result = draw_rtmdet_detections(bgr_image, detections, color=(0, 255, 0))
        assert result.shape == bgr_image.shape


# ---------------------------------------------------------------------------
# export_rtmdet_to_onnx - validation
# ---------------------------------------------------------------------------


class TestExportRtmDetToOnnx:
    def test_requires_mmdet(self, tmp_path) -> None:
        cfg = tmp_path / "model.py"
        cfg.write_text("# dummy config")
        ckpt = tmp_path / "model.pth"
        ckpt.write_bytes(b"\x00" * 10)

        with patch("image_analysis.rtmdet._require_mmdet") as mock_req:
            mock_req.side_effect = ImportError("mmdet missing")
            with pytest.raises(ImportError):
                export_rtmdet_to_onnx(cfg, ckpt)

    def test_raises_for_missing_config(self, tmp_path) -> None:
        ckpt = tmp_path / "model.pth"
        ckpt.write_bytes(b"\x00")
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            export_rtmdet_to_onnx(tmp_path / "nonexistent.py", ckpt)

    def test_raises_for_missing_checkpoint(self, tmp_path) -> None:
        cfg = tmp_path / "model.py"
        cfg.write_text("# dummy config")
        with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
            export_rtmdet_to_onnx(cfg, tmp_path / "nonexistent.pth")

    def test_raises_for_invalid_img_size(self, tmp_path) -> None:
        cfg = tmp_path / "model.py"
        cfg.write_text("# dummy config")
        ckpt = tmp_path / "model.pth"
        ckpt.write_bytes(b"\x00")
        with pytest.raises(ValueError, match="img_size must be a positive multiple of 32"):
            export_rtmdet_to_onnx(cfg, ckpt, img_size=100)

    def test_raises_for_zero_img_size(self, tmp_path) -> None:
        cfg = tmp_path / "model.py"
        cfg.write_text("# dummy config")
        ckpt = tmp_path / "model.pth"
        ckpt.write_bytes(b"\x00")
        with pytest.raises(ValueError):
            export_rtmdet_to_onnx(cfg, ckpt, img_size=0)


# ---------------------------------------------------------------------------
# _load_inferencer_with_retry
# ---------------------------------------------------------------------------


class TestLoadInferencerWithRetry:
    def test_succeeds_on_first_try(self) -> None:
        mock_cls = MagicMock(return_value=MagicMock())
        result = _load_inferencer_with_retry(mock_cls, "rtmdet-nano", max_retries=3)
        assert result is mock_cls.return_value
        assert mock_cls.call_count == 1

    def test_retries_on_failure_then_succeeds(self) -> None:
        mock_instance = MagicMock()
        mock_cls = MagicMock(side_effect=[RuntimeError("fail"), mock_instance])
        result = _load_inferencer_with_retry(
            mock_cls, "rtmdet-nano", max_retries=3, retry_delay=0.0
        )
        assert result is mock_instance
        assert mock_cls.call_count == 2

    def test_raises_runtime_error_after_all_retries_exhausted(self) -> None:
        mock_cls = MagicMock(side_effect=RuntimeError("persistent failure"))
        with pytest.raises(RuntimeError, match="Failed to load RTMDet model"):
            _load_inferencer_with_retry(
                mock_cls, "rtmdet-nano", max_retries=2, retry_delay=0.0
            )
        assert mock_cls.call_count == 2


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


class TestPrivateHelpers:
    def test_class_color_returns_tuple(self) -> None:
        color = _class_color(0)
        assert isinstance(color, tuple)
        assert len(color) == 3

    def test_class_color_wraps_modulo(self) -> None:
        assert _class_color(0) == _class_color(20)  # palette length is 20

    def test_is_dark_color_black(self) -> None:
        assert _is_dark_color((0, 0, 0))

    def test_is_dark_color_white(self) -> None:
        assert not _is_dark_color((255, 255, 255))

    def test_is_dark_color_medium_blue(self) -> None:
        # Pure blue is relatively dark in perceptual brightness
        assert _is_dark_color((255, 0, 0))  # BGR: full blue channel

    def test_is_dark_color_green(self) -> None:
        # Pure green in BGR: (0, 255, 0) - green is bright perceptually
        assert not _is_dark_color((0, 255, 0))
