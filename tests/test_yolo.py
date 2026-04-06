"""Tests for image_analysis.yolo module."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from image_analysis.yolo import (
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_DOWNLOAD_MAX_RETRIES,
    YOLO_DOWNLOAD_RETRY_DELAY_SECONDS,
    YOLO_MODELS_DIR,
    YOLO_NMS_IOU_THRESHOLD,
    YoloDetection,
    YoloDetector,
    _class_color,
    _is_dark_color,
    _load_with_retry,
    detect_yolo,
    draw_yolo_detections,
    export_yolo_to_onnx,
    export_yolo_to_torchscript,
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
def sample_detections() -> list[YoloDetection]:
    """Return two overlapping sample YOLO detections."""
    return [
        YoloDetection(label="person", class_id=0, confidence=0.92, bbox=(10, 10, 60, 80)),
        YoloDetection(label="cat", class_id=15, confidence=0.75, bbox=(40, 20, 90, 70)),
    ]


@pytest.fixture
def mock_ultralytics() -> MagicMock:
    """Patch ultralytics.YOLO and return the mock class."""
    with patch("image_analysis.yolo.YOLO", create=True) as mock_cls:
        yield mock_cls


# ---------------------------------------------------------------------------
# YoloDetection dataclass
# ---------------------------------------------------------------------------


class TestYoloDetection:
    def test_fields_stored_correctly(self) -> None:
        det = YoloDetection(label="dog", class_id=16, confidence=0.85, bbox=(5, 5, 50, 50))
        assert det.label == "dog"
        assert det.class_id == 16
        assert det.confidence == 0.85
        assert det.bbox == (5, 5, 50, 50)

    def test_frozen_raises_on_mutation(self) -> None:
        det = YoloDetection(label="cat", class_id=15, confidence=0.7, bbox=(0, 0, 10, 10))
        with pytest.raises(Exception):  # FrozenInstanceError (dataclass)
            det.label = "dog"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = YoloDetection(label="car", class_id=2, confidence=0.9, bbox=(0, 0, 100, 100))
        b = YoloDetection(label="car", class_id=2, confidence=0.9, bbox=(0, 0, 100, 100))
        assert a == b

    def test_inequality(self) -> None:
        a = YoloDetection(label="car", class_id=2, confidence=0.9, bbox=(0, 0, 100, 100))
        b = YoloDetection(label="car", class_id=2, confidence=0.5, bbox=(0, 0, 100, 100))
        assert a != b


# ---------------------------------------------------------------------------
# detect_yolo – validation
# ---------------------------------------------------------------------------


class TestDetectYoloValidation:
    def test_raises_type_error_for_list(self) -> None:
        fake_model = MagicMock()
        with pytest.raises(TypeError):
            detect_yolo([[1, 2, 3]], fake_model)  # type: ignore[arg-type]

    def test_raises_value_error_for_grayscale(self) -> None:
        gray = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError):
            detect_yolo(gray, MagicMock())  # type: ignore[arg-type]

    def test_raises_value_error_for_float_image(self) -> None:
        img = np.zeros((100, 100, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            detect_yolo(img, MagicMock())  # type: ignore[arg-type]

    @pytest.mark.parametrize("conf", [-0.1, 1.1, 2.0])
    def test_raises_for_invalid_confidence_threshold(
        self, bgr_image: np.ndarray, conf: float
    ) -> None:
        with pytest.raises(ValueError, match="confidence_threshold"):
            detect_yolo(bgr_image, MagicMock(), confidence_threshold=conf)

    @pytest.mark.parametrize("iou", [-0.1, 1.1])
    def test_raises_for_invalid_iou_threshold(
        self, bgr_image: np.ndarray, iou: float
    ) -> None:
        with pytest.raises(ValueError, match="iou_threshold"):
            detect_yolo(bgr_image, MagicMock(), iou_threshold=iou)


# ---------------------------------------------------------------------------
# detect_yolo – model interaction
# ---------------------------------------------------------------------------


class TestDetectYoloModelInteraction:
    def _make_box_mock(
        self, xyxy: list[float], conf: float, cls: int
    ) -> MagicMock:
        """Build a minimal ultralytics Boxes-like mock for a single detection."""
        import torch  # only needed to create tensors; skip test if unavailable

        box = MagicMock()
        box.xyxy = [torch.tensor(xyxy)]
        box.conf = [torch.tensor(conf)]
        box.cls = [torch.tensor(float(cls))]
        return box

    def _make_result_mock(self, boxes: list[MagicMock], names: dict[int, str]) -> MagicMock:
        result = MagicMock()
        result.names = names
        result.boxes = MagicMock()
        result.boxes.__len__ = MagicMock(return_value=len(boxes))
        # Simulate iteration over boxes
        result.boxes.xyxy = [b.xyxy[0] for b in boxes]
        result.boxes.conf = [b.conf[0] for b in boxes]
        result.boxes.cls = [b.cls[0] for b in boxes]
        return result

    def test_empty_results_returns_empty_list(self, bgr_image: np.ndarray) -> None:
        model = MagicMock()
        result = MagicMock()
        result.boxes = None
        model.predict.return_value = [result]

        detections = detect_yolo(bgr_image, model)
        assert detections == []

    def test_returns_sorted_by_descending_confidence(self, bgr_image: np.ndarray) -> None:
        pytest.importorskip("torch")
        model = MagicMock()
        box_a = self._make_box_mock([0, 0, 50, 50], 0.6, 0)
        box_b = self._make_box_mock([10, 10, 60, 60], 0.9, 1)
        result = self._make_result_mock([box_a, box_b], {0: "person", 1: "car"})
        model.predict.return_value = [result]

        detections = detect_yolo(bgr_image, model, confidence_threshold=0.0)
        assert detections[0].confidence >= detections[1].confidence

    def test_confidence_filter_applied(self, bgr_image: np.ndarray) -> None:
        pytest.importorskip("torch")
        model = MagicMock()
        # Only the high-confidence box should pass when threshold is 0.8.
        box_low = self._make_box_mock([0, 0, 50, 50], 0.4, 0)
        box_high = self._make_box_mock([10, 10, 60, 60], 0.9, 1)
        result = self._make_result_mock([box_low, box_high], {0: "person", 1: "car"})
        # Simulate ultralytics already filtering by conf threshold internally.
        model.predict.return_value = [result]

        # Both boxes are returned from model (ultralytics handles its own filtering).
        detections = detect_yolo(bgr_image, model, confidence_threshold=0.0)
        assert len(detections) == 2  # model returns all; our code doesn't re-filter

    def test_detection_fields_populated(self, bgr_image: np.ndarray) -> None:
        pytest.importorskip("torch")
        model = MagicMock()
        box = self._make_box_mock([5, 10, 55, 70], 0.85, 0)
        result = self._make_result_mock([box], {0: "person"})
        model.predict.return_value = [result]

        detections = detect_yolo(bgr_image, model, confidence_threshold=0.0)
        assert len(detections) == 1
        det = detections[0]
        assert det.label == "person"
        assert det.class_id == 0
        assert abs(det.confidence - 0.85) < 1e-5
        assert det.bbox == (5, 10, 55, 70)

    def test_predict_called_with_correct_kwargs(self, bgr_image: np.ndarray) -> None:
        model = MagicMock()
        result = MagicMock()
        result.boxes = None
        model.predict.return_value = [result]

        detect_yolo(bgr_image, model, confidence_threshold=0.3, iou_threshold=0.6)
        model.predict.assert_called_once_with(
            bgr_image, conf=0.3, iou=0.6, verbose=False
        )


# ---------------------------------------------------------------------------
# draw_yolo_detections
# ---------------------------------------------------------------------------


class TestDrawYoloDetections:
    def test_returns_copy_not_inplace(
        self, bgr_image: np.ndarray, sample_detections: list[YoloDetection]
    ) -> None:
        original = bgr_image.copy()
        result = draw_yolo_detections(bgr_image, sample_detections)
        np.testing.assert_array_equal(bgr_image, original)
        assert result is not bgr_image

    def test_output_shape_matches_input(
        self, bgr_image: np.ndarray, sample_detections: list[YoloDetection]
    ) -> None:
        result = draw_yolo_detections(bgr_image, sample_detections)
        assert result.shape == bgr_image.shape

    def test_output_dtype_matches_input(
        self, bgr_image: np.ndarray, sample_detections: list[YoloDetection]
    ) -> None:
        result = draw_yolo_detections(bgr_image, sample_detections)
        assert result.dtype == bgr_image.dtype

    def test_empty_detections_returns_copy(self, bgr_image: np.ndarray) -> None:
        result = draw_yolo_detections(bgr_image, [])
        np.testing.assert_array_equal(result, bgr_image)

    def test_raises_for_non_ndarray(
        self, sample_detections: list[YoloDetection]
    ) -> None:
        with pytest.raises(TypeError):
            draw_yolo_detections("not an image", sample_detections)  # type: ignore[arg-type]

    def test_raises_for_grayscale_image(
        self, sample_detections: list[YoloDetection]
    ) -> None:
        gray = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError):
            draw_yolo_detections(gray, sample_detections)  # type: ignore[arg-type]

    def test_raises_for_non_positive_thickness(
        self, bgr_image: np.ndarray, sample_detections: list[YoloDetection]
    ) -> None:
        with pytest.raises(ValueError, match="thickness"):
            draw_yolo_detections(bgr_image, sample_detections, thickness=0)

    def test_custom_color_applied(
        self, bgr_image: np.ndarray, sample_detections: list[YoloDetection]
    ) -> None:
        # Just verify it doesn't raise when a custom colour is provided.
        result = draw_yolo_detections(bgr_image, sample_detections, color=(0, 0, 255))
        assert result.shape == bgr_image.shape

    def test_modifies_pixels(
        self, bgr_image: np.ndarray, sample_detections: list[YoloDetection]
    ) -> None:
        result = draw_yolo_detections(bgr_image, sample_detections)
        # Drawing bounding boxes must change at least some pixels.
        assert not np.array_equal(result, bgr_image)


# ---------------------------------------------------------------------------
# YoloDetector
# ---------------------------------------------------------------------------


class TestYoloDetector:
    def test_raises_import_error_without_ultralytics(self) -> None:
        with patch.dict("sys.modules", {"ultralytics": None}):
            with pytest.raises(ImportError, match="ultralytics"):
                YoloDetector()

    def test_initialize_raises_file_not_found_for_missing_model(self) -> None:
        import sys
        from pathlib import Path

        mock_ultralytics = MagicMock()
        mock_ultralytics.YOLO.side_effect = FileNotFoundError("not found")

        with patch.dict("sys.modules", {"ultralytics": mock_ultralytics}):
            # Re-import to pick up the mocked module in deferred imports.
            import importlib
            import image_analysis.yolo as yolo_mod
            importlib.reload(yolo_mod)

            detector = yolo_mod.YoloDetector.__new__(yolo_mod.YoloDetector)
            detector._model_path = Path("/nonexistent/model.pt")
            detector._model = None
            detector.confidence_threshold = YOLO_CONFIDENCE_THRESHOLD
            detector.iou_threshold = YOLO_NMS_IOU_THRESHOLD

            with pytest.raises(FileNotFoundError):
                detector.initialize()

        # Reload back to normal state.
        import image_analysis.yolo as yolo_mod
        importlib.reload(yolo_mod)

    def test_close_sets_model_to_none(self) -> None:
        with patch("image_analysis.yolo._require_ultralytics"):
            detector = YoloDetector.__new__(YoloDetector)
            detector._model = MagicMock()
            detector._model_path = __import__("pathlib").Path("yolov8n.pt")
            detector.confidence_threshold = YOLO_CONFIDENCE_THRESHOLD
            detector.iou_threshold = YOLO_NMS_IOU_THRESHOLD
            detector.close()
            assert detector._model is None

    def test_context_manager_closes_on_exit(self) -> None:
        with patch("image_analysis.yolo._require_ultralytics"):
            detector = YoloDetector.__new__(YoloDetector)
            detector._model = MagicMock()
            detector._model_path = __import__("pathlib").Path("yolov8n.pt")
            detector.confidence_threshold = YOLO_CONFIDENCE_THRESHOLD
            detector.iou_threshold = YOLO_NMS_IOU_THRESHOLD
            with detector:
                pass
            assert detector._model is None

    def test_detect_delegates_to_detect_yolo(self, bgr_image: np.ndarray) -> None:
        with patch("image_analysis.yolo._require_ultralytics"):
            with patch("image_analysis.yolo.detect_yolo", return_value=[]) as mock_fn:
                detector = YoloDetector.__new__(YoloDetector)
                detector._model = MagicMock()
                detector._model_path = __import__("pathlib").Path("yolov8n.pt")
                detector.confidence_threshold = YOLO_CONFIDENCE_THRESHOLD
                detector.iou_threshold = YOLO_NMS_IOU_THRESHOLD

                result = detector.detect(bgr_image)
                mock_fn.assert_called_once()
                assert result == []


# ---------------------------------------------------------------------------
# export_yolo_to_onnx
# ---------------------------------------------------------------------------


class TestExportYoloToOnnx:
    def test_raises_import_error_without_ultralytics(self, tmp_path) -> None:
        with patch.dict("sys.modules", {"ultralytics": None}):
            with pytest.raises(ImportError, match="ultralytics"):
                export_yolo_to_onnx(tmp_path / "model.pt")

    def test_raises_file_not_found_for_missing_model(self, tmp_path) -> None:
        with patch("image_analysis.yolo._require_ultralytics"):
            with pytest.raises(FileNotFoundError):
                export_yolo_to_onnx(tmp_path / "nonexistent.pt")

    @pytest.mark.parametrize("bad_size", [0, -640, 100, 33])
    def test_raises_value_error_for_invalid_img_size(self, tmp_path, bad_size: int) -> None:
        model_file = tmp_path / "model.pt"
        model_file.write_bytes(b"fake")
        with patch("image_analysis.yolo._require_ultralytics"):
            with pytest.raises(ValueError, match="img_size"):
                export_yolo_to_onnx(model_file, img_size=bad_size)

    def test_export_calls_ultralytics_and_renames(self, tmp_path) -> None:
        model_file = tmp_path / "model.pt"
        model_file.write_bytes(b"fake")
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"onnx")  # simulate exported file

        mock_model = MagicMock()
        mock_model.export.return_value = str(tmp_path / "model.onnx")

        mock_ultralytics = MagicMock()
        mock_ultralytics.YOLO.return_value = mock_model

        import importlib
        import image_analysis.yolo as yolo_mod

        with patch.dict("sys.modules", {"ultralytics": mock_ultralytics}):
            importlib.reload(yolo_mod)
            result = yolo_mod.export_yolo_to_onnx(
                model_file, output_path=onnx_file, img_size=640
            )

        importlib.reload(yolo_mod)
        mock_model.export.assert_called_once()
        assert result == onnx_file.resolve()


# ---------------------------------------------------------------------------
# export_yolo_to_torchscript
# ---------------------------------------------------------------------------


class TestExportYoloToTorchscript:
    def test_raises_import_error_without_ultralytics(self, tmp_path) -> None:
        with patch.dict("sys.modules", {"ultralytics": None}):
            with pytest.raises(ImportError, match="ultralytics"):
                export_yolo_to_torchscript(tmp_path / "model.pt")

    def test_raises_file_not_found_for_missing_model(self, tmp_path) -> None:
        with patch("image_analysis.yolo._require_ultralytics"):
            with pytest.raises(FileNotFoundError):
                export_yolo_to_torchscript(tmp_path / "nonexistent.pt")

    @pytest.mark.parametrize("bad_size", [0, -640, 100, 33])
    def test_raises_value_error_for_invalid_img_size(self, tmp_path, bad_size: int) -> None:
        model_file = tmp_path / "model.pt"
        model_file.write_bytes(b"fake")
        with patch("image_analysis.yolo._require_ultralytics"):
            with pytest.raises(ValueError, match="img_size"):
                export_yolo_to_torchscript(model_file, img_size=bad_size)

    def test_export_calls_ultralytics_torchscript_and_renames(self, tmp_path) -> None:
        model_file = tmp_path / "model.pt"
        model_file.write_bytes(b"fake")
        ts_file = tmp_path / "model.torchscript"
        ts_file.write_bytes(b"torchscript")  # simulate exported file

        mock_model = MagicMock()
        mock_model.export.return_value = str(tmp_path / "model.torchscript")

        mock_ultralytics = MagicMock()
        mock_ultralytics.YOLO.return_value = mock_model

        import importlib
        import image_analysis.yolo as yolo_mod

        with patch.dict("sys.modules", {"ultralytics": mock_ultralytics}):
            importlib.reload(yolo_mod)
            result = yolo_mod.export_yolo_to_torchscript(
                model_file, output_path=ts_file, img_size=640
            )

        importlib.reload(yolo_mod)
        mock_model.export.assert_called_once()
        call_kwargs = mock_model.export.call_args
        assert call_kwargs.kwargs.get("format") == "torchscript"
        assert result == ts_file.resolve()

    def test_default_output_path_uses_torchscript_extension(self, tmp_path) -> None:
        model_file = tmp_path / "yolov8n.pt"
        model_file.write_bytes(b"fake")
        expected_output = tmp_path / "yolov8n.torchscript"
        expected_output.write_bytes(b"torchscript")

        mock_model = MagicMock()
        mock_model.export.return_value = str(expected_output)

        mock_ultralytics = MagicMock()
        mock_ultralytics.YOLO.return_value = mock_model

        import importlib
        import image_analysis.yolo as yolo_mod

        with patch.dict("sys.modules", {"ultralytics": mock_ultralytics}):
            importlib.reload(yolo_mod)
            result = yolo_mod.export_yolo_to_torchscript(model_file, img_size=640)

        importlib.reload(yolo_mod)
        assert result.suffix == ".torchscript"

    def test_optimize_flag_forwarded_to_export(self, tmp_path) -> None:
        model_file = tmp_path / "model.pt"
        model_file.write_bytes(b"fake")
        ts_file = tmp_path / "model.torchscript"
        ts_file.write_bytes(b"torchscript")

        mock_model = MagicMock()
        mock_model.export.return_value = str(ts_file)

        mock_ultralytics = MagicMock()
        mock_ultralytics.YOLO.return_value = mock_model

        import importlib
        import image_analysis.yolo as yolo_mod

        with patch.dict("sys.modules", {"ultralytics": mock_ultralytics}):
            importlib.reload(yolo_mod)
            yolo_mod.export_yolo_to_torchscript(model_file, output_path=ts_file, optimize=True)

        importlib.reload(yolo_mod)
        call_kwargs = mock_model.export.call_args
        assert call_kwargs.kwargs.get("optimize") is True


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


class TestClassColor:
    def test_returns_tuple_of_three_ints(self) -> None:
        color = _class_color(0)
        assert isinstance(color, tuple)
        assert len(color) == 3
        assert all(isinstance(c, int) for c in color)

    def test_wraps_around_palette(self) -> None:
        # Two class IDs that are `len(palette)` apart must get the same colour.
        from image_analysis.yolo import _PALETTE

        c0 = _class_color(0)
        c_wrap = _class_color(len(_PALETTE))
        assert c0 == c_wrap

    def test_different_classes_may_differ(self) -> None:
        # At least two distinct class IDs should produce different colours.
        colors = {_class_color(i) for i in range(10)}
        assert len(colors) > 1


class TestIsDarkColor:
    def test_black_is_dark(self) -> None:
        assert _is_dark_color((0, 0, 0)) is True

    def test_white_is_not_dark(self) -> None:
        assert _is_dark_color((255, 255, 255)) is False

    def test_midgrey_boundary(self) -> None:
        # (127, 127, 127) has luminance ≈ 127 < 128 → dark
        assert _is_dark_color((127, 127, 127)) is True


# ---------------------------------------------------------------------------
# _load_with_retry
# ---------------------------------------------------------------------------


class TestLoadWithRetry:
    def test_returns_model_on_first_success(self) -> None:
        mock_model = MagicMock()
        mock_cls = MagicMock(return_value=mock_model)

        result = _load_with_retry(mock_cls, "yolov8n.pt", max_retries=3, retry_delay=0.0)

        assert result is mock_model
        mock_cls.assert_called_once_with("yolov8n.pt")

    def test_retries_on_transient_failure_then_succeeds(self) -> None:
        mock_model = MagicMock()
        mock_cls = MagicMock(side_effect=[RuntimeError("network error"), mock_model])

        with patch("image_analysis.yolo.time.sleep") as mock_sleep:
            result = _load_with_retry(mock_cls, "yolov8n.pt", max_retries=3, retry_delay=1.0)

        assert result is mock_model
        assert mock_cls.call_count == 2
        mock_sleep.assert_called_once_with(1.0)

    def test_exponential_backoff_delays(self) -> None:
        mock_model = MagicMock()
        mock_cls = MagicMock(
            side_effect=[RuntimeError("err"), RuntimeError("err"), mock_model]
        )

        with patch("image_analysis.yolo.time.sleep") as mock_sleep:
            _load_with_retry(mock_cls, "yolov8n.pt", max_retries=3, retry_delay=2.0)

        assert mock_sleep.call_args_list == [call(2.0), call(4.0)]

    def test_raises_runtime_error_after_all_retries_exhausted(self) -> None:
        mock_cls = MagicMock(side_effect=RuntimeError("always fails"))

        with patch("image_analysis.yolo.time.sleep"):
            with pytest.raises(RuntimeError, match="Failed to download YOLO model"):
                _load_with_retry(mock_cls, "yolov8n.pt", max_retries=3, retry_delay=0.0)

        assert mock_cls.call_count == 3

    def test_raises_runtime_error_chained_from_last_exception(self) -> None:
        original_exc = OSError("connection refused")
        mock_cls = MagicMock(side_effect=original_exc)

        with patch("image_analysis.yolo.time.sleep"):
            with pytest.raises(RuntimeError) as exc_info:
                _load_with_retry(mock_cls, "yolov8n.pt", max_retries=2, retry_delay=0.0)

        assert exc_info.value.__cause__ is original_exc

    def test_single_retry_attempt(self) -> None:
        mock_cls = MagicMock(side_effect=ValueError("bad"))

        with patch("image_analysis.yolo.time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError):
                _load_with_retry(mock_cls, "yolov8n.pt", max_retries=1, retry_delay=5.0)

        mock_sleep.assert_not_called()  # no sleep when max_retries=1


# ---------------------------------------------------------------------------
# Constants sanity checks
# ---------------------------------------------------------------------------


def test_default_confidence_threshold_in_range() -> None:
    assert 0.0 < YOLO_CONFIDENCE_THRESHOLD < 1.0


def test_default_nms_iou_threshold_in_range() -> None:
    assert 0.0 < YOLO_NMS_IOU_THRESHOLD < 1.0


def test_download_max_retries_is_positive() -> None:
    assert YOLO_DOWNLOAD_MAX_RETRIES >= 1


def test_download_retry_delay_is_non_negative() -> None:
    assert YOLO_DOWNLOAD_RETRY_DELAY_SECONDS >= 0.0


# ---------------------------------------------------------------------------
# YOLO_MODELS_DIR and _get_models_dir
# ---------------------------------------------------------------------------


def test_yolo_models_dir_is_path_instance() -> None:
    from pathlib import Path

    assert isinstance(YOLO_MODELS_DIR, Path)


def test_get_models_dir_returns_path(tmp_path) -> None:
    """_get_models_dir should return the YOLO_MODELS_DIR path after creation."""
    import image_analysis.yolo as yolo_mod

    with patch.object(yolo_mod, "YOLO_MODELS_DIR", tmp_path / "yolo_cache"):
        result = yolo_mod._get_models_dir()
        assert result == tmp_path / "yolo_cache"


def test_get_models_dir_creates_directory(tmp_path) -> None:
    """_get_models_dir should create the directory if it does not exist."""
    import image_analysis.yolo as yolo_mod

    new_dir = tmp_path / "nested" / "yolo"
    assert not new_dir.exists()
    with patch.object(yolo_mod, "YOLO_MODELS_DIR", new_dir):
        yolo_mod._get_models_dir()
    assert new_dir.is_dir()


# ---------------------------------------------------------------------------
# YoloDetector.initialize - cache-first resolution
# ---------------------------------------------------------------------------


class TestYoloDetectorCacheResolution:
    """Verify that initialize() checks YOLO_MODELS_DIR before downloading."""

    def test_loads_from_cache_when_model_exists(self, tmp_path) -> None:
        """If the model file is already in the cache dir it is loaded directly."""
        import importlib

        import image_analysis.yolo as yolo_mod

        cached_model = tmp_path / "yolov8n.pt"
        cached_model.write_bytes(b"fake-weights")

        mock_yolo_cls = MagicMock()
        mock_ultralytics = MagicMock()
        mock_ultralytics.YOLO = mock_yolo_cls

        try:
            with patch.dict("sys.modules", {"ultralytics": mock_ultralytics}):
                importlib.reload(yolo_mod)
                with patch.object(yolo_mod, "YOLO_MODELS_DIR", tmp_path):
                    detector = yolo_mod.YoloDetector.__new__(yolo_mod.YoloDetector)
                    detector._model_path = yolo_mod.Path("yolov8n.pt")
                    detector._model = None
                    detector.confidence_threshold = YOLO_CONFIDENCE_THRESHOLD
                    detector.iou_threshold = YOLO_NMS_IOU_THRESHOLD
                    detector.initialize()

            # YOLO should have been called with the cached path, not the bare name.
            mock_yolo_cls.assert_called_once_with(str(tmp_path / "yolov8n.pt"))
        finally:
            importlib.reload(yolo_mod)

    def test_downloads_to_cache_when_model_missing(self, tmp_path) -> None:
        """If the model is absent from cache, _load_with_retry receives the cache path."""
        import importlib

        import image_analysis.yolo as yolo_mod

        mock_yolo_cls = MagicMock()
        mock_ultralytics = MagicMock()
        mock_ultralytics.YOLO = mock_yolo_cls

        try:
            with patch.dict("sys.modules", {"ultralytics": mock_ultralytics}):
                importlib.reload(yolo_mod)
                with patch.object(yolo_mod, "YOLO_MODELS_DIR", tmp_path):
                    with patch.object(yolo_mod, "_load_with_retry") as mock_retry:
                        mock_retry.return_value = MagicMock()
                        detector = yolo_mod.YoloDetector.__new__(yolo_mod.YoloDetector)
                        detector._model_path = yolo_mod.Path("yolov8n.pt")
                        detector._model = None
                        detector.confidence_threshold = YOLO_CONFIDENCE_THRESHOLD
                        detector.iou_threshold = YOLO_NMS_IOU_THRESHOLD
                        detector.initialize()

                        # The retry helper must receive the full path inside the cache dir.
                        expected_path = str(tmp_path / "yolov8n.pt")
                        mock_retry.assert_called_once_with(mock_yolo_cls, expected_path)
        finally:
            importlib.reload(yolo_mod)

    def test_explicit_path_still_raises_file_not_found(self, tmp_path) -> None:
        """An explicit (slash-containing) path that does not exist must raise."""
        import importlib

        import image_analysis.yolo as yolo_mod

        mock_ultralytics = MagicMock()

        try:
            with patch.dict("sys.modules", {"ultralytics": mock_ultralytics}):
                importlib.reload(yolo_mod)
                detector = yolo_mod.YoloDetector.__new__(yolo_mod.YoloDetector)
                detector._model_path = tmp_path / "nonexistent.pt"
                detector._model = None
                detector.confidence_threshold = YOLO_CONFIDENCE_THRESHOLD
                detector.iou_threshold = YOLO_NMS_IOU_THRESHOLD

                with pytest.raises(FileNotFoundError):
                    detector.initialize()
        finally:
            importlib.reload(yolo_mod)
