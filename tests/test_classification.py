"""Tests for image_analysis.classification module."""

from __future__ import annotations

import numpy as np
import pytest

from image_analysis.classification import (
    DEFAULT_CLASSIFIER_BACKEND,
    ClassificationResult,
    classify_image,
    create_classifier_backend,
    evaluate_classifier,
    register_classifier_backend,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bgr_image() -> np.ndarray:
    """Return a synthetic 64x64 BGR uint8 image."""
    rng = np.random.default_rng(seed=7)
    return rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)


class _FakeClassifierBackend:
    """Pomocniczy backend zwracający kontrolowaną predykcję w testach."""

    def classify(
        self,
        image: np.ndarray,
        confidence_threshold: float,
    ) -> ClassificationResult:
        _ = image
        _ = confidence_threshold
        return ClassificationResult(
            bbox=(0, 0, 1, 1),
            label="cat",
            score=0.93,
            metadata={"source": "fake"},
        )


# ---------------------------------------------------------------------------
# classify_image
# ---------------------------------------------------------------------------


class TestClassifyImage:
    def test_returns_unified_result_structure(self, bgr_image: np.ndarray) -> None:
        result = classify_image(bgr_image)
        assert isinstance(result.label, str)
        assert isinstance(result.score, float)

    def test_stub_returns_unknown_and_zero(self, bgr_image: np.ndarray) -> None:
        result = classify_image(bgr_image)
        assert result.label == "unknown"
        assert result.score == 0.0
        assert result.bbox is None

    def test_raises_for_non_ndarray(self) -> None:
        with pytest.raises(TypeError):
            classify_image("not an image")  # type: ignore[arg-type]

    @pytest.mark.parametrize("threshold", [-0.1, 1.1])
    def test_raises_for_invalid_threshold(self, bgr_image: np.ndarray, threshold: float) -> None:
        with pytest.raises(ValueError):
            classify_image(bgr_image, confidence_threshold=threshold)

    def test_raises_for_invalid_dtype(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.int32)
        with pytest.raises(ValueError):
            classify_image(image)

    def test_raises_for_invalid_channel_count(self) -> None:
        image = np.zeros((32, 32, 2), dtype=np.uint8)
        with pytest.raises(ValueError):
            classify_image(image)

    def test_uses_named_backend_from_registry(self, bgr_image: np.ndarray) -> None:
        register_classifier_backend("fake-classifier", _FakeClassifierBackend)
        result = classify_image(bgr_image, backend="fake-classifier")
        assert result.label == "cat"
        assert result.score == pytest.approx(0.93)

    def test_unknown_backend_falls_back_to_default(self, bgr_image: np.ndarray) -> None:
        backend = create_classifier_backend("missing-backend")
        assert backend.__class__.__name__ == "StubClassifierBackend"

        result = classify_image(bgr_image, backend="missing-backend")
        assert result.label == "unknown"


# ---------------------------------------------------------------------------
# evaluate_classifier
# ---------------------------------------------------------------------------


class TestEvaluateClassifier:
    def test_empty_inputs_return_zero_metrics(self) -> None:
        result = evaluate_classifier([], [])
        assert result["accuracy"] == 0.0
        assert result["avg_confidence"] == 0.0

    def test_perfect_predictions(self) -> None:
        predictions = [("cat", 0.9), ("dog", 0.8)]
        ground_truth = ["cat", "dog"]
        result = evaluate_classifier(predictions, ground_truth)
        assert result["accuracy"] == pytest.approx(1.0)

    def test_no_correct_predictions(self) -> None:
        predictions = [("cat", 0.9), ("cat", 0.8)]
        ground_truth = ["dog", "dog"]
        result = evaluate_classifier(predictions, ground_truth)
        assert result["accuracy"] == pytest.approx(0.0)

    def test_partial_accuracy(self) -> None:
        predictions = [("cat", 1.0), ("dog", 1.0), ("cat", 1.0), ("dog", 1.0)]
        ground_truth = ["cat", "cat", "cat", "cat"]
        result = evaluate_classifier(predictions, ground_truth)
        assert result["accuracy"] == pytest.approx(0.5)

    def test_avg_confidence(self) -> None:
        predictions = [("cat", 0.6), ("dog", 0.4)]
        ground_truth = ["cat", "dog"]
        result = evaluate_classifier(predictions, ground_truth)
        assert result["avg_confidence"] == pytest.approx(0.5)

    def test_raises_for_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            evaluate_classifier([("cat", 0.9)], ["cat", "dog"])

    def test_raises_for_invalid_confidence(self) -> None:
        with pytest.raises(ValueError):
            evaluate_classifier([("cat", 1.5)], ["cat"])

    def test_raises_for_empty_label(self) -> None:
        with pytest.raises(ValueError):
            evaluate_classifier([("", 0.5)], ["cat"])

    def test_raises_for_invalid_ground_truth_label(self) -> None:
        with pytest.raises(ValueError):
            evaluate_classifier([("cat", 0.9)], [""])


def test_default_backend_constant_is_stub() -> None:
    """Ensure default classifier backend key remains stable for CLI and tests."""
    assert DEFAULT_CLASSIFIER_BACKEND == "stub"
