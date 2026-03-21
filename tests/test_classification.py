"""Tests for image_analysis.classification module."""

from __future__ import annotations

import numpy as np
import pytest

from image_analysis.classification import classify_image, evaluate_classifier

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bgr_image() -> np.ndarray:
    """Return a synthetic 64x64 BGR uint8 image."""
    rng = np.random.default_rng(seed=7)
    return rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# classify_image
# ---------------------------------------------------------------------------


class TestClassifyImage:
    def test_returns_tuple_of_label_and_confidence(self, bgr_image: np.ndarray) -> None:
        label, confidence = classify_image(bgr_image)
        assert isinstance(label, str)
        assert isinstance(confidence, float)

    def test_stub_returns_unknown_and_zero(self, bgr_image: np.ndarray) -> None:
        label, confidence = classify_image(bgr_image)
        assert label == "unknown"
        assert confidence == 0.0

    def test_raises_for_non_ndarray(self) -> None:
        with pytest.raises(TypeError):
            classify_image("not an image")  # type: ignore[arg-type]

    @pytest.mark.parametrize("threshold", [-0.1, 1.1])
    def test_raises_for_invalid_threshold(
        self, bgr_image: np.ndarray, threshold: float
    ) -> None:
        with pytest.raises(ValueError):
            classify_image(bgr_image, confidence_threshold=threshold)


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
