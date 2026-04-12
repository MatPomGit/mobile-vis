"""Common backend protocols and unified inference result contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .types import BboxXYXY, BgrImageU8, Image


@dataclass(frozen=True)
class InferenceResult:
    """Unified result structure returned by detector/classifier backends.

    Attributes:
        bbox: Bounding box in ``(x1, y1, x2, y2)`` format or ``None`` when not applicable.
        label: Predicted class label.
        score: Prediction score in ``[0.0, 1.0]``.
        metadata: Optional provider-specific payload.
    """

    bbox: BboxXYXY | None
    label: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        """Backward-compatible alias for legacy confidence naming."""
        return self.score


class DetectorBackend(Protocol):
    """Protocol implemented by object-detection backends."""

    def detect(
        self,
        image: BgrImageU8,
        confidence_threshold: float,
    ) -> list[InferenceResult]:
        """Run detection inference for an input BGR image."""


class ClassifierBackend(Protocol):
    """Protocol implemented by image-classification backends."""

    def classify(
        self,
        image: Image,
        confidence_threshold: float,
    ) -> InferenceResult:
        """Run classification inference for an input image."""
