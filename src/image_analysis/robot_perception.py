"""Utilities for offline robot-perception replay and tracking measurements.

The module enables processing recorded MP4 files frame-by-frame and exporting
per-frame tracking metrics to CSV for state-machine quality comparisons.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import ModuleType
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _require_cv2() -> ModuleType:
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'cv2' package is required for MP4 replay and tracking measurement."
        ) from exc
    return cv2


@dataclass(frozen=True)
class TrackedElement:
    """Tracked element descriptor.

    Attributes:
        element_id: Stable identifier of the tracked object in current run.
        label: Semantic class label.
        bbox_xyxy: Bounding box in pixel format ``(x1, y1, x2, y2)``.
        confidence: Tracker confidence in range ``[0.0, 1.0]``.
    """

    element_id: str
    label: str
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True)
class FrameTrackingMetrics:
    """Per-frame tracking metrics exported to CSV.

    Attributes:
        frame_index: Zero-based frame index in sequence.
        timestamp_s: Frame timestamp in seconds.
        state_stage: Active state-machine stage for frame.
        tracked_count: Number of tracked elements after label filtering.
        mean_confidence: Average confidence for tracked elements.
        centroid_x_px: Mean centroid X (pixels) across tracked elements.
        centroid_y_px: Mean centroid Y (pixels) across tracked elements.
        centroid_speed_px_s: Mean centroid speed against previous frame.
        bbox_area_mean_px2: Mean bounding-box area in pixels squared.
        bbox_area_delta_px2: Mean area delta against previous frame.
        tracking_stability: Stability score in ``[0.0, 1.0]`` based on motion jitter.
    """

    frame_index: int
    timestamp_s: float
    state_stage: str
    tracked_count: int
    mean_confidence: float
    centroid_x_px: float
    centroid_y_px: float
    centroid_speed_px_s: float
    bbox_area_mean_px2: float
    bbox_area_delta_px2: float
    tracking_stability: float


class ElementTracker(Protocol):
    """Protocol for frame-level tracker implementations."""

    def track(self, frame_bgr: NDArray[np.uint8]) -> Sequence[TrackedElement]:
        """Track elements on a single frame."""


class StateMachineStageResolver(Protocol):
    """Protocol for resolving state-machine stage per frame."""

    def resolve_stage(
        self,
        frame_index: int,
        timestamp_s: float,
        tracked_elements: Sequence[TrackedElement],
    ) -> str:
        """Return state-machine stage name for a frame."""


@dataclass(frozen=True)
class ScheduledStateMachine:
    """Simple timeline-based stage resolver.

    ``schedule`` is a list of ``(start_time_seconds, stage_name)`` tuples.
    """

    schedule: Sequence[tuple[float, str]]

    def resolve_stage(
        self,
        frame_index: int,
        timestamp_s: float,
        tracked_elements: Sequence[TrackedElement],
    ) -> str:
        del frame_index, tracked_elements
        stage = "unknown"
        for start_time_s, stage_name in self.schedule:
            if timestamp_s >= start_time_s:
                stage = stage_name
        return stage


@dataclass(frozen=True)
class ReplayFrame:
    """Frame payload generated from MP4 replay."""

    frame_index: int
    timestamp_s: float
    frame_bgr: NDArray[np.uint8]


@dataclass(frozen=True)
class ColorRangeHSV:
    """HSV colour filter range.

    Attributes:
        lower: Lower HSV bound ``(H, S, V)`` in OpenCV scale
            where ``H`` is in ``[0, 179]`` and ``S``/``V`` in ``[0, 255]``.
        upper: Upper HSV bound ``(H, S, V)`` in OpenCV scale.
    """

    lower: tuple[int, int, int]
    upper: tuple[int, int, int]


@dataclass(frozen=True)
class LightSpotDetection:
    """Detected bright ellipse describing robot destination marker.

    Attributes:
        center_xy: Ellipse center in pixels as ``(x, y)``.
        axes_xy: Major/minor axis lengths in pixels as ``(width, height)``.
        angle_deg: Ellipse rotation angle in degrees.
        area_px2: Contour area in pixels squared.
    """

    center_xy: tuple[float, float]
    axes_xy: tuple[float, float]
    angle_deg: float
    area_px2: float


def detect_light_spot(
    image_bgr: NDArray[np.uint8],
    min_brightness: int = 220,
    min_area_px2: float = 20.0,
    allowed_colors_hsv: Sequence[ColorRangeHSV] | None = None,
) -> LightSpotDetection | None:
    """Find the brightest destination spot and return ellipse center coordinates.

    The function detects bright blobs in a BGR image and fits an ellipse to the
    largest valid contour. Optionally, detection can be restricted to selected
    colour ranges in HSV space.

    Args:
        image_bgr: Input image with shape ``(H, W, C)``, dtype ``uint8``,
            where ``C=3`` and channel order is BGR.
        min_brightness: Grayscale threshold in ``[0, 255]`` used to isolate
            bright pixels.
        min_area_px2: Minimum contour area to accept as a destination marker.
        allowed_colors_hsv: Optional list of accepted HSV ranges. If provided,
            only bright pixels inside at least one range are considered.

    Returns:
        ``LightSpotDetection`` for the best matching blob, otherwise ``None``.

    Raises:
        ValueError: If arguments are outside accepted ranges.
    """
    _validate_bgr_image(image_bgr)
    if not (0 <= min_brightness <= 255):
        raise ValueError(f"min_brightness must be in [0, 255], got {min_brightness}")
    if min_area_px2 <= 0.0:
        raise ValueError(f"min_area_px2 must be > 0, got {min_area_px2}")

    cv2 = _require_cv2()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray, min_brightness, 255, cv2.THRESH_BINARY)

    if allowed_colors_hsv:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        color_mask = np.zeros(gray.shape, dtype=np.uint8)
        for color_range in allowed_colors_hsv:
            _validate_hsv_range(color_range)
            partial_mask = cv2.inRange(
                hsv,
                np.array(color_range.lower, dtype=np.uint8),
                np.array(color_range.upper, dtype=np.uint8),
            )
            color_mask = cv2.bitwise_or(color_mask, partial_mask)
        bright_mask = cv2.bitwise_and(bright_mask, color_mask)

    contours, _hierarchy = cv2.findContours(
        bright_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area_px2]
    if not valid_contours:
        return None
    best_contour = max(valid_contours, key=cv2.contourArea)
    area_px2 = float(cv2.contourArea(best_contour))

    if len(best_contour) >= 5:
        (center_x, center_y), (axis_w, axis_h), angle_deg = cv2.fitEllipse(best_contour)
    else:
        moments = cv2.moments(best_contour)
        if moments["m00"] <= 0.0:
            return None
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]
        _x, _y, w, h = cv2.boundingRect(best_contour)
        axis_w, axis_h = float(w), float(h)
        angle_deg = 0.0

    return LightSpotDetection(
        center_xy=(float(center_x), float(center_y)),
        axes_xy=(float(axis_w), float(axis_h)),
        angle_deg=float(angle_deg),
        area_px2=area_px2,
    )


def replay_mp4(video_path: str | Path) -> list[ReplayFrame]:
    """Load MP4 and return frame list suitable for offline processing.

    Args:
        video_path: Path to an MP4 file.

    Returns:
        List of frames with timestamps.

    Raises:
        FileNotFoundError: If MP4 file does not exist.
        ValueError: If video cannot be opened.
    """
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    cv2 = _require_cv2()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video file: {path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    fallback_fps = fps if fps > 0 else 30.0

    frames: list[ReplayFrame] = []
    frame_index = 0
    while True:
        is_ok, frame = capture.read()
        if not is_ok or frame is None:
            break

        timestamp_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
        timestamp_s = frame_index / fallback_fps if timestamp_ms <= 0 else timestamp_ms / 1000.0

        frames.append(
            ReplayFrame(
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                frame_bgr=frame,
            )
        )
        frame_index += 1

    capture.release()
    logger.info("Loaded %d frames from %s", len(frames), path)
    return frames


def _mean_centroid(elements: Sequence[TrackedElement]) -> tuple[float, float]:
    if not elements:
        return 0.0, 0.0
    centroids = [
        ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)
        for box in (e.bbox_xyxy for e in elements)
    ]
    return float(np.mean([c[0] for c in centroids])), float(np.mean([c[1] for c in centroids]))


def _mean_area(elements: Sequence[TrackedElement]) -> float:
    if not elements:
        return 0.0
    areas = [max(0.0, (b[2] - b[0]) * (b[3] - b[1])) for b in (e.bbox_xyxy for e in elements)]
    return float(np.mean(areas))


def _stability_from_speed(speed_px_s: float) -> float:
    speed_norm = min(1.0, speed_px_s / 250.0)
    return float(max(0.0, 1.0 - speed_norm))


def measure_tracking_on_mp4(
    video_path: str | Path,
    tracker: ElementTracker,
    state_resolver: StateMachineStageResolver,
    csv_output_path: str | Path,
    tracked_labels: set[str] | None = None,
) -> list[FrameTrackingMetrics]:
    """Measure tracking metrics frame-by-frame and export CSV.

    Args:
        video_path: Input MP4 path.
        tracker: Tracker implementation returning per-frame elements.
        state_resolver: State-machine stage resolver for each frame.
        csv_output_path: Output CSV file path.
        tracked_labels: Optional label whitelist.

    Returns:
        List of per-frame tracking metrics.
    """
    replay_frames = replay_mp4(video_path)
    rows: list[FrameTrackingMetrics] = []
    prev_centroid: tuple[float, float] | None = None
    prev_area: float | None = None
    prev_ts: float | None = None

    for replay in replay_frames:
        elements = list(tracker.track(replay.frame_bgr))
        if tracked_labels is not None:
            elements = [item for item in elements if item.label in tracked_labels]

        stage = state_resolver.resolve_stage(replay.frame_index, replay.timestamp_s, elements)
        centroid_x, centroid_y = _mean_centroid(elements)
        area_mean = _mean_area(elements)

        speed_px_s = 0.0
        if prev_centroid is not None and prev_ts is not None:
            dt_s = max(1e-6, replay.timestamp_s - prev_ts)
            distance = float(
                np.hypot(
                    centroid_x - prev_centroid[0],
                    centroid_y - prev_centroid[1],
                )
            )
            speed_px_s = distance / dt_s

        area_delta = 0.0 if prev_area is None else area_mean - prev_area
        mean_conf = float(np.mean([e.confidence for e in elements])) if elements else 0.0

        row = FrameTrackingMetrics(
            frame_index=replay.frame_index,
            timestamp_s=round(replay.timestamp_s, 6),
            state_stage=stage,
            tracked_count=len(elements),
            mean_confidence=round(mean_conf, 6),
            centroid_x_px=round(centroid_x, 6),
            centroid_y_px=round(centroid_y, 6),
            centroid_speed_px_s=round(speed_px_s, 6),
            bbox_area_mean_px2=round(area_mean, 6),
            bbox_area_delta_px2=round(area_delta, 6),
            tracking_stability=round(_stability_from_speed(speed_px_s), 6),
        )
        rows.append(row)

        prev_centroid = (centroid_x, centroid_y)
        prev_area = area_mean
        prev_ts = replay.timestamp_s

    output = Path(csv_output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as csv_file:
        column_names = [field.name for field in fields(FrameTrackingMetrics)]
        writer = csv.DictWriter(csv_file, fieldnames=column_names)
        writer.writeheader()
        for item in rows:
            writer.writerow(asdict(item))

    logger.info("Saved %d tracking rows to %s", len(rows), output)
    return rows


def _validate_bgr_image(image: object) -> None:
    """Validate that image is a uint8 BGR array with shape ``(H, W, 3)``."""
    if not isinstance(image, np.ndarray):
        raise ValueError("image_bgr must be a numpy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image_bgr must have shape (H, W, 3), got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"image_bgr must use uint8 dtype, got {image.dtype}")


def _validate_hsv_range(color_range: ColorRangeHSV) -> None:
    """Validate OpenCV HSV range bounds."""
    if len(color_range.lower) != 3 or len(color_range.upper) != 3:
        raise ValueError("HSV range bounds must be 3-element tuples")

    h_low, s_low, v_low = color_range.lower
    h_high, s_high, v_high = color_range.upper
    if not (0 <= h_low <= 179 and 0 <= h_high <= 179):
        raise ValueError("HSV hue must be in [0, 179]")
    if not (0 <= s_low <= 255 and 0 <= s_high <= 255):
        raise ValueError("HSV saturation must be in [0, 255]")
    if not (0 <= v_low <= 255 and 0 <= v_high <= 255):
        raise ValueError("HSV value must be in [0, 255]")
    if (h_low > h_high) or (s_low > s_high) or (v_low > v_high):
        raise ValueError("HSV lower bound must not exceed upper bound")


# Rejestr publicznych symboli modułu używany przez image_analysis.__init__.
PUBLIC_EXPORTS: dict[str, str] = {
    "ColorRangeHSV": "ColorRangeHSV",
    "FrameTrackingMetrics": "FrameTrackingMetrics",
    "LightSpotDetection": "LightSpotDetection",
    "ScheduledStateMachine": "ScheduledStateMachine",
    "TrackedElement": "TrackedElement",
    "detect_light_spot": "detect_light_spot",
    "measure_tracking_on_mp4": "measure_tracking_on_mp4",
    "replay_mp4": "replay_mp4",
}
