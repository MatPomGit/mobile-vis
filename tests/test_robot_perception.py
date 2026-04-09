"""Tests for offline robot-perception replay and measurement helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pytest import MonkeyPatch

import image_analysis.robot_perception as robot_perception


def _opencv_or_skip() -> object:
    try:
        return robot_perception._require_cv2()
    except ImportError as exc:
        pytest.skip(f"OpenCV unavailable in this environment: {exc}")


class _FakeCapture:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self._frames = [
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.zeros((8, 8, 3), dtype=np.uint8),
        ]
        self._idx = 0

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._idx >= len(self._frames):
            return False, None
        frame = self._frames[self._idx]
        self._idx += 1
        return True, frame

    def get(self, prop: int) -> float:
        if prop == 1:
            return 25.0
        if prop == 2:
            return float(self._idx * 40)
        return 0.0

    def release(self) -> None:
        return None


class _ConstantTracker:
    def track(self, frame_bgr: np.ndarray) -> list[robot_perception.TrackedElement]:
        del frame_bgr
        return [
            robot_perception.TrackedElement(
                element_id="obj-1",
                label="target",
                bbox_xyxy=(10.0, 20.0, 30.0, 40.0),
                confidence=0.9,
            )
        ]


class _SimpleStateResolver:
    def resolve_stage(
        self,
        frame_index: int,
        timestamp_s: float,
        tracked_elements: list[robot_perception.TrackedElement],
    ) -> str:
        del timestamp_s, tracked_elements
        return "INIT" if frame_index == 0 else "TRACK"


def test_replay_mp4_reads_all_frames(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"dummy")
    fake_cv2 = SimpleNamespace(
        VideoCapture=_FakeCapture,
        CAP_PROP_FPS=1,
        CAP_PROP_POS_MSEC=2,
    )
    monkeypatch.setattr(robot_perception, "_require_cv2", lambda: fake_cv2)

    frames = robot_perception.replay_mp4(video)

    assert len(frames) == 2
    assert frames[0].frame_index == 0
    assert frames[1].timestamp_s > frames[0].timestamp_s


def test_measure_tracking_exports_csv(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    replay_frames = [
        robot_perception.ReplayFrame(
            frame_index=0,
            timestamp_s=0.0,
            frame_bgr=np.zeros((10, 10, 3), dtype=np.uint8),
        ),
        robot_perception.ReplayFrame(
            frame_index=1,
            timestamp_s=0.1,
            frame_bgr=np.zeros((10, 10, 3), dtype=np.uint8),
        ),
    ]

    monkeypatch.setattr(robot_perception, "replay_mp4", lambda _path: replay_frames)
    output_csv = tmp_path / "metrics" / "tracking.csv"

    rows = robot_perception.measure_tracking_on_mp4(
        video_path="unused.mp4",
        tracker=_ConstantTracker(),
        state_resolver=_SimpleStateResolver(),
        csv_output_path=output_csv,
        tracked_labels={"target"},
    )

    assert len(rows) == 2
    assert rows[0].state_stage == "INIT"
    assert rows[1].state_stage == "TRACK"
    assert output_csv.exists()
    header = output_csv.read_text(encoding="utf-8").splitlines()[0]
    assert "timestamp_s" in header


def test_detect_light_spot_returns_ellipse_center() -> None:
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2 = _opencv_or_skip()
    cv2.ellipse(
        image,
        center=(70, 50),
        axes=(12, 7),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=(255, 255, 255),
        thickness=-1,
    )

    result = robot_perception.detect_light_spot(image, min_brightness=200, min_area_px2=30.0)

    assert result is not None
    assert result.center_xy[0] == pytest.approx(70.0, abs=2.0)
    assert result.center_xy[1] == pytest.approx(50.0, abs=2.0)
    assert result.area_px2 > 100.0


def test_detect_light_spot_filters_by_requested_color() -> None:
    image = np.zeros((160, 160, 3), dtype=np.uint8)
    cv2 = _opencv_or_skip()
    cv2.ellipse(image, (45, 80), (14, 8), 0, 0, 360, (255, 0, 0), -1)  # Blue in BGR
    cv2.ellipse(image, (120, 80), (14, 8), 0, 0, 360, (0, 0, 255), -1)  # Red in BGR

    blue_range = robot_perception.ColorRangeHSV(lower=(100, 120, 120), upper=(130, 255, 255))
    red_range = robot_perception.ColorRangeHSV(lower=(0, 120, 120), upper=(10, 255, 255))

    blue_result = robot_perception.detect_light_spot(
        image,
        min_brightness=20,
        min_area_px2=30.0,
        allowed_colors_hsv=[blue_range],
    )
    red_result = robot_perception.detect_light_spot(
        image,
        min_brightness=20,
        min_area_px2=30.0,
        allowed_colors_hsv=[red_range],
    )

    assert blue_result is not None
    assert red_result is not None
    assert blue_result.center_xy[0] == pytest.approx(45.0, abs=2.0)
    assert red_result.center_xy[0] == pytest.approx(120.0, abs=2.0)
