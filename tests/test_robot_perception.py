"""Tests for offline robot-perception replay and measurement helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pytest import MonkeyPatch

import image_analysis.robot_perception as robot_perception


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
