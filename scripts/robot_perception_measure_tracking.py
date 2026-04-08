#!/usr/bin/env python3
"""Measure tracking quality on MP4 replay and export per-frame CSV metrics."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from image_analysis.robot_perception import (
    ScheduledStateMachine,
    TrackedElement,
    measure_tracking_on_mp4,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ColorRangeTracker:
    """Simple HSV-color tracker for deterministic offline comparisons."""

    lower_hsv: tuple[int, int, int]
    upper_hsv: tuple[int, int, int]
    min_area_px2: float
    label: str

    def track(self, frame_bgr: NDArray[np.uint8]) -> list[TrackedElement]:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array(self.lower_hsv, dtype=np.uint8),
            np.array(self.upper_hsv, dtype=np.uint8),
        )

        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        elements: list[TrackedElement] = []
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < self.min_area_px2:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            frame_area = float(frame_bgr.shape[0] * frame_bgr.shape[1])
            confidence = float(min(1.0, area / max(1.0, frame_area)))
            elements.append(
                TrackedElement(
                    element_id=f"{self.label}-{idx}",
                    label=self.label,
                    bbox_xyxy=(float(x), float(y), float(x + w), float(y + h)),
                    confidence=confidence,
                )
            )
        return elements


def _load_schedule(schedule_path: Path) -> list[tuple[float, str]]:
    with schedule_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    schedule: list[tuple[float, str]] = []
    for item in payload:
        schedule.append((float(item["start_time_s"]), str(item["stage"])))
    schedule.sort(key=lambda pair: pair[0])
    return schedule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="MP4 input recording")
    parser.add_argument("--csv", type=Path, required=True, help="Output CSV path")
    parser.add_argument(
        "--schedule",
        type=Path,
        required=True,
        help="JSON file with ordered entries: [{'start_time_s':0.0,'stage':'INIT'}]",
    )
    parser.add_argument("--label", type=str, default="target", help="Tracked label name")
    parser.add_argument("--h-min", type=int, default=0)
    parser.add_argument("--s-min", type=int, default=120)
    parser.add_argument("--v-min", type=int, default=70)
    parser.add_argument("--h-max", type=int, default=10)
    parser.add_argument("--s-max", type=int, default=255)
    parser.add_argument("--v-max", type=int, default=255)
    parser.add_argument("--min-area", type=float, default=30.0)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    tracker = ColorRangeTracker(
        lower_hsv=(args.h_min, args.s_min, args.v_min),
        upper_hsv=(args.h_max, args.s_max, args.v_max),
        min_area_px2=args.min_area,
        label=args.label,
    )
    state_machine = ScheduledStateMachine(schedule=_load_schedule(args.schedule))

    rows = measure_tracking_on_mp4(
        video_path=args.video,
        tracker=tracker,
        state_resolver=state_machine,
        csv_output_path=args.csv,
        tracked_labels={args.label},
    )

    logger.info("Measurement finished with %d frame rows exported to %s", len(rows), args.csv)


if __name__ == "__main__":
    main()
