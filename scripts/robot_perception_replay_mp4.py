#!/usr/bin/env python3
"""Replay MP4 data for robot-perception pipelines instead of live camera frames."""

from __future__ import annotations

import argparse
import importlib
import logging
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from image_analysis.robot_perception import replay_mp4

logger = logging.getLogger(__name__)

FrameProcessor = Callable[[NDArray[np.uint8], float, int], NDArray[np.uint8]]


def _parse_processor(reference: str) -> FrameProcessor:
    module_name, func_name = reference.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    processor = getattr(module, func_name)
    if not callable(processor):
        raise TypeError(f"Processor is not callable: {reference}")
    return processor


def _identity_processor(
    frame_bgr: NDArray[np.uint8],
    timestamp_s: float,
    frame_index: int,
) -> NDArray[np.uint8]:
    del timestamp_s, frame_index
    return frame_bgr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to source MP4 recording")
    parser.add_argument(
        "--processor",
        type=str,
        default="",
        help="Optional callback in 'module:function' format. "
        "Signature: processor(frame_bgr, timestamp_s, frame_index) -> frame_bgr",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=None,
        help="Optional path where processed MP4 will be saved",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Optional max number of frames")
    parser.add_argument("--log-level", type=str, default="INFO", help="Python logging level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    processor = _parse_processor(args.processor) if args.processor else _identity_processor

    frames = replay_mp4(args.video)
    if args.max_frames > 0:
        frames = frames[: args.max_frames]

    writer: cv2.VideoWriter | None = None
    if args.output_video and frames:
        height, width = frames[0].frame_bgr.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.output_video), fourcc, 30.0, (width, height))

    for replay_frame in frames:
        processed = processor(
            replay_frame.frame_bgr,
            replay_frame.timestamp_s,
            replay_frame.frame_index,
        )

        if writer is not None:
            writer.write(processed)

    if writer is not None:
        writer.release()

    logger.info("Processed %d frames from %s", len(frames), args.video)


if __name__ == "__main__":
    main()
