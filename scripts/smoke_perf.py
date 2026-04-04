"""Smoke performance benchmark for frame-time and FPS regression checks."""

from __future__ import annotations

import argparse
import statistics
import time

import cv2
import numpy as np

from image_analysis.detection import detect_objects
from image_analysis.preprocessing import normalize_image, resize_image


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=120, help="Number of measured iterations.")
    parser.add_argument("--warmup", type=int, default=20, help="Number of warm-up iterations.")
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Synthetic input frame width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Synthetic input frame height in pixels.",
    )
    return parser.parse_args()


def run_smoke_pipeline(frame: np.ndarray) -> None:
    """Run a lightweight smoke pipeline over a single frame."""
    resized = resize_image(frame, width=320, height=240)
    _ = normalize_image(resized)
    _ = detect_objects(resized)
    _ = cv2.Canny(resized, threshold1=50, threshold2=150)


def main() -> None:
    """Execute smoke benchmark and print plain-text metrics for CI parsing."""
    args = parse_args()
    rng = np.random.default_rng(seed=42)
    frame = rng.integers(0, 255, (args.height, args.width, 3), dtype=np.uint8)

    for _ in range(args.warmup):
        run_smoke_pipeline(frame)

    durations_ms: list[float] = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        run_smoke_pipeline(frame)
        durations_ms.append((time.perf_counter() - start) * 1000.0)

    avg_frame_time_ms = statistics.fmean(durations_ms)
    p95_frame_time_ms = np.percentile(durations_ms, 95)
    fps = 1000.0 / avg_frame_time_ms if avg_frame_time_ms > 0.0 else 0.0

    print(f"SMOKE_ITERATIONS={args.iterations}")
    print(f"FRAME_TIME_MS_AVG={avg_frame_time_ms:.3f}")
    print(f"FRAME_TIME_MS_P95={p95_frame_time_ms:.3f}")
    print(f"FPS_AVG={fps:.2f}")


if __name__ == "__main__":
    main()
