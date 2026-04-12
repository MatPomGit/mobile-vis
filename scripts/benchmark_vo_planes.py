"""Run VO/planes benchmark suite and detect regressions against a JSON baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from image_analysis.benchmarking import (
    default_alarm_thresholds,
    default_pr_lite_benchmark_scenarios,
    detect_regressions,
    load_json_file,
    run_benchmark_suite,
    save_json_file,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("benchmarks/vo_planes_baseline.json"),
        help="Path to baseline JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/vo_planes_latest.json"),
        help="Path to latest benchmark output JSON.",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Overwrite baseline with fresh metrics.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero code when regressions are detected.",
    )
    parser.add_argument(
        "--scenario-set",
        choices=("full", "pr-lite"),
        default="full",
        help="Select benchmark scenario set.",
    )
    return parser.parse_args()


def _print_summary(result: dict[str, Any]) -> None:
    summary = result["summary"]
    vo = summary["vo"]
    planes = summary["planes"]

    print("VO_TRACK_LENGTH_MEAN=" + f"{vo['track_length_mean']:.3f}")
    print("VO_INLIER_RATIO_MEAN=" + f"{vo['inlier_ratio_mean']:.5f}")
    print("VO_DRIFT_PER_METER_MEAN=" + f"{vo['drift_per_meter_mean']:.6f}")
    print("VO_REPROJECTION_ERROR_MEAN=" + f"{vo['reprojection_error_mean']:.5f}")
    print("PLANE_IOU_MEAN=" + f"{planes['overlap_region_iou_mean']:.5f}")
    print("PLANE_NORMAL_ERROR_DEG_MEAN=" + f"{planes['normal_error_deg_mean']:.5f}")
    print("PLANE_TEMPORAL_STABILITY_MEAN=" + f"{planes['temporal_stability_mean']:.5f}")


def main() -> int:
    """Execute benchmark workflow."""
    args = parse_args()
    scenarios = None
    if args.scenario_set == "pr-lite":
        # Tryb PR-lite uruchamia krótszy zestaw scen dla szybkiej walidacji pull requestów.
        scenarios = default_pr_lite_benchmark_scenarios()

    result: dict[str, Any] = run_benchmark_suite(scenarios=scenarios)
    save_json_file(args.output, result)
    print(f"Saved benchmark result to: {args.output}")
    _print_summary(result)

    if args.write_baseline:
        save_json_file(args.baseline, result)
        print(f"Baseline updated at: {args.baseline}")
        return 0

    if not args.baseline.exists():
        print(f"Baseline file not found: {args.baseline}")
        print("Run with --write-baseline to create it.")
        return 0

    baseline = load_json_file(args.baseline)
    baseline_summary = baseline.get("summary")
    if not isinstance(baseline_summary, dict):
        raise ValueError("Baseline payload must contain an object under 'summary'.")

    regressions = detect_regressions(
        current_summary=result["summary"],
        baseline_summary=baseline_summary,
        thresholds=default_alarm_thresholds(),
    )

    if regressions:
        print("REGRESSION_STATUS=detected")
        for index, regression in enumerate(regressions, start=1):
            print(f"REGRESSION_{index}={regression}")
        return 1 if args.strict else 0

    print("REGRESSION_STATUS=none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
