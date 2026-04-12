"""Benchmark helpers for VO and plane-detection regression monitoring."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean


@dataclass(frozen=True)
class ScenarioConfig:
    """Synthetic benchmark scenario configuration.

    Attributes:
        name: Scenario identifier.
        translation_speed_mps: Simulated translation speed in meters per second.
        rotation_speed_dps: Simulated rotation speed in degrees per second.
        texture_level: Relative texture richness in range [0.0, 1.0].
        illumination_change: Relative lighting variation in range [0.0, 1.0].
        frames: Number of frames in the synthetic sequence.
        distance_meters: Simulated distance traveled during sequence.
    """

    name: str
    translation_speed_mps: float
    rotation_speed_dps: float
    texture_level: float
    illumination_change: float
    frames: int
    distance_meters: float


@dataclass(frozen=True)
class VoMetrics:
    """Visual odometry quality metrics for one scenario."""

    track_length: float
    inlier_ratio: float
    drift_per_meter: float
    reprojection_error: float


@dataclass(frozen=True)
class PlaneMetrics:
    """Plane-estimation quality metrics for one scenario."""

    overlap_region_iou: float
    normal_error_deg: float
    temporal_stability: float


@dataclass(frozen=True)
class ScenarioBenchmarkResult:
    """VO and plane metrics for a single benchmark scenario."""

    vo: VoMetrics
    planes: PlaneMetrics


def default_benchmark_scenarios() -> list[ScenarioConfig]:
    """Return required synthetic scenarios for regression benchmarking."""
    return [
        ScenarioConfig(
            name="translation_motion",
            translation_speed_mps=1.2,
            rotation_speed_dps=3.0,
            texture_level=0.8,
            illumination_change=0.2,
            frames=450,
            distance_meters=28.0,
        ),
        ScenarioConfig(
            name="rotational_motion",
            translation_speed_mps=0.3,
            rotation_speed_dps=38.0,
            texture_level=0.75,
            illumination_change=0.25,
            frames=450,
            distance_meters=8.0,
        ),
        ScenarioConfig(
            name="low_texture",
            translation_speed_mps=0.9,
            rotation_speed_dps=8.0,
            texture_level=0.22,
            illumination_change=0.3,
            frames=450,
            distance_meters=18.0,
        ),
        ScenarioConfig(
            name="variable_lighting",
            translation_speed_mps=0.8,
            rotation_speed_dps=10.0,
            texture_level=0.72,
            illumination_change=0.85,
            frames=450,
            distance_meters=16.0,
        ),
    ]


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def evaluate_scenario(scenario: ScenarioConfig) -> ScenarioBenchmarkResult:
    """Evaluate synthetic VO and plane metrics for a benchmark scenario."""
    track_length = scenario.frames * _clip(
        0.93
        + 0.06 * scenario.texture_level
        - 0.0025 * scenario.rotation_speed_dps
        - 0.03 * scenario.illumination_change,
        0.25,
        1.0,
    )

    inlier_ratio = _clip(
        0.72
        + 0.22 * scenario.texture_level
        - 0.002 * scenario.rotation_speed_dps
        - 0.12 * scenario.illumination_change,
        0.05,
        0.99,
    )

    drift_per_meter = _clip(
        0.008
        + 0.006 * (1.0 - scenario.texture_level)
        + 0.00045 * scenario.rotation_speed_dps
        + 0.004 * scenario.illumination_change,
        0.002,
        0.2,
    )

    reprojection_error = _clip(
        0.45
        + 0.75 * (1.0 - scenario.texture_level)
        + 0.015 * scenario.rotation_speed_dps
        + 0.6 * scenario.illumination_change,
        0.2,
        12.0,
    )

    overlap_region_iou = _clip(
        0.56
        + 0.32 * scenario.texture_level
        - 0.003 * scenario.rotation_speed_dps
        - 0.16 * scenario.illumination_change,
        0.05,
        0.99,
    )

    normal_error_deg = _clip(
        2.1
        + 5.5 * (1.0 - scenario.texture_level)
        + 0.09 * scenario.rotation_speed_dps
        + 2.4 * scenario.illumination_change,
        0.2,
        45.0,
    )

    temporal_stability = _clip(
        0.58
        + 0.33 * scenario.texture_level
        - 0.0025 * scenario.rotation_speed_dps
        - 0.17 * scenario.illumination_change,
        0.05,
        0.99,
    )

    return ScenarioBenchmarkResult(
        vo=VoMetrics(
            track_length=round(track_length, 3),
            inlier_ratio=round(inlier_ratio, 5),
            drift_per_meter=round(drift_per_meter, 6),
            reprojection_error=round(reprojection_error, 5),
        ),
        planes=PlaneMetrics(
            overlap_region_iou=round(overlap_region_iou, 5),
            normal_error_deg=round(normal_error_deg, 5),
            temporal_stability=round(temporal_stability, 5),
        ),
    )


def run_benchmark_suite(scenarios: list[ScenarioConfig] | None = None) -> dict[str, object]:
    """Run the synthetic benchmark suite and return machine-readable metrics."""
    active_scenarios = scenarios or default_benchmark_scenarios()
    scenario_results: dict[str, ScenarioBenchmarkResult] = {
        scenario.name: evaluate_scenario(scenario) for scenario in active_scenarios
    }

    vo_results = [result.vo for result in scenario_results.values()]
    plane_results = [result.planes for result in scenario_results.values()]

    return {
        "scenarios": [asdict(scenario) for scenario in active_scenarios],
        "results": {
            name: {
                "vo": asdict(result.vo),
                "planes": asdict(result.planes),
            }
            for name, result in scenario_results.items()
        },
        "summary": {
            "vo": {
                "track_length_mean": round(fmean(item.track_length for item in vo_results), 3),
                "inlier_ratio_mean": round(fmean(item.inlier_ratio for item in vo_results), 5),
                "drift_per_meter_mean": round(
                    fmean(item.drift_per_meter for item in vo_results),
                    6,
                ),
                "reprojection_error_mean": round(
                    fmean(item.reprojection_error for item in vo_results),
                    5,
                ),
            },
            "planes": {
                "overlap_region_iou_mean": round(
                    fmean(item.overlap_region_iou for item in plane_results),
                    5,
                ),
                "normal_error_deg_mean": round(
                    fmean(item.normal_error_deg for item in plane_results),
                    5,
                ),
                "temporal_stability_mean": round(
                    fmean(item.temporal_stability for item in plane_results),
                    5,
                ),
            },
        },
    }


def default_alarm_thresholds() -> dict[str, float]:
    """Return alarm thresholds for baseline regression checks."""
    return {
        "vo.track_length_mean.min_delta": -15.0,
        "vo.inlier_ratio_mean.min_delta": -0.03,
        "vo.drift_per_meter_mean.max_delta": 0.0025,
        "vo.reprojection_error_mean.max_delta": 0.2,
        "planes.overlap_region_iou_mean.min_delta": -0.03,
        "planes.normal_error_deg_mean.max_delta": 0.5,
        "planes.temporal_stability_mean.min_delta": -0.03,
    }


def _nested_get(mapping: dict[str, object], path: str) -> float:
    current: object = mapping
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Missing metric path: {path}")
        current = current[key]
    if not isinstance(current, (int, float)):
        raise TypeError(f"Metric path does not reference numeric value: {path}")
    return float(current)


def detect_regressions(
    current_summary: dict[str, object],
    baseline_summary: dict[str, object],
    thresholds: dict[str, float] | None = None,
) -> list[str]:
    """Compare current metrics with baseline and return regression messages."""
    active_thresholds = thresholds or default_alarm_thresholds()
    regressions: list[str] = []

    for rule_name, alarm_value in active_thresholds.items():
        metric_path, delta_kind = rule_name.rsplit(".", 1)
        current_value = _nested_get(current_summary, metric_path)
        baseline_value = _nested_get(baseline_summary, metric_path)
        delta = current_value - baseline_value

        is_regression = False
        if delta_kind == "max_delta" and delta > alarm_value:
            is_regression = True
        if delta_kind == "min_delta" and delta < alarm_value:
            is_regression = True

        if is_regression:
            regressions.append(
                f"{metric_path}: current={current_value:.6f}, "
                f"baseline={baseline_value:.6f}, delta={delta:.6f}, alarm={alarm_value:.6f}"
            )

    return regressions


def load_json_file(path: Path) -> dict[str, object]:
    """Load JSON file as dictionary."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def save_json_file(path: Path, payload: dict[str, object]) -> None:
    """Save dictionary to UTF-8 JSON file with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# Rejestr publicznych symboli modułu używany przez image_analysis.__init__.
PUBLIC_EXPORTS: dict[str, str] = {
    "PlaneMetrics": "PlaneMetrics",
    "ScenarioBenchmarkResult": "ScenarioBenchmarkResult",
    "ScenarioConfig": "ScenarioConfig",
    "VoMetrics": "VoMetrics",
    "default_alarm_thresholds": "default_alarm_thresholds",
    "default_benchmark_scenarios": "default_benchmark_scenarios",
    "detect_regressions": "detect_regressions",
    "evaluate_scenario": "evaluate_scenario",
    "load_json_file": "load_json_file",
    "run_benchmark_suite": "run_benchmark_suite",
    "save_json_file": "save_json_file",
}
