"""Tests for VO/plane benchmarking helpers."""

from __future__ import annotations

from pathlib import Path

from image_analysis.benchmarking import (
    DEFAULT_ALARM_THRESHOLDS,
    default_alarm_thresholds,
    default_benchmark_scenarios,
    default_pr_lite_benchmark_scenarios,
    detect_regressions,
    run_benchmark_suite,
    save_json_file,
)


def test_default_scenarios_cover_required_conditions() -> None:
    """Default suite should contain requested stress scenarios."""
    scenario_names = {scenario.name for scenario in default_benchmark_scenarios()}
    assert scenario_names == {
        "translation_motion",
        "rotational_motion",
        "low_texture",
        "variable_lighting",
    }


def test_suite_returns_expected_summary_shape() -> None:
    """Suite should produce VO and planes summary metrics."""
    result = run_benchmark_suite()

    assert "summary" in result
    assert "vo" in result["summary"]
    assert "planes" in result["summary"]
    assert "track_length_mean" in result["summary"]["vo"]
    assert "normal_error_deg_mean" in result["summary"]["planes"]


def test_pr_lite_scenarios_are_shorter_than_full_suite() -> None:
    """PR-lite scenarios should keep reduced runtime while covering core stressors."""
    full_scenarios = default_benchmark_scenarios()
    pr_lite_scenarios = default_pr_lite_benchmark_scenarios()

    assert len(pr_lite_scenarios) < len(full_scenarios)
    assert all(scenario.frames < 450 for scenario in pr_lite_scenarios)
    assert {scenario.name for scenario in pr_lite_scenarios} == {
        "translation_motion_pr_lite",
        "low_texture_pr_lite",
    }


def test_default_alarm_thresholds_are_defined_from_single_source() -> None:
    """Public threshold helper should expose a copy of the shared defaults."""
    from_helper = default_alarm_thresholds()

    assert from_helper == DEFAULT_ALARM_THRESHOLDS
    assert from_helper is not DEFAULT_ALARM_THRESHOLDS


def test_regression_detection_flags_threshold_breach() -> None:
    """Regression check should emit issues when summary degrades."""
    baseline = run_benchmark_suite()
    current = run_benchmark_suite()

    current["summary"]["vo"]["drift_per_meter_mean"] = (
        baseline["summary"]["vo"]["drift_per_meter_mean"] + 0.01
    )

    regressions = detect_regressions(
        current_summary=current["summary"],
        baseline_summary=baseline["summary"],
        thresholds=default_alarm_thresholds(),
    )

    assert regressions
    assert "vo.drift_per_meter_mean" in regressions[0]


def test_save_json_file_creates_parent_dirs(tmp_path: Path) -> None:
    """Saving benchmark payload should create missing directories."""
    output = tmp_path / "nested" / "metrics.json"
    save_json_file(output, run_benchmark_suite())

    assert output.exists()
