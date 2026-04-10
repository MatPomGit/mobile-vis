"""Tests for CLI version reporting helpers."""

from __future__ import annotations

from pathlib import Path

from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch

from image_analysis import cli
from image_analysis.versioning import get_android_version


def test_get_android_version_reads_gradle_file() -> None:
    """Android version reader should parse version name and code from Gradle."""
    version_name, version_code = get_android_version()

    assert version_name
    assert version_code > 0


def test_cli_version_command_prints_versions(
    capsys: CaptureFixture[str],
    monkeypatch: MonkeyPatch,
) -> None:
    """CLI should print Android and Python package version lines."""
    monkeypatch.setattr("sys.argv", ["mobilecv-version", "version"])

    exit_code = cli.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "MobileCV Android version:" in captured.out
    assert "image-analysis package version:" in captured.out


def test_cli_accepts_custom_project_root(tmp_path: Path) -> None:
    """Version parser should support alternate repository root."""
    android_dir = tmp_path / "android"
    android_dir.mkdir(parents=True)
    gradle_file = android_dir / "build.gradle.kts"
    gradle_file.write_text(
        'val app_version_name by extra("9.9.9")\nval app_version_code by extra(999)\n',
        encoding="utf-8",
    )

    parsed_name, parsed_code = get_android_version(tmp_path)

    assert parsed_name == "9.9.9"
    assert parsed_code == 999
