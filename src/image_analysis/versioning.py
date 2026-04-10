"""Helpers for reading project and Android application versions."""

from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path

ANDROID_VERSION_NAME_PATTERN = re.compile(r'val app_version_name by extra\("([^"]+)"\)')
ANDROID_VERSION_CODE_PATTERN = re.compile(r"val app_version_code by extra\((\d+)\)")
PACKAGE_NAME = "image-analysis"


def get_python_package_version() -> str:
    """Return the installed package version.

    Returns:
        Version string resolved from package metadata. If the package metadata is
        unavailable (for example when running from source tree without installation),
        the function returns ``"unknown"``.
    """
    try:
        return metadata.version(PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return "unknown"


def get_android_version(project_root: Path | None = None) -> tuple[str, int]:
    """Read Android app version name and code from Gradle configuration.

    Args:
        project_root: Repository root path. When ``None``, inferred from this file
            location.

    Returns:
        Tuple of ``(version_name, version_code)``.

    Raises:
        FileNotFoundError: If top-level Android Gradle file does not exist.
        ValueError: If version entries cannot be parsed.
    """
    root = project_root or Path(__file__).resolve().parents[2]
    gradle_file = root / "android" / "build.gradle.kts"
    if not gradle_file.is_file():
        raise FileNotFoundError(f"Android Gradle file not found: {gradle_file}")

    content = gradle_file.read_text(encoding="utf-8")
    name_match = ANDROID_VERSION_NAME_PATTERN.search(content)
    code_match = ANDROID_VERSION_CODE_PATTERN.search(content)
    if name_match is None or code_match is None:
        raise ValueError(
            f"Could not parse app_version_name/app_version_code from {gradle_file}",
        )

    return name_match.group(1), int(code_match.group(1))
