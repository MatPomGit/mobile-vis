"""Tests for stable public API surface and import behavior."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SRC_PATH = Path(__file__).resolve().parents[1] / "src"


def _subprocess_env_with_src_path() -> dict[str, str]:
    """Build subprocess environment that includes local src/ on PYTHONPATH."""
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{SRC_PATH}:{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = str(SRC_PATH)
    return env


def _run_python_script(script: str) -> subprocess.CompletedProcess[str]:
    """Run helper script in isolated subprocess and return process result."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env=_subprocess_env_with_src_path(),
    )


def test_key_public_symbols_are_available() -> None:
    """Package should expose key stable symbols from eager and lazy API sets."""
    script = """
    import importlib

    package = importlib.import_module("image_analysis")

    for required_name in [
        "__version__",
        "setup_logging",
        "validate_image",
        "validate_bbox_xywh",
        "BgrImage",
        "GrayImage",
        "classify_image",
        "detect_objects",
        "load_image",
        "yolo",
        "rtmdet",
        "holistic",
        "iris",
    ]:
        assert required_name in package.__all__, f"Missing {required_name} in __all__"

    assert isinstance(package.__version__, str)
    assert callable(package.setup_logging)
    assert callable(package.validate_image)
    assert callable(package.validate_bbox_xywh)
    assert callable(package.classify_image)
    # Komentarz: Nie dotykamy symboli zależnych od cv2, by uniknąć niestabilności CI.
    assert "image_analysis.detection" not in importlib.sys.modules
    assert "image_analysis.preprocessing" not in importlib.sys.modules

    # Komentarz: Sprawdzamy kontrakt API bez wymuszania importu ciężkich modułów.
    for heavy_module in ("image_analysis.yolo", "image_analysis.rtmdet"):
        assert heavy_module not in importlib.sys.modules
    """

    result = _run_python_script(script)

    if result.returncode != 0:
        msg = (
            "Public API symbols verification failed.\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
        pytest.fail(msg)


def test_import_image_analysis_without_optional_dependencies() -> None:
    """Bare import should work even when optional heavy libs are unavailable."""
    script = """
    import importlib.abc
    import sys


    class BlockOptionalDepsFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            blocked_roots = {"ultralytics", "mmdet", "mediapipe"}
            if fullname.split(".")[0] in blocked_roots:
                raise ModuleNotFoundError(
                    f"Optional dependency blocked in test subprocess: {fullname}"
                )
            return None


    sys.meta_path.insert(0, BlockOptionalDepsFinder())

    import image_analysis

    assert image_analysis.__name__ == "image_analysis"
    assert callable(image_analysis.validate_image)

    for module_name in (
        "image_analysis.yolo",
        "image_analysis.rtmdet",
        "image_analysis.holistic",
        "image_analysis.iris",
    ):
        assert module_name not in sys.modules, f"{module_name} should remain lazy"
    """

    result = _run_python_script(script)

    if result.returncode != 0:
        msg = (
            "Import behavior test failed when optional dependencies were blocked.\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
        pytest.fail(msg)


def test_accessing_lazy_backend_without_dependency_raises_module_error() -> None:
    """Lazy backend access should fail only at first use with clear ModuleNotFoundError."""
    script = """
    import importlib.abc
    import sys


    class BlockOptionalDepsFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in {"ultralytics", "cv2"}:
                raise ModuleNotFoundError(f"Blocked optional dependency: {fullname}")
            return None


    sys.meta_path.insert(0, BlockOptionalDepsFinder())

    import image_analysis

    try:
        _ = image_analysis.YoloDetector
    except ModuleNotFoundError as exc:
        assert "optional dependency" in str(exc).lower()
    else:
        raise AssertionError("Expected ModuleNotFoundError for missing optional backend")
    """

    result = _run_python_script(script)

    if result.returncode != 0:
        msg = (
            "Missing optional backend behavior failed.\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
        pytest.fail(msg)


def test_missing_symbol_raises_attribute_error() -> None:
    """Unknown attributes should raise AttributeError."""
    script = """
    import image_analysis

    try:
        _ = image_analysis.this_symbol_does_not_exist
    except AttributeError:
        pass
    else:
        raise AssertionError("Missing symbol should raise AttributeError")
    """

    result = _run_python_script(script)

    if result.returncode != 0:
        msg = (
            "Missing symbol behavior test failed.\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
        pytest.fail(msg)
