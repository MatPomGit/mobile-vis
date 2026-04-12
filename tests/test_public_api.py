"""Tests for public package API exports and lazy loading behavior."""

from __future__ import annotations

import importlib
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


def test_package_import_does_not_eagerly_import_heavy_submodules() -> None:
    """Importing image_analysis should not eagerly import optional modules."""
    script = """
    import importlib
    import sys

    heavy_modules = [
        "cv2",
        "mediapipe",
        "image_analysis.detection",
        "image_analysis.holistic",
        "image_analysis.iris",
    ]

    for name in heavy_modules:
        assert name not in sys.modules, (
            f"{name} unexpectedly in sys.modules before importing image_analysis"
        )

    importlib.import_module("image_analysis")

    for name in heavy_modules:
        assert name not in sys.modules, (
            f"{name} eagerly imported during image_analysis import"
        )
    """

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env=_subprocess_env_with_src_path(),
    )

    if result.returncode != 0:
        msg = (
            "Subprocess importing image_analysis eagerly imported heavy modules "
            "or failed.\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
        pytest.fail(msg)


def test_lazy_export_for_classification_symbol() -> None:
    """Accessing a symbol should trigger only its submodule import, not others."""
    script = """
    import importlib
    import sys

    importlib.import_module("image_analysis")

    assert "image_analysis.classification" not in sys.modules, (
        "image_analysis.classification was eagerly imported"
    )

    import image_analysis
    symbol = image_analysis.classify_image

    assert callable(symbol), "classify_image should be callable"
    assert "image_analysis.classification" in sys.modules, (
        "image_analysis.classification should be imported after accessing classify_image"
    )
    assert "image_analysis.detection" not in sys.modules, (
        "image_analysis.detection should not be imported when accessing classify_image"
    )
    """

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env=_subprocess_env_with_src_path(),
    )

    if result.returncode != 0:
        msg = (
            "Lazy export verification for classify_image failed.\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
        pytest.fail(msg)


def test_package_all_is_built_from_module_registries() -> None:
    """Package __all__ should be derived from all per-module PUBLIC_EXPORTS."""
    package = importlib.import_module("image_analysis")

    expected_names: set[str] = set()
    for module_name in package._EXPORT_MODULES:
        module_registry, _ = package._load_module_registry(module_name)
        expected_names.update(module_registry)

    assert set(package.__all__) == expected_names
    assert package.__all__ == sorted(package.__all__)


def test_missing_symbol_raises_attribute_error() -> None:
    """Unknown attributes should raise AttributeError."""
    package = importlib.import_module("image_analysis")

    with pytest.raises(AttributeError):
        _ = package.this_symbol_does_not_exist
