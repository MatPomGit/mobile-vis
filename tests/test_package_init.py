"""Tests for lazy package exports in image_analysis."""

from __future__ import annotations

import importlib

import pytest


class TestImageAnalysisPackage:
    """Validate lazy import behavior and export registry validation."""

    def test_classification_export_is_lazy_loaded(self) -> None:
        """The package should expose non-CV utilities without importing CV modules."""
        importlib.sys.modules.pop("image_analysis.april_tags", None)
        package = importlib.import_module("image_analysis")

        assert "classify_image" in package.__all__
        assert "image_analysis.april_tags" not in importlib.sys.modules

        classify_image = package.classify_image
        assert callable(classify_image)
        assert "image_analysis.april_tags" not in importlib.sys.modules

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        """Accessing an unknown package attribute should fail with AttributeError."""
        package = importlib.import_module("image_analysis")

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = package.not_existing_symbol

    def test_build_exports_detects_duplicate_public_names(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Duplicate public names across module registries should raise an error."""
        package = importlib.import_module("image_analysis")

        def fake_load_module_registry(module_name: str) -> tuple[dict[str, str], set[str]]:
            if module_name == "first":
                return {"duplicate_name": "first_attr"}, {"first_attr"}
            return {"duplicate_name": "second_attr"}, {"second_attr"}

        monkeypatch.setattr(package, "_EXPORT_MODULES", ("first", "second"))
        monkeypatch.setattr(package, "_load_module_registry", fake_load_module_registry)

        with pytest.raises(RuntimeError, match="Duplicate public export"):
            package._build_exports()

    def test_build_exports_validates_missing_module_attribute(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Exports should fail when registry points to missing module attribute."""
        package = importlib.import_module("image_analysis")

        def fake_load_module_registry(module_name: str) -> tuple[dict[str, str], set[str]]:
            return {"public_name": "missing_attr"}, {"another_attr"}

        monkeypatch.setattr(package, "_EXPORT_MODULES", ("single",))
        monkeypatch.setattr(package, "_load_module_registry", fake_load_module_registry)

        with pytest.raises(RuntimeError, match="attribute 'missing_attr' is not defined"):
            package._build_exports()
