"""Tests for lazy package exports in image_analysis."""

from __future__ import annotations

import importlib

import pytest


class TestImageAnalysisPackage:
    """Validate lazy import behavior and stable package exports."""

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

    def test_lazy_symbol_is_imported_only_after_access(self) -> None:
        """Lazy symbol should load its module only when accessed."""
        importlib.sys.modules.pop("image_analysis.classification", None)
        package = importlib.import_module("image_analysis")
        if "classify_image" in package.__dict__:
            del package.__dict__["classify_image"]

        assert "image_analysis.classification" not in importlib.sys.modules
        classify_image = package.classify_image
        assert callable(classify_image)
        assert "image_analysis.classification" in importlib.sys.modules

    def test_eager_exports_are_available_without_lazy_loading(self) -> None:
        """Stable eager utilities should be available without touching CV modules."""
        package = importlib.import_module("image_analysis")

        assert callable(package.setup_logging)
        assert callable(package.validate_image)
        assert isinstance(package.__version__, str)
