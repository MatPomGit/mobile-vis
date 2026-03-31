"""Tests for lazy package exports in image_analysis."""

from __future__ import annotations

import importlib

import pytest


class TestImageAnalysisPackage:
    """Validate lazy import behavior for package-level exports."""

    def test_classification_export_is_lazy_loaded(self) -> None:
        """The package should expose non-CV utilities without importing CV modules."""
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
