"""Tests for public package API exports and lazy loading behavior."""

from __future__ import annotations

import importlib

import pytest


def test_package_import_does_not_eagerly_import_heavy_submodules() -> None:
    """Importing image_analysis should not eagerly import optional modules."""
    package = importlib.import_module("image_analysis")

    assert package.__name__ == "image_analysis"


def test_lazy_export_for_classification_symbol() -> None:
    """A symbol exported in __all__ should be resolved lazily on first access."""
    package = importlib.import_module("image_analysis")

    symbol = package.classify_image

    assert callable(symbol)


def test_missing_symbol_raises_attribute_error() -> None:
    """Unknown attributes should raise AttributeError."""
    package = importlib.import_module("image_analysis")

    with pytest.raises(AttributeError):
        _ = package.this_symbol_does_not_exist
