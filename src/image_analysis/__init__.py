"""Image analysis package.

Public API for the :mod:`image_analysis` package.

This module intentionally uses lazy imports so that utilities that do not require
optional computer-vision dependencies (for example ``cv2``) remain importable in
minimal environments.
"""

from __future__ import annotations

import ast
from importlib import import_module
from pathlib import Path

# Kolejność modułów, z których budujemy publiczne API pakietu.
_EXPORT_MODULES: tuple[str, ...] = (
    "april_tags",
    "benchmarking",
    "calibration",
    "cctag",
    "classification",
    "detection",
    "detector_common",
    "effects",
    "holistic",
    "hologram",
    "iris",
    "planes",
    "preprocessing",
    "qr_detection",
    "robot_perception",
    "rtmdet",
    "utils",
    "yolo",
)


def _collect_defined_names(module_ast: ast.Module) -> set[str]:
    """Return top-level names defined in a module AST."""
    names: set[str] = set()

    for node in module_ast.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            continue

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
            continue

        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
            continue

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[-1])

    return names


def _load_module_registry(module_name: str) -> tuple[dict[str, str], set[str]]:
    """Load and validate ``PUBLIC_EXPORTS`` from one module source file."""
    module_path = Path(__file__).resolve().parent / f"{module_name}.py"
    module_ast = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))

    registry_value: object | None = None
    for node in module_ast.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "PUBLIC_EXPORTS":
                    registry_value = ast.literal_eval(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "PUBLIC_EXPORTS"
            and node.value is not None
        ):
            registry_value = ast.literal_eval(node.value)

    if registry_value is None:
        raise RuntimeError(f"Module image_analysis.{module_name} does not define PUBLIC_EXPORTS")
    if not isinstance(registry_value, dict):
        raise RuntimeError(
            f"Module image_analysis.{module_name} has invalid PUBLIC_EXPORTS (expected dict)"
        )

    registry: dict[str, str] = {}
    for public_name, attr_name in registry_value.items():
        if not isinstance(public_name, str) or not isinstance(attr_name, str):
            raise RuntimeError(
                "PUBLIC_EXPORTS entries must map str -> str in "
                f"image_analysis.{module_name}"
            )
        registry[public_name] = attr_name

    return registry, _collect_defined_names(module_ast)


def _build_exports() -> dict[str, tuple[str, str]]:
    """Build package exports by merging per-module ``PUBLIC_EXPORTS`` registries."""
    merged: dict[str, tuple[str, str]] = {}

    for module_name in _EXPORT_MODULES:
        registry, defined_names = _load_module_registry(module_name)

        for public_name, attr_name in registry.items():
            if public_name in merged:
                existing_module, existing_attr = merged[public_name]
                raise RuntimeError(
                    "Duplicate public export "
                    f"{public_name!r} found in image_analysis.{module_name} "
                    f"and image_analysis.{existing_module.lstrip('.')}"
                    f" (existing attr: {existing_attr!r})"
                )

            if attr_name not in defined_names:
                raise RuntimeError(
                    f"Invalid export in image_analysis.{module_name}: "
                    f"attribute {attr_name!r} is not defined in module source"
                )

            merged[public_name] = (f".{module_name}", attr_name)

    return merged


_EXPORTS: dict[str, tuple[str, str]] = _build_exports()

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> object:
    """Lazily load public package attributes.

    Args:
        name: Name requested from the package namespace.

    Returns:
        Exported object referenced by ``name``.

    Raises:
        AttributeError: If ``name`` is not part of the package public API.
    """
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
