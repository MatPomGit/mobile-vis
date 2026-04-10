#!/usr/bin/env python3
"""Export models to mobile artifacts and generate compatibility manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


LOGGER = logging.getLogger("export_mobile_models")


@dataclass(frozen=True)
class ExportConfig:
    """Configuration for mobile model export."""

    model_name: str
    weights_path: Path
    output_dir: Path
    input_shape: tuple[int, int, int, int]
    dtype: str
    version: str
    class_map: dict[str, str]
    preprocess: dict[str, Any]
    postprocess: dict[str, Any]
    tflite_saved_model: Path | None


def parse_args() -> ExportConfig:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Export Torch model to TorchScript/ONNX and optional TFLite.",
    )
    parser.add_argument("--model-name", required=True, help="Base model name without extension.")
    parser.add_argument("--weights", required=True, type=Path, help="Path to PyTorch weights file.")
    parser.add_argument("--output-dir", default=Path("models"), type=Path, help="Artifact directory.")
    parser.add_argument(
        "--input-shape",
        default="1,3,640,640",
        help="Input tensor shape in N,C,H,W format.",
    )
    parser.add_argument("--dtype", default="float32", help="Input dtype metadata.")
    parser.add_argument("--version", default="1.0.0", help="Model version metadata.")
    parser.add_argument(
        "--class-map",
        default="{}",
        help='JSON object with class names, e.g. \'{"0":"person"}\'.',
    )
    parser.add_argument(
        "--preprocess",
        default='{"normalize": true, "mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}',
        help="JSON object describing preprocessing steps.",
    )
    parser.add_argument(
        "--postprocess",
        default='{"output_rank": 3, "min_last_dim": 5}',
        help="JSON object describing output validation requirements.",
    )
    parser.add_argument(
        "--tflite-saved-model",
        type=Path,
        default=None,
        help="Optional TensorFlow SavedModel directory for TFLite conversion.",
    )
    args = parser.parse_args()

    input_shape = tuple(int(value.strip()) for value in args.input_shape.split(","))
    if len(input_shape) != 4:
        raise ValueError("Input shape must contain exactly four values (N,C,H,W).")

    return ExportConfig(
        model_name=args.model_name,
        weights_path=args.weights,
        output_dir=args.output_dir,
        input_shape=input_shape,
        dtype=args.dtype,
        version=args.version,
        class_map=json.loads(args.class_map),
        preprocess=json.loads(args.preprocess),
        postprocess=json.loads(args.postprocess),
        tflite_saved_model=args.tflite_saved_model,
    )


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum for the given file."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def export_torchscript(model: torch.nn.Module, config: ExportConfig) -> Path:
    """Export model to TorchScript format."""
    artifact_path = config.output_dir / f"{config.model_name}.pt"
    example = torch.randn(config.input_shape)
    scripted_model = torch.jit.trace(model, example)
    scripted_model.save(str(artifact_path))
    return artifact_path


def export_onnx(model: torch.nn.Module, config: ExportConfig) -> Path:
    """Export model to ONNX format."""
    artifact_path = config.output_dir / f"{config.model_name}.onnx"
    example = torch.randn(config.input_shape)
    torch.onnx.export(
        model,
        example,
        str(artifact_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        do_constant_folding=True,
    )
    return artifact_path


def export_tflite(config: ExportConfig) -> Path | None:
    """Export TensorFlow SavedModel to TFLite when available."""
    if config.tflite_saved_model is None:
        LOGGER.info("Skipping TFLite export: --tflite-saved-model not provided.")
        return None
    try:
        import tensorflow as tf  # type: ignore
    except ImportError:
        LOGGER.warning("Skipping TFLite export: tensorflow is not installed.")
        return None

    artifact_path = config.output_dir / f"{config.model_name}.tflite"
    converter = tf.lite.TFLiteConverter.from_saved_model(str(config.tflite_saved_model))
    tflite_model = converter.convert()
    artifact_path.write_bytes(tflite_model)
    return artifact_path


def build_manifest_entry(
    *,
    config: ExportConfig,
    artifact_path: Path,
    artifact_format: str,
) -> dict[str, Any]:
    """Build a single manifest entry for one artifact file."""
    return {
        "model_name": artifact_path.name,
        "format": artifact_format,
        "input_shape": list(config.input_shape),
        "dtype": config.dtype,
        "class_map": config.class_map,
        "preprocess": config.preprocess,
        "postprocess": config.postprocess,
        "version": config.version,
        "sha256": compute_sha256(artifact_path),
    }


def save_manifest(output_dir: Path, entries: list[dict[str, Any]]) -> Path:
    """Save manifest JSON file with all exported artifacts."""
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"models": entries}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    """Main entrypoint for export pipeline."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Ładujemy model wag PyTorch i przełączamy do trybu inferencji przed eksportem.
    model = torch.load(config.weights_path, map_location="cpu")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("The loaded weights object must be a torch.nn.Module instance.")
    model.eval()

    manifest_entries: list[dict[str, Any]] = []

    pt_path = export_torchscript(model, config)
    manifest_entries.append(
        build_manifest_entry(config=config, artifact_path=pt_path, artifact_format="pt"),
    )
    LOGGER.info("Exported TorchScript: %s", pt_path)

    onnx_path = export_onnx(model, config)
    manifest_entries.append(
        build_manifest_entry(config=config, artifact_path=onnx_path, artifact_format="onnx"),
    )
    LOGGER.info("Exported ONNX: %s", onnx_path)

    tflite_path = export_tflite(config)
    if tflite_path is not None:
        manifest_entries.append(
            build_manifest_entry(config=config, artifact_path=tflite_path, artifact_format="tflite"),
        )
        LOGGER.info("Exported TFLite: %s", tflite_path)

    manifest_path = save_manifest(config.output_dir, manifest_entries)
    LOGGER.info("Generated manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
