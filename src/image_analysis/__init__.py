"""Image analysis package.

Public API for the :mod:`image_analysis` package.

The module uses lazy imports so that optional heavy dependencies (for
example OpenCV/MediaPipe bindings) are imported only when specific API
symbols are accessed.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "AprilTagDetection",
    "CCTagDetection",
    "CalibrationResult",
    "HolisticLandmark",
    "HolisticResult",
    "IrisLandmark",
    "IrisResult",
    "QRCode",
    "calibrate_camera",
    "classify_image",
    "create_face_mesh_iris",
    "create_holistic",
    "detect_april_tags",
    "detect_cc_tags",
    "detect_objects",
    "detect_qr_codes",
    "draw_april_tags",
    "draw_cc_tags",
    "draw_chessboard_corners",
    "draw_holistic_results",
    "draw_iris_results",
    "draw_qr_codes",
    "estimate_gaze_offset",
    "find_chessboard_corners",
    "load_image",
    "normalize_image",
    "process_holistic",
    "process_iris",
    "resize_image",
    "undistort_image",
]

_EXPORT_MAP: dict[str, tuple[str, str]] = {
    "AprilTagDetection": ("april_tags", "AprilTagDetection"),
    "detect_april_tags": ("april_tags", "detect_april_tags"),
    "draw_april_tags": ("april_tags", "draw_april_tags"),
    "CalibrationResult": ("calibration", "CalibrationResult"),
    "calibrate_camera": ("calibration", "calibrate_camera"),
    "draw_chessboard_corners": ("calibration", "draw_chessboard_corners"),
    "find_chessboard_corners": ("calibration", "find_chessboard_corners"),
    "undistort_image": ("calibration", "undistort_image"),
    "CCTagDetection": ("cctag", "CCTagDetection"),
    "detect_cc_tags": ("cctag", "detect_cc_tags"),
    "draw_cc_tags": ("cctag", "draw_cc_tags"),
    "classify_image": ("classification", "classify_image"),
    "detect_objects": ("detection", "detect_objects"),
    "HolisticLandmark": ("holistic", "HolisticLandmark"),
    "HolisticResult": ("holistic", "HolisticResult"),
    "create_holistic": ("holistic", "create_holistic"),
    "draw_holistic_results": ("holistic", "draw_holistic_results"),
    "process_holistic": ("holistic", "process_holistic"),
    "IrisLandmark": ("iris", "IrisLandmark"),
    "IrisResult": ("iris", "IrisResult"),
    "create_face_mesh_iris": ("iris", "create_face_mesh_iris"),
    "draw_iris_results": ("iris", "draw_iris_results"),
    "estimate_gaze_offset": ("iris", "estimate_gaze_offset"),
    "process_iris": ("iris", "process_iris"),
    "load_image": ("preprocessing", "load_image"),
    "normalize_image": ("preprocessing", "normalize_image"),
    "resize_image": ("preprocessing", "resize_image"),
    "QRCode": ("qr_detection", "QRCode"),
    "detect_qr_codes": ("qr_detection", "detect_qr_codes"),
    "draw_qr_codes": ("qr_detection", "draw_qr_codes"),
}


def __getattr__(name: str) -> object:
    """Lazily load symbols from submodules.

    Args:
        name: Exported symbol name.

    Returns:
        Resolved object from the mapped submodule.

    Raises:
        AttributeError: If *name* is not exported by this package.
    """
    try:
        module_name, attr_name = _EXPORT_MAP[name]
    except KeyError as exc:
        raise AttributeError(f"module 'image_analysis' has no attribute '{name}'") from exc

    module = import_module(f". {module_name}".replace(" ", ""), __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module attributes for improved introspection support."""
    return sorted(set(globals()) | set(__all__))
