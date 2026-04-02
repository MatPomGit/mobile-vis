"""Image analysis package.

Public API for the :mod:`image_analysis` package.

This module intentionally uses lazy imports so that utilities that do not require
optional computer-vision dependencies (for example ``cv2``) remain importable in
minimal environments.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, tuple[str, str]] = {
    "AprilTagDetection": (".april_tags", "AprilTagDetection"),
    "YoloDetection": (".yolo", "YoloDetection"),
    "YoloDetector": (".yolo", "YoloDetector"),
    "detect_yolo": (".yolo", "detect_yolo"),
    "draw_yolo_detections": (".yolo", "draw_yolo_detections"),
    "export_yolo_to_onnx": (".yolo", "export_yolo_to_onnx"),
    "yolo_detector": (".yolo", "yolo_detector"),
    "PlaneDetection": (".planes", "PlaneDetection"),
    "VanishingPoint": (".planes", "VanishingPoint"),
    "CCTagDetection": (".cctag", "CCTagDetection"),
    "CalibrationResult": (".calibration", "CalibrationResult"),
    "HolisticLandmark": (".holistic", "HolisticLandmark"),
    "HolisticResult": (".holistic", "HolisticResult"),
    "IrisLandmark": (".iris", "IrisLandmark"),
    "IrisResult": (".iris", "IrisResult"),
    "QRCode": (".qr_detection", "QRCode"),
    "calibrate_camera": (".calibration", "calibrate_camera"),
    "classify_image": (".classification", "classify_image"),
    "create_face_mesh_iris": (".iris", "create_face_mesh_iris"),
    "create_holistic": (".holistic", "create_holistic"),
    "detect_april_tags": (".april_tags", "detect_april_tags"),
    "detect_cc_tags": (".cctag", "detect_cc_tags"),
    "estimate_cctag_pose": (".cctag", "estimate_cctag_pose"),
    "detect_objects": (".detection", "detect_objects"),
    "detect_planes": (".planes", "detect_planes"),
    "detect_qr_codes": (".qr_detection", "detect_qr_codes"),
    "detect_vanishing_points": (".planes", "detect_vanishing_points"),
    "draw_april_tags": (".april_tags", "draw_april_tags"),
    "draw_cc_tags": (".cctag", "draw_cc_tags"),
    "draw_chessboard_corners": (".calibration", "draw_chessboard_corners"),
    "draw_holistic_results": (".holistic", "draw_holistic_results"),
    "draw_iris_results": (".iris", "draw_iris_results"),
    "draw_planes": (".planes", "draw_planes"),
    "draw_qr_codes": (".qr_detection", "draw_qr_codes"),
    "estimate_gaze_offset": (".iris", "estimate_gaze_offset"),
    "estimate_plane_pose": (".planes", "estimate_plane_pose"),
    "fit_plane_ransac": (".planes", "fit_plane_ransac"),
    "find_chessboard_corners": (".calibration", "find_chessboard_corners"),
    "load_image": (".preprocessing", "load_image"),
    "normalize_image": (".preprocessing", "normalize_image"),
    "process_holistic": (".holistic", "process_holistic"),
    "process_iris": (".iris", "process_iris"),
    "resize_image": (".preprocessing", "resize_image"),
    "undistort_image": (".calibration", "undistort_image"),
}

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
