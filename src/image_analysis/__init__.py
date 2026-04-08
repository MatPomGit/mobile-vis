"""Image analysis package.

Public API for the :mod:`image_analysis` package.

This module intentionally uses lazy imports so that utilities that do not require
optional computer-vision dependencies (for example ``cv2``) remain importable in
minimal environments.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, tuple[str, str]] = {
    # April Tags
    "AprilTagDetection": (".april_tags", "AprilTagDetection"),
    "DEFAULT_APRILTAG_FAMILY": (".april_tags", "DEFAULT_APRILTAG_FAMILY"),
    # Calibration
    "CalibrationResult": (".calibration", "CalibrationResult"),
    "DEFAULT_BOARD_WIDTH": (".calibration", "DEFAULT_BOARD_WIDTH"),
    "DEFAULT_BOARD_HEIGHT": (".calibration", "DEFAULT_BOARD_HEIGHT"),
    "MIN_CALIBRATION_FRAMES": (".calibration", "MIN_CALIBRATION_FRAMES"),
    # Classification
    "CLASSIFICATION_CONFIDENCE_THRESHOLD": (
        ".classification",
        "CLASSIFICATION_CONFIDENCE_THRESHOLD",
    ),
    # Benchmarking
    "ScenarioConfig": (".benchmarking", "ScenarioConfig"),
    "VoMetrics": (".benchmarking", "VoMetrics"),
    "PlaneMetrics": (".benchmarking", "PlaneMetrics"),
    "ScenarioBenchmarkResult": (".benchmarking", "ScenarioBenchmarkResult"),
    # CCTag
    "CCTagDetection": (".cctag", "CCTagDetection"),
    # Detection
    "Detection": (".detection", "Detection"),
    "DETECTION_CONFIDENCE_THRESHOLD": (".detection", "DETECTION_CONFIDENCE_THRESHOLD"),
    "NMS_IOU_THRESHOLD": (".detection", "NMS_IOU_THRESHOLD"),
    # Effects
    "PIXELATE_BLOCK_SIZE": (".effects", "PIXELATE_BLOCK_SIZE"),
    # Holistic
    "HolisticLandmark": (".holistic", "HolisticLandmark"),
    "HolisticResult": (".holistic", "HolisticResult"),
    # Hologram
    "FaceOrientation": (".hologram", "FaceOrientation"),
    "HologramResult": (".hologram", "HologramResult"),
    "HOLOGRAM_EDGE_COLOUR": (".hologram", "HOLOGRAM_EDGE_COLOUR"),
    "HOLOGRAM_SIZE_FRACTION": (".hologram", "HOLOGRAM_SIZE_FRACTION"),
    "MAX_YAW_DEGREES": (".hologram", "MAX_YAW_DEGREES"),
    "MAX_PITCH_DEGREES": (".hologram", "MAX_PITCH_DEGREES"),
    # Iris
    "IrisLandmark": (".iris", "IrisLandmark"),
    "IrisResult": (".iris", "IrisResult"),
    # Planes
    "PlaneDetection": (".planes", "PlaneDetection"),
    "VanishingPoint": (".planes", "VanishingPoint"),
    # QR Detection
    "QRCode": (".qr_detection", "QRCode"),
    # RTMDet
    "RTMDET_CONFIDENCE_THRESHOLD": (".rtmdet", "RTMDET_CONFIDENCE_THRESHOLD"),
    "RTMDET_NMS_IOU_THRESHOLD": (".rtmdet", "RTMDET_NMS_IOU_THRESHOLD"),
    "DEFAULT_RTMDET_MODEL": (".rtmdet", "DEFAULT_RTMDET_MODEL"),
    "RTMDET_DOWNLOAD_MAX_RETRIES": (".rtmdet", "RTMDET_DOWNLOAD_MAX_RETRIES"),
    "RTMDET_DOWNLOAD_RETRY_DELAY_SECONDS": (".rtmdet", "RTMDET_DOWNLOAD_RETRY_DELAY_SECONDS"),
    "RtmDetDetection": (".rtmdet", "RtmDetDetection"),
    "RtmDetDetector": (".rtmdet", "RtmDetDetector"),
    # YOLO Detection
    "YOLO_CONFIDENCE_THRESHOLD": (".yolo", "YOLO_CONFIDENCE_THRESHOLD"),
    "YOLO_NMS_IOU_THRESHOLD": (".yolo", "YOLO_NMS_IOU_THRESHOLD"),
    "DEFAULT_YOLO_MODEL": (".yolo", "DEFAULT_YOLO_MODEL"),
    "YOLO_DOWNLOAD_MAX_RETRIES": (".yolo", "YOLO_DOWNLOAD_MAX_RETRIES"),
    "YOLO_DOWNLOAD_RETRY_DELAY_SECONDS": (".yolo", "YOLO_DOWNLOAD_RETRY_DELAY_SECONDS"),
    "YOLO_MODELS_DIR": (".yolo", "YOLO_MODELS_DIR"),
    "YoloDetection": (".yolo", "YoloDetection"),
    "YoloDetector": (".yolo", "YoloDetector"),
    # Functions from various modules
    "apply_cartoon": (".effects", "apply_cartoon"),
    "apply_emboss": (".effects", "apply_emboss"),
    "apply_invert": (".effects", "apply_invert"),
    "apply_nms": (".detection", "apply_nms"),
    "apply_pixelate": (".effects", "apply_pixelate"),
    "apply_sepia": (".effects", "apply_sepia"),
    "calibrate_camera": (".calibration", "calibrate_camera"),
    "classify_image": (".classification", "classify_image"),
    "compute_face_orientation": (".hologram", "compute_face_orientation"),
    "create_face_mesh_hologram": (".hologram", "create_face_mesh_hologram"),
    "create_face_mesh_iris": (".iris", "create_face_mesh_iris"),
    "create_holistic": (".holistic", "create_holistic"),
    "detect_april_tags": (".april_tags", "detect_april_tags"),
    "detect_cc_tags": (".cctag", "detect_cc_tags"),
    "detect_objects": (".detection", "detect_objects"),
    "detect_planes": (".planes", "detect_planes"),
    "detect_qr_codes": (".qr_detection", "detect_qr_codes"),
    "detect_rtmdet": (".rtmdet", "detect_rtmdet"),
    "detect_vanishing_points": (".planes", "detect_vanishing_points"),
    "detect_yolo": (".yolo", "detect_yolo"),
    "draw_april_tags": (".april_tags", "draw_april_tags"),
    "default_alarm_thresholds": (".benchmarking", "default_alarm_thresholds"),
    "default_benchmark_scenarios": (".benchmarking", "default_benchmark_scenarios"),
    "detect_regressions": (".benchmarking", "detect_regressions"),
    "evaluate_scenario": (".benchmarking", "evaluate_scenario"),
    "load_json_file": (".benchmarking", "load_json_file"),
    "run_benchmark_suite": (".benchmarking", "run_benchmark_suite"),
    "save_json_file": (".benchmarking", "save_json_file"),
    "draw_bounding_boxes": (".detection", "draw_bounding_boxes"),
    "draw_cc_tags": (".cctag", "draw_cc_tags"),
    "draw_chessboard_corners": (".calibration", "draw_chessboard_corners"),
    "draw_hologram_3d": (".hologram", "draw_hologram_3d"),
    "draw_holistic_results": (".holistic", "draw_holistic_results"),
    "draw_iris_results": (".iris", "draw_iris_results"),
    "draw_planes": (".planes", "draw_planes"),
    "draw_qr_codes": (".qr_detection", "draw_qr_codes"),
    "draw_rtmdet_detections": (".rtmdet", "draw_rtmdet_detections"),
    "draw_yolo_detections": (".yolo", "draw_yolo_detections"),
    "estimate_cctag_pose": (".cctag", "estimate_cctag_pose"),
    "estimate_gaze_offset": (".iris", "estimate_gaze_offset"),
    "estimate_plane_pose": (".planes", "estimate_plane_pose"),
    "evaluate_classifier": (".classification", "evaluate_classifier"),
    "export_rtmdet_to_onnx": (".rtmdet", "export_rtmdet_to_onnx"),
    "export_yolo_to_onnx": (".yolo", "export_yolo_to_onnx"),
    "export_yolo_to_torchscript": (".yolo", "export_yolo_to_torchscript"),
    "find_chessboard_corners": (".calibration", "find_chessboard_corners"),
    "fit_plane_ransac": (".planes", "fit_plane_ransac"),
    "get_project_root": (".utils", "get_project_root"),
    "list_images": (".utils", "list_images"),
    "load_classifier": (".classification", "load_classifier"),
    "load_image": (".preprocessing", "load_image"),
    "normalize_image": (".preprocessing", "normalize_image"),
    "process_hologram": (".hologram", "process_hologram"),
    "process_holistic": (".holistic", "process_holistic"),
    "process_iris": (".iris", "process_iris"),
    "render_hologram_3d": (".hologram", "render_hologram_3d"),
    "resize_image": (".preprocessing", "resize_image"),
    "rtmdet_detector": (".rtmdet", "rtmdet_detector"),
    "safe_makedirs": (".utils", "safe_makedirs"),
    "setup_logging": (".utils", "setup_logging"),
    "undistort_image": (".calibration", "undistort_image"),
    "validate_image": (".utils", "validate_image"),
    "yolo_detector": (".yolo", "yolo_detector"),
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
