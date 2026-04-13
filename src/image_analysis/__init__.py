"""Public API for :mod:`image_analysis` with selective eager imports.

This package keeps import-time side effects minimal by exposing only stable utility
symbols eagerly. Heavier computer-vision modules are resolved lazily via ``__getattr__``.
"""

from __future__ import annotations

from importlib import import_module

from .utils import (
    get_project_root,
    list_images,
    safe_makedirs,
    setup_logging,
    validate_bbox_xywh,
    validate_bbox_xyxy,
    validate_bgr_image,
    validate_gray_image,
    validate_image,
)
from .versioning import get_python_package_version

# Stabilna wersja pakietu wyznaczana na starcie bez importowania ciężkich modułów CV.
__version__ = get_python_package_version()

# Minimalny zestaw symboli dostępnych od razu po `import image_analysis`.
_EAGER_EXPORTS: dict[str, object] = {
    "__version__": __version__,
    "get_project_root": get_project_root,
    "list_images": list_images,
    "safe_makedirs": safe_makedirs,
    "setup_logging": setup_logging,
    "validate_bbox_xyxy": validate_bbox_xyxy,
    "validate_bbox_xywh": validate_bbox_xywh,
    "validate_bgr_image": validate_bgr_image,
    "validate_gray_image": validate_gray_image,
    "validate_image": validate_image,
}

# Mapa publicznych symboli ładowanych leniwie (nazwa API -> (moduł, atrybut)).
_LAZY_ATTR_EXPORTS: dict[str, tuple[str, str]] = {
    "AprilTagDetection": (".april_tags", "AprilTagDetection"),
    "CCTagDetection": (".cctag", "CCTagDetection"),
    "CLASSIFICATION_CONFIDENCE_THRESHOLD": (
        ".classification",
        "CLASSIFICATION_CONFIDENCE_THRESHOLD",
    ),
    "CalibrationResult": (".calibration", "CalibrationResult"),
    "ColorRangeHSV": (".robot_perception", "ColorRangeHSV"),
    "DEFAULT_APRILTAG_FAMILY": (".april_tags", "DEFAULT_APRILTAG_FAMILY"),
    "DEFAULT_BOARD_HEIGHT": (".calibration", "DEFAULT_BOARD_HEIGHT"),
    "DEFAULT_BOARD_WIDTH": (".calibration", "DEFAULT_BOARD_WIDTH"),
    "DEFAULT_RTMDET_MODEL": (".rtmdet", "DEFAULT_RTMDET_MODEL"),
    "DEFAULT_YOLO_MODEL": (".yolo", "DEFAULT_YOLO_MODEL"),
    "DETECTION_CONFIDENCE_THRESHOLD": (
        ".detection",
        "DETECTION_CONFIDENCE_THRESHOLD",
    ),
    "Detection": (".detection", "Detection"),
    "FaceOrientation": (".hologram", "FaceOrientation"),
    "FrameTrackingMetrics": (".robot_perception", "FrameTrackingMetrics"),
    "HOLOGRAM_EDGE_COLOUR": (".hologram", "HOLOGRAM_EDGE_COLOUR"),
    "HOLOGRAM_SIZE_FRACTION": (".hologram", "HOLOGRAM_SIZE_FRACTION"),
    "HolisticLandmark": (".holistic", "HolisticLandmark"),
    "HolisticResult": (".holistic", "HolisticResult"),
    "HologramResult": (".hologram", "HologramResult"),
    "IrisLandmark": (".iris", "IrisLandmark"),
    "IrisResult": (".iris", "IrisResult"),
    "LightSpotDetection": (".robot_perception", "LightSpotDetection"),
    "MAX_PITCH_DEGREES": (".hologram", "MAX_PITCH_DEGREES"),
    "MAX_YAW_DEGREES": (".hologram", "MAX_YAW_DEGREES"),
    "MIN_CALIBRATION_FRAMES": (".calibration", "MIN_CALIBRATION_FRAMES"),
    "NMS_IOU_THRESHOLD": (".detection", "NMS_IOU_THRESHOLD"),
    "PIXELATE_BLOCK_SIZE": (".effects", "PIXELATE_BLOCK_SIZE"),
    "PlaneDetection": (".planes", "PlaneDetection"),
    "PlaneMetrics": (".benchmarking", "PlaneMetrics"),
    "QRCode": (".qr_detection", "QRCode"),
    "RTMDET_CONFIDENCE_THRESHOLD": (".rtmdet", "RTMDET_CONFIDENCE_THRESHOLD"),
    "RTMDET_DOWNLOAD_MAX_RETRIES": (".rtmdet", "RTMDET_DOWNLOAD_MAX_RETRIES"),
    "RTMDET_DOWNLOAD_RETRY_DELAY_SECONDS": (
        ".rtmdet",
        "RTMDET_DOWNLOAD_RETRY_DELAY_SECONDS",
    ),
    "RTMDET_NMS_IOU_THRESHOLD": (".rtmdet", "RTMDET_NMS_IOU_THRESHOLD"),
    "RtmDetDetection": (".rtmdet", "RtmDetDetection"),
    "RtmDetDetector": (".rtmdet", "RtmDetDetector"),
    "ScenarioBenchmarkResult": (".benchmarking", "ScenarioBenchmarkResult"),
    "ScenarioConfig": (".benchmarking", "ScenarioConfig"),
    "ScheduledStateMachine": (".robot_perception", "ScheduledStateMachine"),
    "TrackedElement": (".robot_perception", "TrackedElement"),
    "VanishingPoint": (".planes", "VanishingPoint"),
    "VoMetrics": (".benchmarking", "VoMetrics"),
    "YOLO_CONFIDENCE_THRESHOLD": (".yolo", "YOLO_CONFIDENCE_THRESHOLD"),
    "YOLO_DOWNLOAD_MAX_RETRIES": (".yolo", "YOLO_DOWNLOAD_MAX_RETRIES"),
    "YOLO_DOWNLOAD_RETRY_DELAY_SECONDS": (".yolo", "YOLO_DOWNLOAD_RETRY_DELAY_SECONDS"),
    "YOLO_MODELS_DIR": (".yolo", "YOLO_MODELS_DIR"),
    "YOLO_NMS_IOU_THRESHOLD": (".yolo", "YOLO_NMS_IOU_THRESHOLD"),
    "YoloDetection": (".yolo", "YoloDetection"),
    "YoloDetector": (".yolo", "YoloDetector"),
    "BboxXYWH": (".types", "BboxXYWH"),
    "BboxXYXY": (".types", "BboxXYXY"),
    "BgrImage": (".types", "BgrImage"),
    "BgrImageF32": (".types", "BgrImageF32"),
    "BgrImageU8": (".types", "BgrImageU8"),
    "GrayImage": (".types", "GrayImage"),
    "GrayImageF32": (".types", "GrayImageF32"),
    "GrayImageU8": (".types", "GrayImageU8"),
    "Image": (".types", "Image"),
    "ImageF32": (".types", "ImageF32"),
    "ImageU8": (".types", "ImageU8"),
    "apply_cartoon": (".effects", "apply_cartoon"),
    "apply_emboss": (".effects", "apply_emboss"),
    "apply_invert": (".effects", "apply_invert"),
    "apply_nms": (".detection", "apply_nms"),
    "apply_pixelate": (".effects", "apply_pixelate"),
    "apply_sepia": (".effects", "apply_sepia"),
    "calibrate_camera": (".calibration", "calibrate_camera"),
    "center_crop": (".preprocessing", "center_crop"),
    "classify_image": (".classification", "classify_image"),
    "compute_face_orientation": (".hologram", "compute_face_orientation"),
    "create_face_mesh_hologram": (".hologram", "create_face_mesh_hologram"),
    "create_face_mesh_iris": (".iris", "create_face_mesh_iris"),
    "create_holistic": (".holistic", "create_holistic"),
    "default_alarm_thresholds": (".benchmarking", "default_alarm_thresholds"),
    "default_benchmark_scenarios": (".benchmarking", "default_benchmark_scenarios"),
    "default_pr_lite_benchmark_scenarios": (
        ".benchmarking",
        "default_pr_lite_benchmark_scenarios",
    ),
    "detect_april_tags": (".april_tags", "detect_april_tags"),
    "detect_cc_tags": (".cctag", "detect_cc_tags"),
    "detect_light_spot": (".robot_perception", "detect_light_spot"),
    "detect_objects": (".detection", "detect_objects"),
    "detect_planes": (".planes", "detect_planes"),
    "detect_qr_codes": (".qr_detection", "detect_qr_codes"),
    "detect_regressions": (".benchmarking", "detect_regressions"),
    "detect_rtmdet": (".rtmdet", "detect_rtmdet"),
    "detect_vanishing_points": (".planes", "detect_vanishing_points"),
    "detect_yolo": (".yolo", "detect_yolo"),
    "draw_april_tags": (".april_tags", "draw_april_tags"),
    "draw_bounding_boxes": (".detection", "draw_bounding_boxes"),
    "draw_cc_tags": (".cctag", "draw_cc_tags"),
    "draw_chessboard_corners": (".calibration", "draw_chessboard_corners"),
    "draw_holistic_results": (".holistic", "draw_holistic_results"),
    "draw_hologram_3d": (".hologram", "draw_hologram_3d"),
    "draw_iris_results": (".iris", "draw_iris_results"),
    "draw_planes": (".planes", "draw_planes"),
    "draw_qr_codes": (".qr_detection", "draw_qr_codes"),
    "draw_rtmdet_detections": (".rtmdet", "draw_rtmdet_detections"),
    "draw_yolo_detections": (".yolo", "draw_yolo_detections"),
    "estimate_cctag_pose": (".cctag", "estimate_cctag_pose"),
    "estimate_gaze_offset": (".iris", "estimate_gaze_offset"),
    "estimate_plane_pose": (".planes", "estimate_plane_pose"),
    "evaluate_classifier": (".classification", "evaluate_classifier"),
    "evaluate_scenario": (".benchmarking", "evaluate_scenario"),
    "export_rtmdet_to_onnx": (".rtmdet", "export_rtmdet_to_onnx"),
    "export_yolo_to_onnx": (".yolo", "export_yolo_to_onnx"),
    "export_yolo_to_torchscript": (".yolo", "export_yolo_to_torchscript"),
    "find_chessboard_corners": (".calibration", "find_chessboard_corners"),
    "fit_plane_ransac": (".planes", "fit_plane_ransac"),
    "load_classifier": (".classification", "load_classifier"),
    "load_image": (".preprocessing", "load_image"),
    "load_json_file": (".benchmarking", "load_json_file"),
    "load_with_retry": (".detector_common", "load_with_retry"),
    "measure_tracking_on_mp4": (".robot_perception", "measure_tracking_on_mp4"),
    "normalize_image": (".preprocessing", "normalize_image"),
    "process_holistic": (".holistic", "process_holistic"),
    "process_hologram": (".hologram", "process_hologram"),
    "process_iris": (".iris", "process_iris"),
    "render_hologram_3d": (".hologram", "render_hologram_3d"),
    "replay_mp4": (".robot_perception", "replay_mp4"),
    "resize_image": (".preprocessing", "resize_image"),
    "resize_with_aspect_ratio": (".preprocessing", "resize_with_aspect_ratio"),
    "rtmdet_detector": (".rtmdet", "rtmdet_detector"),
    "run_benchmark_suite": (".benchmarking", "run_benchmark_suite"),
    "save_json_file": (".benchmarking", "save_json_file"),
    "undistort_image": (".calibration", "undistort_image"),
    "validate_bgr_uint8_image": (".detector_common", "validate_bgr_uint8_image"),
    "yolo_detector": (".yolo", "yolo_detector"),
}

# Nazwy modułów opcjonalnych dostępne jako `image_analysis.yolo`, `image_analysis.iris`, itd.
_LAZY_MODULE_EXPORTS: dict[str, str] = {
    "yolo": ".yolo",
    "rtmdet": ".rtmdet",
    "holistic": ".holistic",
    "iris": ".iris",
}

__all__ = sorted({*_EAGER_EXPORTS, *_LAZY_ATTR_EXPORTS, *_LAZY_MODULE_EXPORTS})


def __getattr__(name: str) -> object:
    """Lazily resolve public package attributes and heavy optional modules."""
    if name in _LAZY_MODULE_EXPORTS:
        module_name = _LAZY_MODULE_EXPORTS[name]
        try:
            module = import_module(module_name, __name__)
        except ModuleNotFoundError as error:
            # Komentarz: Ułatwiamy diagnostykę brakujących backendów opcjonalnych.
            raise ModuleNotFoundError(
                f"Cannot import optional module {module_name!r} for image_analysis.{name}. "
                "Install optional dependencies first."
            ) from error
        globals()[name] = module
        return module

    if name in _LAZY_ATTR_EXPORTS:
        module_name, attr_name = _LAZY_ATTR_EXPORTS[name]
        try:
            module = import_module(module_name, __name__)
        except ModuleNotFoundError as error:
            # Komentarz: Nie zrywamy `import image_analysis`, błąd pojawia się przy użyciu symbolu.
            raise ModuleNotFoundError(
                f"Cannot import optional dependency required by {__name__}.{name}. "
                f"Failed while loading module {module_name!r}."
            ) from error
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return stable introspection results for editors and auto-completion."""
    return sorted(set(globals()) | set(__all__))
