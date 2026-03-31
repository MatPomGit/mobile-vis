"""Image analysis package.

Public API for the image_analysis module.
"""

from .april_tags import AprilTagDetection, detect_april_tags, draw_april_tags
from .calibration import (
    CalibrationResult,
    calibrate_camera,
    draw_chessboard_corners,
    find_chessboard_corners,
    undistort_image,
)
from .cctag import CCTagDetection, detect_cc_tags, draw_cc_tags
from .classification import classify_image
from .detection import detect_objects
from .holistic import (
    HolisticLandmark,
    HolisticResult,
    create_holistic,
    draw_holistic_results,
    process_holistic,
)
from .iris import (
    IrisLandmark,
    IrisResult,
    create_face_mesh_iris,
    draw_iris_results,
    estimate_gaze_offset,
    process_iris,
)
from .preprocessing import load_image, normalize_image, resize_image
from .qr_detection import QRCode, detect_qr_codes, draw_qr_codes

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
