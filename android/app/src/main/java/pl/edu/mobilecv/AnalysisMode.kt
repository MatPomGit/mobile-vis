package pl.edu.mobilecv

/**
 * Groups [OpenCvFilter] entries into high-level image-analysis modes
 * displayed as bottom-bar tabs in [MainActivity].
 *
 * Each mode exposes a [filters] list that drives the chip group shown
 * in the bottom bar when the tab is active.
 */
enum class AnalysisMode(val displayName: String, val filters: List<OpenCvFilter>) {

    /** Basic image-processing operations: colour conversion, blur, threshold. */
    FILTERS(
        "Filtry",
        listOf(
            OpenCvFilter.ORIGINAL,
            OpenCvFilter.GRAYSCALE,
            OpenCvFilter.GAUSSIAN_BLUR,
            OpenCvFilter.MEDIAN_BLUR,
            OpenCvFilter.BILATERAL_FILTER,
            OpenCvFilter.BOX_FILTER,
            OpenCvFilter.THRESHOLD,
            OpenCvFilter.ADAPTIVE_THRESHOLD,
            OpenCvFilter.HISTOGRAM_EQUALIZATION,
        )
    ),

    /** Edge-detection algorithms: Canny, Sobel, Laplacian. */
    EDGES(
        "Krawędzie",
        listOf(
            OpenCvFilter.CANNY_EDGES,
            OpenCvFilter.SOBEL,
            OpenCvFilter.SCHARR,
            OpenCvFilter.LAPLACIAN,
            OpenCvFilter.PREWITT,
            OpenCvFilter.ROBERTS,
        )
    ),

    /** Morphological operations: dilation, erosion, open, close, gradient, top-hat, black-hat. */
    MORPHOLOGY(
        "Morfologia",
        listOf(
            OpenCvFilter.DILATE,
            OpenCvFilter.ERODE,
            OpenCvFilter.OPEN,
            OpenCvFilter.CLOSE,
            OpenCvFilter.GRADIENT,
            OpenCvFilter.TOP_HAT,
            OpenCvFilter.BLACK_HAT,
        )
    ),

    /** Visual marker detection: AprilTags, ArUco markers, QR codes, and CCTags. */
    MARKERS(
        "Markery",
        listOf(
            OpenCvFilter.APRIL_TAGS,
            OpenCvFilter.APRIL_TAG_3D,
            OpenCvFilter.ARUCO,
            OpenCvFilter.ARUCO_3D,
            OpenCvFilter.QR_CODE,
            OpenCvFilter.QR_CODE_3D,
            OpenCvFilter.CCTAG,
            OpenCvFilter.MARKER_UKF,
        )
    ),

    /** Human body, face and hand tracking using MediaPipe Holistic and Iris. */
    POSE(
        "Poza / Twarz",
        listOf(
            OpenCvFilter.HOLISTIC_BODY,
            OpenCvFilter.HOLISTIC_HANDS,
            OpenCvFilter.HOLISTIC_FACE,
            OpenCvFilter.IRIS,
            OpenCvFilter.EYE_TRACKING,
            OpenCvFilter.HOLOGRAM_3D,
            OpenCvFilter.OBJECTRON,
            OpenCvFilter.GESTURE_RECOGNIZER,
            OpenCvFilter.FACE_DETECTION_BLAZE,
            OpenCvFilter.EMOTION_RECOGNITION,
        )
    ),

    /** Unified odometry workflow: VO, full pose, trajectory and sparse map. */
    ODOMETRY_UNIFIED(
        "Odometria",
        listOf(
            OpenCvFilter.VISUAL_ODOMETRY,
            OpenCvFilter.FULL_ODOMETRY,
            OpenCvFilter.ODOMETRY_TRAJECTORY,
            OpenCvFilter.ODOMETRY_MAP,
        )
    ),

    /** Dedicated SLAM workflows: pure VO map vs marker-aided fusion. */
    SLAM(
        "SLAM",
        listOf(
            OpenCvFilter.SLAM_POINTS,
            OpenCvFilter.SLAM_MARKERS_FUSED,
        )
    ),

    /** Camera calibration using chessboard patterns. */
    CALIBRATION(
        "Kalibracja",
        listOf(
            OpenCvFilter.CHESSBOARD_CALIBRATION,
            OpenCvFilter.UNDISTORT,
        )
    ),

    /** YOLO family models for object detection and scene understanding. */
    YOLO(
        "YOLO",
        listOf(
            OpenCvFilter.YOLO_DETECT,
            OpenCvFilter.YOLO_SEGMENT,
            OpenCvFilter.YOLO_POSE,
            OpenCvFilter.YOLO_CLASSIFY,
            OpenCvFilter.YOLO_OBB,
        )
    ),

    /**
     * Artistic and visual effects: colour inversion, sepia tone, emboss relief,
     * pixel-art pixelation, and cartoon / comic-book rendering.
     */
    EFFECTS(
        "Efekty",
        listOf(
            OpenCvFilter.INVERT,
            OpenCvFilter.SEPIA,
            OpenCvFilter.EMBOSS,
            OpenCvFilter.PIXELATE,
            OpenCvFilter.CARTOON,
        )
    ),
}
