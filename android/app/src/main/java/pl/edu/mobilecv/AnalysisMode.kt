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
            OpenCvFilter.ARUCO,
            OpenCvFilter.QR_CODE,
            OpenCvFilter.CCTAG,
        )
    ),

    /**
     * Human body, face and hand tracking using MediaPipe Holistic and Iris.
     *
     * Includes full-body pose estimation, bilateral hand tracking,
     * 468-landmark face mesh, and iris/gaze tracking.
     */
    POSE(
        "Poza / Twarz",
        listOf(
            OpenCvFilter.HOLISTIC_BODY,
            OpenCvFilter.HOLISTIC_HANDS,
            OpenCvFilter.HOLISTIC_FACE,
            OpenCvFilter.IRIS,
        )
    ),

    /**
     * Visual odometry and pseudo point-cloud reconstruction from monocular video.
     */
    ODOMETRY(
        "Odometria 3D",
        listOf(
            OpenCvFilter.VISUAL_ODOMETRY,
            OpenCvFilter.POINT_CLOUD,
        )
    ),

    /**
     * Full monocular visual-odometry pipeline as implemented in practice:
     * Essential-matrix estimation, pose recovery, accumulated world-frame
     * trajectory, and sparse 3-D map built via point triangulation.
     */
    FULL_ODOMETRY_3D(
        "Pełna Odometria 3D",
        listOf(
            OpenCvFilter.FULL_ODOMETRY,
            OpenCvFilter.ODOMETRY_TRAJECTORY,
            OpenCvFilter.ODOMETRY_MAP,
        )
    ),

    /**
     * 3-D geometry analysis: plane detection, vanishing-point extraction,
     * and point-cloud visualisation.
     */
    GEOMETRY(
        "Geometria 3D",
        listOf(
            OpenCvFilter.PLANE_DETECTION,
            OpenCvFilter.VANISHING_POINTS,
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

    /**
     * Real-time object detection, instance segmentation and pose estimation
     * powered by YOLOv8-nano models running on the OpenCV DNN backend.
     *
     * Models are downloaded on demand the first time this tab is selected.
     */
    YOLO(
        "YOLO",
        listOf(
            OpenCvFilter.YOLO_DETECT,
            OpenCvFilter.YOLO_SEGMENT,
            OpenCvFilter.YOLO_POSE,
        )
    ),

    /**
     * Real-time object detection with axis-aligned and rotated bounding boxes
     * powered by RTMDet-nano (OpenMMLab) models running on the OpenCV DNN
     * backend.
     *
     * RTMDet (Real-Time instance detection) delivers state-of-the-art accuracy
     * at competitive inference speeds.  The rotated variant (RTMDet-R) supports
     * oriented bounding boxes for aerial and scene-text imagery.
     *
     * Models are downloaded on demand the first time this tab is selected.
     */
    RTMDET(
        "RTMDet",
        listOf(
            OpenCvFilter.RTMDET_DETECT,
            OpenCvFilter.RTMDET_ROTATED,
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
