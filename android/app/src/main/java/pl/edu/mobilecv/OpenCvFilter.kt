package pl.edu.mobilecv

/**
 * Enumeration of all available image-processing filters and detectors.
 *
 * Each entry bundles a human-readable [displayName] shown in the UI.
 */
enum class OpenCvFilter(val displayName: String) {
    /** No processing – raw camera feed passed through unchanged. */
    ORIGINAL("Original"),

    /** Converts the frame to grayscale. */
    GRAYSCALE("Grayscale"),

    /** Detects edges using the Canny algorithm. */
    CANNY_EDGES("Canny Edges"),

    /** Applies a 15×15 Gaussian blur. */
    GAUSSIAN_BLUR("Gaussian Blur"),

    /** Binary threshold at 127 (Otsu-style midpoint). */
    THRESHOLD("Threshold"),

    /** Gradient magnitude via combined Sobel X and Y operators. */
    SOBEL("Sobel Edges"),

    /** Second-order derivative edges via the Laplacian operator. */
    LAPLACIAN("Laplacian"),

    /** Morphological dilation with a configurable rectangular kernel. */
    DILATE("Dilate"),

    /** Morphological erosion with a configurable rectangular kernel. */
    ERODE("Erode"),

    /** Morphological opening (erosion then dilation) – removes small bright spots. */
    OPEN("Open"),

    /** Morphological closing (dilation then erosion) – fills small dark holes. */
    CLOSE("Close"),

    /** Morphological gradient (dilation minus erosion) – highlights object boundaries. */
    GRADIENT("Gradient"),

    /** White top-hat transform (original minus opening) – extracts bright fine details. */
    TOP_HAT("Top-Hat"),

    /** Black top-hat transform (closing minus original) – extracts dark fine details. */
    BLACK_HAT("Black-Hat"),

    /** Detects AprilTag fiducial markers and overlays position info. */
    APRIL_TAGS("AprilTag"),

    /** Detects ArUco markers (4×4 dictionary) and overlays position info. */
    ARUCO("ArUco"),

    /** Detects QR codes and overlays position info. */
    QR_CODE("QR Code"),

    /** Detects CCTag (Circular Concentric Tag) markers and overlays ring count and position. */
    CCTAG("CCTag"),

    /** Detects chessboard corners for camera calibration. */
    CHESSBOARD_CALIBRATION("Szachownica"),

    /** Shows the undistorted camera feed using stored calibration data. */
    UNDISTORT("Korekcja"),

    // ------------------------------------------------------------------
    // MediaPipe Holistic & Iris
    // ------------------------------------------------------------------

    /**
     * Detects and tracks full-body pose landmarks (33 points) using
     * MediaPipe Pose Landmarker.
     */
    HOLISTIC_BODY("Poza ciała"),

    /**
     * Detects and tracks hand landmarks (21 points per hand) for both
     * hands using MediaPipe Hand Landmarker.
     */
    HOLISTIC_HANDS("Ręce"),

    /**
     * Detects and tracks face-mesh landmarks (468 points) using
     * MediaPipe Face Landmarker.
     */
    HOLISTIC_FACE("Twarz"),

    /**
     * Tracks iris and eye landmarks (478 refined face points including
     * 10 iris-specific landmarks) using MediaPipe Face Landmarker with
     * iris refinement enabled.
     */
    IRIS("Wzrok (Iris)"),

    /** Estimates sparse monocular visual odometry from tracked feature points. */
    VISUAL_ODOMETRY("Odometria wizyjna"),

    /** Builds a pseudo 3D point cloud view from frame-to-frame parallax. */
    POINT_CLOUD("Chmura punktów"),

    /**
     * Detects planar surfaces using vanishing-point analysis and RANSAC-based
     * line clustering.  Detected planes are highlighted with semi-transparent
     * colour overlays and normal-direction arrows.
     */
    PLANE_DETECTION("Detekcja płaszczyzn"),

    /**
     * Detects vanishing points from groups of parallel line segments and
     * visualizes them with converging-line bundles and point markers.
     */
    VANISHING_POINTS("Punkty zbieżności"),

    /** Median blur filter. */
    MEDIAN_BLUR("Median Blur"),

    /** Bilateral filter. */
    BILATERAL_FILTER("Bilateral Filter"),

    /** Box blur filter. */
    BOX_FILTER("Box Filter"),

    /** Adaptive thresholding. */
    ADAPTIVE_THRESHOLD("Adaptive Threshold"),

    /** Histogram equalization. */
    HISTOGRAM_EQUALIZATION("Histogram Equalization"),

    /** Scharr operator for edge detection. */
    SCHARR("Scharr Edges"),

    /** Prewitt operator (simulated via custom kernel). */
    PREWITT("Prewitt Edges"),

    /** Roberts cross operator (simulated via custom kernel). */
    ROBERTS("Roberts Edges"),
}

/**
 * Returns ``true`` if this filter requires the MediaPipe processing pipeline
 * rather than the OpenCV pipeline.
 */
val OpenCvFilter.isMediaPipe: Boolean
    get() = this == OpenCvFilter.HOLISTIC_BODY ||
        this == OpenCvFilter.HOLISTIC_HANDS ||
        this == OpenCvFilter.HOLISTIC_FACE ||
        this == OpenCvFilter.IRIS
