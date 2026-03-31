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

    /** Morphological dilation with a 9×9 rectangular kernel. */
    DILATE("Dilate"),

    /** Morphological erosion with a 9×9 rectangular kernel. */
    ERODE("Erode"),

    /** Detects AprilTag fiducial markers and overlays position info. */
    APRIL_TAGS("AprilTag"),

    /** Detects ArUco markers (4×4 dictionary) and overlays position info. */
    ARUCO("ArUco"),

    /** Detects QR codes and overlays position info. */
    QR_CODE("QR Code"),

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
