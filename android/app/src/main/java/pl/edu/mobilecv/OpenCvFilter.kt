package pl.edu.mobilecv

/**
 * Enumeration of all available OpenCV processing filters.
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

    /** Detects QR codes and overlays position info. */
    QR_CODE("QR Code"),
}
