package pl.edu.mobilecv

/**
 * Groups [OpenCvFilter] entries into high-level image-analysis modes
 * displayed as top-bar tabs in [MainActivity].
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
            OpenCvFilter.THRESHOLD,
        )
    ),

    /** Edge-detection algorithms: Canny, Sobel, Laplacian. */
    EDGES(
        "Krawędzie",
        listOf(
            OpenCvFilter.CANNY_EDGES,
            OpenCvFilter.SOBEL,
            OpenCvFilter.LAPLACIAN,
        )
    ),

    /** Morphological operations: dilation and erosion. */
    MORPHOLOGY(
        "Morfologia",
        listOf(
            OpenCvFilter.DILATE,
            OpenCvFilter.ERODE,
        )
    ),

    /** Visual marker detection: AprilTags, ArUco markers, and QR codes. */
    MARKERS(
        "Markery",
        listOf(
            OpenCvFilter.APRIL_TAGS,
            OpenCvFilter.ARUCO,
            OpenCvFilter.QR_CODE,
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
}
