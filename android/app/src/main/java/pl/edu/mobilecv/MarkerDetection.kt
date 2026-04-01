package pl.edu.mobilecv

/**
 * Sealed hierarchy representing a single visual-marker detection result.
 *
 * Produced by [ImageProcessor] during MARKERS mode frame processing and
 * forwarded to [RosBridgeClient] for publication to ROS2 topics.
 *
 * All corner coordinates are in **image pixels** (origin at top-left corner).
 *
 * @property timestampMs System clock milliseconds at detection time.
 * @property corners     Four corner points of the detected marker [(x, y), …],
 *                       ordered: top-left, top-right, bottom-right, bottom-left.
 */
sealed class MarkerDetection {
    abstract val timestampMs: Long
    abstract val corners: List<Pair<Float, Float>>

    /**
     * AprilTag (tag36h11 family) detection.
     *
     * @param id      AprilTag numeric identifier.
     * @param corners Four corner pixel coordinates.
     */
    data class AprilTag(
        val id: Int,
        override val corners: List<Pair<Float, Float>>,
        override val timestampMs: Long = System.currentTimeMillis(),
    ) : MarkerDetection()

    /**
     * ArUco marker (4×4_50 dictionary) detection.
     *
     * @param id      ArUco marker numeric identifier.
     * @param corners Four corner pixel coordinates.
     */
    data class Aruco(
        val id: Int,
        override val corners: List<Pair<Float, Float>>,
        override val timestampMs: Long = System.currentTimeMillis(),
    ) : MarkerDetection()

    /**
     * QR code detection.
     *
     * @param text    Decoded QR code content string.
     * @param corners Four corner pixel coordinates.
     */
    data class QrCode(
        val text: String,
        override val corners: List<Pair<Float, Float>>,
        override val timestampMs: Long = System.currentTimeMillis(),
    ) : MarkerDetection()

    /**
     * CCTag (Circular Concentric Tag) detection.
     *
     * CCTag markers are concentric black-and-white rings.  The [id] equals
     * the number of concentric ring boundaries detected (2–5).
     *
     * @param id      Ring count used as the tag identifier (2–5).
     * @param center  Tag centre in image pixels as ``(x, y)``.
     * @param radius  Radius of the outermost detected ring in pixels.
     * @param corners Four corners of the bounding box in pixel coordinates,
     *                ordered: top-left, top-right, bottom-right, bottom-left.
     */
    data class CCTag(
        val id: Int,
        val center: Pair<Float, Float>,
        val radius: Float,
        override val corners: List<Pair<Float, Float>>,
        override val timestampMs: Long = System.currentTimeMillis(),
    ) : MarkerDetection()
}
