package pl.edu.mobilecv

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.core.TermCriteria
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.ArucoDetector
import org.opencv.objdetect.DetectorParameters
import org.opencv.objdetect.Dictionary
import org.opencv.objdetect.Objdetect
import org.opencv.objdetect.QRCodeDetector

/**
 * Applies OpenCV image-processing filters to Android [Bitmap] frames.
 *
 * All input bitmaps must use [Bitmap.Config.ARGB_8888].  Internally the
 * bitmap is converted to a **RGBA** [Mat] (OpenCV's representation of
 * ARGB_8888), the chosen [OpenCvFilter] is applied, and the result is
 * converted back to an ARGB_8888 bitmap suitable for display.
 *
 * This class is **not thread-safe**; create one instance per thread or
 * synchronise access externally.
 */
class ImageProcessor {

    /** Reference to the shared [CameraCalibrator]; set by [MainActivity]. */
    var calibrator: CameraCalibrator? = null

    /**
     * Reference to the [MediaPipeProcessor]; set by [MainActivity] after the processor
     * has been initialised with downloaded models.
     */
    var mediaPipeProcessor: MediaPipeProcessor? = null

    /**
     * Overlay label for the frame counter shown in [CHESSBOARD_CALIBRATION] mode.
     * Set by [MainActivity] from the string resource for proper localisation.
     */
    var labelFrameCountSuffix: String = "klatek"

    /**
     * Overlay label shown when no chessboard is visible.
     * Set by [MainActivity] from the string resource for proper localisation.
     */
    var labelBoardNotFound: String = "Brak szachownicy"

    /**
     * Overlay label shown in [UNDISTORT] mode when no calibration is available.
     * Set by [MainActivity] from the string resource for proper localisation.
     */
    var labelNoCalibration: String = "Brak kalibracji"

    /** Overlay label prefix for visual odometry track statistics. */
    var labelOdometryTracks: String = "Ścieżki"

    /** Overlay label prefix for pseudo point-cloud statistics. */
    var labelPointCloud: String = "Chmura"

    /**
     * Optional callback invoked after each frame in which at least one marker
     * (AprilTag, ArUco, or QR code) was detected.
     *
     * Called on the image-analysis executor thread; the receiver must switch
     * to the main thread before updating UI.  Wired by [MainActivity] to
     * [RosBridgeClient.publishMarkers].
     */
    var onMarkersDetected: ((List<MarkerDetection>) -> Unit)? = null

    /** Enables Active Vision ROI optimisation pipeline. */
    var isActiveVisionEnabled: Boolean = false

    /**
     * Kernel half-size for all morphological operations (DILATE, ERODE, OPEN, CLOSE,
     * GRADIENT, TOP_HAT, BLACK_HAT).  The actual structuring element will be a square
     * with side length `2 * morphKernelSize + 1`.
     *
     * Valid range: 1–20 (inclusive).  Values outside the range are silently clamped.
     */
    @Volatile
    var morphKernelSize: Int = 4

    private val activeVisionOptimizer = ActiveVisionOptimizer()

    // ------------------------------------------------------------------
    // Cached detector instances – created lazily so that native OpenCV
    // methods are not called before the library is loaded.
    // ------------------------------------------------------------------

    private val aprilTagDictionary: Dictionary by lazy {
        Objdetect.getPredefinedDictionary(Objdetect.DICT_APRILTAG_36h11)
    }
    private val aprilTagDetector: ArucoDetector by lazy {
        ArucoDetector(aprilTagDictionary, DetectorParameters())
    }
    private val arucoDictionary: Dictionary by lazy {
        Objdetect.getPredefinedDictionary(Objdetect.DICT_4X4_50)
    }
    private val arucoDetector: ArucoDetector by lazy {
        ArucoDetector(arucoDictionary, DetectorParameters())
    }
    private val qrCodeDetector: QRCodeDetector by lazy { QRCodeDetector() }

    private val visualOdometryEngine = VisualOdometryEngine()

    // ------------------------------------------------------------------
    // Display constants
    // ------------------------------------------------------------------

    companion object {
        /** Half-length of the crosshair gap on each side of the centre point (pixels). */
        private const val CROSSHAIR_GAP = 30

        /** Maximum number of QR-code characters shown in the HUD label. */
        private const val MAX_QR_TEXT_DISPLAY_LENGTH = 20

        // ------------------------------------------------------------------
        // CCTag detection constants (mirror the Python cctag.py defaults)
        // ------------------------------------------------------------------

        /** Minimum area in pixels² for a contour to be a CCTag ring candidate. */
        private const val CCTAG_MIN_CONTOUR_AREA = 50.0

        /** Minimum circularity score [0, 1] for a contour to be treated as a ring. */
        private const val CCTAG_MIN_CIRCULARITY = 0.5

        /** Maximum centre-to-centre distance (fraction of outer radius) for concentricity. */
        private const val CCTAG_MAX_CENTRE_OFFSET_FRACTION = 0.25

        /** Minimum number of concentric ring boundaries for a valid CCTag. */
        private const val CCTAG_MIN_RINGS = 2

        /** Maximum number of concentric ring boundaries for a valid CCTag. */
        private const val CCTAG_MAX_RINGS = 5
    }

    /**
     * Process a single [Bitmap] frame with the given [filter].
     *
     * @param bitmap ARGB_8888 bitmap to process.
     * @param filter Filter to apply.
     * @return New ARGB_8888 bitmap with the filter applied.
     */
    fun processFrame(bitmap: Bitmap, filter: OpenCvFilter): Bitmap {
        if (filter != OpenCvFilter.VISUAL_ODOMETRY && filter != OpenCvFilter.POINT_CLOUD) {
            visualOdometryEngine.reset()
        }

        // Delegate MediaPipe filters to MediaPipeProcessor.
        if (filter.isMediaPipe) {
            return mediaPipeProcessor?.processFrame(bitmap, filter)
                ?: bitmap.copy(Bitmap.Config.ARGB_8888, false)
        }

        val src = Mat()
        // bitmapToMat converts ARGB_8888 → RGBA Mat (4 channels)
        Utils.bitmapToMat(bitmap, src)

        val baseFrame = if (isActiveVisionEnabled) {
            activeVisionOptimizer.optimize(src)
        } else {
            src.clone()
        }

        val processed: Mat = when (filter) {
            OpenCvFilter.ORIGINAL -> src.clone()
            OpenCvFilter.GRAYSCALE -> applyGrayscale(src)
            OpenCvFilter.CANNY_EDGES -> applyCanny(src)
            OpenCvFilter.GAUSSIAN_BLUR -> applyGaussianBlur(src)
            OpenCvFilter.THRESHOLD -> applyThreshold(src)
            OpenCvFilter.SOBEL -> applySobel(src)
            OpenCvFilter.LAPLACIAN -> applyLaplacian(src)
            OpenCvFilter.DILATE -> applyDilate(src)
            OpenCvFilter.ERODE -> applyErode(src)
            OpenCvFilter.OPEN -> applyMorphEx(src, Imgproc.MORPH_OPEN)
            OpenCvFilter.CLOSE -> applyMorphEx(src, Imgproc.MORPH_CLOSE)
            OpenCvFilter.GRADIENT -> applyMorphEx(src, Imgproc.MORPH_GRADIENT)
            OpenCvFilter.TOP_HAT -> applyMorphEx(src, Imgproc.MORPH_TOPHAT)
            OpenCvFilter.BLACK_HAT -> applyMorphEx(src, Imgproc.MORPH_BLACKHAT)
            OpenCvFilter.APRIL_TAGS -> applyAprilTagDetection(src)
            OpenCvFilter.ARUCO -> applyArucoDetection(src)
            OpenCvFilter.QR_CODE -> applyQrCodeDetection(src)
            OpenCvFilter.CCTAG -> applyCCTagDetection(src)
            OpenCvFilter.CHESSBOARD_CALIBRATION -> applyChessboardCalibration(src)
            OpenCvFilter.UNDISTORT -> applyUndistort(src)
            OpenCvFilter.VISUAL_ODOMETRY -> applyVisualOdometry(src)
            OpenCvFilter.POINT_CLOUD -> applyPointCloud(src)
            // MediaPipe filters are already handled above; exhaustive branch prevents warning.
            OpenCvFilter.HOLISTIC_BODY,
            OpenCvFilter.HOLISTIC_HANDS,
            OpenCvFilter.HOLISTIC_FACE,
            OpenCvFilter.IRIS -> baseFrame.clone()
        }

        val result = Bitmap.createBitmap(processed.cols(), processed.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(processed, result)

        src.release()
        baseFrame.release()
        processed.release()

        return result
    }

    // ------------------------------------------------------------------
    // Private filter implementations
    // ------------------------------------------------------------------

    /**
     * Convert the frame to grayscale and back to RGBA for display.
     *
     * Input/output: RGBA Mat (shape H × W × 4).
     */
    private fun applyGrayscale(src: Mat): Mat {
        val gray = Mat()
        val result = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.cvtColor(gray, result, Imgproc.COLOR_GRAY2RGBA)
        gray.release()
        return result
    }

    /**
     * Detect edges using the Canny algorithm.
     *
     * Pre-blurs with a 5×5 Gaussian kernel to reduce noise.
     * Thresholds: low = 50, high = 150.
     */
    private fun applyCanny(src: Mat): Mat {
        val gray = Mat()
        val blurred = Mat()
        val edges = Mat()
        val result = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
        Imgproc.Canny(blurred, edges, 50.0, 150.0)
        Imgproc.cvtColor(edges, result, Imgproc.COLOR_GRAY2RGBA)
        gray.release()
        blurred.release()
        edges.release()
        return result
    }

    /**
     * Apply a 15×15 (5x5) Gaussian blur to soften the image.
     */
    private fun applyGaussianBlur(src: Mat): Mat {
        val result = Mat()
        Imgproc.GaussianBlur(src, result, Size(5.0, 5.0), 0.0)
        return result
    }

    /**
     * Apply binary threshold at pixel value 127 (range [0, 255]).
     */
    private fun applyThreshold(src: Mat): Mat {
        val gray = Mat()
        val thresh = Mat()
        val result = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.threshold(gray, thresh, 127.0, 255.0, Imgproc.THRESH_BINARY)
        Imgproc.cvtColor(thresh, result, Imgproc.COLOR_GRAY2RGBA)
        gray.release()
        thresh.release()
        return result
    }

    /**
     * Compute the gradient magnitude via combined Sobel X and Y operators.
     *
     * Each derivative is computed at [CvType.CV_16S] depth to avoid
     * overflow, then scaled back to 8-bit with [Core.convertScaleAbs].
     * The two gradients are averaged with [Core.addWeighted].
     */
    private fun applySobel(src: Mat): Mat {
        val gray = Mat()
        val sobelX = Mat()
        val sobelY = Mat()
        val absX = Mat()
        val absY = Mat()
        val combined = Mat()
        val result = Mat()

        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.Sobel(gray, sobelX, CvType.CV_16S, 1, 0)
        Imgproc.Sobel(gray, sobelY, CvType.CV_16S, 0, 1)
        Core.convertScaleAbs(sobelX, absX)
        Core.convertScaleAbs(sobelY, absY)
        Core.addWeighted(absX, 0.5, absY, 0.5, 0.0, combined)
        Imgproc.cvtColor(combined, result, Imgproc.COLOR_GRAY2RGBA)

        gray.release()
        sobelX.release()
        sobelY.release()
        absX.release()
        absY.release()
        combined.release()
        return result
    }

    /**
     * Compute second-order derivative edges with the Laplacian operator.
     *
     * Computed at [CvType.CV_16S] depth then scaled back to 8-bit.
     */
    private fun applyLaplacian(src: Mat): Mat {
        val gray = Mat()
        val laplacian = Mat()
        val abs = Mat()
        val result = Mat()

        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.Laplacian(gray, laplacian, CvType.CV_16S)
        Core.convertScaleAbs(laplacian, abs)
        Imgproc.cvtColor(abs, result, Imgproc.COLOR_GRAY2RGBA)

        gray.release()
        laplacian.release()
        abs.release()
        return result
    }

    /**
     * Apply morphological dilation with a rectangular structuring element whose
     * side length is determined by [morphKernelSize].
     *
     * Brightens bright regions, useful for closing small dark holes.
     */
    private fun applyDilate(src: Mat): Mat {
        val result = Mat()
        val kernel = buildMorphKernel()
        Imgproc.dilate(src, result, kernel)
        kernel.release()
        return result
    }

    /**
     * Apply morphological erosion with a rectangular structuring element whose
     * side length is determined by [morphKernelSize].
     *
     * Darkens dark regions, useful for removing small bright specks.
     */
    private fun applyErode(src: Mat): Mat {
        val result = Mat()
        val kernel = buildMorphKernel()
        Imgproc.erode(src, result, kernel)
        kernel.release()
        return result
    }

    /**
     * Apply a compound morphological operation via [Imgproc.morphologyEx].
     *
     * Supports MORPH_OPEN, MORPH_CLOSE, MORPH_GRADIENT, MORPH_TOPHAT, and
     * MORPH_BLACKHAT.  The structuring element size is controlled by [morphKernelSize].
     *
     * @param src   Input RGBA Mat.
     * @param op    One of the `Imgproc.MORPH_*` constants.
     * @return      Result RGBA Mat.
     */
    private fun applyMorphEx(src: Mat, op: Int): Mat {
        val result = Mat()
        val kernel = buildMorphKernel()
        Imgproc.morphologyEx(src, result, op, kernel)
        kernel.release()
        return result
    }

    /**
     * Build a square MORPH_RECT structuring element with side length
     * `2 * clamp(morphKernelSize, 1, 20) + 1`.
     */
    private fun buildMorphKernel(): Mat {
        val half = morphKernelSize.coerceIn(1, 20)
        val side = (2 * half + 1).toDouble()
        return Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(side, side))
    }

    // ------------------------------------------------------------------
    // Marker detection: AprilTags and QR codes
    // ------------------------------------------------------------------

    /**
     * Detect AprilTag (tag36h11 family) fiducial markers in the frame.
     *
     * Draws a crosshair at the image centre, outlines each detected tag,
     * and overlays its ID and pixel offset (Δx, Δy) from the centre.
     *
     * Input/output: RGBA Mat (shape H × W × 4).
     */
    private fun applyAprilTagDetection(src: Mat): Mat {
        val result = src.clone()
        val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)

        val cx = src.cols() / 2
        val cy = src.rows() / 2
        drawCrosshair(result, cx, cy)

        val corners = ArrayList<Mat>()
        val ids = Mat()
        aprilTagDetector.detectMarkers(gray, corners, ids)

        val color = Scalar(0.0, 255.0, 255.0, 255.0) // cyan (RGBA)
        val detections = mutableListOf<MarkerDetection>()
        val timestampMs = System.currentTimeMillis()

        for (i in corners.indices) {
            val cornerMat = corners[i] // shape (1, 4), CV_32FC2
            val pts = Array(4) { j ->
                val raw = cornerMat.get(0, j)
                Point(raw[0].toDouble(), raw[1].toDouble())
            }

            // Draw tag outline
            val polygon = MatOfPoint(*pts)
            Imgproc.polylines(result, listOf(polygon), true, color, 2)

            // Draw center dot
            val markerCx = pts.map { it.x }.average()
            val markerCy = pts.map { it.y }.average()
            Imgproc.circle(result, Point(markerCx, markerCy), 6, color, -1)

            // Draw ID and offset from screen centre
            val tagId = if (i < ids.rows()) ids.get(i, 0)[0].toInt() else -1
            val dx = (markerCx - cx).toInt()
            val dy = (markerCy - cy).toInt()
            val label = "id=$tagId  dx=$dx  dy=$dy"
            Imgproc.putText(
                result, label,
                Point(pts[0].x, maxOf(pts[0].y - 10.0, 12.0)),
                Imgproc.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
            )

            detections += MarkerDetection.AprilTag(
                id = tagId,
                corners = pts.map { Pair(it.x.toFloat(), it.y.toFloat()) },
                timestampMs = timestampMs,
            )

            polygon.release()
        }

        gray.release()
        ids.release()
        corners.forEach { it.release() }

        if (detections.isNotEmpty()) {
            onMarkersDetected?.invoke(detections)
        }
        return result
    }

    /**
     * Detect ArUco markers (4×4_50 dictionary) in the frame.
     *
     * Draws a crosshair at the image centre, outlines each detected marker
     * in magenta, and overlays its ID and pixel offset (Δx, Δy) from the centre.
     *
     * Input/output: RGBA Mat (shape H × W × 4).
     */
    private fun applyArucoDetection(src: Mat): Mat {
        val result = src.clone()
        val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)

        val cx = src.cols() / 2
        val cy = src.rows() / 2
        drawCrosshair(result, cx, cy)

        val corners = ArrayList<Mat>()
        val ids = Mat()
        arucoDetector.detectMarkers(gray, corners, ids)

        val color = Scalar(255.0, 0.0, 255.0, 255.0) // magenta (RGBA)
        val detections = mutableListOf<MarkerDetection>()
        val timestampMs = System.currentTimeMillis()

        for (i in corners.indices) {
            val cornerMat = corners[i] // shape (1, 4), CV_32FC2
            val pts = Array(4) { j ->
                val raw = cornerMat.get(0, j)
                Point(raw[0].toDouble(), raw[1].toDouble())
            }

            // Draw marker outline
            val polygon = MatOfPoint(*pts)
            Imgproc.polylines(result, listOf(polygon), true, color, 2)

            // Draw center dot
            val markerCx = pts.map { it.x }.average()
            val markerCy = pts.map { it.y }.average()
            Imgproc.circle(result, Point(markerCx, markerCy), 6, color, -1)

            // Draw ID and offset from screen centre
            val markerId = if (i < ids.rows()) ids.get(i, 0)[0].toInt() else -1
            val dx = (markerCx - cx).toInt()
            val dy = (markerCy - cy).toInt()
            val label = "id=$markerId  dx=$dx  dy=$dy"
            Imgproc.putText(
                result, label,
                Point(pts[0].x, maxOf(pts[0].y - 10.0, 12.0)),
                Imgproc.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
            )

            detections += MarkerDetection.Aruco(
                id = markerId,
                corners = pts.map { Pair(it.x.toFloat(), it.y.toFloat()) },
                timestampMs = timestampMs,
            )

            polygon.release()
        }

        gray.release()
        ids.release()
        corners.forEach { it.release() }

        if (detections.isNotEmpty()) {
            onMarkersDetected?.invoke(detections)
        }
        return result
    }

    /**
     * Detect QR codes in the frame.
     *
     * Draws a crosshair at the image centre, outlines each detected QR code,
     * and overlays its decoded text and pixel offset (Δx, Δy) from the centre.
     *
     * Input/output: RGBA Mat (shape H × W × 4).
     */
    private fun applyQrCodeDetection(src: Mat): Mat {
        val result = src.clone()
        val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)

        val cx = src.cols() / 2
        val cy = src.rows() / 2
        drawCrosshair(result, cx, cy)

        val points = Mat()
        val texts = ArrayList<String>()
        val straightCodes = ArrayList<Mat>()
        val found = qrCodeDetector.detectAndDecodeMulti(gray, texts, points, straightCodes)

        val detections = mutableListOf<MarkerDetection>()
        val timestampMs = System.currentTimeMillis()

        if (found && !points.empty()) {
            val color = Scalar(0.0, 255.0, 0.0, 255.0) // green (RGBA)
            for (i in 0 until points.rows()) {
                val pts = Array(4) { j ->
                    val raw = points.get(i, j)
                    Point(raw[0].toDouble(), raw[1].toDouble())
                }

                // Draw QR code outline
                val polygon = MatOfPoint(*pts)
                Imgproc.polylines(result, listOf(polygon), true, color, 2)

                // Compute QR centre
                val qrCx = pts.map { it.x }.average()
                val qrCy = pts.map { it.y }.average()
                Imgproc.circle(result, Point(qrCx, qrCy), 6, color, -1)

                // Draw decoded text and offset from screen centre
                val text = if (i < texts.size) texts[i] else ""
                val dx = (qrCx - cx).toInt()
                val dy = (qrCy - cy).toInt()
                val shortText = if (text.length > MAX_QR_TEXT_DISPLAY_LENGTH) {
                    text.take(MAX_QR_TEXT_DISPLAY_LENGTH) + "…"
                } else {
                    text
                }
                val label = "$shortText  dx=$dx  dy=$dy"
                Imgproc.putText(
                    result, label,
                    Point(pts[0].x, maxOf(pts[0].y - 10.0, 12.0)),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
                )

                detections += MarkerDetection.QrCode(
                    text = text,
                    corners = pts.map { Pair(it.x.toFloat(), it.y.toFloat()) },
                    timestampMs = timestampMs,
                )

                polygon.release()
            }
        }

        gray.release()
        points.release()
        straightCodes.forEach { it.release() }

        if (detections.isNotEmpty()) {
            onMarkersDetected?.invoke(detections)
        }
        return result
    }

    /**
     * Detect CCTag (Circular Concentric Tag) markers in the frame.
     *
     * CCTag markers consist of concentric black-and-white rings.  The ring
     * count (2–5) is used as the tag identifier.
     *
     * Algorithm:
     * 1. Convert RGBA → grayscale.
     * 2. Apply 5×5 Gaussian blur.
     * 3. Binary threshold (Otsu method).
     * 4. Extract contours with full hierarchy (RETR_TREE).
     * 5. Filter contours by minimum area and circularity (4π·A/P²).
     * 6. Group circles whose centres are within 25 % of the outer radius.
     * 7. Groups with 2–5 rings become detections; overlaid in orange.
     *
     * Input/output: RGBA Mat (shape H × W × 4).
     */
    private fun applyCCTagDetection(src: Mat): Mat {
        val result = src.clone()
        val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)

        val cx = src.cols() / 2
        val cy = src.rows() / 2
        drawCrosshair(result, cx, cy)

        // Gaussian blur + Otsu threshold
        val blurred = Mat()
        Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
        val binary = Mat()
        Imgproc.threshold(blurred, binary, 0.0, 255.0, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU)

        // Extract contours with hierarchy for parent-child relationships
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(
            binary, contours, hierarchy,
            Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE,
        )

        // Filter contours: skip top-level (no parent), require area and circularity thresholds.
        // Each entry is Triple(cx, cy, radius).
        val circles = mutableListOf<Triple<Float, Float, Float>>()
        if (!hierarchy.empty() && contours.isNotEmpty()) {
            for (i in contours.indices) {
                // hierarchy row: [next, prev, firstChild, parent]; parent == -1 → top-level
                val parent = hierarchy.get(0, i)[3].toInt()
                if (parent == -1) continue

                val contour = contours[i]
                val area = Imgproc.contourArea(contour)
                if (area < CCTAG_MIN_CONTOUR_AREA) continue

                val contour2f = MatOfPoint2f(*contour.toArray())
                val perimeter = Imgproc.arcLength(contour2f, true)
                val circularity = if (perimeter > 0.0) {
                    4.0 * Math.PI * area / (perimeter * perimeter)
                } else {
                    0.0
                }
                if (circularity < CCTAG_MIN_CIRCULARITY) {
                    contour2f.release()
                    continue
                }

                val center = Point()
                val radiusArr = FloatArray(1)
                Imgproc.minEnclosingCircle(contour2f, center, radiusArr)
                contour2f.release()

                circles.add(Triple(center.x.toFloat(), center.y.toFloat(), radiusArr[0]))
            }
        }

        // Group concentric circles: sort descending by radius, then greedily assign inner circles.
        circles.sortByDescending { it.third }
        val used = BooleanArray(circles.size)
        val groups = mutableListOf<List<Triple<Float, Float, Float>>>()

        for (i in circles.indices) {
            if (used[i]) continue
            val outer = circles[i]
            val group = mutableListOf(outer)

            for (j in circles.indices) {
                if (i == j || used[j]) continue
                val inner = circles[j]
                if (inner.third >= outer.third) continue
                val dist = Math.hypot(
                    (inner.first - outer.first).toDouble(),
                    (inner.second - outer.second).toDouble(),
                )
                if (dist <= CCTAG_MAX_CENTRE_OFFSET_FRACTION * outer.third) {
                    group.add(inner)
                    used[j] = true
                }
            }

            if (group.size >= CCTAG_MIN_RINGS) {
                used[i] = true
                groups.add(group)
            }
        }

        // Draw detections and collect results.
        val color = Scalar(0.0, 165.0, 255.0, 255.0) // orange (RGBA)
        val detections = mutableListOf<MarkerDetection>()
        val timestampMs = System.currentTimeMillis()

        for (group in groups) {
            val ringsCount = group.size
            if (ringsCount > CCTAG_MAX_RINGS) continue

            val outer = group[0]
            val markerCx = outer.first.toDouble()
            val markerCy = outer.second.toDouble()
            val radius = maxOf(1, outer.third.toInt())

            // Draw outer circle and centre dot
            Imgproc.circle(result, Point(markerCx, markerCy), radius, color, 2)
            Imgproc.circle(result, Point(markerCx, markerCy), 6, color, -1)

            // Draw label: id=<rings> dx=<Δx> dy=<Δy>
            val dx = (markerCx - cx).toInt()
            val dy = (markerCy - cy).toInt()
            val label = "id=$ringsCount  dx=$dx  dy=$dy"
            Imgproc.putText(
                result, label,
                Point(markerCx - radius, maxOf(markerCy - radius - 8.0, 12.0)),
                Imgproc.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )

            // Build bounding-box corners (TL, TR, BR, BL) for MarkerDetection
            val x1 = maxOf(0, (markerCx - radius).toInt()).toFloat()
            val y1 = maxOf(0, (markerCy - radius).toInt()).toFloat()
            val x2 = (markerCx + radius).toInt().toFloat()
            val y2 = (markerCy + radius).toInt().toFloat()

            detections += MarkerDetection.CCTag(
                id = ringsCount,
                center = Pair(outer.first, outer.second),
                radius = outer.third,
                corners = listOf(
                    Pair(x1, y1), Pair(x2, y1), Pair(x2, y2), Pair(x1, y2),
                ),
                timestampMs = timestampMs,
            )
        }

        // Cleanup
        gray.release()
        blurred.release()
        binary.release()
        hierarchy.release()
        contours.forEach { it.release() }

        if (detections.isNotEmpty()) {
            onMarkersDetected?.invoke(detections)
        }
        return result
    }

    /**
     * Draw a centre crosshair on [mat] made of four lines extending from
     * the centre gap to the respective image edges.
     *
     * The 30-pixel gap keeps the crosshair centre unobstructed so the
     * operator can see small markers located exactly at the frame centre.
     *
     * @param mat  RGBA Mat to draw onto (modified in-place).
     * @param cx   X coordinate of the centre point.
     * @param cy   Y coordinate of the centre point.
     */
    private fun drawCrosshair(mat: Mat, cx: Int, cy: Int) {
        val color = Scalar(255.0, 255.0, 255.0, 255.0) // white (RGBA)
        val thickness = 2
        val w = mat.cols()
        val h = mat.rows()

        // Horizontal arms
        Imgproc.line(mat, Point(0.0, cy.toDouble()), Point((cx - CROSSHAIR_GAP).toDouble(), cy.toDouble()), color, thickness)
        Imgproc.line(mat, Point((cx + CROSSHAIR_GAP).toDouble(), cy.toDouble()), Point(w.toDouble(), cy.toDouble()), color, thickness)

        // Vertical arms
        Imgproc.line(mat, Point(cx.toDouble(), 0.0), Point(cx.toDouble(), (cy - CROSSHAIR_GAP).toDouble()), color, thickness)
        Imgproc.line(mat, Point(cx.toDouble(), (cy + CROSSHAIR_GAP).toDouble()), Point(cx.toDouble(), h.toDouble()), color, thickness)
    }

    private fun applyVisualOdometry(src: Mat): Mat {
        val result = src.clone()
        val state = visualOdometryEngine.updateOdometry(src)

        if (state == null) {
            Imgproc.putText(
                result,
                "$labelOdometryTracks: inicjalizacja...",
                Point(20.0, 40.0),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.7,
                Scalar(0.0, 255.0, 255.0, 255.0),
                2,
            )
            return result
        }

        Imgproc.putText(
            result,
            "$labelOdometryTracks: ${state.tracksCount} (inliers: ${state.inliersCount})",
            Point(20.0, 40.0),
            Imgproc.FONT_HERSHEY_SIMPLEX,
            0.65,
            Scalar(0.0, 255.0, 255.0, 255.0),
            2,
        )
        Imgproc.putText(
            result,
            "T: ${"%.3f".format(state.translationNorm)}  R: ${"%.2f".format(state.rotationDeg)}°",
            Point(20.0, 70.0),
            Imgproc.FONT_HERSHEY_SIMPLEX,
            0.65,
            Scalar(255.0, 255.0, 0.0, 255.0),
            2,
        )
        return result
    }

    private fun applyPointCloud(src: Mat): Mat {
        val result = src.clone()
        val cloudState = visualOdometryEngine.updatePointCloud(src)

        if (cloudState == null) {
            Imgproc.putText(
                result,
                "$labelPointCloud: inicjalizacja...",
                Point(20.0, 40.0),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.7,
                Scalar(255.0, 180.0, 0.0, 255.0),
                2,
            )
            return result
        }

        val maxPoints = minOf(cloudState.points.size, 120)
        for (i in 0 until maxPoints) {
            val point = cloudState.points[i]
            val depthRatio = ((point.y + src.rows()) / (2.0 * src.rows())).coerceIn(0.0, 1.0)
            val color = Scalar(
                255.0 * (1.0 - depthRatio),
                255.0 * depthRatio,
                200.0,
                255.0,
            )
            Imgproc.circle(result, point, 2, color, -1)
        }

        Imgproc.putText(
            result,
            "$labelPointCloud: ${cloudState.points.size} pkt  p=${"%.2f".format(cloudState.meanParallax)}",
            Point(20.0, 40.0),
            Imgproc.FONT_HERSHEY_SIMPLEX,
            0.65,
            Scalar(255.0, 255.0, 0.0, 255.0),
            2,
        )
        return result
    }

    // ------------------------------------------------------------------
    // Calibration filters
    // ------------------------------------------------------------------

    /**
     * Detect chessboard corners in the frame and visualise them.
     *
     * Passes the detected corners to [calibrator] (if set) so that the
     * user can collect frames via [CameraCalibrator.collectLastFrame].
     *
     * A green border indicates that the full pattern was found; red means
     * not found.  The current frame count is shown as an overlay.
     *
     * Input/output: RGBA Mat (shape H × W × 4).
     */
    private fun applyChessboardCalibration(src: Mat): Mat {
        val result = src.clone()
        val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)

        val cal = calibrator
        val boardWidth = cal?.boardWidth ?: CameraCalibrator.DEFAULT_BOARD_WIDTH
        val boardHeight = cal?.boardHeight ?: CameraCalibrator.DEFAULT_BOARD_HEIGHT
        val patternSize = Size(boardWidth.toDouble(), boardHeight.toDouble())

        val corners = MatOfPoint2f()
        val found = Calib3d.findChessboardCorners(
            gray,
            patternSize,
            corners,
            Calib3d.CALIB_CB_ADAPTIVE_THRESH or Calib3d.CALIB_CB_NORMALIZE_IMAGE,
        )

        if (found && !corners.empty()) {
            // Sub-pixel refinement
            val criteria = TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 30, 0.001)
            Imgproc.cornerSubPix(
                gray, corners,
                Size(11.0, 11.0), Size(-1.0, -1.0), criteria
            )

            // Draw corners (OpenCV draws directly onto a BGR Mat)
            val bgrMat = Mat()
            Imgproc.cvtColor(result, bgrMat, Imgproc.COLOR_RGBA2BGR)
            Calib3d.drawChessboardCorners(bgrMat, patternSize, corners, true)
            Imgproc.cvtColor(bgrMat, result, Imgproc.COLOR_BGR2RGBA)
            bgrMat.release()

            // Green border: board detected
            val green = Scalar(0.0, 255.0, 0.0, 255.0)
            Imgproc.rectangle(
                result,
                Point(4.0, 4.0),
                Point(src.cols() - 4.0, src.rows() - 4.0),
                green, 4
            )

            // Update calibrator with latest corners
            cal?.storeDetectedCorners(corners, Size(src.cols().toDouble(), src.rows().toDouble()))
        } else {
            // Red border: board not visible
            val red = Scalar(255.0, 0.0, 0.0, 255.0)
            Imgproc.rectangle(
                result,
                Point(4.0, 4.0),
                Point(src.cols() - 4.0, src.rows() - 4.0),
                red, 4
            )
            Imgproc.putText(
                result, labelBoardNotFound,
                Point(16.0, 48.0),
                Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0.0, 0.0, 0.0, 200.0), 4
            )
            Imgproc.putText(
                result, labelBoardNotFound,
                Point(16.0, 48.0),
                Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255.0, 0.0, 0.0, 255.0), 2
            )
            cal?.storeDetectedCorners(null, Size(src.cols().toDouble(), src.rows().toDouble()))
        }

        // Frame count overlay
        val frameCount = cal?.frameCount ?: 0
        val min = CameraCalibrator.MIN_FRAMES
        val countLabel = "$frameCount/$min $labelFrameCountSuffix"
        val labelColor = if (frameCount >= min) Scalar(0.0, 255.0, 0.0, 255.0)
        else Scalar(255.0, 255.0, 255.0, 255.0)
        Imgproc.putText(
            result, countLabel,
            Point(16.0, 48.0),
            Imgproc.FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0.0, 0.0, 0.0, 200.0), 4
        )
        Imgproc.putText(
            result, countLabel,
            Point(16.0, 48.0),
            Imgproc.FONT_HERSHEY_SIMPLEX, 1.2, labelColor, 2
        )

        gray.release()
        corners.release()
        return result
    }

    /**
     * Apply lens-distortion correction using the stored [CameraCalibrator] result.
     *
     * If no calibration has been computed yet a red "Brak kalibracji" banner
     * is drawn over the raw frame to inform the user.
     *
     * Input/output: RGBA Mat (shape H × W × 4).
     */
    private fun applyUndistort(src: Mat): Mat {
        val calData = calibrator?.calibrationResult
        if (calData == null) {
            // No calibration data – show original with informational overlay.
            val result = src.clone()
            Imgproc.putText(
                result, labelNoCalibration,
                Point(16.0, 48.0),
                Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0.0, 0.0, 0.0, 200.0), 4
            )
            Imgproc.putText(
                result, labelNoCalibration,
                Point(16.0, 48.0),
                Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0.0, 100.0, 255.0, 255.0), 2
            )
            return result
        }

        // Convert RGBA → BGR, undistort, convert back.
        val bgr = Mat()
        Imgproc.cvtColor(src, bgr, Imgproc.COLOR_RGBA2BGR)
        val undistorted = Mat()
        Calib3d.undistort(bgr, undistorted, calData.cameraMatrix, calData.distCoeffs)
        bgr.release()

        val result = Mat()
        Imgproc.cvtColor(undistorted, result, Imgproc.COLOR_BGR2RGBA)
        undistorted.release()
        return result
    }
}
