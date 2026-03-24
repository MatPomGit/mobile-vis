package pl.edu.mobilecv

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
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
 * bitmap is converted to a **BGRA** [Mat] (OpenCV's representation of
 * ARGB_8888), the chosen [OpenCvFilter] is applied, and the result is
 * converted back to an ARGB_8888 bitmap suitable for display.
 *
 * This class is **not thread-safe**; create one instance per thread or
 * synchronise access externally.
 */
class ImageProcessor {

    // ------------------------------------------------------------------
    // Cached detector instances – created once and reused across frames
    // ------------------------------------------------------------------

    private val aprilTagDictionary: Dictionary =
        Objdetect.getPredefinedDictionary(Objdetect.DICT_APRILTAG_36h11)
    private val aprilTagDetector = ArucoDetector(aprilTagDictionary, DetectorParameters())
    private val arucoDictionary: Dictionary =
        Objdetect.getPredefinedDictionary(Objdetect.DICT_4X4_50)
    private val arucoDetector = ArucoDetector(arucoDictionary, DetectorParameters())
    private val qrCodeDetector = QRCodeDetector()

    // ------------------------------------------------------------------
    // Display constants
    // ------------------------------------------------------------------

    companion object {
        /** Half-length of the crosshair gap on each side of the centre point (pixels). */
        private const val CROSSHAIR_GAP = 30

        /** Maximum number of QR-code characters shown in the HUD label. */
        private const val MAX_QR_TEXT_DISPLAY_LENGTH = 20
    }

    /**
     * Process a single [Bitmap] frame with the given [filter].
     *
     * @param bitmap ARGB_8888 bitmap to process.
     * @param filter OpenCV filter to apply.
     * @return New ARGB_8888 bitmap with the filter applied.
     */
    fun processFrame(bitmap: Bitmap, filter: OpenCvFilter): Bitmap {
        val src = Mat()
        // bitmapToMat converts ARGB_8888 → BGRA Mat (4 channels)
        Utils.bitmapToMat(bitmap, src)

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
            OpenCvFilter.APRIL_TAGS -> applyAprilTagDetection(src)
            OpenCvFilter.ARUCO -> applyArucoDetection(src)
            OpenCvFilter.QR_CODE -> applyQrCodeDetection(src)
        }

        val result = Bitmap.createBitmap(processed.cols(), processed.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(processed, result)

        src.release()
        processed.release()

        return result
    }

    // ------------------------------------------------------------------
    // Private filter implementations
    // ------------------------------------------------------------------

    /**
     * Convert the frame to grayscale and back to BGRA for display.
     *
     * Input/output: BGRA Mat (shape H × W × 4).
     */
    private fun applyGrayscale(src: Mat): Mat {
        val gray = Mat()
        val result = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGRA2GRAY)
        Imgproc.cvtColor(gray, result, Imgproc.COLOR_GRAY2BGRA)
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
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGRA2GRAY)
        Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
        Imgproc.Canny(blurred, edges, 50.0, 150.0)
        Imgproc.cvtColor(edges, result, Imgproc.COLOR_GRAY2BGRA)
        gray.release()
        blurred.release()
        edges.release()
        return result
    }

    /**
     * Apply a 15×15 Gaussian blur to soften the image.
     */
    private fun applyGaussianBlur(src: Mat): Mat {
        val result = Mat()
        Imgproc.GaussianBlur(src, result, Size(15.0, 15.0), 0.0)
        return result
    }

    /**
     * Apply binary threshold at pixel value 127 (range [0, 255]).
     */
    private fun applyThreshold(src: Mat): Mat {
        val gray = Mat()
        val thresh = Mat()
        val result = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGRA2GRAY)
        Imgproc.threshold(gray, thresh, 127.0, 255.0, Imgproc.THRESH_BINARY)
        Imgproc.cvtColor(thresh, result, Imgproc.COLOR_GRAY2BGRA)
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

        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGRA2GRAY)
        Imgproc.Sobel(gray, sobelX, CvType.CV_16S, 1, 0)
        Imgproc.Sobel(gray, sobelY, CvType.CV_16S, 0, 1)
        Core.convertScaleAbs(sobelX, absX)
        Core.convertScaleAbs(sobelY, absY)
        Core.addWeighted(absX, 0.5, absY, 0.5, 0.0, combined)
        Imgproc.cvtColor(combined, result, Imgproc.COLOR_GRAY2BGRA)

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

        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGRA2GRAY)
        Imgproc.Laplacian(gray, laplacian, CvType.CV_16S)
        Core.convertScaleAbs(laplacian, abs)
        Imgproc.cvtColor(abs, result, Imgproc.COLOR_GRAY2BGRA)

        gray.release()
        laplacian.release()
        abs.release()
        return result
    }

    /**
     * Apply morphological dilation with a 9×9 rectangular structuring element.
     *
     * Brightens bright regions, useful for closing small dark holes.
     */
    private fun applyDilate(src: Mat): Mat {
        val result = Mat()
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(9.0, 9.0))
        Imgproc.dilate(src, result, kernel)
        kernel.release()
        return result
    }

    /**
     * Apply morphological erosion with a 9×9 rectangular structuring element.
     *
     * Darkens dark regions, useful for removing small bright specks.
     */
    private fun applyErode(src: Mat): Mat {
        val result = Mat()
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(9.0, 9.0))
        Imgproc.erode(src, result, kernel)
        kernel.release()
        return result
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
     * Input/output: BGRA Mat (shape H × W × 4).
     */
    private fun applyAprilTagDetection(src: Mat): Mat {
        val result = src.clone()
        val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGRA2GRAY)

        val cx = src.cols() / 2
        val cy = src.rows() / 2
        drawCrosshair(result, cx, cy)

        val corners = ArrayList<Mat>()
        val ids = Mat()
        aprilTagDetector.detectMarkers(gray, corners, ids)

        val color = Scalar(0.0, 255.0, 255.0, 255.0) // cyan (BGRA)
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

            polygon.release()
        }

        gray.release()
        ids.release()
        corners.forEach { it.release() }
        return result
    }

    /**
     * Detect ArUco markers (4×4_50 dictionary) in the frame.
     *
     * Draws a crosshair at the image centre, outlines each detected marker
     * in magenta, and overlays its ID and pixel offset (Δx, Δy) from the centre.
     *
     * Input/output: BGRA Mat (shape H × W × 4).
     */
    private fun applyArucoDetection(src: Mat): Mat {
        val result = src.clone()
        val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGRA2GRAY)

        val cx = src.cols() / 2
        val cy = src.rows() / 2
        drawCrosshair(result, cx, cy)

        val corners = ArrayList<Mat>()
        val ids = Mat()
        arucoDetector.detectMarkers(gray, corners, ids)

        val color = Scalar(255.0, 0.0, 255.0, 255.0) // magenta (BGRA)
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

            polygon.release()
        }

        gray.release()
        ids.release()
        corners.forEach { it.release() }
        return result
    }

    /**
     * Detect QR codes in the frame.
     *
     * Draws a crosshair at the image centre, outlines each detected QR code,
     * and overlays its decoded text and pixel offset (Δx, Δy) from the centre.
     *
     * Input/output: BGRA Mat (shape H × W × 4).
     */
    private fun applyQrCodeDetection(src: Mat): Mat {
        val result = src.clone()
        val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGRA2GRAY)

        val cx = src.cols() / 2
        val cy = src.rows() / 2
        drawCrosshair(result, cx, cy)

        val points = Mat()
        val texts = ArrayList<String>()
        val straightCodes = ArrayList<Mat>()
        val found = qrCodeDetector.detectAndDecodeMulti(gray, texts, points, straightCodes)

        if (found && !points.empty()) {
            val color = Scalar(0.0, 255.0, 0.0, 255.0) // green (BGRA)
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

                polygon.release()
            }
        }

        gray.release()
        points.release()
        straightCodes.forEach { it.release() }
        return result
    }

    /**
     * Draw a centre crosshair on [mat] made of four lines extending from
     * the centre gap to the respective image edges.
     *
     * The 30-pixel gap keeps the crosshair centre unobstructed so the
     * operator can see small markers located exactly at the frame centre.
     *
     * @param mat  BGRA Mat to draw onto (modified in-place).
     * @param cx   X coordinate of the centre point.
     * @param cy   Y coordinate of the centre point.
     */
    private fun drawCrosshair(mat: Mat, cx: Int, cy: Int) {
        val color = Scalar(255.0, 255.0, 255.0, 255.0) // white (BGRA)
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
}
