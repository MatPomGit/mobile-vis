package pl.edu.mobilecv

import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.core.TermCriteria
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.ArucoDetector
import org.opencv.objdetect.DetectorParameters
import org.opencv.objdetect.Objdetect
import org.opencv.objdetect.QRCodeDetector
import androidx.core.graphics.createBitmap
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot

/**
 * Applies OpenCV image-processing filters to Android [Bitmap] frames.
 */
class ImageProcessor {

    var calibrator: CameraCalibrator? = null
    var mediaPipeProcessor: MediaPipeProcessor? = null
    var yoloProcessor: YoloProcessor? = null

    var labelFrameCountSuffix: String = "klatek"
    var labelBoardNotFound: String = "Brak szachownicy"
    var labelNoCalibration: String = "Brak kalibracji"
    var labelOdometryTracks: String = "Ścieżki"
    var labelPointCloud: String = "Chmura"
    var labelVoMaxFeaturesDesc: String = "Max features (more = accurate, slower)"
    var labelVoMinParallaxDesc: String = "Min parallax [px] (motion threshold)"
    var labelVoColorDepthDesc: String = "Color = depth (bright=near, dark=far)"
    var labelNoPlanes: String = "Brak płaszczyzn"
    var labelNoVanishingPoints: String = "Brak punktów zbieżności"
    var labelNoLines: String = "Brak linii w scenie"
    var labelPlanes: String = "Płaszczyzny"
    var labelLines: String = "Linie"
    var labelGroups: String = "Grupy"
    var labelGeometryError: String = "Błąd geometrii"
    var labelVpError: String = "Błąd VP"

    var onMarkersDetected: ((List<MarkerDetection>) -> Unit)? = null
    var isActiveVisionEnabled: Boolean = false
    var isActiveVisionVisualizationEnabled: Boolean = false

    val lastPointCloud: VisualOdometryEngine.PointCloudState?
        get() = visualOdometryEngine.lastPointCloud

    @Volatile
    var morphKernelSize: Int = 4

    @Volatile
    var voMaxFeatures: Int = 300
        set(value) { field = value; visualOdometryEngine.maxFeatures = value }

    @Volatile
    var voMinParallax: Double = 1.0
        set(value) { field = value; visualOdometryEngine.minParallax = value }

    @Volatile
    var isVoMeshEnabled: Boolean = false
        set(value) { field = value; visualOdometryEngine.isMeshEnabled = value }

    private val activeVisionOptimizer = ActiveVisionOptimizer()
    private val visualOdometryEngine = VisualOdometryEngine().also {
        it.maxFeatures = voMaxFeatures
        it.minParallax = voMinParallax
        it.isMeshEnabled = isVoMeshEnabled
    }

    private val detectorParameters by lazy {
        DetectorParameters().apply {
            set_adaptiveThreshWinSizeMin(3)
            set_adaptiveThreshWinSizeMax(23)
            set_adaptiveThreshWinSizeStep(10)
        }
    }

    /** Cached 3×3 rectangular kernel for edge dilation in plane detection. */
    private val dilationKernel3x3 by lazy {
        Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
    }

    private val aprilTagDetector by lazy {
        ArucoDetector(Objdetect.getPredefinedDictionary(Objdetect.DICT_APRILTAG_36h11), detectorParameters)
    }
    private val arucoDetector by lazy {
        ArucoDetector(Objdetect.getPredefinedDictionary(Objdetect.DICT_4X4_50), detectorParameters)
    }
    private val qrCodeDetector by lazy { QRCodeDetector() }

    fun processFrame(bitmap: Bitmap, filter: OpenCvFilter): Bitmap {
        if (filter != OpenCvFilter.VISUAL_ODOMETRY && filter != OpenCvFilter.POINT_CLOUD) {
            visualOdometryEngine.reset()
        }
        if (filter.isMediaPipe) {
            return mediaPipeProcessor?.processFrame(bitmap, filter) ?: bitmap.copy(Bitmap.Config.ARGB_8888, false)
        }
        if (filter.isYolo) {
            return yoloProcessor?.processFrame(bitmap, filter, onMarkersDetected)
                ?: bitmap.copy(Bitmap.Config.ARGB_8888, false)
        }

        val src = Mat()
        Utils.bitmapToMat(bitmap, src)
        val baseFrame = if (isActiveVisionEnabled) {
            activeVisionOptimizer.optimize(src, visualizeWork = isActiveVisionVisualizationEnabled)
        } else {
            src.clone()
        }

        val processed: Mat = when (filter) {
            OpenCvFilter.ORIGINAL -> baseFrame.clone()
            OpenCvFilter.GRAYSCALE -> applyGrayscale(baseFrame)
            OpenCvFilter.CANNY_EDGES -> applyCanny(baseFrame)
            OpenCvFilter.GAUSSIAN_BLUR -> applyGaussianBlur(baseFrame)
            OpenCvFilter.THRESHOLD -> applyThreshold(baseFrame)
            OpenCvFilter.SOBEL -> applySobel(baseFrame)
            OpenCvFilter.LAPLACIAN -> applyLaplacian(baseFrame)
            OpenCvFilter.DILATE -> applyDilate(baseFrame)
            OpenCvFilter.ERODE -> applyErode(baseFrame)
            OpenCvFilter.OPEN -> applyMorphEx(baseFrame, Imgproc.MORPH_OPEN)
            OpenCvFilter.CLOSE -> applyMorphEx(baseFrame, Imgproc.MORPH_CLOSE)
            OpenCvFilter.GRADIENT -> applyMorphEx(baseFrame, Imgproc.MORPH_GRADIENT)
            OpenCvFilter.TOP_HAT -> applyMorphEx(baseFrame, Imgproc.MORPH_TOPHAT)
            OpenCvFilter.BLACK_HAT -> applyMorphEx(baseFrame, Imgproc.MORPH_BLACKHAT)
            OpenCvFilter.APRIL_TAGS -> applyAprilTagDetection(baseFrame)
            OpenCvFilter.ARUCO -> applyArucoDetection(baseFrame)
            OpenCvFilter.QR_CODE -> applyQrCodeDetection(baseFrame)
            OpenCvFilter.CCTAG -> applyCCTagDetection(baseFrame)
            OpenCvFilter.CHESSBOARD_CALIBRATION -> applyChessboardCalibration(baseFrame)
            OpenCvFilter.UNDISTORT -> applyUndistort(baseFrame)
            OpenCvFilter.VISUAL_ODOMETRY -> applyVisualOdometry(baseFrame)
            OpenCvFilter.POINT_CLOUD -> applyPointCloud(baseFrame)
            OpenCvFilter.PLANE_DETECTION -> applyPlaneDetection(baseFrame)
            OpenCvFilter.VANISHING_POINTS -> applyVanishingPoints(baseFrame)
            OpenCvFilter.MEDIAN_BLUR -> applyMedianBlur(baseFrame)
            OpenCvFilter.BILATERAL_FILTER -> applyBilateralFilter(baseFrame)
            OpenCvFilter.BOX_FILTER -> applyBoxFilter(baseFrame)
            OpenCvFilter.ADAPTIVE_THRESHOLD -> applyAdaptiveThreshold(baseFrame)
            OpenCvFilter.HISTOGRAM_EQUALIZATION -> applyHistogramEqualization(baseFrame)
            OpenCvFilter.SCHARR -> applyScharr(baseFrame)
            OpenCvFilter.PREWITT -> applyPrewitt(baseFrame)
            OpenCvFilter.ROBERTS -> applyRoberts(baseFrame)
            else -> baseFrame.clone()
        }

        val result = createBitmap(processed.cols(), processed.rows())
        Utils.matToBitmap(processed, result)
        src.release(); baseFrame.release(); processed.release()
        return result
    }

    private fun applyGrayscale(src: Mat): Mat {
        val res = Mat(); Imgproc.cvtColor(src, res, Imgproc.COLOR_RGBA2GRAY)
        val out = Mat(); Imgproc.cvtColor(res, out, Imgproc.COLOR_GRAY2RGBA)
        res.release(); return out
    }

    private fun applyCanny(src: Mat): Mat {
        val gray = Mat(); val blurred = Mat(); val edges = Mat(); val res = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
        Imgproc.Canny(blurred, edges, 50.0, 150.0)
        Imgproc.cvtColor(edges, res, Imgproc.COLOR_GRAY2RGBA)
        gray.release(); blurred.release(); edges.release(); return res
    }

    private fun applyGaussianBlur(src: Mat): Mat {
        val res = Mat(); Imgproc.GaussianBlur(src, res, Size(5.0, 5.0), 0.0); return res
    }

    private fun applyThreshold(src: Mat): Mat {
        val gray = Mat(); val thresh = Mat(); val res = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.threshold(gray, thresh, 127.0, 255.0, Imgproc.THRESH_BINARY)
        Imgproc.cvtColor(thresh, res, Imgproc.COLOR_GRAY2RGBA)
        gray.release(); thresh.release(); return res
    }

    private fun applySobel(src: Mat): Mat {
        val gray = Mat(); val sx = Mat(); val sy = Mat(); val ax = Mat(); val ay = Mat(); val c = Mat(); val res = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.Sobel(gray, sx, CvType.CV_16S, 1, 0)
        Imgproc.Sobel(gray, sy, CvType.CV_16S, 0, 1)
        Core.convertScaleAbs(sx, ax); Core.convertScaleAbs(sy, ay)
        Core.addWeighted(ax, 0.5, ay, 0.5, 0.0, c)
        Imgproc.cvtColor(c, res, Imgproc.COLOR_GRAY2RGBA)
        gray.release(); sx.release(); sy.release(); ax.release(); ay.release(); c.release(); return res
    }

    private fun applyLaplacian(src: Mat): Mat {
        val gray = Mat(); val lap = Mat(); val abs = Mat(); val res = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.Laplacian(gray, lap, CvType.CV_16S)
        Core.convertScaleAbs(lap, abs)
        Imgproc.cvtColor(abs, res, Imgproc.COLOR_GRAY2RGBA)
        gray.release(); lap.release(); abs.release(); return res
    }

    private fun applyDilate(src: Mat): Mat {
        val res = Mat(); val k = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size((2*morphKernelSize+1).toDouble(), (2*morphKernelSize+1).toDouble()))
        Imgproc.dilate(src, res, k); k.release(); return res
    }

    private fun applyErode(src: Mat): Mat {
        val res = Mat(); val k = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size((2*morphKernelSize+1).toDouble(), (2*morphKernelSize+1).toDouble()))
        Imgproc.erode(src, res, k); k.release(); return res
    }

    private fun applyMorphEx(src: Mat, op: Int): Mat {
        val res = Mat(); val k = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size((2*morphKernelSize+1).toDouble(), (2*morphKernelSize+1).toDouble()))
        Imgproc.morphologyEx(src, res, op, k); k.release(); return res
    }

    private fun applyAprilTagDetection(src: Mat): Mat {
        val res = src.clone(); val corners = ArrayList<Mat>(); val ids = Mat()
        val gray = Mat(); Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        aprilTagDetector.detectMarkers(gray, corners, ids)
        gray.release()
        if (ids.rows() > 0) {
            Objdetect.drawDetectedMarkers(res, corners, ids, Scalar(0.0, 255.0, 0.0, 255.0))
            val detections = ArrayList<MarkerDetection>()
            for (i in 0 until corners.size) {
                val c = corners[i]
                val pts = listOf(
                    Pair(c.get(0,0)[0].toFloat(), c.get(0,0)[1].toFloat()),
                    Pair(c.get(0,1)[0].toFloat(), c.get(0,1)[1].toFloat()),
                    Pair(c.get(0,2)[0].toFloat(), c.get(0,2)[1].toFloat()),
                    Pair(c.get(0,3)[0].toFloat(), c.get(0,3)[1].toFloat())
                )
                detections.add(MarkerDetection.AprilTag(ids.get(i,0)[0].toInt(), pts))
            }
            onMarkersDetected?.invoke(detections)
        }
        ids.release(); return res
    }

    private fun applyArucoDetection(src: Mat): Mat {
        val res = src.clone(); val corners = ArrayList<Mat>(); val ids = Mat()
        val gray = Mat(); Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        arucoDetector.detectMarkers(gray, corners, ids)
        gray.release()
        if (ids.rows() > 0) {
            Objdetect.drawDetectedMarkers(res, corners, ids, Scalar(255.0, 255.0, 0.0, 255.0))
            val detections = ArrayList<MarkerDetection>()
            for (i in 0 until corners.size) {
                val c = corners[i]
                val pts = ptsToList(c)
                detections.add(MarkerDetection.Aruco(ids.get(i,0)[0].toInt(), pts))
            }
            onMarkersDetected?.invoke(detections)
        }
        ids.release(); return res
    }

    private fun ptsToList(c: Mat): List<Pair<Float, Float>> {
        return listOf(
            Pair(c.get(0,0)[0].toFloat(), c.get(0,0)[1].toFloat()),
            Pair(c.get(0,1)[0].toFloat(), c.get(0,1)[1].toFloat()),
            Pair(c.get(0,2)[0].toFloat(), c.get(0,2)[1].toFloat()),
            Pair(c.get(0,3)[0].toFloat(), c.get(0,3)[1].toFloat())
        )
    }

    private fun applyQrCodeDetection(src: Mat): Mat {
        val res = src.clone(); val points = Mat()
        val gray = Mat(); Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        val data = qrCodeDetector.detectAndDecode(gray, points)
        if (!data.isNullOrEmpty()) {
            val pts = MatOfPoint2f()
            points.convertTo(pts, CvType.CV_32F)
            val ptsList = ArrayList<Point>()
            for (i in 0 until pts.rows()) {
                val p = Point(pts.get(i, 0)[0], pts.get(i, 0)[1])
                ptsList.add(p)
                Imgproc.line(res, p, Point(pts.get((i+1)%pts.rows(), 0)[0], pts.get((i+1)%pts.rows(), 0)[1]), Scalar(255.0, 0.0, 0.0, 255.0), 3)
            }
            onMarkersDetected?.invoke(listOf(MarkerDetection.QrCode(data, ptsList.map { Pair(it.x.toFloat(), it.y.toFloat()) })))
            pts.release()
        }
        gray.release(); points.release(); return res
    }

    private fun applyCCTagDetection(src: Mat): Mat {
        // CCTag is not natively in OpenCV, placeholder or custom impl needed.
        return src.clone()
    }

    private fun applyChessboardCalibration(src: Mat): Mat {
        val res = src.clone()
        val gray = Mat(); Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        val corners = calibrator?.detectCorners(gray, src.size())
        if (corners != null) {
            val boardSize = Size(calibrator?.boardWidth?.toDouble() ?: 9.0, calibrator?.boardHeight?.toDouble() ?: 6.0)
            Calib3d.drawChessboardCorners(res, boardSize, corners, true)
        } else {
            Imgproc.putText(res, labelBoardNotFound, Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255.0, 0.0, 0.0, 255.0), 2)
        }
        val frameCount = calibrator?.frameCount ?: 0
        Imgproc.putText(res, "$frameCount $labelFrameCountSuffix", Point(30.0, res.rows() - 30.0), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0.0, 255.0, 0.0, 255.0), 2)
        gray.release(); return res
    }

    private fun applyUndistort(src: Mat): Mat {
        val res = Mat()
        val calib = calibrator?.calibrationResult
        if (calib != null) {
            Calib3d.undistort(src, res, calib.cameraMatrix, calib.distCoeffs)
        } else {
            src.copyTo(res)
            Imgproc.putText(res, labelNoCalibration, Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255.0, 0.0, 0.0, 255.0), 2)
        }
        return res
    }

    private fun applyVisualOdometry(src: Mat): Mat {
        val res = src.clone()
        visualOdometryEngine.processFrameRgba(src, calibrator)
        val tracks = visualOdometryEngine.currentTracks
        for (track in tracks) {
            if (track.size < 2) continue
            for (i in 0 until track.size - 1) {
                Imgproc.line(res, track[i], track[i+1], Scalar(0.0, 255.0, 0.0, 255.0), 1)
            }
            Imgproc.circle(res, track.last(), 3, Scalar(0.0, 0.0, 255.0, 255.0), -1)
        }
        Imgproc.putText(res, "$labelOdometryTracks: ${tracks.size}", Point(30.0, 30.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255.0, 255.0, 255.0), 2)
        return res
    }

    private fun applyPointCloud(src: Mat): Mat {
        val res = src.clone()
        visualOdometryEngine.processFrameRgba(src, calibrator)
        val cloud = visualOdometryEngine.lastPointCloud
        if (cloud != null) {
            Imgproc.putText(res, "$labelPointCloud: ${cloud.points.size}", Point(30.0, 30.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255.0, 255.0, 255.0), 2)
        }
        return res
    }

    companion object {
        private const val TAG = "ImageProcessor"
        private const val CLUSTER_ANGLE_THRESHOLD_DEG = 8.0
        private const val MAX_LINE_DIRECTION_CLUSTERS = 4
    }

    /**
     * Segments the image into planes using a combination of edge detection, 
     * Hough line clustering, and vanishing point analysis.
     *
     * Visualises planes by:
     * - Drawing inlier lines of dominant directions in unique colours.
     * - Drawing a convex hull over the detected plane area.
     * - Marking the vanishing point for each direction.
     * - Confidence label (percentage of lines belonging to the plane).
     */
    private fun applyPlaneDetection(src: Mat): Mat {
        val res = src.clone()
        val gray = Mat(); val clahe = Mat(); val blurred = Mat(); val edges = Mat()
        try {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)

            // CLAHE – contrast-limited adaptive histogram equalization for even-lighting robustness
            val claheObj = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
            claheObj.apply(gray, clahe)

            Imgproc.GaussianBlur(clahe, blurred, Size(5.0, 5.0), 0.0)

            // Adaptive Canny: thresholds derived from the median intensity of the blurred image
            val medianVal = _medianIntensity(blurred)
            val sigma = 0.33
            val lower = maxOf(0.0, (1.0 - sigma) * medianVal)
            val upper = minOf(255.0, (1.0 + sigma) * medianVal)
            Imgproc.Canny(blurred, edges, lower, upper)

            // Dilate edges to bridge small gaps between line segments
            Imgproc.dilate(edges, edges, dilationKernel3x3)

            val lines = Mat()
            Imgproc.HoughLinesP(edges, lines, 1.0, Math.PI / 180.0, 30, 15.0, 5.0)

            // Cluster lines by angle; use length-weighted mean angle per cluster
            val clusters = ArrayList<ArrayList<DoubleArray>>()  // each segment: [x1,y1,x2,y2,length]
            val clusterAngles = ArrayList<Double>()
            for (i in 0 until lines.rows()) {
                val seg = lines.get(i, 0).takeIf { it.isNotEmpty() } ?: continue
                if (seg.size < 4) continue
                val x1 = seg[0]; val y1 = seg[1]; val x2 = seg[2]; val y2 = seg[3]
                val length = hypot(x2 - x1, y2 - y1)
                val angle = Math.toDegrees(atan2(y2 - y1, x2 - x1)).let { if (it < 0) it + 180.0 else it } % 180.0
                var assigned = false
                for (k in clusterAngles.indices) {
                    var diff = abs(angle - clusterAngles[k]); diff = minOf(diff, 180.0 - diff)
                    if (diff <= CLUSTER_ANGLE_THRESHOLD_DEG) {
                        clusters[k].add(doubleArrayOf(x1, y1, x2, y2, length))
                        // Update cluster mean angle (length-weighted)
                        clusterAngles[k] = _weightedAngleMean(clusters[k])
                        assigned = true; break
                    }
                }
                if (!assigned) {
                    clusters.add(arrayListOf(doubleArrayOf(x1, y1, x2, y2, length)))
                    clusterAngles.add(angle)
                }
            }

            val totalLines = lines.rows()
            val planeColors = arrayOf(
                Scalar(0.0, 255.0, 0.0),
                Scalar(0.0, 120.0, 255.0),
                Scalar(0.0, 200.0, 255.0),
            )
            val sortedClusters = clusters.sortedByDescending { it.size }.take(MAX_LINE_DIRECTION_CLUSTERS)
            var planeIdx = 0

            for (i in sortedClusters.indices) {
                for (j in i + 1 until sortedClusters.size) {
                    if (planeIdx >= 3) break
                    val c1 = sortedClusters[i]; val c2 = sortedClusters[j]
                    if (c1.size + c2.size < 4) continue
                    val color = planeColors[planeIdx % planeColors.size]
                    val planeLineCount = c1.size + c2.size
                    val confidence = if (totalLines > 0) ((planeLineCount * 100.0 / totalLines).toInt()).coerceAtMost(100) else 0

                    // Draw inlier lines with thickness proportional to confidence
                    val thickness = if (confidence >= 50) 2 else 1
                    for (seg in c1) Imgproc.line(res, Point(seg[0], seg[1]), Point(seg[2], seg[3]), color, thickness)
                    for (seg in c2) Imgproc.line(res, Point(seg[0], seg[1]), Point(seg[2], seg[3]), color, thickness)

                    // Semi-transparent convex hull over all endpoints of both clusters
                    val allPoints = ArrayList<Point>()
                    for (seg in c1) { allPoints.add(Point(seg[0], seg[1])); allPoints.add(Point(seg[2], seg[3])) }
                    for (seg in c2) { allPoints.add(Point(seg[0], seg[1])); allPoints.add(Point(seg[2], seg[3])) }
                    _drawPlaneOverlay(res, allPoints, color)

                    // Vanishing points for each sub-cluster
                    val vp1 = _computeVanishingPoint(c1.map { intArrayOf(it[0].toInt(), it[1].toInt(), it[2].toInt(), it[3].toInt()) })
                    val vp2 = _computeVanishingPoint(c2.map { intArrayOf(it[0].toInt(), it[1].toInt(), it[2].toInt(), it[3].toInt()) })

                    // Centroid of all cluster endpoints for arrow base
                    val cx = allPoints.map { it.x }.average()
                    val cy = allPoints.map { it.y }.average()
                    val centroid = Point(cx, cy)

                    for (vp in listOfNotNull(vp1, vp2)) {
                        Imgproc.circle(res, vp, 8, color, -1)
                        // Arrow from centroid toward vanishing point (capped at 80 px)
                        val dx = vp.x - cx; val dy = vp.y - cy
                        val dist = hypot(dx, dy)
                        if (dist > 1.0) {
                            val arrowEnd = Point(cx + dx / dist * minOf(80.0, dist), cy + dy / dist * minOf(80.0, dist))
                            Imgproc.arrowedLine(res, centroid, arrowEnd, color, 2, Imgproc.LINE_8, 0, 0.3)
                        }
                    }

                    Imgproc.putText(res, "P${planeIdx + 1} ($confidence%)", Point(cx + 8, cy), Imgproc.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    planeIdx++
                }
            }

            // Handle single dominant cluster (all lines nearly parallel)
            if (planeIdx == 0 && sortedClusters.isNotEmpty() && sortedClusters[0].size >= 3) {
                val c = sortedClusters[0]
                val color = planeColors[0]
                val confidence = if (totalLines > 0) ((c.size * 100.0 / totalLines).toInt()).coerceAtMost(100) else 0
                for (seg in c) Imgproc.line(res, Point(seg[0], seg[1]), Point(seg[2], seg[3]), color, 2)
                val allPoints = c.flatMap { listOf(Point(it[0], it[1]), Point(it[2], it[3])) }
                _drawPlaneOverlay(res, allPoints, color)
                val vp = _computeVanishingPoint(c.map { intArrayOf(it[0].toInt(), it[1].toInt(), it[2].toInt(), it[3].toInt()) })
                if (vp != null) Imgproc.circle(res, vp, 8, color, -1)
                val cx = allPoints.map { it.x }.average()
                val cy = allPoints.map { it.y }.average()
                Imgproc.putText(res, "P1 ($confidence%)", Point(cx + 8, cy), Imgproc.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                planeIdx = 1
            }

            if (planeIdx == 0) {
                Imgproc.putText(res, "$labelNoPlanes ($labelLines: $totalLines)", Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, Scalar(200.0, 200.0, 200.0), 2)
            } else {
                Imgproc.putText(res, "$labelPlanes: $planeIdx | $labelLines: $totalLines", Point(30.0, 30.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255.0, 255.0, 255.0), 2)
            }
            lines.release()
        } catch (e: Exception) {
            Imgproc.putText(res, "$labelGeometryError: ${e.message?.take(30)}", Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255.0, 100.0, 100.0), 2)
        } finally {
            gray.release(); clahe.release(); blurred.release(); edges.release()
        }
        return res
    }

    /**
     * Draws a semi-transparent filled convex hull over the given point list on [dst].
     *
     * The fill uses an addWeighted blend so the original image detail is preserved
     * beneath the coloured plane overlay.
     */
    private fun _drawPlaneOverlay(dst: Mat, points: List<Point>, color: Scalar) {
        if (points.size < 3) return
        try {
            val contourMat = MatOfPoint()
            contourMat.fromArray(*points.toTypedArray())
            val hullIdx = MatOfInt()
            Imgproc.convexHull(contourMat, hullIdx)
            val hullPts = hullIdx.toArray().map { i -> points[i] }
            hullIdx.release()
            if (hullPts.size < 3) { contourMat.release(); return }
            val hullMat = MatOfPoint()
            hullMat.fromArray(*hullPts.toTypedArray())
            val overlay = dst.clone()
            Imgproc.fillConvexPoly(overlay, hullMat, Scalar(color.`val`[0], color.`val`[1], color.`val`[2], 255.0))
            Core.addWeighted(dst, 0.75, overlay, 0.25, 0.0, dst)
            overlay.release(); hullMat.release(); contourMat.release()
        } catch (e: Exception) {
            Log.w(TAG, "Plane overlay rendering failed: ${e.message}")
        }
    }

    /**
     * Returns the median intensity of a single-channel [mat].
     * Used to compute adaptive Canny thresholds.
     */
    private fun _medianIntensity(mat: Mat): Double {
        val hist = Mat()
        val images = listOf(mat)
        val channels = MatOfInt(0)
        val mask = Mat()
        val histSize = MatOfInt(256)
        val ranges = MatOfFloat(0f, 256f)
        Imgproc.calcHist(images, channels, mask, hist, histSize, ranges)
        val total = mat.rows().toLong() * mat.cols().toLong()
        var cumulative = 0.0
        for (i in 0 until 256) {
            cumulative += hist.get(i, 0)[0]
            if (cumulative >= total / 2.0) {
                hist.release(); channels.release(); histSize.release(); ranges.release(); mask.release()
                return i.toDouble()
            }
        }
        hist.release(); channels.release(); histSize.release(); ranges.release(); mask.release()
        return 128.0
    }

    /**
     * Returns the length-weighted mean angle for a cluster of segments.
     * Each segment element is [x1, y1, x2, y2, length].
     */
    private fun _weightedAngleMean(cluster: List<DoubleArray>): Double {
        var sumSin = 0.0
        var sumCos = 0.0
        var totalWeight = 0.0
        for (seg in cluster) {
            val x1 = seg[0]; val y1 = seg[1]; val x2 = seg[2]; val y2 = seg[3]; val length = seg[4]
            val angleRad = atan2(y2 - y1, x2 - x1)
            // Double the angle to map 180° periodicity to 360° for circular averaging
            sumSin += length * kotlin.math.sin(2.0 * angleRad)
            sumCos += length * kotlin.math.cos(2.0 * angleRad)
            totalWeight += length
        }
        if (totalWeight == 0.0) return 0.0
        val mean = Math.toDegrees(atan2(sumSin, sumCos)).let { if (it < 0) it + 180.0 else it } % 180.0
        return mean
    }

    /**
     * Detects and visualises vanishing points from parallel line-segment groups.
     *
     * Draws each cluster of parallel lines in a distinct colour and marks the
     * estimated vanishing point with a filled circle.
     */
    private fun applyVanishingPoints(src: Mat): Mat {
        val res = src.clone()
        val gray = Mat(); val blurred = Mat(); val edges = Mat()
        try {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
            Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
            Imgproc.Canny(blurred, edges, 50.0, 150.0)

            val lines = Mat()
            Imgproc.HoughLinesP(edges, lines, 1.0, Math.PI / 180.0, 40, 20.0, 8.0)

            val clusters = ArrayList<ArrayList<IntArray>>()
            val clusterAngles = ArrayList<Double>()
            for (i in 0 until lines.rows()) {
                val seg = lines.get(i, 0).takeIf { it.isNotEmpty() } ?: continue
                if (seg.size < 4) continue
                val x1 = seg[0].toInt(); val y1 = seg[1].toInt()
                val x2 = seg[2].toInt(); val y2 = seg[3].toInt()
                val angle = Math.toDegrees(atan2((y2 - y1).toDouble(), (x2 - x1).toDouble())).let { if (it < 0) it + 180.0 else it } % 180.0
                var assigned = false
                for (k in clusterAngles.indices) {
                    var diff = abs(angle - clusterAngles[k]); diff = minOf(diff, 180.0 - diff)
                    if (diff <= 8.0) { clusters[k].add(intArrayOf(x1, y1, x2, y2)); assigned = true; break }
                }
                if (!assigned) { clusters.add(arrayListOf(intArrayOf(x1, y1, x2, y2))); clusterAngles.add(angle) }
            }

            val vpColors = arrayOf(Scalar(0.0, 255.0, 0.0), Scalar(0.0, 0.0, 255.0), Scalar(0.0, 165.0, 255.0), Scalar(255.0, 255.0, 0.0))
            var foundVP = false
            for (i in 0 until minOf(clusters.size, MAX_LINE_DIRECTION_CLUSTERS)) {
                val cluster = clusters.sortedByDescending { it.size }[i]
                if (cluster.size < 2) continue
                val color = vpColors[i % vpColors.size]
                for (seg in cluster) {
                    Imgproc.line(res, Point(seg[0].toDouble(), seg[1].toDouble()), Point(seg[2].toDouble(), seg[3].toDouble()), color, 1)
                }
                val vp = _computeVanishingPoint(cluster)
                if (vp != null) {
                    Imgproc.circle(res, vp, 10, color, -1)
                    Imgproc.putText(res, "VP${i + 1}", Point(vp.x + 12, vp.y + 5), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    foundVP = true
                }
            }

            when {
                lines.rows() == 0 -> Imgproc.putText(res, labelNoLines, Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, Scalar(200.0, 200.0, 200.0), 2)
                !foundVP -> Imgproc.putText(res, "$labelNoVanishingPoints ($labelLines: ${lines.rows()})", Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, Scalar(200.0, 200.0, 200.0), 2)
                else -> Imgproc.putText(res, "$labelLines: ${lines.rows()} | $labelGroups: ${clusters.size}", Point(30.0, 30.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255.0, 255.0, 255.0), 2)
            }
            lines.release()
        } catch (e: Exception) {
            Imgproc.putText(res, "$labelVpError: ${e.message?.take(30)}", Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255.0, 100.0, 100.0), 2)
        } finally {
            gray.release(); blurred.release(); edges.release()
        }
        return res
    }

    /**
     * Estimates a vanishing point for a cluster of line segments using the
     * least-squares intersection of their line equations.
     *
     * Returns ``null`` when the system is rank-deficient (parallel lines that
     * intersect at infinity).
     */
    private fun _computeVanishingPoint(lines: List<IntArray>): Point? {
        if (lines.size < 2) return null
        val aMat = Mat(lines.size, 2, CvType.CV_64F)
        val bMat = Mat(lines.size, 1, CvType.CV_64F)
        for (i in lines.indices) {
            val x1 = lines[i][0].toDouble(); val y1 = lines[i][1].toDouble()
            val x2 = lines[i][2].toDouble(); val y2 = lines[i][3].toDouble()
            val a = y1 - y2
            val b = x2 - x1
            val c = a * x1 + b * y1
            aMat.put(i, 0, a); aMat.put(i, 1, b)
            bMat.put(i, 0, c)
        }
        val solution = Mat()
        val solved = Core.solve(aMat, bMat, solution, Core.DECOMP_SVD)
        val res = if (solved && solution.rows() >= 2) Point(solution.get(0, 0)[0], solution.get(1, 0)[0]) else null
        aMat.release(); bMat.release(); solution.release()
        return res
    }

    private fun applyMedianBlur(src: Mat): Mat {
        val res = Mat(); Imgproc.medianBlur(src, res, 5); return res
    }

    private fun applyBilateralFilter(src: Mat): Mat {
        val res = Mat(); val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2RGB)
        Imgproc.bilateralFilter(gray, res, 9, 75.0, 75.0)
        val out = Mat(); Imgproc.cvtColor(res, out, Imgproc.COLOR_RGB2RGBA)
        gray.release(); res.release(); return out
    }

    private fun applyBoxFilter(src: Mat): Mat {
        val res = Mat(); Imgproc.boxFilter(src, res, -1, Size(5.0, 5.0)); return res
    }

    private fun applyAdaptiveThreshold(src: Mat): Mat {
        val gray = Mat(); val thresh = Mat(); val res = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.adaptiveThreshold(gray, thresh, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 2.0)
        Imgproc.cvtColor(thresh, res, Imgproc.COLOR_GRAY2RGBA)
        gray.release(); thresh.release(); return res
    }

    private fun applyHistogramEqualization(src: Mat): Mat {
        val gray = Mat(); val equ = Mat(); val res = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.equalizeHist(gray, equ)
        Imgproc.cvtColor(equ, res, Imgproc.COLOR_GRAY2RGBA)
        gray.release(); equ.release(); return res
    }

    private fun applyScharr(src: Mat): Mat {
        val gray = Mat(); val sx = Mat(); val sy = Mat(); val ax = Mat(); val ay = Mat(); val c = Mat(); val res = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.Scharr(gray, sx, CvType.CV_16S, 1, 0)
        Imgproc.Scharr(gray, sy, CvType.CV_16S, 0, 1)
        Core.convertScaleAbs(sx, ax); Core.convertScaleAbs(sy, ay)
        Core.addWeighted(ax, 0.5, ay, 0.5, 0.0, c)
        Imgproc.cvtColor(c, res, Imgproc.COLOR_GRAY2RGBA)
        gray.release(); sx.release(); sy.release(); ax.release(); ay.release(); c.release(); return res
    }

    private fun applyPrewitt(src: Mat): Mat {
        val gray = Mat(); val res = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        val kernelX = Mat(3, 3, CvType.CV_32F)
        kernelX.put(0, 0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0)
        val kernelY = Mat(3, 3, CvType.CV_32F)
        kernelY.put(0, 0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        val gradX = Mat(); val gradY = Mat()
        Imgproc.filter2D(gray, gradX, -1, kernelX)
        Imgproc.filter2D(gray, gradY, -1, kernelY)
        val absX = Mat(); val absY = Mat()
        Core.convertScaleAbs(gradX, absX); Core.convertScaleAbs(gradY, absY)
        val combined = Mat()
        Core.addWeighted(absX, 0.5, absY, 0.5, 0.0, combined)
        Imgproc.cvtColor(combined, res, Imgproc.COLOR_GRAY2RGBA)
        gray.release(); kernelX.release(); kernelY.release(); gradX.release(); gradY.release(); absX.release(); absY.release(); combined.release()
        return res
    }

    private fun applyRoberts(src: Mat): Mat {
        val gray = Mat(); val res = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        val kernelX = Mat(2, 2, CvType.CV_32F)
        kernelX.put(0, 0, 1.0, 0.0, 0.0, -1.0)
        val kernelY = Mat(2, 2, CvType.CV_32F)
        kernelY.put(0, 0, 0.0, 1.0, -1.0, 0.0)
        val gradX = Mat(); val gradY = Mat()
        Imgproc.filter2D(gray, gradX, -1, kernelX)
        Imgproc.filter2D(gray, gradY, -1, kernelY)
        val absX = Mat(); val absY = Mat()
        Core.convertScaleAbs(gradX, absX); Core.convertScaleAbs(gradY, absY)
        val combined = Mat()
        Core.addWeighted(absX, 0.5, absY, 0.5, 0.0, combined)
        Imgproc.cvtColor(combined, res, Imgproc.COLOR_GRAY2RGBA)
        gray.release(); kernelX.release(); kernelY.release(); gradX.release(); gradY.release(); absX.release(); absY.release(); combined.release()
        return res
    }
}
