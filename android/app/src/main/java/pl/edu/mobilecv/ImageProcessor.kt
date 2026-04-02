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
import org.opencv.imgproc.CLAHE as OpencvClahe
import org.opencv.objdetect.ArucoDetector
import org.opencv.objdetect.DetectorParameters
import org.opencv.objdetect.Dictionary
import org.opencv.objdetect.Objdetect
import org.opencv.objdetect.QRCodeDetector
import androidx.core.graphics.createBitmap
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.sqrt

/**
 * Applies OpenCV image-processing filters to Android [Bitmap] frames.
 */
class ImageProcessor {

    var calibrator: CameraCalibrator? = null
    var mediaPipeProcessor: MediaPipeProcessor? = null

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
            onMarkersDetected?.invoke(corners.indices.map { i ->
                val c = corners[i]
                val pts = listOf(
                    Pair(c.get(0,0)[0].toFloat(), c.get(0,0)[1].toFloat()),
                    Pair(c.get(0,1)[0].toFloat(), c.get(0,1)[1].toFloat()),
                    Pair(c.get(0,2)[0].toFloat(), c.get(0,2)[1].toFloat()),
                    Pair(c.get(0,3)[0].toFloat(), c.get(0,3)[1].toFloat())
                )
                MarkerDetection.AprilTag(ids.get(i,0)[0].toInt(), pts)
            })
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
            onMarkersDetected?.invoke(corners.indices.map { i ->
                val c = corners[i]
                val pts = ptsToList(c)
                MarkerDetection.Aruco(ids.get(i,0)[0].toInt(), pts)
            })
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
        val res = src.clone()
        val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        val points = Mat()
        val text = qrCodeDetector.detectAndDecode(gray, points)
        if (!points.empty()) {
            val pts = (0 until 4).map { i ->
                val data = points.get(0, i)
                if (data != null) Point(data[0], data[1]) else Point(0.0, 0.0)
            }
            val color = Scalar(255.0, 0.0, 255.0, 255.0)
            for (i in 0 until 4) {
                Imgproc.line(res, pts[i], pts[(i + 1) % 4], color, 3)
            }
            if (text.isNotEmpty()) {
                Imgproc.putText(res, text.take(MAX_QR_TEXT_DISPLAY_LENGTH), Point(pts[0].x, maxOf(20.0, pts[0].y - 10)), Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                val corners = pts.map { Pair(it.x.toFloat(), it.y.toFloat()) }
                onMarkersDetected?.invoke(listOf(MarkerDetection.QrCode(text, corners)))
            }
        }
        gray.release(); points.release(); return res
    }

    private fun applyCCTagDetection(src: Mat): Mat {
        val res = src.clone(); val gray = Mat(); val thresh = Mat(); Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.adaptiveThreshold(gray, thresh, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 2.0)
        val contours = ArrayList<MatOfPoint>(); val hierarchy = Mat()
        Imgproc.findContours(thresh, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE)
        val candidates = ArrayList<Pair<Point, Double>>()
        for (c in contours) {
            val area = Imgproc.contourArea(c)
            if (area < 50.0) continue
            val perim = Imgproc.arcLength(MatOfPoint2f(*c.toArray()), true)
            if (perim <= 0) continue
            if (4 * Math.PI * area / (perim * perim) < 0.5) continue
            val center = Point(); val radius = FloatArray(1); Imgproc.minEnclosingCircle(MatOfPoint2f(*c.toArray()), center, radius)
            candidates.add(Pair(center, radius[0].toDouble()))
        }
        val tags = ArrayList<Pair<Point, Int>>(); val sorted = candidates.sortedByDescending { it.second }; val used = BooleanArray(sorted.size)
        val detections = ArrayList<MarkerDetection>()
        for (i in sorted.indices) {
            if (used[i]) continue
            used[i] = true; val outer = sorted[i]; var count = 1
            for (j in i + 1 until sorted.size) {
                if (used[j]) continue
                val inner = sorted[j]; val d =
                    sqrt((outer.first.x - inner.first.x) * (outer.first.x - inner.first.x) + (outer.first.y - inner.first.y) * (outer.first.y - inner.first.y))
                if (d < outer.second * 0.25) { count++; used[j] = true }
            }
            if (count in 2..5) {
                tags.add(Pair(outer.first, count))
                val r = outer.second.toFloat()
                val corners = listOf(
                    Pair(outer.first.x.toFloat() - r, outer.first.y.toFloat() - r),
                    Pair(outer.first.x.toFloat() + r, outer.first.y.toFloat() - r),
                    Pair(outer.first.x.toFloat() + r, outer.first.y.toFloat() + r),
                    Pair(outer.first.x.toFloat() - r, outer.first.y.toFloat() + r)
                )
                detections.add(MarkerDetection.CCTag(count, Pair(outer.first.x.toFloat(), outer.first.y.toFloat()), r, corners))
            }
        }
        for (t in tags) {
            Imgproc.circle(res, t.first, 10, Scalar(0.0, 255.0, 255.0, 255.0), -1)
            Imgproc.putText(res, "CCTag (${t.second})", Point(t.first.x+15, t.first.y+5), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255.0, 255.0, 255.0, 255.0), 2)
        }
        if (detections.isNotEmpty()) onMarkersDetected?.invoke(detections)
        gray.release(); thresh.release(); hierarchy.release(); contours.forEach { it.release() }; return res
    }

    private fun applyChessboardCalibration(src: Mat): Mat {
        val res = src.clone(); val gray = Mat(); Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        val corners = MatOfPoint2f(); val pattern = Size(9.0, 6.0); val found = Calib3d.findChessboardCorners(gray, pattern, corners)
        if (found) {
            Imgproc.cornerSubPix(gray, corners, Size(11.0, 11.0), Size(-1.0, -1.0), TermCriteria(TermCriteria.EPS + TermCriteria.COUNT, 30, 0.1))
            Calib3d.drawChessboardCorners(res, pattern, corners, true)
            Imgproc.putText(res, "Board Detected", Point(30.0, 40.0), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0.0, 255.0, 0.0, 255.0), 2)
        } else Imgproc.putText(res, labelBoardNotFound, Point(30.0, 40.0), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255.0, 0.0, 0.0, 255.0), 2)
        calibrator?.storeDetectedCorners(
            if (found) corners else null,
            Size(src.cols().toDouble(), src.rows().toDouble()),
        )
        Imgproc.putText(res, "${calibrator?.frameCount ?: 0} $labelFrameCountSuffix", Point(30.0, 80.0), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255.0, 255.0, 255.0, 255.0), 2)
        gray.release(); corners.release(); return res
    }

    private fun applyUndistort(src: Mat): Mat {
        val r = calibrator?.calibrationResult; val m = r?.cameraMatrix; val d = r?.distCoeffs
        if (m == null) {
            val out = src.clone(); Imgproc.putText(out, labelNoCalibration, Point(30.0, 60.0), Imgproc.FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255.0, 0.0, 0.0, 255.0), 3)
            return out
        }
        val out = Mat(); Calib3d.undistort(src, out, m, d); return out
    }

    private fun applyVisualOdometry(src: Mat): Mat {
        val res = src.clone(); val state = visualOdometryEngine.updateOdometry(src) ?: return res
        Imgproc.putText(res, "$labelOdometryTracks: ${state.tracksCount} (inliers: ${state.inliersCount})", Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0.0, 255.0, 0.0, 255.0), 2)
        Imgproc.putText(res, "Move: %.2f | Rot: %.1f deg".format(state.translationNorm, state.rotationDeg), Point(30.0, 90.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0.0, 255.0, 0.0, 255.0), 2)
        Imgproc.putText(res, "Max features: ${visualOdometryEngine.maxFeatures} — $labelVoMaxFeaturesDesc", Point(30.0, 130.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200.0, 200.0, 255.0, 255.0), 2)
        Imgproc.putText(res, "Min parallax: %.1f px — $labelVoMinParallaxDesc".format(visualOdometryEngine.minParallax), Point(30.0, 158.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200.0, 200.0, 255.0, 255.0), 2)
        val cx = res.cols()/2; val cy = res.rows()/2
        Imgproc.line(res, Point(cx-30.0, cy.toDouble()), Point(cx+30.0, cy.toDouble()), Scalar(255.0, 255.0, 255.0, 255.0), 2)
        Imgproc.line(res, Point(cx.toDouble(), cy-30.0), Point(cx.toDouble(), cy+30.0), Scalar(255.0, 255.0, 255.0, 255.0), 2)
        return res
    }

    private fun applyPointCloud(src: Mat): Mat {
        val res = Mat.zeros(src.size(), src.type()); val state = visualOdometryEngine.updatePointCloud(src) ?: return res
        Imgproc.putText(res, "$labelPointCloud: ${state.points.size} pts | parallax: %.1f px".format(state.meanParallax), Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255.0, 255.0, 255.0, 255.0), 2)
        Imgproc.putText(res, "Max features: ${visualOdometryEngine.maxFeatures} | Min parallax: %.1f px".format(visualOdometryEngine.minParallax), Point(30.0, 85.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200.0, 200.0, 255.0, 255.0), 2)
        Imgproc.putText(res, labelVoColorDepthDesc, Point(30.0, 115.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, Scalar(180.0, 180.0, 180.0, 255.0), 1)
        if (isVoMeshEnabled) {
            for (e in state.edges) Imgproc.line(res, e.first, e.second, Scalar(100.0, 100.0, 100.0, 255.0), 1)
        }
        for (p in state.points) {
            val b = 150.0 + 105.0 * (1.0 - p.y / src.rows())
            Imgproc.circle(res, p, 2, Scalar(b, b, 255.0, 255.0), -1)
        }
        return res
    }

    /**
     * Detects planar surfaces by extracting line segments with HoughLinesP,
     * clustering them by orientation, computing vanishing points, and deriving
     * plane normals from pairs of vanishing points.
     *
     * Improvements over the baseline:
     * - CLAHE normalisation for contrast-invariant edge detection.
     * - Adaptive Canny thresholds based on the median pixel intensity.
     * - Morphological dilation to connect nearby broken edge segments.
     * - Weighted cluster representative angle using line-length as weight.
     * - Semi-transparent filled convex-hull overlay per detected plane.
     * - Confidence label (percentage of lines belonging to the plane).
     * - Arrow from plane centroid toward estimated vanishing point.
     */
    private fun applyPlaneDetection(src: Mat): Mat {
        val res = src.clone()
        val gray = Mat(); val clahe = Mat(); val blurred = Mat(); val edges = Mat()
        var claheObj: OpencvClahe? = null
        try {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)

            // CLAHE – contrast-limited adaptive histogram equalization for even-lighting robustness
            claheObj = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
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
            claheObj?.release()
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
        if (cluster.isEmpty()) return 0.0
        var sumSin = 0.0; var sumCos = 0.0
        for (seg in cluster) {
            val angle = Math.toDegrees(atan2(seg[3] - seg[1], seg[2] - seg[0])).let { if (it < 0) it + 180.0 else it } % 180.0
            val w = seg[4]  // length
            sumSin += w * Math.sin(Math.toRadians(angle))
            sumCos += w * Math.cos(Math.toRadians(angle))
        }
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
            for ((idx, cluster) in clusters.sortedByDescending { it.size }.take(MAX_LINE_DIRECTION_CLUSTERS).withIndex()) {
                if (cluster.size < 2) continue
                val color = vpColors[idx % vpColors.size]
                for (seg in cluster) {
                    Imgproc.line(res, Point(seg[0].toDouble(), seg[1].toDouble()), Point(seg[2].toDouble(), seg[3].toDouble()), color, 1)
                }
                val vp = _computeVanishingPoint(cluster)
                if (vp != null) {
                    Imgproc.circle(res, vp, 10, color, -1)
                    Imgproc.putText(res, "VP${idx + 1}", Point(vp.x + 12, vp.y + 5), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
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
     * truly do not converge within the image).
     */
    private fun _computeVanishingPoint(cluster: List<IntArray>): Point? {
        if (cluster.size < 2) return null
        var a11 = 0.0; var a12 = 0.0; var a22 = 0.0; var b1 = 0.0; var b2 = 0.0
        for (seg in cluster) {
            val x1 = seg[0].toDouble(); val y1 = seg[1].toDouble()
            val x2 = seg[2].toDouble(); val y2 = seg[3].toDouble()
            val dy = y2 - y1; val dx = x2 - x1
            val c = dy * x1 - dx * y1
            a11 += dy * dy; a12 -= dy * dx; a22 += dx * dx
            b1 += dy * c; b2 -= dx * c
        }
        val det = a11 * a22 - a12 * a12
        if (abs(det) < 1e-10) return null
        val vx = (a22 * b1 - a12 * b2) / det
        val vy = (a11 * b2 - a12 * b1) / det
        return Point(vx, vy)
    }

    private fun applyMedianBlur(src: Mat): Mat {
        val res = Mat()
        Imgproc.medianBlur(src, res, 5)
        return res
    }

    private fun applyBilateralFilter(src: Mat): Mat {
        val res = Mat()
        val rgb = Mat()
        Imgproc.cvtColor(src, rgb, Imgproc.COLOR_RGBA2RGB)
        Imgproc.bilateralFilter(rgb, res, 9, 75.0, 75.0)
        val out = Mat()
        Imgproc.cvtColor(res, out, Imgproc.COLOR_RGB2RGBA)
        rgb.release(); res.release(); return out
    }

    private fun applyBoxFilter(src: Mat): Mat {
        val res = Mat()
        Imgproc.boxFilter(src, res, -1, Size(5.0, 5.0))
        return res
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

    private companion object {
        private const val TAG = "ImageProcessor"
        private const val CROSSHAIR_GAP = 30
        private const val MAX_QR_TEXT_DISPLAY_LENGTH = 20
        /** Maximum number of line-direction clusters used for plane and VP detection. */
        private const val MAX_LINE_DIRECTION_CLUSTERS = 4
        /** Angle tolerance (degrees) for assigning a line segment to a direction cluster. */
        private const val CLUSTER_ANGLE_THRESHOLD_DEG = 10.0
    }
}
