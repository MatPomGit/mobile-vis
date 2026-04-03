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
import org.opencv.core.MatOfPoint3f
import org.opencv.core.Point
import org.opencv.core.Point3
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
import kotlin.math.sqrt

/**
 * Applies OpenCV image-processing filters to Android [Bitmap] frames.
 */
class ImageProcessor {
    data class RuntimeBenchmarkSnapshot(
        val filter: OpenCvFilter,
        val samples: Int,
        val avgBeforeMs: Double,
        val avgAfterMs: Double,
        val fpsBefore: Double,
        val fpsAfter: Double,
    )

    data class PoseOverlayMetrics(
        val distanceMeters: Double,
        val yawDeg: Double,
        val pitchDeg: Double,
        val rollDeg: Double,
        val reprojectionErrorPx: Double,
        val confidence: Double,
    )

    private data class PoseState(
        val tvec: Mat,
        val yawDeg: Double,
        val pitchDeg: Double,
        val rollDeg: Double,
        val zAxisCamera: DoubleArray,
    )

    private data class MarkerPoseEstimate(
        val rvec: List<Double>,
        val tvec: List<Double>,
        val quality: MarkerDetection.Quality,
        val metrics: PoseOverlayMetrics,
    )

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
    private val markerPoseStates = HashMap<String, PoseState>()

    companion object {
        private const val TAG = "ImageProcessor"
        private const val MARKER_SIZE_METERS = 0.06
        private const val AXIS_LENGTH_METERS = MARKER_SIZE_METERS * 0.6
        private const val ORIENTATION_SMOOTH_ALPHA = 0.35
        private const val MAX_ANGLE_STEP_DEG = 20.0
        private const val LABEL_LINE_HEIGHT = 24.0
        private const val LABEL_FONT_SCALE = 0.55
        private val OVERLAY_CORNER_COLORS = listOf(
            Scalar(255.0, 70.0, 70.0, 255.0),
            Scalar(70.0, 255.0, 70.0, 255.0),
            Scalar(70.0, 120.0, 255.0, 255.0),
            Scalar(255.0, 220.0, 70.0, 255.0),
        )
        private const val CLUSTER_ANGLE_THRESHOLD_DEG = 8.0
        private const val MAX_LINE_DIRECTION_CLUSTERS = 4
        private const val POINT_CLOUD_CIRCLE_RADIUS = 3
        private const val POINT_CLOUD_MESH_THICKNESS = 1
        private const val PIXELATE_BLOCK_SIZE = 16
    }

    // Reusable mats for the hottest filters to reduce per-frame allocations.
    private val srcBuffer = Mat()
    private val baseFrameBuffer = Mat()
    private val originalBuffer = Mat()
    private val grayBuffer = Mat()
    private val grayRgbaBuffer = Mat()
    private val cannyBlurBuffer = Mat()
    private val cannyEdgesBuffer = Mat()
    private val cannyRgbaBuffer = Mat()
    private val gaussianBuffer = Mat()

    private data class RuntimeBenchmarkAccumulator(
        var samples: Int = 0,
        var beforeNs: Long = 0,
        var afterNs: Long = 0,
    )

    private val benchmarkFilters = setOf(
        OpenCvFilter.ORIGINAL,
        OpenCvFilter.GRAYSCALE,
        OpenCvFilter.CANNY_EDGES,
        OpenCvFilter.GAUSSIAN_BLUR,
    )
    private val benchmarkAccumulators = mutableMapOf<OpenCvFilter, RuntimeBenchmarkAccumulator>()
    @Volatile
    var benchmarkSampleLimit: Int = 30

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

        val src = ensureMat(srcBuffer, bitmap.height, bitmap.width, CvType.CV_8UC4)
        Utils.bitmapToMat(bitmap, src)
        val baseFrame = if (isActiveVisionEnabled) {
            activeVisionOptimizer.optimize(src, visualizeWork = isActiveVisionVisualizationEnabled)
        } else {
            val base = ensureMat(baseFrameBuffer, src.rows(), src.cols(), src.type())
            src.copyTo(base)
            base
        }

        val shouldBenchmark = filter in benchmarkFilters
        var benchmarkBeforeNs = 0L
        if (shouldBenchmark) {
            val beforeStart = System.nanoTime()
            val before = processHotFilterLegacy(baseFrame, filter)
            benchmarkBeforeNs = System.nanoTime() - beforeStart
            before.release()
        }

        val processedPair = when (filter) {
            OpenCvFilter.ORIGINAL,
            OpenCvFilter.GRAYSCALE,
            OpenCvFilter.CANNY_EDGES,
            OpenCvFilter.GAUSSIAN_BLUR -> processHotFilterBuffered(baseFrame, filter)
            OpenCvFilter.THRESHOLD -> Triple(applyThreshold(baseFrame), true, 0L)
            OpenCvFilter.SOBEL -> Triple(applySobel(baseFrame), true, 0L)
            OpenCvFilter.LAPLACIAN -> Triple(applyLaplacian(baseFrame), true, 0L)
            OpenCvFilter.DILATE -> Triple(applyDilate(baseFrame), true, 0L)
            OpenCvFilter.ERODE -> Triple(applyErode(baseFrame), true, 0L)
            OpenCvFilter.OPEN -> Triple(applyMorphEx(baseFrame, Imgproc.MORPH_OPEN), true, 0L)
            OpenCvFilter.CLOSE -> Triple(applyMorphEx(baseFrame, Imgproc.MORPH_CLOSE), true, 0L)
            OpenCvFilter.GRADIENT -> Triple(applyMorphEx(baseFrame, Imgproc.MORPH_GRADIENT), true, 0L)
            OpenCvFilter.TOP_HAT -> Triple(applyMorphEx(baseFrame, Imgproc.MORPH_TOPHAT), true, 0L)
            OpenCvFilter.BLACK_HAT -> Triple(applyMorphEx(baseFrame, Imgproc.MORPH_BLACKHAT), true, 0L)
            OpenCvFilter.APRIL_TAGS -> Triple(applyAprilTagDetection(baseFrame), true, 0L)
            OpenCvFilter.ARUCO -> Triple(applyArucoDetection(baseFrame), true, 0L)
            OpenCvFilter.QR_CODE -> Triple(applyQrCodeDetection(baseFrame), true, 0L)
            OpenCvFilter.CCTAG -> Triple(applyCCTagDetection(baseFrame), true, 0L)
            OpenCvFilter.CHESSBOARD_CALIBRATION -> Triple(applyChessboardCalibration(baseFrame), true, 0L)
            OpenCvFilter.UNDISTORT -> Triple(applyUndistort(baseFrame), true, 0L)
            OpenCvFilter.VISUAL_ODOMETRY -> Triple(applyVisualOdometry(baseFrame), true, 0L)
            OpenCvFilter.POINT_CLOUD -> Triple(applyPointCloud(baseFrame), true, 0L)
            OpenCvFilter.PLANE_DETECTION -> Triple(applyPlaneDetection(baseFrame), true, 0L)
            OpenCvFilter.VANISHING_POINTS -> Triple(applyVanishingPoints(baseFrame), true, 0L)
            OpenCvFilter.MEDIAN_BLUR -> Triple(applyMedianBlur(baseFrame), true, 0L)
            OpenCvFilter.BILATERAL_FILTER -> Triple(applyBilateralFilter(baseFrame), true, 0L)
            OpenCvFilter.BOX_FILTER -> Triple(applyBoxFilter(baseFrame), true, 0L)
            OpenCvFilter.ADAPTIVE_THRESHOLD -> Triple(applyAdaptiveThreshold(baseFrame), true, 0L)
            OpenCvFilter.HISTOGRAM_EQUALIZATION -> Triple(applyHistogramEqualization(baseFrame), true, 0L)
            OpenCvFilter.SCHARR -> Triple(applyScharr(baseFrame), true, 0L)
            OpenCvFilter.PREWITT -> Triple(applyPrewitt(baseFrame), true, 0L)
            OpenCvFilter.ROBERTS -> Triple(applyRoberts(baseFrame), true, 0L)
            OpenCvFilter.INVERT -> Triple(applyInvert(baseFrame), true, 0L)
            OpenCvFilter.SEPIA -> Triple(applySepia(baseFrame), true, 0L)
            OpenCvFilter.EMBOSS -> Triple(applyEmboss(baseFrame), true, 0L)
            OpenCvFilter.PIXELATE -> Triple(applyPixelate(baseFrame), true, 0L)
            OpenCvFilter.CARTOON -> Triple(applyCartoon(baseFrame), true, 0L)
            else -> Triple(baseFrame.clone(), true, 0L)
        }
        val processed = processedPair.first
        val shouldReleaseProcessed = processedPair.second

        val benchmarkAfterNs = if (shouldBenchmark) {
            processedPair.third
        } else {
            0L
        }

        val result = createBitmap(processed.cols(), processed.rows())
        Utils.matToBitmap(processed, result)
        if (shouldReleaseProcessed) {
            processed.release()
        }
        if (isActiveVisionEnabled) {
            baseFrame.release()
        }
        if (shouldBenchmark && benchmarkAfterNs > 0L) {
            updateBenchmark(filter, benchmarkBeforeNs, benchmarkAfterNs)
        }
        return result
    }

    private fun ensureMat(mat: Mat, rows: Int, cols: Int, type: Int): Mat {
        if (mat.rows() != rows || mat.cols() != cols || mat.type() != type) {
            mat.create(rows, cols, type)
        }
        return mat
    }

    private fun processHotFilterBuffered(src: Mat, filter: OpenCvFilter): Triple<Mat, Boolean, Long> {
        val start = System.nanoTime()
        val output = when (filter) {
            OpenCvFilter.ORIGINAL -> {
                val out = ensureMat(originalBuffer, src.rows(), src.cols(), src.type())
                src.copyTo(out)
                out
            }
            OpenCvFilter.GRAYSCALE -> {
                val gray = ensureMat(grayBuffer, src.rows(), src.cols(), CvType.CV_8UC1)
                val out = ensureMat(grayRgbaBuffer, src.rows(), src.cols(), CvType.CV_8UC4)
                Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
                Imgproc.cvtColor(gray, out, Imgproc.COLOR_GRAY2RGBA)
                out
            }
            OpenCvFilter.CANNY_EDGES -> {
                val gray = ensureMat(grayBuffer, src.rows(), src.cols(), CvType.CV_8UC1)
                val blur = ensureMat(cannyBlurBuffer, src.rows(), src.cols(), CvType.CV_8UC1)
                val edges = ensureMat(cannyEdgesBuffer, src.rows(), src.cols(), CvType.CV_8UC1)
                val out = ensureMat(cannyRgbaBuffer, src.rows(), src.cols(), CvType.CV_8UC4)
                Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
                Imgproc.GaussianBlur(gray, blur, Size(5.0, 5.0), 0.0)
                Imgproc.Canny(blur, edges, 50.0, 150.0)
                Imgproc.cvtColor(edges, out, Imgproc.COLOR_GRAY2RGBA)
                out
            }
            OpenCvFilter.GAUSSIAN_BLUR -> {
                val out = ensureMat(gaussianBuffer, src.rows(), src.cols(), src.type())
                Imgproc.GaussianBlur(src, out, Size(5.0, 5.0), 0.0)
                out
            }
            else -> src
        }
        return Triple(output, false, System.nanoTime() - start)
    }

    private fun processHotFilterLegacy(src: Mat, filter: OpenCvFilter): Mat {
        return when (filter) {
            OpenCvFilter.ORIGINAL -> src.clone()
            OpenCvFilter.GRAYSCALE -> applyGrayscale(src)
            OpenCvFilter.CANNY_EDGES -> applyCanny(src)
            OpenCvFilter.GAUSSIAN_BLUR -> applyGaussianBlur(src)
            else -> src.clone()
        }
    }

    private fun updateBenchmark(filter: OpenCvFilter, beforeNs: Long, afterNs: Long) {
        val acc = benchmarkAccumulators.getOrPut(filter) { RuntimeBenchmarkAccumulator() }
        if (acc.samples >= benchmarkSampleLimit) {
            return
        }
        acc.samples += 1
        acc.beforeNs += beforeNs
        acc.afterNs += afterNs
    }

    fun consumeBenchmarkSnapshot(filter: OpenCvFilter): RuntimeBenchmarkSnapshot? {
        val acc = benchmarkAccumulators[filter] ?: return null
        if (acc.samples == 0 || acc.samples < benchmarkSampleLimit) {
            return null
        }
        benchmarkAccumulators.remove(filter)
        val avgBeforeMs = acc.beforeNs.toDouble() / acc.samples / 1_000_000.0
        val avgAfterMs = acc.afterNs.toDouble() / acc.samples / 1_000_000.0
        val fpsBefore = if (avgBeforeMs > 0.0) 1000.0 / avgBeforeMs else 0.0
        val fpsAfter = if (avgAfterMs > 0.0) 1000.0 / avgAfterMs else 0.0
        return RuntimeBenchmarkSnapshot(
            filter = filter,
            samples = acc.samples,
            avgBeforeMs = avgBeforeMs,
            avgAfterMs = avgAfterMs,
            fpsBefore = fpsBefore,
            fpsAfter = fpsAfter,
        )
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
                val pts = ptsToList(c)
                val markerId = ids.get(i, 0)[0].toInt()
                val poseEstimate = drawMarkerPoseOverlay(
                    res = res,
                    corners = pts,
                    markerKey = "april:$markerId",
                    markerLabel = "AprilTag#$markerId",
                )
                val detection = MarkerDetection.AprilTag(
                    markerId = markerId,
                    corners = pts,
                    rvec = poseEstimate?.rvec,
                    tvec = poseEstimate?.tvec,
                    quality = poseEstimate?.quality ?: MarkerDetection.Quality(),
                )
                drawCornerOutlineWithOrder(res, pts)
                drawMarkerLabel(res, detection, poseEstimate?.metrics)
                logMarkerDiagnostics(detection)
                detections.add(detection)
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
                val markerId = ids.get(i, 0)[0].toInt()
                val poseEstimate = drawMarkerPoseOverlay(
                    res = res,
                    corners = pts,
                    markerKey = "aruco:$markerId",
                    markerLabel = "ArUco#$markerId",
                )
                val detection = MarkerDetection.Aruco(
                    markerId = markerId,
                    corners = pts,
                    rvec = poseEstimate?.rvec,
                    tvec = poseEstimate?.tvec,
                    quality = poseEstimate?.quality ?: MarkerDetection.Quality(),
                )
                drawCornerOutlineWithOrder(res, pts)
                drawMarkerLabel(res, detection, poseEstimate?.metrics)
                logMarkerDiagnostics(detection)
                detections.add(detection)
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

    private fun drawCornerOutlineWithOrder(res: Mat, corners: List<Pair<Float, Float>>) {
        if (corners.size < 4) return
        for (i in corners.indices) {
            val a = corners[i]
            val b = corners[(i + 1) % corners.size]
            val pa = Point(a.first.toDouble(), a.second.toDouble())
            val pb = Point(b.first.toDouble(), b.second.toDouble())
            val color = OVERLAY_CORNER_COLORS[i % OVERLAY_CORNER_COLORS.size]
            Imgproc.line(res, pa, pb, color, 3)
            Imgproc.circle(res, pa, 4, color, -1)
            Imgproc.putText(
                res,
                "$i",
                Point(pa.x + 6.0, pa.y - 6.0),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
            )
        }
    }

    private fun drawMarkerPoseOverlay(
        res: Mat,
        corners: List<Pair<Float, Float>>,
        markerKey: String,
        markerLabel: String,
    ): MarkerPoseEstimate? {
        val calib = calibrator?.calibrationResult ?: return null
        if (corners.size != 4) return null
        val imagePoints = MatOfPoint2f(
            Point(corners[0].first.toDouble(), corners[0].second.toDouble()),
            Point(corners[1].first.toDouble(), corners[1].second.toDouble()),
            Point(corners[2].first.toDouble(), corners[2].second.toDouble()),
            Point(corners[3].first.toDouble(), corners[3].second.toDouble()),
        )
        val half = MARKER_SIZE_METERS / 2.0
        val objectPoints = MatOfPoint3f(
            Point3(-half, half, 0.0),
            Point3(half, half, 0.0),
            Point3(half, -half, 0.0),
            Point3(-half, -half, 0.0),
        )
        val rvec = Mat()
        val tvec = Mat()
        val solved = Calib3d.solvePnP(
            objectPoints,
            imagePoints,
            calib.cameraMatrix,
            calib.distCoeffs,
            rvec,
            tvec,
            false,
            Calib3d.SOLVEPNP_IPPE_SQUARE,
        ) || Calib3d.solvePnP(
            objectPoints,
            imagePoints,
            calib.cameraMatrix,
            calib.distCoeffs,
            rvec,
            tvec,
        )
        if (!solved) {
            imagePoints.release()
            objectPoints.release()
            rvec.release()
            tvec.release()
            return null
        }

        val rmat = Mat()
        Calib3d.Rodrigues(rvec, rmat)
        val zAxis = doubleArrayOf(rmat.get(0, 2)[0], rmat.get(1, 2)[0], rmat.get(2, 2)[0])

        val euler = rotationMatrixToEuler(rmat)
        val prev = markerPoseStates[markerKey]
        val smoothed = if (prev != null) {
            val smoothYaw = smoothAngle(prev.yawDeg, euler[0])
            val smoothPitch = smoothAngle(prev.pitchDeg, euler[1])
            val smoothRoll = smoothAngle(prev.rollDeg, euler[2])
            val mixedTvec = Mat()
            Core.addWeighted(prev.tvec, 1.0 - ORIENTATION_SMOOTH_ALPHA, tvec, ORIENTATION_SMOOTH_ALPHA, 0.0, mixedTvec)
            PoseState(
                tvec = mixedTvec,
                yawDeg = smoothYaw,
                pitchDeg = smoothPitch,
                rollDeg = smoothRoll,
                zAxisCamera = zAxis,
            )
        } else {
            PoseState(tvec.clone(), euler[0], euler[1], euler[2], zAxis)
        }

        val stableRvec = eulerToRvec(smoothed.yawDeg, smoothed.pitchDeg, smoothed.rollDeg)
        val zAligned = ensureZAxisConsistency(
            markerKey = markerKey,
            zAxis = smoothed.zAxisCamera,
            markerLabel = markerLabel,
        )
        if (!zAligned) {
            Log.w(TAG, "Potential axis inconsistency for $markerLabel")
        }
        Calib3d.drawFrameAxes(
            res,
            calib.cameraMatrix,
            calib.distCoeffs,
            stableRvec,
            smoothed.tvec,
            AXIS_LENGTH_METERS.toFloat(),
        )
        val reprojectionError = computeReprojectionError(
            objectPoints,
            imagePoints,
            stableRvec,
            smoothed.tvec,
            calib.cameraMatrix,
            calib.distCoeffs,
        )
        val distance = norm3(smoothed.tvec)
        val confidence = 1.0 / (1.0 + reprojectionError)

        markerPoseStates[markerKey]?.tvec?.release()
        markerPoseStates[markerKey] = PoseState(
            smoothed.tvec.clone(),
            smoothed.yawDeg,
            smoothed.pitchDeg,
            smoothed.rollDeg,
            smoothed.zAxisCamera,
        )

        val poseRvec = List(3) { index -> stableRvec.get(index, 0)[0] }
        val poseTvec = List(3) { index -> smoothed.tvec.get(index, 0)[0] }
        val metrics = PoseOverlayMetrics(
            distanceMeters = distance,
            yawDeg = smoothed.yawDeg,
            pitchDeg = smoothed.pitchDeg,
            rollDeg = smoothed.rollDeg,
            reprojectionErrorPx = reprojectionError,
            confidence = confidence,
        )
        imagePoints.release()
        objectPoints.release()
        rvec.release()
        tvec.release()
        rmat.release()
        stableRvec.release()
        smoothed.tvec.release()
        return MarkerPoseEstimate(
            rvec = poseRvec,
            tvec = poseTvec,
            quality = MarkerDetection.Quality(
                confidence = confidence,
                reprojectionErrorPx = reprojectionError,
            ),
            metrics = metrics,
        )
    }

    private fun drawMarkerLabel(
        res: Mat,
        detection: MarkerDetection,
        metrics: PoseOverlayMetrics?,
    ) {
        val anchor = detection.corners.minByOrNull { it.second } ?: Pair(30f, 30f)
        val x = anchor.first.toDouble()
        val y = (anchor.second - 12f).toDouble().coerceAtLeast(30.0)
        val lines = buildList {
            add("${detection.type}:${detection.id}")
            if (metrics != null) {
                add("dist=%.2fm".format(metrics.distanceMeters))
                add(
                    "yaw/pitch/roll=%.1f/%.1f/%.1f".format(
                        metrics.yawDeg,
                        metrics.pitchDeg,
                        metrics.rollDeg,
                    ),
                )
                add(detection.quality.toOverlayString())
            } else {
                add(detection.quality.toOverlayString())
            }
        }
        lines.forEachIndexed { i, line ->
            Imgproc.putText(
                res,
                line,
                Point(x, y + i * LABEL_LINE_HEIGHT),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                LABEL_FONT_SCALE,
                Scalar(240.0, 240.0, 240.0, 255.0),
                2,
            )
        }
    }

    private fun ensureZAxisConsistency(markerKey: String, zAxis: DoubleArray, markerLabel: String): Boolean {
        val dotWithCamera = zAxis.getOrElse(2) { 0.0 }
        val prev = markerPoseStates[markerKey]
        if (dotWithCamera < 0.0) {
            Log.w(TAG, "$markerLabel has negative Z-axis dot with camera")
            return false
        }
        if (prev != null) {
            val dot = prev.zAxisCamera.zip(zAxis).sumOf { it.first * it.second }
            if (dot < 0.0) {
                Log.w(TAG, "$markerLabel Z-axis flipped between frames")
                return false
            }
        }
        return true
    }

    private fun computeReprojectionError(
        objectPoints: MatOfPoint3f,
        imagePoints: MatOfPoint2f,
        rvec: Mat,
        tvec: Mat,
        cameraMatrix: Mat,
        distCoeffs: Mat,
    ): Double {
        val projected = MatOfPoint2f()
        Calib3d.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, projected)
        val observed = imagePoints.toArray()
        val reproj = projected.toArray()
        projected.release()
        if (observed.isEmpty() || observed.size != reproj.size) return Double.POSITIVE_INFINITY
        var errSq = 0.0
        for (i in observed.indices) {
            val dx = observed[i].x - reproj[i].x
            val dy = observed[i].y - reproj[i].y
            errSq += dx * dx + dy * dy
        }
        return sqrt(errSq / observed.size)
    }

    private fun norm3(tvec: Mat): Double {
        val x = tvec.get(0, 0)[0]
        val y = tvec.get(1, 0)[0]
        val z = tvec.get(2, 0)[0]
        return sqrt(x * x + y * y + z * z)
    }

    private fun smoothAngle(prevDeg: Double, currentDeg: Double): Double {
        val delta = normalizeAngle(currentDeg - prevDeg).coerceIn(-MAX_ANGLE_STEP_DEG, MAX_ANGLE_STEP_DEG)
        return normalizeAngle(prevDeg + ORIENTATION_SMOOTH_ALPHA * delta)
    }

    private fun normalizeAngle(angleDeg: Double): Double {
        var out = angleDeg
        while (out > 180.0) out -= 360.0
        while (out < -180.0) out += 360.0
        return out
    }

    private fun rotationMatrixToEuler(rmat: Mat): DoubleArray {
        val r00 = rmat.get(0, 0)[0]
        val r10 = rmat.get(1, 0)[0]
        val r20 = rmat.get(2, 0)[0]
        val r21 = rmat.get(2, 1)[0]
        val r22 = rmat.get(2, 2)[0]
        val sy = sqrt(r00 * r00 + r10 * r10)
        val singular = sy < 1e-6

        val yaw: Double
        val pitch: Double
        val roll: Double
        if (!singular) {
            yaw = Math.toDegrees(atan2(r10, r00))
            pitch = Math.toDegrees(atan2(-r20, sy))
            roll = Math.toDegrees(atan2(r21, r22))
        } else {
            yaw = Math.toDegrees(atan2(-rmat.get(0, 1)[0], rmat.get(1, 1)[0]))
            pitch = Math.toDegrees(atan2(-r20, sy))
            roll = 0.0
        }
        return doubleArrayOf(yaw, pitch, roll)
    }

    private fun eulerToRvec(yawDeg: Double, pitchDeg: Double, rollDeg: Double): Mat {
        val y = Math.toRadians(yawDeg)
        val p = Math.toRadians(pitchDeg)
        val r = Math.toRadians(rollDeg)
        val cy = kotlin.math.cos(y)
        val sy = kotlin.math.sin(y)
        val cp = kotlin.math.cos(p)
        val sp = kotlin.math.sin(p)
        val cr = kotlin.math.cos(r)
        val sr = kotlin.math.sin(r)
        val rot = Mat(3, 3, CvType.CV_64F)
        rot.put(0, 0, cy * cp)
        rot.put(0, 1, cy * sp * sr - sy * cr)
        rot.put(0, 2, cy * sp * cr + sy * sr)
        rot.put(1, 0, sy * cp)
        rot.put(1, 1, sy * sp * sr + cy * cr)
        rot.put(1, 2, sy * sp * cr - cy * sr)
        rot.put(2, 0, -sp)
        rot.put(2, 1, cp * sr)
        rot.put(2, 2, cp * cr)
        val rvec = Mat()
        Calib3d.Rodrigues(rot, rvec)
        rot.release()
        return rvec
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
            val detection = MarkerDetection.QrCode(
                text = data,
                corners = ptsList.map { Pair(it.x.toFloat(), it.y.toFloat()) },
            )
            logMarkerDiagnostics(detection)
            onMarkersDetected?.invoke(listOf(detection))
            pts.release()
        }
        gray.release(); points.release(); return res
    }

    private fun logMarkerDiagnostics(detection: MarkerDetection) {
        Log.d(TAG, "marker_detection ${detection.toDiagnosticSummary()}")
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
        visualOdometryEngine.processFrameRgba(src, calibrator)
        val res = Mat.zeros(src.rows(), src.cols(), src.type())
        val cloud = visualOdometryEngine.lastPointCloud
        if (cloud != null) {
            if (isVoMeshEnabled) {
                for ((p1, p2) in cloud.edges) {
                    Imgproc.line(res, p1, p2, Scalar(0.0, 128.0, 255.0, 255.0), POINT_CLOUD_MESH_THICKNESS)
                }
            }
            for (pt in cloud.points) {
                Imgproc.circle(res, pt, POINT_CLOUD_CIRCLE_RADIUS, Scalar(0.0, 255.0, 255.0, 255.0), -1)
            }
            Imgproc.putText(res, "$labelPointCloud: ${cloud.points.size}", Point(30.0, 30.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255.0, 255.0, 255.0, 255.0), 2)
        }
        return res
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
            val usedClusters = mutableSetOf<Int>()

            for (i in sortedClusters.indices) {
                for (j in i + 1 until sortedClusters.size) {
                    if (planeIdx >= 3) break
                    if (i in usedClusters || j in usedClusters) continue
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
                    usedClusters.add(i)
                    usedClusters.add(j)
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

    /** Inverts all colour channels using a bitwise NOT operation. */
    private fun applyInvert(src: Mat): Mat {
        val res = Mat(); Core.bitwise_not(src, res); return res
    }

    /**
     * Applies a warm sepia-tone effect using the Adobe Photoshop sepia preset matrix.
     * The frame is transformed from RGBA to RGB, each pixel is linearly projected
     * through the sepia colour matrix, and the result is converted back to RGBA.
     * Matches the Python [apply_sepia] implementation exactly.
     */
    private fun applySepia(src: Mat): Mat {
        val rgb = Mat(); Imgproc.cvtColor(src, rgb, Imgproc.COLOR_RGBA2RGB)
        val rgb32f = Mat(); rgb.convertTo(rgb32f, CvType.CV_32FC3)
        // Adobe Photoshop sepia matrix (rows = output RGB channels, cols = input RGB channels):
        // out_R = 0.393*R + 0.769*G + 0.189*B
        // out_G = 0.349*R + 0.686*G + 0.168*B
        // out_B = 0.272*R + 0.534*G + 0.131*B
        val kernel = Mat(3, 3, CvType.CV_32F)
        kernel.put(0, 0, floatArrayOf(0.393f, 0.769f, 0.189f))  // output R row
        kernel.put(1, 0, floatArrayOf(0.349f, 0.686f, 0.168f))  // output G row
        kernel.put(2, 0, floatArrayOf(0.272f, 0.534f, 0.131f))  // output B row
        val sepia32f = Mat(); Core.transform(rgb32f, sepia32f, kernel)
        val sepia8u = Mat(); sepia32f.convertTo(sepia8u, CvType.CV_8UC3)  // saturate_cast clips [0,255]
        val res = Mat(); Imgproc.cvtColor(sepia8u, res, Imgproc.COLOR_RGB2RGBA)
        rgb.release(); rgb32f.release(); kernel.release(); sepia32f.release(); sepia8u.release()
        return res
    }

    /**
     * Produces an emboss / relief effect using a directional 3×3 kernel that
     * highlights edges as raised surface texture against a mid-grey background.
     */
    private fun applyEmboss(src: Mat): Mat {
        val gray = Mat(); val embossed = Mat(); val res = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        val kernel = Mat(3, 3, CvType.CV_32F)
        kernel.put(0, 0, -2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0)
        Imgproc.filter2D(gray, embossed, -1, kernel)
        Core.add(embossed, Scalar(128.0), embossed)       // shift to mid-grey baseline
        Imgproc.cvtColor(embossed, res, Imgproc.COLOR_GRAY2RGBA)
        gray.release(); embossed.release(); kernel.release(); return res
    }

    /**
     * Pixelates the frame by downscaling to a tiny resolution and then scaling
     * back up with nearest-neighbour interpolation, creating a blocky pixel-art look.
     *
     * Uses [PIXELATE_BLOCK_SIZE] as the pixel block side length. Downsampling uses
     * INTER_AREA for better quality area-averaging (consistent with the Python implementation).
     */
    private fun applyPixelate(src: Mat): Mat {
        val small = Mat()
        val res = Mat()
        Imgproc.resize(src, small, Size(src.cols().toDouble() / PIXELATE_BLOCK_SIZE, src.rows().toDouble() / PIXELATE_BLOCK_SIZE), 0.0, 0.0, Imgproc.INTER_AREA)
        Imgproc.resize(small, res, Size(src.cols().toDouble(), src.rows().toDouble()), 0.0, 0.0, Imgproc.INTER_NEAREST)
        small.release(); return res
    }

    /**
     * Creates a cartoon / comic-book effect by combining a heavily smoothed
     * (bilateral-filtered) colour image with binary Canny edges drawn in black.
     */
    private fun applyCartoon(src: Mat): Mat {
        // Step 1: colour-smooth with bilateral filter (applied on RGB, not RGBA)
        val rgb = Mat(); val smoothed = Mat()
        Imgproc.cvtColor(src, rgb, Imgproc.COLOR_RGBA2RGB)
        Imgproc.bilateralFilter(rgb, smoothed, 9, 75.0, 75.0)

        // Step 2: extract edges on grayscale
        val gray = Mat(); val blurred = Mat(); val edges = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.medianBlur(gray, blurred, 7)
        Imgproc.Canny(blurred, edges, 80.0, 200.0)

        // Step 3: invert edges → black lines on white mask
        val edgesInv = Mat(); Core.bitwise_not(edges, edgesInv)

        // Step 4: dilate the edge mask to make lines slightly thicker
        val dilatedEdges = Mat()
        Imgproc.dilate(edgesInv, dilatedEdges, dilationKernel3x3)

        // Step 5: apply edge mask to smoothed colour image (black out edge pixels)
        val edgeMask3ch = Mat()
        Imgproc.cvtColor(dilatedEdges, edgeMask3ch, Imgproc.COLOR_GRAY2RGB)
        val maskedSmoothed = Mat()
        Core.bitwise_and(smoothed, edgeMask3ch, maskedSmoothed)

        // Step 6: convert back to RGBA
        val res = Mat()
        Imgproc.cvtColor(maskedSmoothed, res, Imgproc.COLOR_RGB2RGBA)

        rgb.release(); smoothed.release(); gray.release(); blurred.release()
        edges.release(); edgesInv.release(); dilatedEdges.release()
        edgeMask3ch.release(); maskedSmoothed.release()
        return res
    }
}
