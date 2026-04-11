package pl.edu.mobilecv

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import org.opencv.android.Utils
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvException
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.MatOfPoint3f
import org.opencv.core.Point
import org.opencv.core.Point3
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.ArucoDetector
import org.opencv.objdetect.DetectorParameters
import org.opencv.objdetect.Objdetect
import org.opencv.objdetect.QRCodeDetector
import androidx.core.graphics.createBitmap
import pl.edu.mobilecv.vision.CameraCalibrator
import pl.edu.mobilecv.odometry.VisualOdometryEngine
import pl.edu.mobilecv.odometry.FullOdometryEngine
import pl.edu.mobilecv.tracking.ObjectPoseTracker
import kotlin.math.atan2
import kotlin.math.min
import kotlin.math.sqrt
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

/**
 * Applies OpenCV image-processing filters to Android [Bitmap] frames.
 */
class ImageProcessor {
    /**
     * Bazowy stan wejściowy przekazywany do modułów przetwarzania klatek.
     */
    sealed interface ModuleState

    /**
     * Stan modułu morfologii. Przechowuje rozmiar pół-jądra operacji.
     */
    data class MorphologyState(
        val kernelHalfSize: Int = 4,
    ) : ModuleState

    /**
     * Stan modułu odometrii wizualnej.
     */
    data class OdometryState(
        val maxFeatures: Int = 300,
        val minParallax: Double = 1.0,
        val meshEnabled: Boolean = false,
    ) : ModuleState

    /**
     * Stan modułu geometrii 3D.
     */
    data class GeometryState(
        val maxPlanes: Int = 3,
    ) : ModuleState

    /**
     * Domyślny stan dla filtrów, które nie wymagają dodatkowych parametrów.
     */
    data object EmptyState : ModuleState

    /**
     * Interfejs modułu przetwarzania pojedynczej klatki.
     */
    private interface FrameModule<S : ModuleState> {
        fun process(frame: Mat, filter: OpenCvFilter, state: S): Mat
        fun reset()
    }

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
        val temporalDebug: PoseTemporalDebugMetrics,
    )

    private data class PoseState(
        val zAxisCamera: DoubleArray,
    )

    data class PoseTemporalDebugMetrics(
        val mode: PoseOutputMode,
        val jitterRawTvecMm: Double?,
        val jitterRawRvecDeg: Double?,
        val jitterSmoothTvecMm: Double?,
        val jitterSmoothRvecDeg: Double?,
        val stableStaticScene: Boolean,
    )

    private data class MarkerPoseEstimate(
        val rvec: List<Double>,
        val tvec: List<Double>,
        val quality: MarkerDetection.Quality,
        val metrics: PoseOverlayMetrics,
    )

    private data class CircleData(
        val center: Point,
        val radius: Float,
        val circularity: Double
    )

    var calibrator: CameraCalibrator? = null
    var mediaPipeProcessor: MediaPipeProcessor? = null
    var yoloProcessor: YoloProcessor? = null
    var tfliteProcessor: TfliteProcessor? = null

    var labelFrameCountSuffix: String = "klatek"
    var labelBoardNotFound: String = "Brak szachownicy"
    var labelNoCalibration: String = "Brak kalibracji"
    var labelOdometryTracks: String = "Ścieżki"
    var labelPointCloud: String = "Chmura"
    var labelNoPlanes: String = "Brak płaszczyzn"
    var labelNoVanishingPoints: String = "Brak punktów zbieżności"
    var labelNoLines: String = "Brak linii w scenie"
    var labelPlanes: String = "Płaszczyzny"
    var labelLines: String = "Linie"
    var labelGroups: String = "Grupy"
    var labelVpError: String = "Błąd VP"
    var labelGeometryError: String = "Błąd geometrii"
    var labelVoMaxFeaturesDesc: String = ""
    var labelVoMinParallaxDesc: String = ""
    var labelVoColorDepthDesc: String = ""
    var labelFullOdometryTracks: String = "Tory"
    var labelFullOdometryInliers: String = "Inlierów"
    var labelFullOdometryFrames: String = "Klatki"
    var labelFullOdometrySteps: String = "Kroki"
    var labelFullOdometryPos: String = "Poz"
    var labelTrajectory: String = "Trajektoria"
    var labelMap3D: String = "Mapa 3D"
    var labelOdometryPoints: String = "Punkty"
    var labelCollectingData: String = "Zbieranie danych..."
    var labelCollectingPoints: String = "Zbieranie punktów..."

    var isActiveVisionEnabled: Boolean = false
    var isActiveVisionVisualizationEnabled: Boolean = false

    val lastPointCloud: VisualOdometryEngine.PointCloudState?
        get() = visualOdometryEngine.lastPointCloud

    val currentSlamMap: FullOdometryEngine.MapState
        get() = fullOdometryEngine.currentMap

    @Volatile
    var debugUndistortBeforeGeometry: Boolean = false

    var morphKernelSize: Int = 4
    var voMaxFeatures: Int = 300
    var voMinParallax: Double = 1.0
    var isVoMeshEnabled: Boolean = false
    var geometryMaxPlanes: Int = 3

    private val geometryProcessor = GeometryProcessor(::logExceptionTelemetry)
    private val activeVisionOptimizer = ActiveVisionOptimizer()
    private val visualOdometryEngine = VisualOdometryEngine().also {
        it.maxFeatures = OdometryState().maxFeatures
        it.minParallax = OdometryState().minParallax
        it.isMeshEnabled = OdometryState().meshEnabled
    }
    val fullOdometryEngine = FullOdometryEngine().also {
        it.maxFeatures = OdometryState().maxFeatures
        it.minParallax = OdometryState().minParallax
    }

    /** Callback for automatic map export. */
    var onLargeMapDetected: ((FullOdometryEngine.MapState) -> Unit)? = null
        set(value) {
            field = value
            fullOdometryEngine.onLargeMapDetected = value
        }

    /** Callback for loop closure events. */
    var onLoopClosed: ((String) -> Unit)? = null
        set(value) {
            field = value
            fullOdometryEngine.onLoopClosed = value
        }

    private val detectorParameters by lazy {
        DetectorParameters().apply {
            _adaptiveThreshWinSizeMin = 3
            _adaptiveThreshWinSizeMax = 33
            _adaptiveThreshWinSizeStep = 10
            _cornerRefinementMethod = Objdetect.CORNER_REFINE_SUBPIX
            _cornerRefinementWinSize = 5
            _cornerRefinementMaxIterations = 30
            _cornerRefinementMinAccuracy = 0.1
        }
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
        private const val LABEL_LINE_HEIGHT = 24.0
        private const val LABEL_FONT_SCALE = 0.55
        private val OVERLAY_CORNER_COLORS = listOf(
            Scalar(255.0, 70.0, 70.0, 255.0),
            Scalar(70.0, 255.0, 70.0, 255.0),
            Scalar(70.0, 120.0, 255.0, 255.0),
            Scalar(255.0, 220.0, 70.0, 255.0),
        )
        private const val POINT_CLOUD_MIN_CIRCLE_RADIUS = 3
        private const val POINT_CLOUD_MAX_CIRCLE_RADIUS = 7
        private const val POINT_CLOUD_MESH_THICKNESS = 1
        private const val MAP_POINT_MIN_CIRCLE_RADIUS = 2
        private const val MAP_POINT_MAX_CIRCLE_RADIUS = 5
        private const val PIXELATE_BLOCK_SIZE = 16
        private const val FULL_ODOMETRY_HUD_X = 20.0
        private const val FULL_ODOMETRY_HUD_LINE_HEIGHT = 28.0
        private const val EPSILON_THRESHOLD = 1e-6

        private const val CCTAG_MIN_CONTOUR_AREA = 50.0
        private const val CCTAG_MIN_CIRCULARITY = 0.5
        private const val CCTAG_MAX_CENTRE_OFFSET_FRACTION = 0.25
        private const val CCTAG_MIN_RINGS = 2
        private const val CCTAG_MAX_RINGS = 5
    }

    private val srcBuffer = Mat()
    private val baseFrameBuffer = Mat()
    private val originalBuffer = Mat()

    fun release() {
        srcBuffer.release()
        baseFrameBuffer.release()
        originalBuffer.release()
        aprilTagDetector.apply { /* No explicit release in some versions, but good to check */ }
        arucoDetector.apply { }
    }

    /**
     * Resetuje zasoby modułu powiązanego z danym trybem analizy.
     */
    fun resetModule(mode: AnalysisMode) {
        when (mode) {
            AnalysisMode.ODOMETRY -> {
                odometryModule.reset()
                fullOdometryEngine.reset()
            }
            AnalysisMode.SLAM -> fullOdometryEngine.reset()
            else -> {
                // Pozostałe moduły nie utrzymują trwałego stanu między klatkami.
            }
        }
    }

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
    @Volatile
    var showFpsOverlay: Boolean = true
    @Volatile
    var poseSmoothingEnabled: Boolean = true
    @Volatile
    var poseTemporalFilterType: PoseTemporalFilterType = PoseTemporalFilterType.EMA
    @Volatile
    var poseOutputMode: PoseOutputMode = PoseOutputMode.SMOOTHED
    @Volatile
    var poseEmaAlpha: Double = 0.35
    @Volatile
    var poseOneEuroBeta: Double = 0.02
    private val poseTemporalFilter = PoseTemporalFilter()
    private val objectPoseTracker = ObjectPoseTracker()
    private val exceptionTelemetry = ConcurrentHashMap<String, AtomicInteger>()
    @Volatile
    private var lastCalibrationDiagnosticsKey: String? = null
    private val morphologyModule = object : FrameModule<MorphologyState> {
        override fun process(frame: Mat, filter: OpenCvFilter, state: MorphologyState): Mat = when (filter) {
            OpenCvFilter.DILATE -> LegacyFilters.applyMorphology(frame, -1, morphKernelSize)
            OpenCvFilter.ERODE -> LegacyFilters.applyMorphology(frame, -2, morphKernelSize)
            OpenCvFilter.OPEN ->
                LegacyFilters.applyMorphology(frame, Imgproc.MORPH_OPEN, morphKernelSize)
            OpenCvFilter.CLOSE ->
                LegacyFilters.applyMorphology(frame, Imgproc.MORPH_CLOSE, morphKernelSize)
            OpenCvFilter.GRADIENT ->
                LegacyFilters.applyMorphology(frame, Imgproc.MORPH_GRADIENT, morphKernelSize)
            OpenCvFilter.TOP_HAT ->
                LegacyFilters.applyMorphology(frame, Imgproc.MORPH_TOPHAT, morphKernelSize)
            OpenCvFilter.BLACK_HAT ->
                LegacyFilters.applyMorphology(frame, Imgproc.MORPH_BLACKHAT, morphKernelSize)
            else -> frame.clone()
        }

        override fun reset() {
            // Brak stanu wewnętrznego modułu morfologii do czyszczenia.
        }
    }

    private val odometryModule = object : FrameModule<OdometryState> {
        override fun process(frame: Mat, filter: OpenCvFilter, state: OdometryState): Mat {
            visualOdometryEngine.maxFeatures = voMaxFeatures
            visualOdometryEngine.minParallax = voMinParallax
            visualOdometryEngine.isMeshEnabled = isVoMeshEnabled
            return when (filter) {
                OpenCvFilter.VISUAL_ODOMETRY -> applyVisualOdometry(frame)
                OpenCvFilter.POINT_CLOUD -> applyPointCloud(frame, state)
                else -> frame.clone()
            }
        }

        override fun reset() {
            visualOdometryEngine.reset()
        }
    }

    private fun logExceptionTelemetry(scope: String, category: String, error: Throwable) {
        val key = "$scope:$category"
        val count = exceptionTelemetry.getOrPut(key) { AtomicInteger(0) }.incrementAndGet()
        Log.i(TAG, "Exception telemetry key=$key count=$count type=${error::class.java.simpleName}")
    }

    fun processFrame(bitmap: Bitmap, filter: OpenCvFilter, moduleState: ModuleState): Bitmap {
        if (filter != OpenCvFilter.VISUAL_ODOMETRY && filter != OpenCvFilter.POINT_CLOUD) {
            visualOdometryEngine.reset()
        }
        if (!filter.isFullOdometry) {
            fullOdometryEngine.reset()
        }
        if (filter.isMediaPipe) {
            val result = mediaPipeProcessor?.processFrame(bitmap, filter) ?: bitmap.copy(Bitmap.Config.ARGB_8888, false)
            if (showFpsOverlay) drawFpsOnBitmap(result)
            return result
        }
        if (filter.isYolo) {
            val result = yoloProcessor?.processFrame(bitmap, filter)
                ?: bitmap.copy(Bitmap.Config.ARGB_8888, false)
            if (showFpsOverlay) drawFpsOnBitmap(result)
            return result
        }
        if (filter.isTflite) {
            val result = tfliteProcessor?.processFrame(bitmap, filter)
                ?: bitmap.copy(Bitmap.Config.ARGB_8888, false)
            if (showFpsOverlay) drawFpsOnBitmap(result)
            return result
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
            OpenCvFilter.THRESHOLD -> Triple(LegacyFilters.applyThreshold(baseFrame), true, 0L)
            OpenCvFilter.SOBEL -> Triple(LegacyFilters.applySobel(baseFrame), true, 0L)
            OpenCvFilter.LAPLACIAN -> Triple(LegacyFilters.applyLaplacian(baseFrame), true, 0L)
            OpenCvFilter.DILATE,
            OpenCvFilter.ERODE,
            OpenCvFilter.OPEN,
            OpenCvFilter.CLOSE,
            OpenCvFilter.GRADIENT,
            OpenCvFilter.TOP_HAT,
            OpenCvFilter.BLACK_HAT -> Triple(
                runModuleSafely(
                    moduleName = "morphology",
                    filter = filter,
                    baseFrame = baseFrame,
                    module = morphologyModule,
                    state = moduleState as? MorphologyState ?: MorphologyState(),
                ),
                true,
                0L,
            )
            OpenCvFilter.APRIL_TAGS -> Triple(applyAprilTagDetection(baseFrame), true, 0L)
            OpenCvFilter.APRIL_TAG_3D -> Triple(applyAprilTag3D(baseFrame), true, 0L)
            OpenCvFilter.ARUCO -> Triple(applyArucoDetection(baseFrame), true, 0L)
            OpenCvFilter.ARUCO_3D -> Triple(applyAruco3D(baseFrame), true, 0L)
            OpenCvFilter.MARKER_UKF -> Triple(applyMarkerUkf(baseFrame), true, 0L)
            OpenCvFilter.QR_CODE -> Triple(applyQrCodeDetection(baseFrame), true, 0L)
            OpenCvFilter.QR_CODE_3D -> Triple(applyQrCode3D(baseFrame), true, 0L)
            OpenCvFilter.CCTAG -> Triple(applyCCTagDetection(baseFrame), true, 0L)
            OpenCvFilter.CHESSBOARD_CALIBRATION -> Triple(applyChessboardCalibration(baseFrame), true, 0L)
            OpenCvFilter.UNDISTORT -> Triple(applyUndistort(baseFrame), true, 0L)
            OpenCvFilter.VISUAL_ODOMETRY,
            OpenCvFilter.POINT_CLOUD -> Triple(
                runModuleSafely(
                    moduleName = "odometry",
                    filter = filter,
                    baseFrame = baseFrame,
                    module = odometryModule,
                    state = moduleState as? OdometryState ?: OdometryState(),
                ),
                true,
                0L,
            )
            OpenCvFilter.PLANE_DETECTION -> Triple(
                applyPlaneDetection(
                    baseFrame,
                    moduleState as? GeometryState ?: GeometryState(),
                ),
                true,
                0L,
            )
            OpenCvFilter.VANISHING_POINTS -> Triple(applyVanishingPoints(baseFrame), true, 0L)
            OpenCvFilter.MEDIAN_BLUR -> Triple(LegacyFilters.applyMedianBlur(baseFrame), true, 0L)
            OpenCvFilter.BILATERAL_FILTER -> Triple(LegacyFilters.applyBilateralFilter(baseFrame), true, 0L)
            OpenCvFilter.BOX_FILTER -> Triple(LegacyFilters.applyBoxFilter(baseFrame), true, 0L)
            OpenCvFilter.ADAPTIVE_THRESHOLD -> Triple(LegacyFilters.applyAdaptiveThreshold(baseFrame), true, 0L)
            OpenCvFilter.HISTOGRAM_EQUALIZATION -> Triple(LegacyFilters.applyHistogramEqualization(baseFrame), true, 0L)
            OpenCvFilter.SCHARR -> Triple(LegacyFilters.applyScharr(baseFrame), true, 0L)
            OpenCvFilter.PREWITT -> Triple(LegacyFilters.applyPrewitt(baseFrame), true, 0L)
            OpenCvFilter.ROBERTS -> Triple(LegacyFilters.applyRoberts(baseFrame), true, 0L)
            OpenCvFilter.INVERT -> Triple(LegacyFilters.applyInvert(baseFrame), true, 0L)
            OpenCvFilter.SEPIA -> Triple(LegacyFilters.applySepia(baseFrame), true, 0L)
            OpenCvFilter.EMBOSS -> Triple(LegacyFilters.applyEmboss(baseFrame), true, 0L)
            OpenCvFilter.PIXELATE -> Triple(LegacyFilters.applyPixelate(baseFrame, PIXELATE_BLOCK_SIZE), true, 0L)
            OpenCvFilter.CARTOON -> Triple(LegacyFilters.applyCartoon(baseFrame), true, 0L)
            OpenCvFilter.FULL_ODOMETRY -> Triple(applyFullOdometry(baseFrame), true, 0L)
            OpenCvFilter.ODOMETRY_TRAJECTORY -> Triple(applyOdometryTrajectory(baseFrame), true, 0L)
            OpenCvFilter.ODOMETRY_MAP -> Triple(applyOdometryMap(baseFrame), true, 0L)
            OpenCvFilter.DISTANCE_ESTIMATION -> Triple(applyDistanceEstimation(baseFrame), true, 0L)
            else -> Triple(baseFrame.clone(), true, 0L)
        }
        val processed = processedPair.first
        val shouldReleaseProcessed = processedPair.second

        val benchmarkAfterNs = if (shouldBenchmark) processedPair.third else 0L

        val result = createBitmap(processed.cols(), processed.rows())
        Utils.matToBitmap(processed, result)

        if (showFpsOverlay) {
            drawFpsOnBitmap(result)
        }

        if (shouldReleaseProcessed) processed.release()
        if (isActiveVisionEnabled) baseFrame.release()
        if (shouldBenchmark && benchmarkAfterNs > 0L) updateBenchmark(filter, benchmarkBeforeNs, benchmarkAfterNs)
        return result
    }

    private val fpsPaint = Paint().apply {
        color = Color.GREEN
        textSize = 40f
        isFakeBoldText = true
        setShadowLayer(5f, 0f, 0f, Color.BLACK)
    }
    private val fpsCounter = FpsCounter()

    private fun drawFpsOnBitmap(bitmap: Bitmap) {
        fpsCounter.onFrame()
        val canvas = Canvas(bitmap)
        val text = "FPS: ${"%.1f".format(fpsCounter.fps)}"
        canvas.drawText(text, 20f, bitmap.height - 20f, fpsPaint)
    }

    private fun ensureMat(mat: Mat, rows: Int, cols: Int, type: Int): Mat {
        if (mat.rows() != rows || mat.cols() != cols || mat.type() != type) {
            mat.create(rows, cols, type)
        }
        return mat
    }

    /**
     * Izoluje błąd pojedynczego modułu, aby nie wpływał na inne moduły.
     */
    private fun <S : ModuleState> runModuleSafely(
        moduleName: String,
        filter: OpenCvFilter,
        baseFrame: Mat,
        module: FrameModule<S>,
        state: S,
    ): Mat = try {
        module.process(baseFrame, filter, state)
    } catch (error: Throwable) {
        logExceptionTelemetry("frame_module", moduleName, error)
        Log.e(TAG, "FrameModule=$moduleName failed for filter=${filter.name}. Local fallback applied.", error)
        module.reset()
        baseFrame.clone()
    }

    private fun processHotFilterBuffered(src: Mat, filter: OpenCvFilter): Triple<Mat, Boolean, Long> {
        val start = System.nanoTime()
        val output = when (filter) {
            OpenCvFilter.ORIGINAL -> {
                val out = ensureMat(originalBuffer, src.rows(), src.cols(), src.type())
                src.copyTo(out)
                out
            }
            OpenCvFilter.GRAYSCALE -> LegacyFilters.applyGrayscale(src)
            OpenCvFilter.CANNY_EDGES -> LegacyFilters.applyCanny(src)
            OpenCvFilter.GAUSSIAN_BLUR -> LegacyFilters.applyGaussianBlur(src)
            else -> src
        }
        return Triple(output, output != src, System.nanoTime() - start)
    }

    private fun processHotFilterLegacy(src: Mat, filter: OpenCvFilter): Mat {
        return when (filter) {
            OpenCvFilter.ORIGINAL -> src.clone()
            OpenCvFilter.GRAYSCALE -> LegacyFilters.applyGrayscale(src)
            OpenCvFilter.CANNY_EDGES -> LegacyFilters.applyCanny(src)
            OpenCvFilter.GAUSSIAN_BLUR -> LegacyFilters.applyGaussianBlur(src)
            else -> src.clone()
        }
    }

    private fun updateBenchmark(filter: OpenCvFilter, beforeNs: Long, afterNs: Long) {
        val acc = benchmarkAccumulators.getOrPut(filter) { RuntimeBenchmarkAccumulator() }
        if (acc.samples >= benchmarkSampleLimit) return
        acc.samples += 1
        acc.beforeNs += beforeNs
        acc.afterNs += afterNs
    }

    fun consumeBenchmarkSnapshot(filter: OpenCvFilter): RuntimeBenchmarkSnapshot? {
        val acc = benchmarkAccumulators[filter] ?: return null
        if (acc.samples == 0 || acc.samples < benchmarkSampleLimit) return null
        benchmarkAccumulators.remove(filter)
        val avgBeforeMs = acc.beforeNs.toDouble() / acc.samples / 1_000_000.0
        val avgAfterMs = acc.afterNs.toDouble() / acc.samples / 1_000_000.0
        val fpsBefore = if (avgBeforeMs > 0.0) 1000.0 / avgBeforeMs else 0.0
        val fpsAfter = if (avgAfterMs > 0.0) 1000.0 / avgAfterMs else 0.0
        return RuntimeBenchmarkSnapshot(filter, acc.samples, avgBeforeMs, avgAfterMs, fpsBefore, fpsAfter)
    }

    private fun applyAprilTag3D(src: Mat): Mat {
        val res = src.clone()
        val detections = detectAprilTags(src)
        detections.forEach { detection ->
            val poseEstimate = drawMarker3DObject(res, detection.corners, "april3d:${detection.id}")
            drawCornerOutlineWithOrder(res, detection.corners)
            drawMarkerLabel(res, detection, poseEstimate?.metrics)
        }
        return res
    }

    private fun applyAruco3D(src: Mat): Mat {
        val res = src.clone()
        val detections = detectArucoMarkers(src)
        detections.forEach { detection ->
            val poseEstimate = drawMarker3DObject(res, detection.corners, "aruco3d:${detection.id}")
            drawCornerOutlineWithOrder(res, detection.corners)
            drawMarkerLabel(res, detection, poseEstimate?.metrics)
        }
        return res
    }

    private fun applyQrCode3D(src: Mat): Mat {
        val res = src.clone(); val points = Mat(); val gray = Mat()
        try {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
            val data = qrCodeDetector.detectAndDecode(gray, points)
            if (!data.isNullOrEmpty() && points.rows() >= 4) {
                val pts = ptsToList(points)
                val poseEstimate = drawMarker3DObject(res, pts, "qr3d:$data")
                val detection = MarkerDetection.QrCode(data, pts, poseEstimate?.rvec, poseEstimate?.tvec, poseEstimate?.quality ?: MarkerDetection.Quality())
                drawCornerOutlineWithOrder(res, pts)
                drawMarkerLabel(res, detection, poseEstimate?.metrics)
            }
        } finally {
            gray.release(); points.release()
        }
        return res
    }

    private fun drawMarker3DObject(res: Mat, corners: List<Pair<Float, Float>>, markerKey: String): MarkerPoseEstimate? {
        val calib = calibrator?.calibrationResult ?: return null
        if (corners.size != 4) return null
        
        val imagePoints = MatOfPoint2f(
            Point(corners[0].first.toDouble(), corners[0].second.toDouble()),
            Point(corners[1].first.toDouble(), corners[1].second.toDouble()),
            Point(corners[2].first.toDouble(), corners[2].second.toDouble()),
            Point(corners[3].first.toDouble(), corners[3].second.toDouble())
        )
        
        val half = MARKER_SIZE_METERS / 2.0
        val objectPoints = MatOfPoint3f(
            Point3(-half, half, 0.0),
            Point3(half, half, 0.0),
            Point3(half, -half, 0.0),
            Point3(-half, -half, 0.0)
        )
        
        val rvec = Mat(); val tvec = Mat()
        val solved = Calib3d.solvePnP(objectPoints, imagePoints, calib.cameraMatrix, calib.distCoeffs, rvec, tvec, false, Calib3d.SOLVEPNP_IPPE_SQUARE) || Calib3d.solvePnP(objectPoints, imagePoints, calib.cameraMatrix, calib.distCoeffs, rvec, tvec)
        
        if (!solved) {
            imagePoints.release(); objectPoints.release(); rvec.release(); tvec.release()
            return null
        }

        val filtered = selectPoseForRender(markerKey, tvec, rvec)
        val rmat = Mat(); Calib3d.Rodrigues(filtered.second, rmat)
        val euler = rotationMatrixToEuler(rmat)
        
        // Draw 3D Cube on the marker
        draw3DCubeOnMarker(res, filtered.second, filtered.first, calib.cameraMatrix, calib.distCoeffs)
        
        val reprojectionError = computeReprojectionError(objectPoints, imagePoints, filtered.second, filtered.first, calib.cameraMatrix, calib.distCoeffs)
        val distance = norm3(filtered.first)
        val confidence = 1.0 / (1.0 + reprojectionError)

        val poseRvec = List(3) { filtered.second.get(it, 0)[0] }
        val poseTvec = List(3) { filtered.first.get(it, 0)[0] }
        val metrics = PoseOverlayMetrics(distance, euler[0], euler[1], euler[2], reprojectionError, confidence, filtered.third)
        
        imagePoints.release(); objectPoints.release(); rvec.release(); tvec.release(); rmat.release(); filtered.first.release(); filtered.second.release()
        return MarkerPoseEstimate(poseRvec, poseTvec, MarkerDetection.Quality(confidence, reprojectionError), metrics)
    }

    private fun draw3DCubeOnMarker(res: Mat, rvec: Mat, tvec: Mat, cameraMatrix: Mat, distCoeffs: MatOfDouble) {
        val h = MARKER_SIZE_METERS / 2.0
        // Cube vertices in 3D (marker coordinate system)
        // Marker is in XY plane at Z=0. We draw the cube "above" the marker.
        // In OpenCV's default solvePnP for this setup, Z axis points into the marker (away from camera).
        // To make the cube appear in front of the marker (towards camera), we use positive Z if Z points towards camera, 
        // or negative Z if Z points away. Usually for solvePnP, Z-forward is into the scene.
        // Let's use -MARKER_SIZE_METERS to bring it towards the camera if Z is into the marker.
        val s = MARKER_SIZE_METERS
        val boxPoints = MatOfPoint3f(
            Point3(-h, -h, 0.0), Point3(h, -h, 0.0), Point3(h, h, 0.0), Point3(-h, h, 0.0),
            Point3(-h, -h, -s), Point3(h, -h, -s), Point3(h, h, -s), Point3(-h, h, -s)
        )
        
        val projectedPoints = MatOfPoint2f()
        Calib3d.projectPoints(boxPoints, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints)
        val pts = projectedPoints.toArray()
        
        if (pts.size >= 8) {
            val color = Scalar(0.0, 255.0, 255.0, 255.0) // Cyan
            val thickness = 3
            
            // Draw base
            for (i in 0..3) {
                Imgproc.line(res, pts[i], pts[(i + 1) % 4], color, thickness)
            }
            // Draw top
            for (i in 0..3) {
                Imgproc.line(res, pts[i + 4], pts[((i + 1) % 4) + 4], color, thickness)
            }
            // Draw pillars
            for (i in 0..3) {
                Imgproc.line(res, pts[i], pts[i + 4], color, thickness)
            }
            
            // Optional: fill the "front" face with semi-transparent color if we had a way to determine depth easily here
            // or just use a fixed alpha blend for the whole cube if desired.
        }
        
        boxPoints.release()
        projectedPoints.release()
    }

    private fun applyAprilTagDetection(src: Mat): Mat {
        val res = src.clone()
        val detections = detectAprilTags(src)
        detections.forEach { detection ->
            val poseEstimate = drawMarkerPoseOverlay(res, detection.corners, "april:${detection.id}", "AprilTag#${detection.id}")
            drawCornerOutlineWithOrder(res, detection.corners)
            drawMarkerLabel(res, detection, poseEstimate?.metrics)
        }
        return res
    }

    private fun detectAprilTags(src: Mat): List<MarkerDetection.AprilTag> {
        val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        val corners = ArrayList<Mat>()
        val ids = Mat()
        val detections = mutableListOf<MarkerDetection.AprilTag>()
        try {
            aprilTagDetector.detectMarkers(gray, corners, ids)
            for (i in 0 until corners.size) {
                val pts = ptsToList(corners[i])
                val markerId = ids.get(i, 0)[0].toInt()
                detections.add(MarkerDetection.AprilTag(markerId, pts))
            }
        } finally {
            gray.release(); ids.release(); corners.forEach { it.release() }
        }
        return detections
    }

    private fun applyArucoDetection(src: Mat): Mat {
        val res = src.clone()
        val detections = detectArucoMarkers(src)
        detections.forEach { detection ->
            val poseEstimate = drawMarkerPoseOverlay(res, detection.corners, "aruco:${detection.id}", "ArUco#${detection.id}")
            drawCornerOutlineWithOrder(res, detection.corners)
            drawMarkerLabel(res, detection, poseEstimate?.metrics)
        }
        return res
    }

    private fun detectArucoMarkers(src: Mat): List<MarkerDetection.Aruco> {
        val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        val corners = ArrayList<Mat>()
        val ids = Mat()
        val detections = mutableListOf<MarkerDetection.Aruco>()
        try {
            arucoDetector.detectMarkers(gray, corners, ids)
            for (i in 0 until corners.size) {
                val pts = ptsToList(corners[i])
                val markerId = ids.get(i, 0)[0].toInt()
                detections.add(MarkerDetection.Aruco(markerId, pts))
            }
        } finally {
            gray.release(); ids.release(); corners.forEach { it.release() }
        }
        return detections
    }

    private fun applyMarkerUkf(src: Mat): Mat {
        val res = src.clone()
        val oldSmoothing = poseSmoothingEnabled
        val oldFilterType = poseTemporalFilterType
        poseSmoothingEnabled = true
        poseTemporalFilterType = PoseTemporalFilterType.UKF

        try {
            val aruco = detectArucoMarkers(src)
            aruco.forEach { 
                val pose = drawMarkerPoseOverlay(res, it.corners, "aruco:${it.id}", "ArUco#${it.id}")
                drawCornerOutlineWithOrder(res, it.corners)
                drawMarkerLabel(res, it, pose?.metrics)
            }
            val april = detectAprilTags(src)
            april.forEach {
                val pose = drawMarkerPoseOverlay(res, it.corners, "april:${it.id}", "AprilTag#${it.id}")
                drawCornerOutlineWithOrder(res, it.corners)
                drawMarkerLabel(res, it, pose?.metrics)
            }
        } finally {
            poseSmoothingEnabled = oldSmoothing
            poseTemporalFilterType = oldFilterType
        }
        return res
    }

    private fun ptsToList(c: Mat): List<Pair<Float, Float>> {
        if (c.rows() == 1 && c.cols() == 4) {
            // ArUco/AprilTag format (1x4)
            return listOf(
                Pair(c.get(0, 0)[0].toFloat(), c.get(0, 0)[1].toFloat()),
                Pair(c.get(0, 1)[0].toFloat(), c.get(0, 1)[1].toFloat()),
                Pair(c.get(0, 2)[0].toFloat(), c.get(0, 2)[1].toFloat()),
                Pair(c.get(0, 3)[0].toFloat(), c.get(0, 3)[1].toFloat())
            )
        } else if (c.rows() == 4 && c.cols() == 1) {
            // QR Code format (4x1)
            return listOf(
                Pair(c.get(0, 0)[0].toFloat(), c.get(0, 0)[1].toFloat()),
                Pair(c.get(1, 0)[0].toFloat(), c.get(1, 0)[1].toFloat()),
                Pair(c.get(2, 0)[0].toFloat(), c.get(2, 0)[1].toFloat()),
                Pair(c.get(3, 0)[0].toFloat(), c.get(3, 0)[1].toFloat())
            )
        }
        return emptyList()
    }

    private fun drawCornerOutlineWithOrder(res: Mat, corners: List<Pair<Float, Float>>) {
        if (corners.size < 4) return
        for (i in corners.indices) {
            val pa = Point(corners[i].first.toDouble(), corners[i].second.toDouble())
            val pb = Point(corners[(i + 1) % corners.size].first.toDouble(), corners[(i + 1) % corners.size].second.toDouble())
            val color = OVERLAY_CORNER_COLORS[i % OVERLAY_CORNER_COLORS.size]
            Imgproc.line(res, pa, pb, color, 3)
            Imgproc.circle(res, pa, 4, color, -1)
            Imgproc.putText(res, "$i", Point(pa.x + 6.0, pa.y - 6.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        }
    }

    private fun drawMarkerPoseOverlay(res: Mat, corners: List<Pair<Float, Float>>, markerKey: String, markerLabel: String): MarkerPoseEstimate? {
        val calib = calibrator?.calibrationResult ?: return null
        if (corners.size != 4) return null

        val intrinsics = ObjectPoseTracker.CameraIntrinsics(
            fx = calib.cameraMatrix.get(0, 0)[0],
            fy = calib.cameraMatrix.get(1, 1)[0],
            cx = calib.cameraMatrix.get(0, 2)[0],
            cy = calib.cameraMatrix.get(1, 2)[0],
            distCoeffs = calib.distCoeffs,
        )

        val half = MARKER_SIZE_METERS / 2.0
        val poseInput = ObjectPoseTracker.PoseInput(
            trackId = markerKey,
            confidence = 1.0,
            cameraIntrinsics = intrinsics,
            imageLandmarks = corners.map { ObjectPoseTracker.ImagePoint(it.first.toDouble(), it.second.toDouble()) },
            objectLandmarks = listOf(
                ObjectPoseTracker.ObjectPoint(-half, half, 0.0),
                ObjectPoseTracker.ObjectPoint(half, half, 0.0),
                ObjectPoseTracker.ObjectPoint(half, -half, 0.0),
                ObjectPoseTracker.ObjectPoint(-half, -half, 0.0),
            ),
            referenceObjectSizeMeters = MARKER_SIZE_METERS,
        )
        val trackedPose = objectPoseTracker.estimatePose(poseInput)
        if (trackedPose.status == ObjectPoseTracker.PoseStatus.NO_POSE) return null

        val filteredTvec = Mat(3, 1, CvType.CV_64F).apply {
            put(0, 0, trackedPose.translationMeters[0])
            put(1, 0, trackedPose.translationMeters[1])
            put(2, 0, trackedPose.translationMeters[2])
        }
        val filteredRvec = Mat(3, 1, CvType.CV_64F).apply {
            put(0, 0, trackedPose.rotationRvec[0])
            put(1, 0, trackedPose.rotationRvec[1])
            put(2, 0, trackedPose.rotationRvec[2])
        }

        val filtered = Triple(
            filteredTvec,
            filteredRvec,
            PoseTemporalDebugMetrics(
                mode = poseOutputMode,
                jitterRawTvecMm = null,
                jitterRawRvecDeg = null,
                jitterSmoothTvecMm = null,
                jitterSmoothRvecDeg = null,
                stableStaticScene = trackedPose.filterStatus != "UKF_INIT",
            ),
        )
        val rmat = Mat(); Calib3d.Rodrigues(filtered.second, rmat)
        val zAxis = doubleArrayOf(rmat.get(0, 2)[0], rmat.get(1, 2)[0], rmat.get(2, 2)[0])
        val euler = rotationMatrixToEuler(rmat)
        ensureZAxisConsistency(markerKey, zAxis, markerLabel)
        
        Calib3d.drawFrameAxes(res, calib.cameraMatrix, calib.distCoeffs, filtered.second, filtered.first, AXIS_LENGTH_METERS.toFloat())
        val reprojectionError = trackedPose.reprojectionErrorPx ?: Double.NaN
        val distance = norm3(filtered.first)
        val confidence = trackedPose.confidence

        markerPoseStates[markerKey] = PoseState(zAxisCamera = zAxis)
        val poseRvec = List(3) { filtered.second.get(it, 0)[0] }
        val poseTvec = List(3) { filtered.first.get(it, 0)[0] }
        val metrics = PoseOverlayMetrics(distance, euler[0], euler[1], euler[2], reprojectionError, confidence, filtered.third)
        
        rmat.release(); filtered.first.release(); filtered.second.release()
        return MarkerPoseEstimate(poseRvec, poseTvec, MarkerDetection.Quality(confidence, reprojectionError), metrics)
    }

    private fun drawMarkerLabel(res: Mat, detection: MarkerDetection, metrics: PoseOverlayMetrics?) {
        val anchor = detection.corners.minByOrNull { it.second } ?: Pair(30f, 30f)
        val x = anchor.first.toDouble(); val y = (anchor.second - 12f).toDouble().coerceAtLeast(30.0)
        val lines = buildList {
            add("${detection.type}:${detection.id}")
            if (metrics != null) {
                add("dist=%.2fm".format(metrics.distanceMeters))
                add("yaw/pitch/roll=%.1f/%.1f/%.1f".format(metrics.yawDeg, metrics.pitchDeg, metrics.rollDeg))
                add(detection.quality.toOverlayString())
                add("mode=${metrics.temporalDebug.mode.name} ${if (metrics.temporalDebug.stableStaticScene) "jitter OK" else "jitter HIGH"}")
            } else {
                add(detection.quality.toOverlayString())
            }
        }
        lines.forEachIndexed { i, line -> Imgproc.putText(res, line, Point(x, y + i * LABEL_LINE_HEIGHT), Imgproc.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, Scalar(255.0, 0.0, 0.0, 255.0), 2) }
    }

    private fun ensureZAxisConsistency(markerKey: String, zAxis: DoubleArray, markerLabel: String): Boolean {
        val dotWithCamera = zAxis.getOrElse(2) { 0.0 }
        val prev = markerPoseStates[markerKey]
        if (dotWithCamera < 0.0) return false
        if (prev != null) {
            val dot = prev.zAxisCamera.zip(zAxis).sumOf { it.first * it.second }
            if (dot < 0.0) return false
        }
        return true
    }

    private fun computeReprojectionError(objectPoints: MatOfPoint3f, imagePoints: MatOfPoint2f, rvec: Mat, tvec: Mat, cameraMatrix: Mat, distCoeffs: MatOfDouble): Double {
        val projected = MatOfPoint2f(); Calib3d.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, projected)
        val observed = imagePoints.toArray(); val reproj = projected.toArray(); projected.release()
        if (observed.isEmpty() || observed.size != reproj.size) return Double.POSITIVE_INFINITY
        var errSq = 0.0
        for (i in observed.indices) {
            val dx = observed[i].x - reproj[i].x; val dy = observed[i].y - reproj[i].y
            errSq += dx * dx + dy * dy
        }
        return sqrt(errSq / observed.size)
    }

    private fun norm3(tvec: Mat): Double {
        val x = tvec.get(0, 0)[0]; val y = tvec.get(1, 0)[0]; val z = tvec.get(2, 0)[0]
        return sqrt(x * x + y * y + z * z)
    }

    private fun selectPoseForRender(rawKey: String, rawTvec: Mat, rawRvec: Mat): Triple<Mat, Mat, PoseTemporalDebugMetrics> {
        poseTemporalFilter.updateConfig(PoseTemporalConfig(poseSmoothingEnabled, poseTemporalFilterType, poseEmaAlpha, poseOneEuroBeta))
        val result = poseTemporalFilter.process(rawKey, doubleArrayOf(rawTvec.get(0, 0)[0], rawTvec.get(1, 0)[0], rawTvec.get(2, 0)[0]), doubleArrayOf(rawRvec.get(0, 0)[0], rawRvec.get(1, 0)[0], rawRvec.get(2, 0)[0]), System.nanoTime())
        val useRaw = poseOutputMode == PoseOutputMode.RAW
        val selectedT = if (useRaw) result.rawTvec else result.smoothedTvec
        val selectedR = if (useRaw) result.rawRvec else result.smoothedRvec
        val tMat = Mat(3, 1, CvType.CV_64F).apply { put(0, 0, selectedT[0]); put(1, 0, selectedT[1]); put(2, 0, selectedT[2]) }
        val rMat = Mat(3, 1, CvType.CV_64F).apply { put(0, 0, selectedR[0]); put(1, 0, selectedR[1]); put(2, 0, selectedR[2]) }
        val debug = PoseTemporalDebugMetrics(poseOutputMode, result.rawJitter?.tvecMm, result.rawJitter?.rvecDeg, result.smoothedJitter?.tvecMm, result.smoothedJitter?.rvecDeg, poseTemporalFilter.isStaticSceneStable(result.smoothedJitter))
        return Triple(tMat, rMat, debug)
    }

    private fun rotationMatrixToEuler(rmat: Mat): DoubleArray {
        val r00 = rmat.get(0, 0)[0]; val r10 = rmat.get(1, 0)[0]; val r20 = rmat.get(2, 0)[0]; val r21 = rmat.get(2, 1)[0]; val r22 = rmat.get(2, 2)[0]
        val sy = sqrt(r00 * r00 + r10 * r10); val singular = sy < 1e-6
        return if (!singular) {
            doubleArrayOf(Math.toDegrees(atan2(r10, r00)), Math.toDegrees(atan2(-r20, sy)), Math.toDegrees(atan2(r21, r22)))
        } else {
            doubleArrayOf(Math.toDegrees(atan2(-rmat.get(0, 1)[0], rmat.get(1, 1)[0])), Math.toDegrees(atan2(-r20, sy)), 0.0)
        }
    }

    private fun applyQrCodeDetection(src: Mat): Mat {
        val res = src.clone(); val points = Mat(); val gray = Mat()
        try {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
            val data = qrCodeDetector.detectAndDecode(gray, points)
            if (!data.isNullOrEmpty() && points.rows() >= 4) {
                val pts = ptsToList(points)
                val poseEstimate = drawMarkerPoseOverlay(res, pts, "qr:$data", "QR:$data")
                val detection = MarkerDetection.QrCode(data, pts, poseEstimate?.rvec, poseEstimate?.tvec, poseEstimate?.quality ?: MarkerDetection.Quality())
                drawCornerOutlineWithOrder(res, pts)
                drawMarkerLabel(res, detection, poseEstimate?.metrics)
            }
        } finally {
            gray.release()
            points.release()
        }
        return res
    }

    private fun applyCCTagDetection(src: Mat): Mat {
        val res = src.clone()
        val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        
        val binary = Mat()
        Imgproc.adaptiveThreshold(gray, binary, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 21, 5.0)

        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE)

        if (hierarchy.empty()) {
            gray.release(); binary.release(); hierarchy.release()
            return res
        }

        val candidates = mutableListOf<CircleData>()
        for (i in contours.indices) {
            val h = hierarchy.get(0, i)
            if (h == null) continue

            val contour = contours[i]
            val area = Imgproc.contourArea(contour)
            if (area < CCTAG_MIN_CONTOUR_AREA) continue

            val contour2f = MatOfPoint2f(*contour.toArray())
            val perimeter = Imgproc.arcLength(contour2f, true)
            if (perimeter <= 0) {
                contour2f.release()
                continue
            }
            val circularity = 4.0 * Math.PI * area / (perimeter * perimeter)
            if (circularity < CCTAG_MIN_CIRCULARITY) {
                contour2f.release()
                continue
            }

            val center = Point()
            val radius = FloatArray(1)
            Imgproc.minEnclosingCircle(contour2f, center, radius)
            candidates.add(CircleData(center, radius[0], circularity))
            contour2f.release()
        }

        // Group concentric circles with higher precision
        val sortedCandidates = candidates.sortedByDescending { it.radius }
        val used = BooleanArray(sortedCandidates.size)
        val groups = mutableListOf<List<CircleData>>()

        for (i in sortedCandidates.indices) {
            if (used[i]) continue
            val outer = sortedCandidates[i]
            val group = mutableListOf(outer)
            used[i] = true

            for (j in sortedCandidates.indices) {
                if (i == j || used[j]) continue
                val inner = sortedCandidates[j]
                if (inner.radius >= outer.radius * 0.95) continue // Must be significantly smaller

                val dx = outer.center.x - inner.center.x
                val dy = outer.center.y - inner.center.y
                val dist = sqrt(dx * dx + dy * dy)
                // Offset must be relative to the outer radius
                if (dist <= CCTAG_MAX_CENTRE_OFFSET_FRACTION * outer.radius) {
                    group.add(inner)
                    used[j] = true
                }
            }

            if (group.size in CCTAG_MIN_RINGS..CCTAG_MAX_RINGS) {
                groups.add(group)
            }
        }

        for (group in groups) {
            val outer = group[0]
            val tagId = group.size
            val confidence = group.map { it.circularity }.average()

            // Draw circle and center
            Imgproc.circle(res, outer.center, outer.radius.toInt(), Scalar(255.0, 165.0, 0.0, 255.0), 3)
            Imgproc.circle(res, outer.center, 3, Scalar(255.0, 165.0, 0.0, 255.0), -1)

            val r = outer.radius.toDouble()
            val cx = outer.center.x
            val cy = outer.center.y
            // Generate a better approximation for CCTag corners for PnP
            val corners = listOf(
                Pair((cx - r).toFloat(), (cy - r).toFloat()),
                Pair((cx + r).toFloat(), (cy - r).toFloat()),
                Pair((cx + r).toFloat(), (cy + r).toFloat()),
                Pair((cx - r).toFloat(), (cy + r).toFloat())
            )

            val poseEstimate = drawMarkerPoseOverlay(res, corners, "cctag:$tagId", "CCTag#$tagId")
            val detection = MarkerDetection.CCTag(
                tagId,
                Pair(cx.toFloat(), cy.toFloat()),
                outer.radius,
                corners,
                poseEstimate?.rvec,
                poseEstimate?.tvec,
                poseEstimate?.quality ?: MarkerDetection.Quality(confidence)
            )
            drawCornerOutlineWithOrder(res, corners)
            drawMarkerLabel(res, detection, poseEstimate?.metrics)
        }

        gray.release(); binary.release(); hierarchy.release()
        contours.forEach { it.release() }
        return res
    }

    private fun applyChessboardCalibration(src: Mat): Mat {
        val res = src.clone(); val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        val corners = calibrator?.detectCorners(gray, src.size())
        if (corners != null) {
            Calib3d.drawChessboardCorners(res, Size(calibrator?.boardWidth?.toDouble() ?: 9.0, calibrator?.boardHeight?.toDouble() ?: 6.0), corners, true)
        } else {
            Imgproc.putText(res, labelBoardNotFound, Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255.0, 0.0, 0.0, 255.0), 2)
        }
        Imgproc.putText(res, "${calibrator?.frameCount ?: 0} $labelFrameCountSuffix", Point(30.0, res.rows() - 30.0), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0.0, 255.0, 0.0, 255.0), 2)
        gray.release(); return res
    }

    private fun applyUndistort(src: Mat): Mat {
        val profile = calibrator?.getCalibrationProfile(src.size())
        val res = Mat()
        if (profile?.isCompatible == true) {
            Calib3d.undistort(src, res, profile.calibration.cameraMatrix, profile.calibration.distCoeffs)
        } else {
            src.copyTo(res)
            Imgproc.putText(res, labelNoCalibration, Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255.0, 0.0, 0.0, 255.0), 2)
        }
        return res
    }

    private fun applyVisualOdometry(src: Mat): Mat {
        val geometryInput = prepareGeometryInput(src, "vo")
        val res = src.clone()
        val markers = detectMarkersForOdometry(geometryInput)
        visualOdometryEngine.processFrameRgba(geometryInput, calibrator)
        // ... (remaining code using markers if needed)
        val tracks = visualOdometryEngine.currentTracks
        for (track in tracks) {
            if (track.size < 2) continue
            for (i in 0 until track.size - 1) Imgproc.line(res, track[i], track[i+1], Scalar(0.0, 255.0, 0.0, 255.0), 1)
            Imgproc.circle(res, track.last(), 3, Scalar(0.0, 0.0, 255.0, 255.0), -1)
        }
        Imgproc.putText(res, "$labelOdometryTracks: ${tracks.size}", Point(30.0, 30.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255.0, 255.0, 255.0), 2)
        geometryInput.release(); return res
    }

    private fun applyPointCloud(src: Mat, state: OdometryState): Mat {
        val geometryInput = prepareGeometryInput(src, "point_cloud")
        visualOdometryEngine.processFrameRgba(geometryInput, calibrator)
        val res = Mat.zeros(src.rows(), src.cols(), src.type())
        val cloud = visualOdometryEngine.lastPointCloud
        val pointRadius = computeAdaptivePointRadius(res.cols(), res.rows(), cloud?.points?.size ?: 0)
        if (cloud != null) {
            if (isVoMeshEnabled) {
                for ((p1, p2) in cloud.edges) Imgproc.line(res, p1, p2, Scalar(0.0, 128.0, 255.0, 255.0), POINT_CLOUD_MESH_THICKNESS)
            }
            cloud.points.forEachIndexed { i, pt ->
                val color = cloud.colors.getOrNull(i) ?: Scalar(0.0, 255.0, 255.0, 255.0)
                Imgproc.circle(res, pt, pointRadius, color, -1)
            }
            Imgproc.putText(res, "$labelPointCloud: ${cloud.points.size}", Point(30.0, 30.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255.0, 255.0, 255.0, 255.0), 2)
        }
        geometryInput.release(); return res
    }

    private fun computeAdaptivePointRadius(width: Int, height: Int, pointCount: Int): Int {
        val shortSide = min(width, height).toDouble()
        val baseRadius = (shortSide * 0.006).toInt()
        val densityPenalty = if (pointCount > 0) (pointCount / 800).coerceAtMost(2) else 0
        return (baseRadius - densityPenalty)
            .coerceIn(POINT_CLOUD_MIN_CIRCLE_RADIUS, POINT_CLOUD_MAX_CIRCLE_RADIUS)
    }

    private data class XzBounds(val minX: Double, val maxX: Double, val minZ: Double, val maxZ: Double)

    private fun computeXzBounds(points: List<Point3>): XzBounds {
        var minX = Double.MAX_VALUE; var maxX = -Double.MAX_VALUE; var minZ = Double.MAX_VALUE; var maxZ = -Double.MAX_VALUE
        for (p in points) {
            if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x
            if (p.z < minZ) minZ = p.z; if (p.z > maxZ) maxZ = p.z
        }
        return XzBounds(minX, maxX, minZ, maxZ)
    }

    private fun computeXzScale(bounds: XzBounds, drawW: Int, drawH: Int): Double {
        val rangeX = bounds.maxX - bounds.minX; val rangeZ = bounds.maxZ - bounds.minZ
        if (rangeX < EPSILON_THRESHOLD && rangeZ < EPSILON_THRESHOLD) return 1.0
        return minOf(drawW.toDouble() / (if (rangeX < EPSILON_THRESHOLD) 1.0 else rangeX), drawH.toDouble() / (if (rangeZ < EPSILON_THRESHOLD) 1.0 else rangeZ))
    }

    private fun drawMapGrid(canvas: Mat, margin: Int, drawW: Int, drawH: Int, steps: Int = 4) {
        val gridColor = Scalar(40.0, 40.0, 40.0, 255.0)
        for (g in 0..steps) {
            val gx = margin + g * drawW / steps; val gy = margin + 50 + g * drawH / steps
            Imgproc.line(canvas, Point(gx.toDouble(), (margin + 50).toDouble()), Point(gx.toDouble(), (margin + 50 + drawH).toDouble()), gridColor, 1)
            Imgproc.line(canvas, Point(margin.toDouble(), gy.toDouble()), Point((margin + drawW).toDouble(), gy.toDouble()), gridColor, 1)
        }
    }

    private fun drawXzAxisLabels(canvas: Mat, margin: Int, drawW: Int, drawH: Int) {
        val axisColor = Scalar(100.0, 220.0, 100.0, 255.0)
        Imgproc.putText(canvas, "X", Point((margin + drawW + 4).toDouble(), (margin + 50 + drawH / 2).toDouble()), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, axisColor, 1)
        Imgproc.putText(canvas, "Z", Point((margin + drawW / 2).toDouble(), (margin + 50 - 6).toDouble()), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, axisColor, 1)
    }

    private fun applyFullOdometry(src: Mat): Mat {
        val geometryInput = prepareGeometryInput(src, "full_odometry")
        val res = src.clone()
        val markers = detectMarkersForOdometry(geometryInput)
        fullOdometryEngine.processFrameRgba(geometryInput, calibrator, markers)
        val tracks = fullOdometryEngine.currentTracks
        for (track in tracks) {
            val color = if (track.isInlier) Scalar(0.0, 255.0, 0.0, 255.0) else Scalar(255.0, 0.0, 0.0, 200.0)
            Imgproc.line(res, track.p1, track.p2, color, 1)
            if (track.isInlier) {
                Imgproc.circle(res, track.p2, 3, Scalar(255.0, 80.0, 0.0, 255.0), -1)
            }
        }
        val state = fullOdometryEngine.lastOdometryState; var y = 40.0
        Imgproc.putText(res, "$labelFullOdometryTracks: ${tracks.size}", Point(FULL_ODOMETRY_HUD_X, y), Imgproc.FONT_HERSHEY_SIMPLEX, 0.65, Scalar(255.0, 255.0, 255.0, 255.0), 2)
        if (state != null) {
            y += FULL_ODOMETRY_HUD_LINE_HEIGHT
            Imgproc.putText(res, "$labelFullOdometryInliers: ${state.inliersCount}/${state.tracksCount}", Point(FULL_ODOMETRY_HUD_X, y), Imgproc.FONT_HERSHEY_SIMPLEX, 0.65, Scalar(255.0, 255.0, 255.0, 255.0), 2)
            y += FULL_ODOMETRY_HUD_LINE_HEIGHT
            Imgproc.putText(res, "$labelFullOdometryFrames: ${state.frameCount}  $labelFullOdometrySteps: ${state.totalSteps}", Point(FULL_ODOMETRY_HUD_X, y), Imgproc.FONT_HERSHEY_SIMPLEX, 0.65, Scalar(255.0, 255.0, 255.0, 255.0), 2)
            val currentPose = state.currentPose
            if (currentPose != null) {
                y += FULL_ODOMETRY_HUD_LINE_HEIGHT
                Imgproc.putText(res, "$labelFullOdometryPos: (%.2f, %.2f, %.2f)".format(currentPose.position.x, currentPose.position.y, currentPose.position.z), Point(FULL_ODOMETRY_HUD_X, y), Imgproc.FONT_HERSHEY_SIMPLEX, 0.55, Scalar(255.0, 255.0, 255.0, 255.0), 2)
                y += FULL_ODOMETRY_HUD_LINE_HEIGHT
                Imgproc.putText(res, "R: %.1f°  inlier: %.0f%%".format(currentPose.rotationDeg, currentPose.inlierRatio * 100), Point(FULL_ODOMETRY_HUD_X, y), Imgproc.FONT_HERSHEY_SIMPLEX, 0.55, Scalar(255.0, 255.0, 255.0, 255.0), 2)
            }
        }
        geometryInput.release(); return res
    }

    private fun applyOdometryTrajectory(src: Mat): Mat {
        val geometryInput = prepareGeometryInput(src, "odometry_trajectory")
        val markers = detectMarkersForOdometry(geometryInput)
        fullOdometryEngine.processFrameRgba(geometryInput, calibrator, markers)
        geometryInput.release()
        val res = Mat.zeros(src.rows(), src.cols(), src.type())
        val positions = fullOdometryEngine.currentTrajectory.positions
        Imgproc.putText(res, "$labelTrajectory: ${positions.size}", Point(FULL_ODOMETRY_HUD_X, 36.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.65, Scalar(200.0, 200.0, 200.0, 255.0), 2)
        if (positions.size < 2) {
            Imgproc.putText(res, labelCollectingData, Point(FULL_ODOMETRY_HUD_X, 72.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.55, Scalar(150.0, 150.0, 150.0, 255.0), 1)
            return res
        }
        val bounds = computeXzBounds(positions); val margin = 40; val drawW = res.cols() - 2 * margin; val drawH = res.rows() - 2 * margin - 50; val scale = computeXzScale(bounds, drawW, drawH)
        
        // Helper to project 3D point (X, Z) to 2D screen coordinates for top-down map
        fun toScreenCoord(pt: Point3): Point = Point(
            margin + (pt.x - bounds.minX) * scale,
            margin + 50 + (pt.z - bounds.minZ) * scale
        )
        
        drawMapGrid(res, margin, drawW, drawH)
        for (i in 1 until positions.size) {
            Imgproc.line(res, toScreenCoord(positions[i - 1]), toScreenCoord(positions[i]), Scalar(180.0, 180.0, 180.0, 255.0), 1)
        }
        for (i in positions.indices) {
            Imgproc.circle(res, toScreenCoord(positions[i]), 2, Scalar(255.0, 255.0, 255.0, 255.0), -1)
        }
        
        val current = fullOdometryEngine.currentTrajectory.currentPosition
        if (current != null) {
            val sc = toScreenCoord(current)
            Imgproc.circle(res, sc, 6, Scalar(0.0, 0.0, 255.0, 255.0), -1)
            Imgproc.line(res, Point(sc.x - 10.0, sc.y), Point(sc.x + 10.0, sc.y), Scalar(0.0, 0.0, 255.0, 255.0), 2)
            Imgproc.line(res, Point(sc.x, sc.y - 10.0), Point(sc.x, sc.y + 10.0), Scalar(0.0, 0.0, 255.0, 255.0), 2)
        }
        drawXzAxisLabels(res, margin, drawW, drawH); return res
    }

    private fun applyOdometryMap(src: Mat): Mat {
        val geometryInput = prepareGeometryInput(src, "odometry_map")
        val markers = detectMarkersForOdometry(geometryInput)
        fullOdometryEngine.processFrameRgba(geometryInput, calibrator, markers)
        geometryInput.release()
        val res = Mat.zeros(src.rows(), src.cols(), src.type())
        val mapState = fullOdometryEngine.currentMap
        val points = mapState.points3d
        val colors = mapState.colors
        Imgproc.putText(res, "$labelMap3D: ${points.size} $labelOdometryPoints", Point(FULL_ODOMETRY_HUD_X, 36.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.65, Scalar(200.0, 200.0, 200.0, 255.0), 2)
        if (points.isEmpty() && mapState.markers.isEmpty()) {
            Imgproc.putText(res, labelCollectingPoints, Point(FULL_ODOMETRY_HUD_X, 72.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.55, Scalar(150.0, 150.0, 150.0, 255.0), 1)
            return res
        }
        val camPos = mapState.cameraPosition
        val markerPoints = mapState.markers.map { it.position }
        val combinedPoints = if (camPos != null) points + markerPoints + camPos else points + markerPoints
        val bounds = computeXzBounds(combinedPoints)
        val margin = 40; val drawW = res.cols() - 2 * margin; val drawH = res.rows() - 2 * margin - 50; val scale = computeXzScale(bounds, drawW, drawH)
        
        fun toScreenCoord(px: Double, pz: Double): Point = Point(
            margin + (px - bounds.minX) * scale,
            margin + 50 + (pz - bounds.minZ) * scale
        )
        
        drawMapGrid(res, margin, drawW, drawH)
        val mapPointRadius = computeAdaptiveMapPointRadius(points.size)
        for (i in points.indices) {
            val pt = points[i]
            val c = colors[i]
            val scalar = Scalar(android.graphics.Color.red(c).toDouble(), android.graphics.Color.green(c).toDouble(), android.graphics.Color.blue(c).toDouble(), 255.0)
            Imgproc.circle(res, toScreenCoord(pt.x, pt.z), mapPointRadius, scalar, -1)
        }

        // Draw Markers on Map
        for (m in mapState.markers) {
            val sc = toScreenCoord(m.position.x, m.position.z)
            Imgproc.drawMarker(res, sc, Scalar(0.0, 255.0, 255.0, 255.0), Imgproc.MARKER_SQUARE, 15, 2)
            Imgproc.putText(res, m.label, Point(sc.x + 10, sc.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0.0, 255.0, 255.0, 255.0), 1)
        }
        
        if (camPos != null) {
            val sc = toScreenCoord(camPos.x, camPos.z)
            Imgproc.circle(res, sc, 7, Scalar(255.0, 80.0, 0.0, 255.0), 2)
            Imgproc.line(res, Point(sc.x - 12.0, sc.y), Point(sc.x + 12.0, sc.y), Scalar(255.0, 80.0, 0.0, 255.0), 2)
            Imgproc.line(res, Point(sc.x, sc.y - 12.0), Point(sc.x, sc.y + 12.0), Scalar(255.0, 80.0, 0.0, 255.0), 2)
        }
        drawXzAxisLabels(res, margin, drawW, drawH); return res
    }

    private fun computeAdaptiveMapPointRadius(pointCount: Int): Int {
        val densityPenalty = if (pointCount > 0) (pointCount / 1500).coerceAtMost(2) else 0
        val boostedRadius = MAP_POINT_MAX_CIRCLE_RADIUS - densityPenalty
        return boostedRadius.coerceIn(MAP_POINT_MIN_CIRCLE_RADIUS, MAP_POINT_MAX_CIRCLE_RADIUS)
    }

    private fun applyPlaneDetection(src: Mat, state: GeometryState): Mat {
        geometryProcessor.maxPlanes = geometryMaxPlanes
        val geometryInput = prepareGeometryInput(src, "plane_detection")
        val profile = calibrator?.getCalibrationProfile(geometryInput.size())
        if (profile != null && !profile.isCompatible) {
            val blocked = src.clone(); Imgproc.putText(blocked, labelNoCalibration, Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255.0, 0.0, 0.0, 255.0), 2)
            geometryInput.release(); return blocked
        }
        val res = geometryInput.clone()
        val markers = detectMarkersForOdometry(geometryInput)
        val labels = mapOf("noPlanes" to labelNoPlanes, "lines" to labelLines, "planes" to labelPlanes)
        val planeIdx = geometryProcessor.detectPlanes(geometryInput, res, labels)
        if (planeIdx == 0) {
            Imgproc.putText(res, labelNoPlanes, Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, Scalar(200.0, 200.0, 200.0), 2)
        } else {
            Imgproc.putText(res, "$labelPlanes: $planeIdx", Point(30.0, 30.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255.0, 255.0, 255.0), 2)
        }
        geometryInput.release(); return res
    }

    private fun applyVanishingPoints(src: Mat): Mat {
        val res = src.clone()
        val labels = mapOf("noLines" to labelNoLines, "noVp" to labelNoVanishingPoints, "lines" to labelLines, "groups" to labelGroups, "vpError" to labelVpError)
        geometryProcessor.detectVanishingPoints(src, res, labels)
        return res
    }

    private fun applyDistanceEstimation(src: Mat): Mat {
        val res = src.clone()
        val calib = calibrator?.calibrationResult
        if (calib == null) {
            Imgproc.putText(res, labelNoCalibration, Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255.0, 0.0, 0.0, 255.0), 2)
            return res
        }

        // Detect all types of markers to estimate distance
        val markers = detectMarkersForOdometry(src).toMutableList()
        
        // Add CCTags manually if not already in detectMarkersForOdometry
        val gray = Mat(); Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        val binary = Mat()
        Imgproc.adaptiveThreshold(gray, binary, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 21, 5.0)
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE)

        if (!hierarchy.empty()) {
            val candidates = mutableListOf<CircleData>()
            for (i in contours.indices) {
                val contour = contours[i]
                val area = Imgproc.contourArea(contour)
                if (area < CCTAG_MIN_CONTOUR_AREA) continue
                val contour2f = MatOfPoint2f(*contour.toArray())
                val perimeter = Imgproc.arcLength(contour2f, true)
                if (perimeter <= 0) { contour2f.release(); continue }
                val circularity = 4.0 * Math.PI * area / (perimeter * perimeter)
                if (circularity < CCTAG_MIN_CIRCULARITY) { contour2f.release(); continue }
                val center = Point(); val radius = FloatArray(1)
                Imgproc.minEnclosingCircle(contour2f, center, radius)
                candidates.add(CircleData(center, radius[0], circularity))
                contour2f.release()
            }
            val sortedCandidates = candidates.sortedByDescending { it.radius }
            val used = BooleanArray(sortedCandidates.size)
            for (i in sortedCandidates.indices) {
                if (used[i]) continue
                val outer = sortedCandidates[i]
                val group = mutableListOf(outer); used[i] = true
                for (j in sortedCandidates.indices) {
                    if (i == j || used[j]) continue
                    val inner = sortedCandidates[j]
                    if (inner.radius >= outer.radius * 0.95) continue
                    val dx = outer.center.x - inner.center.x; val dy = outer.center.y - inner.center.y
                    if (sqrt(dx * dx + dy * dy) <= CCTAG_MAX_CENTRE_OFFSET_FRACTION * outer.radius) {
                        group.add(inner); used[j] = true
                    }
                }
                if (group.size in CCTAG_MIN_RINGS..CCTAG_MAX_RINGS) {
                    val r = outer.radius.toDouble(); val cx = outer.center.x; val cy = outer.center.y
                    val corners = listOf(Pair((cx-r).toFloat(), (cy-r).toFloat()), Pair((cx+r).toFloat(), (cy-r).toFloat()), Pair((cx+r).toFloat(), (cy+r).toFloat()), Pair((cx-r).toFloat(), (cy+r).toFloat()))
                    val pose = estimateMarkerPoseRaw(corners)
                    markers.add(MarkerDetection.CCTag(group.size, Pair(cx.toFloat(), cy.toFloat()), outer.radius, corners, pose?.first, pose?.second, MarkerDetection.Quality(group.map { it.circularity }.average())))
                }
            }
        }

        markers.forEach { m ->
            val pose = drawMarkerPoseOverlay(res, m.corners, "${m.type}:${m.id}", "${m.type}#${m.id}")
            if (pose != null) {
                drawCornerOutlineWithOrder(res, m.corners)
                drawMarkerLabel(res, m, pose.metrics)
            }
        }

        if (markers.isEmpty()) {
            Imgproc.putText(res, "Szukaj markerów do pomiaru...", Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255.0, 255.0, 255.0), 2)
        }

        gray.release(); binary.release(); hierarchy.release(); contours.forEach { it.release() }
        return res
    }

    private fun prepareGeometryInput(src: Mat, stage: String): Mat {
        val profile = calibrator?.getCalibrationProfile(src.size())
        if (!debugUndistortBeforeGeometry || profile?.isCompatible != true) return src.clone()
        val undistorted = Mat(); Calib3d.undistort(src, undistorted, profile.calibration.cameraMatrix, profile.calibration.distCoeffs)
        return undistorted
    }

    private fun detectMarkersForOdometry(src: Mat): List<MarkerDetection> {
        val markers = mutableListOf<MarkerDetection>()
        
        // 1. ArUco
        detectArucoMarkers(src).forEach { 
            val pose = estimateMarkerPoseRaw(it.corners)
            markers.add(it.copy(rvec = pose?.first, tvec = pose?.second))
        }

        // 2. AprilTags
        detectAprilTags(src).forEach {
            val pose = estimateMarkerPoseRaw(it.corners)
            markers.add(it.copy(rvec = pose?.first, tvec = pose?.second))
        }

        // 3. QR Codes
        val gray = Mat(); Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        val points = Mat()
        try {
            val data = qrCodeDetector.detectAndDecode(gray, points)
            if (!data.isNullOrEmpty() && points.rows() >= 4) {
                val pts = ptsToList(points)
                val pose = estimateMarkerPoseRaw(pts)
                markers.add(MarkerDetection.QrCode(data, pts, pose?.first, pose?.second))
            }
        } finally {
            gray.release(); points.release()
        }

        return markers
    }

    private fun estimateMarkerPoseRaw(corners: List<Pair<Float, Float>>): Pair<List<Double>, List<Double>>? {
        val calib = calibrator?.calibrationResult ?: return null
        if (corners.size < 4) return null
        val imagePoints = MatOfPoint2f(
            Point(corners[0].first.toDouble(), corners[0].second.toDouble()),
            Point(corners[1].first.toDouble(), corners[1].second.toDouble()),
            Point(corners[2].first.toDouble(), corners[2].second.toDouble()),
            Point(corners[3].first.toDouble(), corners[3].second.toDouble())
        )
        val half = MARKER_SIZE_METERS / 2.0
        val objectPoints = MatOfPoint3f(Point3(-half, half, 0.0), Point3(half, half, 0.0), Point3(half, -half, 0.0), Point3(-half, -half, 0.0))
        val rvec = Mat(); val tvec = Mat()
        val solved = Calib3d.solvePnP(objectPoints, imagePoints, calib.cameraMatrix, calib.distCoeffs, rvec, tvec, false, Calib3d.SOLVEPNP_IPPE_SQUARE) || Calib3d.solvePnP(objectPoints, imagePoints, calib.cameraMatrix, calib.distCoeffs, rvec, tvec)
        if (!solved) { imagePoints.release(); objectPoints.release(); rvec.release(); tvec.release(); return null }
        val r = List(3) { rvec.get(it, 0)[0] }; val t = List(3) { tvec.get(it, 0)[0] }
        imagePoints.release(); objectPoints.release(); rvec.release(); tvec.release()
        return Pair(r, t)
    }
}
