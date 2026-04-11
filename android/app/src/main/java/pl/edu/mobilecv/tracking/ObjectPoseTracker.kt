package pl.edu.mobilecv.tracking

import android.graphics.RectF
import org.opencv.calib3d.Calib3d
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfPoint2f
import org.opencv.core.MatOfPoint3f
import org.opencv.core.Point
import org.opencv.core.Point3
import pl.edu.mobilecv.util.UnscentedKalmanFilter
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Wspólny tracker pozycji 3D obiektu działający na danych z markerów, detektorów bbox
 * oraz landmarków 2D. Zapewnia estymację pozycji i rotacji oraz wygładzanie per track-id.
 */
class ObjectPoseTracker {
    /** Parametry kamery używane do estymacji 2D->3D i solvePnP. */
    data class CameraIntrinsics(
        val fx: Double,
        val fy: Double,
        val cx: Double,
        val cy: Double,
        val distCoeffs: MatOfDouble = MatOfDouble(),
    )

    /** Prostokąt detekcji w pikselach obrazu. */
    data class BoundingBox(
        val x1: Double,
        val y1: Double,
        val x2: Double,
        val y2: Double,
    ) {
        val width: Double get() = (x2 - x1).coerceAtLeast(0.0)
        val height: Double get() = (y2 - y1).coerceAtLeast(0.0)
        val centerX: Double get() = x1 + width * 0.5
        val centerY: Double get() = y1 + height * 0.5

        companion object {
            fun fromRectF(rect: RectF): BoundingBox =
                BoundingBox(rect.left.toDouble(), rect.top.toDouble(), rect.right.toDouble(), rect.bottom.toDouble())
        }
    }

    /** Punkt 2D w pikselach. */
    data class ImagePoint(val x: Double, val y: Double)

    /** Punkt 3D w metrach. */
    data class ObjectPoint(val x: Double, val y: Double, val z: Double)

    /** Wspólny kontrakt wejścia dla modułów marker/YOLO/MediaPipe. */
    data class PoseInput(
        val trackId: String,
        val confidence: Double,
        val cameraIntrinsics: CameraIntrinsics,
        val boundingBox: BoundingBox? = null,
        val imageLandmarks: List<ImagePoint> = emptyList(),
        val objectLandmarks: List<ObjectPoint> = emptyList(),
        val referenceObjectSizeMeters: Double? = null,
    )

    enum class PoseStatus { SOLVE_PNP, FALLBACK_2D, NO_POSE }

    data class PoseResult(
        val trackId: String,
        val translationMeters: DoubleArray,
        val rotationRvec: DoubleArray,
        val confidence: Double,
        val reprojectionErrorPx: Double?,
        val status: PoseStatus,
        val filterStatus: String,
    )

    private data class TrackSmoother(
        val ukf: UnscentedKalmanFilter,
        var initialized: Boolean = false,
    )

    private val smoothers = HashMap<String, TrackSmoother>()

    /**
     * Estymuje pozycję obiektu i zwraca wynik po filtracji UKF.
     * Najpierw próbuje solvePnP, a przy braku pełnych korespondencji uruchamia fallback 2D->3D.
     */
    fun estimatePose(input: PoseInput): PoseResult {
        val fromPnp = estimateWithSolvePnP(input)
        val raw = fromPnp ?: estimateWith2dFallback(input)
        val rawResult = raw ?: return PoseResult(
            trackId = input.trackId,
            translationMeters = doubleArrayOf(0.0, 0.0, 0.0),
            rotationRvec = doubleArrayOf(0.0, 0.0, 0.0),
            confidence = input.confidence,
            reprojectionErrorPx = null,
            status = PoseStatus.NO_POSE,
            filterStatus = "NO_MEASUREMENT",
        )

        val smoothed = smoothPose(input.trackId, rawResult.translationMeters, rawResult.rotationRvec)
        return rawResult.copy(
            translationMeters = smoothed.first,
            rotationRvec = smoothed.second,
            filterStatus = smoothed.third,
        )
    }

    private fun estimateWithSolvePnP(input: PoseInput): PoseResult? {
        if (input.imageLandmarks.size < 4 || input.objectLandmarks.size < 4) return null
        if (input.imageLandmarks.size != input.objectLandmarks.size) return null

        val imagePoints = MatOfPoint2f(*input.imageLandmarks.map { Point(it.x, it.y) }.toTypedArray())
        val objectPoints = MatOfPoint3f(*input.objectLandmarks.map { Point3(it.x, it.y, it.z) }.toTypedArray())

        val rvec = Mat()
        val tvec = Mat()
        val cameraMatrix = cameraMatrix(input.cameraIntrinsics)
        val solved = Calib3d.solvePnP(
            objectPoints,
            imagePoints,
            cameraMatrix,
            input.cameraIntrinsics.distCoeffs,
            rvec,
            tvec,
            false,
            Calib3d.SOLVEPNP_ITERATIVE,
        )

        if (!solved) {
            imagePoints.release(); objectPoints.release(); rvec.release(); tvec.release(); cameraMatrix.release()
            return null
        }

        val reprojectionError = computeReprojectionError(
            objectPoints,
            imagePoints,
            rvec,
            tvec,
            cameraMatrix,
            input.cameraIntrinsics.distCoeffs,
        )
        val translation = DoubleArray(3) { i -> tvec.get(i, 0)[0] }
        val rotation = DoubleArray(3) { i -> rvec.get(i, 0)[0] }

        imagePoints.release(); objectPoints.release(); rvec.release(); tvec.release(); cameraMatrix.release()
        return PoseResult(
            trackId = input.trackId,
            translationMeters = translation,
            rotationRvec = rotation,
            confidence = input.confidence,
            reprojectionErrorPx = reprojectionError,
            status = PoseStatus.SOLVE_PNP,
            filterStatus = "RAW",
        )
    }

    private fun estimateWith2dFallback(input: PoseInput): PoseResult? {
        val bbox = input.boundingBox ?: return null
        val intrinsics = input.cameraIntrinsics
        val pixelSize = max(1.0, max(bbox.width, bbox.height))
        val depth = input.referenceObjectSizeMeters?.let { (intrinsics.fx * it) / pixelSize }
            ?: (1.0 + (1.0 - input.confidence.coerceIn(0.0, 1.0)) * 2.0)

        val x = (bbox.centerX - intrinsics.cx) * depth / intrinsics.fx
        val y = (bbox.centerY - intrinsics.cy) * depth / intrinsics.fy

        return PoseResult(
            trackId = input.trackId,
            translationMeters = doubleArrayOf(x, y, depth),
            rotationRvec = doubleArrayOf(0.0, 0.0, 0.0),
            confidence = input.confidence,
            reprojectionErrorPx = null,
            status = PoseStatus.FALLBACK_2D,
            filterStatus = "RAW",
        )
    }

    private fun smoothPose(trackId: String, translation: DoubleArray, rotation: DoubleArray): Triple<DoubleArray, DoubleArray, String> {
        val smoother = smoothers.getOrPut(trackId) {
            val ukf = UnscentedKalmanFilter(6, 6)
            ukf.processNoiseCov = Mat.eye(6, 6, CvType.CV_64F)
            ukf.measurementNoiseCov = Mat.eye(6, 6, CvType.CV_64F)
            TrackSmoother(ukf)
        }

        val measurement = Mat(6, 1, CvType.CV_64F).apply {
            put(0, 0, translation[0])
            put(1, 0, translation[1])
            put(2, 0, translation[2])
            put(3, 0, rotation[0])
            put(4, 0, rotation[1])
            put(5, 0, rotation[2])
        }

        val status = if (!smoother.initialized) {
            smoother.ukf.statePost = measurement.clone()
            smoother.initialized = true
            "UKF_INIT"
        } else {
            smoother.ukf.predict { sigma -> sigma.clone() }
            smoother.ukf.update(measurement) { sigma -> sigma.clone() }
            "UKF_TRACKING"
        }

        val state = smoother.ukf.statePost
        val t = doubleArrayOf(state.get(0, 0)[0], state.get(1, 0)[0], state.get(2, 0)[0])
        val r = doubleArrayOf(state.get(3, 0)[0], state.get(4, 0)[0], state.get(5, 0)[0])
        measurement.release()
        return Triple(t, r, status)
    }

    private fun cameraMatrix(intrinsics: CameraIntrinsics): Mat = Mat(3, 3, CvType.CV_64F).apply {
        put(0, 0, intrinsics.fx)
        put(0, 1, 0.0)
        put(0, 2, intrinsics.cx)
        put(1, 0, 0.0)
        put(1, 1, intrinsics.fy)
        put(1, 2, intrinsics.cy)
        put(2, 0, 0.0)
        put(2, 1, 0.0)
        put(2, 2, 1.0)
    }

    private fun computeReprojectionError(
        objectPoints: MatOfPoint3f,
        imagePoints: MatOfPoint2f,
        rvec: Mat,
        tvec: Mat,
        cameraMatrix: Mat,
        distCoeffs: MatOfDouble,
    ): Double {
        val projected = MatOfPoint2f()
        Calib3d.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, projected)
        val observed = imagePoints.toArray()
        val reproj = projected.toArray()
        projected.release()
        if (observed.isEmpty() || observed.size != reproj.size) return Double.NaN

        var errSq = 0.0
        for (i in observed.indices) {
            val dx = observed[i].x - reproj[i].x
            val dy = observed[i].y - reproj[i].y
            errSq += dx * dx + dy * dy
        }
        return sqrt(errSq / observed.size)
    }
}
