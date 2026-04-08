package pl.edu.mobilecv.odometry

import kotlin.math.acos
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.DMatch
import org.opencv.core.KeyPoint
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfFloat4
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.features2d.AKAZE
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.ORB
import org.opencv.imgproc.Imgproc
import org.opencv.imgproc.Subdiv2D
import pl.edu.mobilecv.vision.CameraCalibrator

/**
 * Tracks sparse visual odometry signals and creates a lightweight pseudo point cloud.
 */
class VisualOdometryEngine {

    data class OdometryState(
        val tracksCount: Int,
        val inliersCount: Int,
        val translationNorm: Double,
        val rotationDeg: Double,
    )

    data class PointCloudState(
        val points: List<Point>,
        val depths: List<Double>,
        val colors: List<org.opencv.core.Scalar>,
        val confidences: List<Double>,
        val timestampsMs: List<Long>,
        val edges: List<Pair<Point, Point>>,
        val meanParallax: Double
    )

    companion object {
        private const val TAG = "VisualOdometryEngine"
        private const val MIN_MATCH_COUNT = 25
        private const val MIN_INLIERS_COUNT = 18
        private const val MIN_SPATIAL_BINS = 4
        private const val GRID_ROWS = 3
        private const val GRID_COLS = 3
        private const val RATIO_TEST_THRESHOLD = 0.75f
        private const val RANSAC_REPROJECTION_THRESHOLD = 1.5
        private const val PERSPECTIVE_FACTOR = 0.5
        private const val MAX_MESH_EDGE_DIST_SQ = 50000.0
        private const val MAX_REPROJECTION_ERROR_PX = 2.0
        private const val MAX_DEPTH_RELATIVE_DELTA = 0.35
        private const val DEPTH_STABILITY_EPS = 1e-6
        private const val DEPTH_CLIP_MAD_FACTOR = 3.5
    }

    enum class FeatureDetectorType {
        ORB,
        AKAZE,
    }

    private var prevGray = Mat()
    private var calibrator: CameraCalibrator? = null
    private var lastTracks = mutableListOf<List<Point>>()

    var maxFeatures = 300
    var minParallax = 2.0
    var isMeshEnabled = false
    var featureDetectorType: FeatureDetectorType = FeatureDetectorType.ORB

    var lastPointCloud: PointCloudState? = null
        private set

    /**
     * Returns the current tracked segments (start point in previous frame, end point in current frame).
     */
    val currentTracks: List<List<Point>>
        get() = synchronized(this) { lastTracks }

    fun updateOdometry(src: Mat): OdometryState? {
        val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        val (state, _) = processFrameInternal(gray, src)
        gray.release()
        return state
    }

    fun updatePointCloud(src: Mat): PointCloudState? {
        val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        val (_, cloud) = processFrameInternal(gray, src)
        gray.release()
        return cloud
    }

    /**
     * Process an RGBA frame and update internal state.
     */
    fun processFrameRgba(src: Mat, calib: CameraCalibrator? = null) {
        val gray = Mat()
        if (src.channels() > 1) {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        } else {
            src.copyTo(gray)
        }
        this.calibrator = calib
        processFrameInternal(gray, src)
        gray.release()
    }

    fun processFrame(gray: Mat, srcRgba: Mat, calib: CameraCalibrator? = null): Pair<OdometryState?, PointCloudState?> {
        this.calibrator = calib
        return processFrameInternal(gray, srcRgba)
    }

    private fun processFrameInternal(gray: Mat, srcRgba: Mat): Pair<OdometryState?, PointCloudState?> {
        if (prevGray.empty()) {
            gray.copyTo(prevGray)
            return null to null
        }

        val prevKeyPoints = MatOfKeyPoint()
        val currKeyPoints = MatOfKeyPoint()
        val prevDescriptors = Mat()
        val currDescriptors = Mat()

        detectAndCompute(prevGray, prevKeyPoints, prevDescriptors)
        detectAndCompute(gray, currKeyPoints, currDescriptors)

        if (prevDescriptors.empty() || currDescriptors.empty()) {
            prevKeyPoints.release()
            currKeyPoints.release()
            releaseTrackingMats(prevDescriptors, currDescriptors)
            return trackingLost(gray, "tracking lost / reinit needed: empty descriptors")
        }

        val goodMatches = findSymmetricMatches(prevDescriptors, currDescriptors)
        if (goodMatches.size < MIN_MATCH_COUNT) {
            prevKeyPoints.release()
            currKeyPoints.release()
            releaseTrackingMats(prevDescriptors, currDescriptors)
            return trackingLost(gray, "tracking lost / reinit needed: not enough matches")
        }

        val prevMatched = mutableListOf<Point>()
        val currMatched = mutableListOf<Point>()
        val prevKPArray = prevKeyPoints.toArray()
        val currKPArray = currKeyPoints.toArray()
        for (match in goodMatches) {
            prevMatched.add(prevKPArray[match.queryIdx].pt)
            currMatched.add(currKPArray[match.trainIdx].pt)
        }

        prevKeyPoints.release()
        currKeyPoints.release()

        synchronized(this) {
            lastTracks = prevMatched.indices.map { i -> listOf(prevMatched[i], currMatched[i]) }.toMutableList()
        }

        val prevPoints = MatOfPoint2f(*prevMatched.toTypedArray())
        val currPoints = MatOfPoint2f(*currMatched.toTypedArray())

        val calibrationProfile = calibrator?.getCalibrationProfile(gray.size())
        if (calibrationProfile != null && !calibrationProfile.isCompatible) {
            android.util.Log.w(
                TAG,
                "Skipping 3D estimation due to incompatible calibration profile " +
                    "active=${gray.cols()}x${gray.rows()} " +
                    "calibrated=${calibrationProfile.calibration.calibrationImageSize.width.toInt()}x" +
                    "${calibrationProfile.calibration.calibrationImageSize.height.toInt()} " +
                    "rms=%.4f".format(calibrationProfile.calibration.rmsError),
            )
            prevPoints.release()
            currPoints.release()
            releaseTrackingMats(prevDescriptors, currDescriptors)
            gray.copyTo(prevGray)
            lastPointCloud = null
            return null to null
        }

        val inlierPair = filterInliersWithRansac(prevPoints, currPoints, calibrationProfile)
        val inlierPrev = inlierPair.first
        val inlierCurr = inlierPair.second

        val thresholdsMet =
            inlierPrev.rows() >= MIN_INLIERS_COUNT &&
                hasSufficientSpatialDistribution(inlierCurr, gray.cols(), gray.rows()) &&
                meanParallax(inlierPrev, inlierCurr) >= minParallax

        prevPoints.release()
        currPoints.release()
        releaseTrackingMats(prevDescriptors, currDescriptors)

        if (!thresholdsMet) {
            inlierPrev.release()
            inlierCurr.release()
            return trackingLost(gray, "tracking lost / reinit needed: thresholds not met")
        }

        val (state, points) = estimateMotionAndPoints(inlierPrev, inlierCurr, srcRgba, calibrationProfile)
        inlierPrev.release()
        inlierCurr.release()

        lastPointCloud = points
        gray.copyTo(prevGray)
        return state to points
    }

    private fun trackingLost(gray: Mat, reason: String): Pair<OdometryState?, PointCloudState?> {
        android.util.Log.w(TAG, reason)
        gray.copyTo(prevGray)
        lastPointCloud = null
        synchronized(this) { lastTracks.clear() }
        return null to null
    }

    private fun detectAndCompute(gray: Mat, keypoints: MatOfKeyPoint, descriptors: Mat) {
        when (featureDetectorType) {
            FeatureDetectorType.ORB -> {
                val detector = ORB.create(maxFeatures)
                detector.detectAndCompute(gray, Mat(), keypoints, descriptors)
            }
            FeatureDetectorType.AKAZE -> {
                val detector = AKAZE.create()
                detector.detectAndCompute(gray, Mat(), keypoints, descriptors)
            }
        }
    }

    private fun findSymmetricMatches(prevDescriptors: Mat, currDescriptors: Mat): List<DMatch> {
        val matcher = BFMatcher.create(Core.NORM_HAMMING, false)
        val forward = ArrayList<MatOfDMatch>()
        val backward = ArrayList<MatOfDMatch>()
        matcher.knnMatch(prevDescriptors, currDescriptors, forward, 2)
        matcher.knnMatch(currDescriptors, prevDescriptors, backward, 2)

        val forwardBest = bestMatchesAfterRatioTest(forward)
        val backwardBest = bestMatchesAfterRatioTest(backward)

        val symmetricMatches = mutableListOf<DMatch>()
        for ((queryIdx, match) in forwardBest) {
            val reverse = backwardBest[match.trainIdx]
            if (reverse != null && reverse.trainIdx == queryIdx) {
                symmetricMatches.add(match)
            }
        }

        for (mat in forward) mat.release()
        for (mat in backward) mat.release()

        return symmetricMatches
    }

    private fun bestMatchesAfterRatioTest(knnMatches: List<MatOfDMatch>): Map<Int, DMatch> {
        val accepted = mutableMapOf<Int, DMatch>()
        for (entry in knnMatches) {
            val candidates = entry.toArray()
            if (candidates.size < 2) continue
            val best = candidates[0]
            val second = candidates[1]
            if (best.distance < RATIO_TEST_THRESHOLD * second.distance) {
                accepted[best.queryIdx] = best
            }
        }
        return accepted
    }

    private fun filterInliersWithRansac(
        prev: MatOfPoint2f,
        next: MatOfPoint2f,
        calibrationProfile: CameraCalibrator.CalibrationProfile?,
    ): Pair<MatOfPoint2f, MatOfPoint2f> {
        val mask = Mat()
        val model = if (calibrationProfile != null) {
            val k = calibrationProfile.calibration.cameraMatrix
            val essential = Calib3d.findEssentialMat(prev, next, k, Calib3d.RANSAC, 0.999, RANSAC_REPROJECTION_THRESHOLD)
            if (!essential.empty()) {
                val r = Mat()
                val t = Mat()
                Calib3d.recoverPose(essential, prev, next, k, r, t, mask)
                r.release()
                t.release()
            }
            essential
        } else {
            Calib3d.findFundamentalMat(prev, next, Calib3d.FM_RANSAC, 3.0, 0.99, mask)
        }

        if (model.empty() || mask.empty()) {
            mask.release()
            model.release()
            return MatOfPoint2f() to MatOfPoint2f()
        }

        val prevArr = prev.toArray()
        val nextArr = next.toArray()
        val maskArr = ByteArray(mask.rows() * mask.cols())
        mask.get(0, 0, maskArr)

        val inlierPrev = mutableListOf<Point>()
        val inlierNext = mutableListOf<Point>()
        for (i in maskArr.indices) {
            if (maskArr[i].toInt() != 0) {
                inlierPrev.add(prevArr[i])
                inlierNext.add(nextArr[i])
            }
        }

        mask.release()
        model.release()

        return MatOfPoint2f(*inlierPrev.toTypedArray()) to MatOfPoint2f(*inlierNext.toTypedArray())
    }

    private fun hasSufficientSpatialDistribution(points: MatOfPoint2f, width: Int, height: Int): Boolean {
        if (points.empty() || width <= 0 || height <= 0) {
            return false
        }

        val occupied = HashSet<Int>()
        val cellWidth = width.toDouble() / GRID_COLS
        val cellHeight = height.toDouble() / GRID_ROWS

        for (point in points.toArray()) {
            val col = min(GRID_COLS - 1, max(0, (point.x / cellWidth).toInt()))
            val row = min(GRID_ROWS - 1, max(0, (point.y / cellHeight).toInt()))
            occupied.add(row * GRID_COLS + col)
        }

        return occupied.size >= MIN_SPATIAL_BINS
    }

    private fun meanParallax(prev: MatOfPoint2f, next: MatOfPoint2f): Double {
        if (prev.empty() || next.empty()) return 0.0

        val prevArr = prev.toArray()
        val nextArr = next.toArray()
        var parallaxSum = 0.0

        for (i in prevArr.indices) {
            val dx = nextArr[i].x - prevArr[i].x
            val dy = nextArr[i].y - prevArr[i].y
            parallaxSum += sqrt(dx * dx + dy * dy)
        }

        return parallaxSum / prevArr.size
    }

    private fun estimateMotionAndPoints(
        prev: MatOfPoint2f,
        next: MatOfPoint2f,
        srcRgba: Mat, // Pass RGBA to get colors
        calibrationProfile: CameraCalibrator.CalibrationProfile?,
    ): Pair<OdometryState, PointCloudState> {
        val k = calibrationProfile?.calibration?.cameraMatrix ?: Mat.eye(3, 3, CvType.CV_64F)
        val essential = Calib3d.findEssentialMat(prev, next, k, Calib3d.RANSAC, 0.999, 1.0)
        val r = Mat()
        val t = Mat()
        val mask = Mat()
        Calib3d.recoverPose(essential, prev, next, k, r, t, mask)

        var inlierCount = 0
        if (!mask.empty()) {
            val maskData = ByteArray(mask.rows() * mask.cols())
            mask.get(0, 0, maskData)
            for (v in maskData) if (v.toInt() != 0) inlierCount++
        }

        val transNorm =
            android.opengl.Matrix.length(
                t.get(0, 0)[0].toFloat(),
                t.get(1, 0)[0].toFloat(),
                t.get(2, 0)[0].toFloat(),
            ).toDouble()

        val trace = r.get(0, 0)[0] + r.get(1, 1)[0] + r.get(2, 2)[0]
        val rotDeg = Math.toDegrees(acos(min(1.0, max(-1.0, (trace - 1.0) / 2.0))))

        val state = OdometryState(prev.rows(), inlierCount, transNorm, rotDeg)

        val cloudPoints = mutableListOf<Point>()
        val cloudDepths = mutableListOf<Double>()
        val cloudColors = mutableListOf<org.opencv.core.Scalar>()
        val cloudConfidences = mutableListOf<Double>()
        val cloudTimestamps = mutableListOf<Long>()
        val prevArr = prev.toArray()
        val nextArr = next.toArray()
        val triPoints = Mat()
        val projection1 = Mat.zeros(3, 4, CvType.CV_64F)
        val projection2 = Mat.zeros(3, 4, CvType.CV_64F)
        projection1.put(0, 0, 1.0)
        projection1.put(1, 1, 1.0)
        projection1.put(2, 2, 1.0)
        val rt = Mat.zeros(3, 4, CvType.CV_64F)
        for (row in 0..2) {
            for (col in 0..2) {
                rt.put(row, col, r.get(row, col)[0])
            }
            rt.put(row, 3, t.get(row, 0)[0])
        }
        Core.gemm(k, projection1, 1.0, Mat(), 0.0, projection1)
        Core.gemm(k, rt, 1.0, Mat(), 0.0, projection2)
        Calib3d.triangulatePoints(projection1, projection2, prev, next, triPoints)
        val triangulated = mutableListOf<Point3Data>()
        for (i in 0 until triPoints.cols()) {
            val w = triPoints.get(3, i)[0]
            if (abs(w) <= DEPTH_STABILITY_EPS) {
                triangulated.add(Point3Data(0.0, 0.0, 0.0))
                continue
            }
            triangulated.add(
                Point3Data(
                    triPoints.get(0, i)[0] / w,
                    triPoints.get(1, i)[0] / w,
                    triPoints.get(2, i)[0] / w,
                ),
            )
        }
        val depthsCam1 = triangulated.map { it.z }
        val depthsCam2 = triangulated.map { p ->
            val z2 = r.get(2, 0)[0] * p.x + r.get(2, 1)[0] * p.y + r.get(2, 2)[0] * p.z + t.get(2, 0)[0]
            z2
        }
        val clipRange = robustDepthClipRange(depthsCam1.map { abs(it) })
        val frameTimestampMs = System.currentTimeMillis()
        var parallaxSum = 0.0

        for (i in nextArr.indices) {
            val prevPt = prevArr[i]
            val nextPt = nextArr[i]
            val dx = nextPt.x - prevPt.x
            val dy = nextPt.y - prevPt.y
            val dist = sqrt(dx * dx + dy * dy)
            parallaxSum += dist

            val point3d = triangulated.getOrNull(i) ?: continue
            val depth1 = depthsCam1.getOrNull(i) ?: continue
            val depth2 = depthsCam2.getOrNull(i) ?: continue
            val reprojErr = computeReprojectionError(point3d, prevPt, nextPt, k, r, t)
            if (reprojErr > MAX_REPROJECTION_ERROR_PX) {
                continue
            }
            val depthStable =
                abs(depth1 - depth2) / max(abs(depth1), DEPTH_STABILITY_EPS) <= MAX_DEPTH_RELATIVE_DELTA
            if (!depthStable) {
                continue
            }
            val depthMagnitude = abs(depth1)
            if (depthMagnitude < clipRange.first || depthMagnitude > clipRange.second) {
                continue
            }

            val zScale = if (dist < minParallax) 0.0 else (dist - minParallax) / 10.0
            cloudPoints.add(Point(nextPt.x, nextPt.y - zScale * PERSPECTIVE_FACTOR))
            cloudDepths.add(depth1)
            
            // Sample color from RGBA src
            val ix = nextPt.x.toInt().coerceIn(0, srcRgba.cols() - 1)
            val iy = nextPt.y.toInt().coerceIn(0, srcRgba.rows() - 1)
            val rgba = srcRgba.get(iy, ix)
            cloudColors.add(org.opencv.core.Scalar(rgba[0], rgba[1], rgba[2]))

            val reprojScore = 1.0 - (reprojErr / MAX_REPROJECTION_ERROR_PX).coerceIn(0.0, 1.0)
            val stabilityRatio = abs(depth1 - depth2) / max(abs(depth1), DEPTH_STABILITY_EPS)
            val depthScore = 1.0 - (stabilityRatio / MAX_DEPTH_RELATIVE_DELTA).coerceIn(0.0, 1.0)
            val centerDepth = (clipRange.first + clipRange.second) * 0.5
            val spread = max((clipRange.second - clipRange.first) * 0.5, DEPTH_STABILITY_EPS)
            val clipScore = 1.0 - (abs(depthMagnitude - centerDepth) / spread).coerceIn(0.0, 1.0)
            val confidence = (0.5 * reprojScore + 0.3 * depthScore + 0.2 * clipScore).coerceIn(0.0, 1.0)
            cloudConfidences.add(confidence)
            cloudTimestamps.add(frameTimestampMs)
        }

        val meanParallax = if (nextArr.isEmpty()) 0.0 else parallaxSum / nextArr.size
        val meshEdges = mutableListOf<Pair<Point, Point>>()

        if (isMeshEnabled && cloudPoints.size >= 3) {
            try {
                var minX = Double.MAX_VALUE
                var minY = Double.MAX_VALUE
                var maxX = Double.MIN_VALUE
                var maxY = Double.MIN_VALUE
                for (p in cloudPoints) {
                    if (p.x < minX) minX = p.x
                    if (p.y < minY) minY = p.y
                    if (p.x > maxX) maxX = p.x
                    if (p.y > maxY) maxY = p.y
                }
                val rect = Rect(
                    (minX - 1).toInt(),
                    (minY - 1).toInt(),
                    (maxX - minX + 2).toInt(),
                    (maxY - minY + 2).toInt(),
                )
                val subdivide = Subdiv2D(rect)
                for (p in cloudPoints) subdivide.insert(p)

                val edgeList = MatOfFloat4()
                subdivide.getEdgeList(edgeList)
                val edges = edgeList.toArray()
                for (i in 0 until edges.size step 4) {
                    val p1 = Point(edges[i].toDouble(), edges[i + 1].toDouble())
                    val p2 = Point(edges[i + 2].toDouble(), edges[i + 3].toDouble())
                    if (rect.contains(p1) && rect.contains(p2)) {
                        val d2 = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y)
                        if (d2 < MAX_MESH_EDGE_DIST_SQ) {
                            meshEdges.add(Pair(p1, p2))
                        }
                    }
                }
                edgeList.release()
            } catch (e: Exception) {
                android.util.Log.w(TAG, "Mesh generation failed", e)
            }
        }

        essential.release()
        r.release()
        t.release()
        mask.release()
        triPoints.release()
        projection1.release()
        projection2.release()
        rt.release()

        return state to PointCloudState(
            cloudPoints,
            cloudDepths,
            cloudColors,
            cloudConfidences,
            cloudTimestamps,
            meshEdges,
            meanParallax,
        )
    }

    private data class Point3Data(val x: Double, val y: Double, val z: Double)

    private fun robustDepthClipRange(depths: List<Double>): Pair<Double, Double> {
        if (depths.isEmpty()) return 0.0 to Double.MAX_VALUE
        val sorted = depths.sorted()
        val median = sorted[sorted.size / 2]
        val absDev = sorted.map { abs(it - median) }.sorted()
        val mad = max(absDev[absDev.size / 2], DEPTH_STABILITY_EPS)
        val lower = max(0.0, median - DEPTH_CLIP_MAD_FACTOR * mad)
        val upper = median + DEPTH_CLIP_MAD_FACTOR * mad
        return lower to upper
    }

    private fun computeReprojectionError(
        point3d: Point3Data,
        prevObservation: Point,
        nextObservation: Point,
        k: Mat,
        r: Mat,
        t: Mat,
    ): Double {
        val fx = k.get(0, 0)[0]
        val fy = k.get(1, 1)[0]
        val cx = k.get(0, 2)[0]
        val cy = k.get(1, 2)[0]
        val z1 = if (abs(point3d.z) < DEPTH_STABILITY_EPS) DEPTH_STABILITY_EPS else point3d.z
        val u1 = fx * point3d.x / z1 + cx
        val v1 = fy * point3d.y / z1 + cy
        val err1 = sqrt((u1 - prevObservation.x) * (u1 - prevObservation.x) + (v1 - prevObservation.y) * (v1 - prevObservation.y))

        val x2 = r.get(0, 0)[0] * point3d.x + r.get(0, 1)[0] * point3d.y + r.get(0, 2)[0] * point3d.z + t.get(0, 0)[0]
        val y2 = r.get(1, 0)[0] * point3d.x + r.get(1, 1)[0] * point3d.y + r.get(1, 2)[0] * point3d.z + t.get(1, 0)[0]
        val z2raw = r.get(2, 0)[0] * point3d.x + r.get(2, 1)[0] * point3d.y + r.get(2, 2)[0] * point3d.z + t.get(2, 0)[0]
        val z2 = if (abs(z2raw) < DEPTH_STABILITY_EPS) DEPTH_STABILITY_EPS else z2raw
        val u2 = fx * x2 / z2 + cx
        val v2 = fy * y2 / z2 + cy
        val err2 = sqrt((u2 - nextObservation.x) * (u2 - nextObservation.x) + (v2 - nextObservation.y) * (v2 - nextObservation.y))
        return (err1 + err2) * 0.5
    }

    private fun releaseTrackingMats(vararg mats: Mat) {
        for (mat in mats) {
            mat.release()
        }
    }

    fun reset() {
        synchronized(this) {
            prevGray.release()
            prevGray = Mat()
            lastPointCloud = null
            lastTracks.clear()
        }
    }
}
