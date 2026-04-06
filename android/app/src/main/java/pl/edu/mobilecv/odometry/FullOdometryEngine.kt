package pl.edu.mobilecv.odometry

import android.graphics.Color
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.hypot
import kotlin.math.sqrt
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvException
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Point3
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video
import pl.edu.mobilecv.vision.CameraCalibrator

/**
 * Full monocular visual odometry engine implementing a standard VO pipeline:
 *
 * 1. Feature detection (Shi-Tomasi good features to track).
 * 2. Optical-flow tracking (Lucas-Kanade pyramid LK).
 * 3. Essential-matrix estimation with RANSAC (5-point algorithm).
 * 4. Relative pose recovery (R, t) via [Calib3d.recoverPose].
 * 5. Accumulated pose integration in world frame.
 * 6. 3D point triangulation for sparse-map building.
 * 7. Camera trajectory history storage.
 *
 * This is the "full odometry" counterpart to [VisualOdometryEngine], which
 * only provides per-frame parallax estimates without global pose accumulation.
 */
class FullOdometryEngine {

    /** Snapshot of the accumulated camera pose at one point in time. */
    data class PoseFrame(
        /** Camera position in world frame (unit-scale accumulation). */
        val position: Point3,
        /** Cumulative rotation magnitude in degrees. */
        val rotationDeg: Double,
        /** Translation magnitude of the latest step (unit-scale). */
        val translationStep: Double,
        /** Ratio of RANSAC inliers to tracked points [0,1]. */
        val inlierRatio: Double,
    )

    /** State returned after processing each frame. */
    data class FullOdometryState(
        val tracksCount: Int,
        val inliersCount: Int,
        val frameCount: Int,
        val totalSteps: Int,
        val currentPose: PoseFrame?,
    )

    /** Snapshot of the accumulated camera trajectory for display. */
    data class TrajectoryState(
        /** Ordered list of camera positions (world frame, unit scale). */
        val positions: List<Point3>,
        /** Current camera position (world frame), or null if no frames processed. */
        val currentPosition: Point3?,
    )

    /** Snapshot of the sparse 3D map built from triangulated correspondences. */
    data class MapState(
        /** Triangulated 3D points in world frame. */
        val points3d: List<Point3>,
        /** Colors for each 3D point (RGBA). */
        val colors: List<Int>,
        /** Current camera position in world frame. */
        val cameraPosition: Point3?,
        /** Current camera orientation (world frame). */
        val cameraRotation: Mat? = null,
    )

    companion object {
        private const val TAG = "FullOdometryEngine"
        /** Minimum tracked points required before attempting pose estimation. */
        private const val MIN_TRACK_COUNT = 30
        private const val MAX_TRAJECTORY_POINTS = 1000
        private const val MAX_MAP_POINTS = 5000
        private const val MAX_DEPTH = 500.0
        private const val MIN_DEPTH = 0.1
        private const val FEATURE_QUALITY_LEVEL = 0.005
        private const val FEATURE_MIN_DISTANCE = 5.0
        private const val RANSAC_CONFIDENCE = 0.999
        private const val RANSAC_THRESHOLD = 0.5
        private const val MIN_HOMOGENEOUS_COORDINATE = 1e-8
        private const val MIN_TRANSLATION_NORM = 1e-6
    }

    /** Computes the L2 norm of a 3×1 column-vector [Mat]. */
    private fun vectorNorm(v: Mat): Double {
        val x = v.get(0, 0)[0]
        val y = v.get(1, 0)[0]
        val z = v.get(2, 0)[0]
        return sqrt(x * x + y * y + z * z)
    }

    // ---------------------------------------------------------------
    // Feature-tracking state
    // ---------------------------------------------------------------

    private var prevGray = Mat()
    private var prevPts = MatOfPoint2f()
    private var calibratorRef: CameraCalibrator? = null

    // ---------------------------------------------------------------
    // Accumulated pose (world frame)
    // ---------------------------------------------------------------

    /** Cumulative rotation matrix R_w (world ← camera). */
    private var globalR = Mat.eye(3, 3, CvType.CV_64F)

    /** Cumulative translation vector t_w in world frame. */
    private var globalT = Mat.zeros(3, 1, CvType.CV_64F)

    private var frameCount = 0
    private var stepCount = 0

    // ---------------------------------------------------------------
    // Sparse Map & Landmark Management
    // ---------------------------------------------------------------

    private class Landmark(
        var position: Point3,
        var color: Int,
        var lastSeenPoint: Point,
        var observedCount: Int = 1
    )

    private val activeLandmarks = mutableMapOf<Int, Landmark>()
    private var nextLandmarkId = 0
    private val trajectoryHistory = mutableListOf<Point3>()
    private var lastTracks = mutableListOf<List<Point>>()
    private var lastState: FullOdometryState? = null

    // Tracking indices between frames
    private var prevPointIds = IntArray(0)

    // ---------------------------------------------------------------
    // Settings (may be changed between frames)
    // ---------------------------------------------------------------

    var maxFeatures: Int = 800
    var minParallax: Double = 0.5

    // ---------------------------------------------------------------
    // Public accessors
    // ---------------------------------------------------------------

    val currentTracks: List<List<Point>>
        get() = synchronized(this) { lastTracks.toList() }

    val lastOdometryState: FullOdometryState?
        get() = synchronized(this) { lastState }

    val currentTrajectory: TrajectoryState
        get() = synchronized(this) {
            val copy = trajectoryHistory.toList()
            TrajectoryState(copy, copy.lastOrNull())
        }

    val currentMap: MapState
        get() = synchronized(this) {
            MapState(
                activeLandmarks.values.map { it.position },
                activeLandmarks.values.map { it.color },
                trajectoryHistory.lastOrNull(),
                globalR.clone()
            )
        }

    // ---------------------------------------------------------------
    // Entry point
    // ---------------------------------------------------------------

    /**
     * Process one RGBA (or grayscale) frame and update internal odometry state.
     *
     * @param src  Input frame (RGBA or GRAY Mat).
     * @param calib Optional [CameraCalibrator] used to retrieve the camera matrix K.
     */
    fun processFrameRgba(src: Mat, calib: CameraCalibrator? = null) {
        val gray = Mat()
        if (src.channels() > 1) {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        } else {
            src.copyTo(gray)
        }
        calibratorRef = calib
        processFrameInternal(gray, src)
        gray.release()
    }

    // ---------------------------------------------------------------
    // Internal pipeline
    // ---------------------------------------------------------------

    private fun processFrameInternal(gray: Mat, srcRgba: Mat) {
        if (prevGray.empty()) {
            gray.copyTo(prevGray)
            detectNewFeatures(gray)
            // Record initial position
            synchronized(this) {
                val origin = Point3(0.0, 0.0, 0.0)
                trajectoryHistory.add(origin)
            }
            return
        }

        // --- Optical-flow tracking -----------------------------------------
        val nextPts = MatOfPoint2f()
        val status = MatOfByte()
        val err = MatOfFloat()
        Video.calcOpticalFlowPyrLK(prevGray, gray, prevPts, nextPts, status, err)

        val statusArr = status.toArray()
        val prevPtsArr = prevPts.toArray()
        val nextPtsArr = nextPts.toArray()

        val goodPrevList = mutableListOf<Point>()
        val goodNextList = mutableListOf<Point>()
        val goodIds = mutableListOf<Int>()
        
        for (i in statusArr.indices) {
            if (statusArr[i].toInt() == 1) {
                goodPrevList.add(prevPtsArr[i])
                goodNextList.add(nextPtsArr[i])
                if (i < prevPointIds.size) {
                    goodIds.add(prevPointIds[i])
                } else {
                    goodIds.add(-1) // Should not happen with current logic
                }
            }
        }
        status.release()
        err.release()

        if (goodNextList.size < MIN_TRACK_COUNT) {
            gray.copyTo(prevGray)
            detectNewFeatures(gray)
            nextPts.release()
            synchronized(this) { 
                lastTracks.clear() 
                activeLandmarks.clear() // Clear map on loss of tracking for simplicity in this module
                nextLandmarkId = 0
            }
            return
        }

        synchronized(this) {
            lastTracks = goodNextList.indices
                .map { i -> listOf(goodPrevList[i], goodNextList[i]) }
                .toMutableList()
        }

        val goodPrev = MatOfPoint2f(*goodPrevList.toTypedArray())
        val goodNext = MatOfPoint2f(*goodNextList.toTypedArray())

        // --- Camera intrinsics -----------------------------------------------
        val calibProfile = calibratorRef?.getCalibrationProfile(gray.size())
        val k = calibProfile?.calibration?.cameraMatrix
            ?: buildDefaultK(gray.cols().toDouble(), gray.rows().toDouble())

        // --- Essential matrix + relative pose --------------------------------
        val essential = try {
            Calib3d.findEssentialMat(goodPrev, goodNext, k, Calib3d.RANSAC, RANSAC_CONFIDENCE, RANSAC_THRESHOLD)
        } catch (e: CvException) {
            android.util.Log.w(TAG, "findEssentialMat failed (pts=${goodNextList.size}): ${e.message}")
            goodPrev.release(); goodNext.release(); nextPts.release()
            return
        }

        val relR = Mat()
        val relT = Mat()
        val poseMask = Mat()
        val inlierCount = try {
            Calib3d.recoverPose(essential, goodPrev, goodNext, k, relR, relT, poseMask)
        } catch (e: CvException) {
            android.util.Log.w(TAG, "recoverPose failed: ${e.message}")
            essential.release(); relR.release(); relT.release(); poseMask.release()
            goodPrev.release(); goodNext.release(); nextPts.release()
            return
        }
        essential.release()

        // --- Collect RANSAC inliers -------------------------
        val maskArr = ByteArray(poseMask.rows() * poseMask.cols())
        if (!poseMask.empty()) poseMask.get(0, 0, maskArr)
        poseMask.release()

        val inlierPrev = mutableListOf<Point>()
        val inlierNext = mutableListOf<Point>()
        val inlierIds = mutableListOf<Int>()
        
        for (i in maskArr.indices) {
            if (i < goodPrevList.size && maskArr[i].toInt() != 0) {
                inlierPrev.add(goodPrevList[i])
                inlierNext.add(goodNextList[i])
                inlierIds.add(goodIds[i])
            }
        }

        // --- Pose accumulation -----------------------------------------------
        val tNorm = vectorNorm(relT)

        if (tNorm > MIN_TRANSLATION_NORM) {
            val scaledRelT = Mat()
            Core.multiply(relT, org.opencv.core.Scalar(1.0 / tNorm), scaledRelT)

            val worldStep = Mat()
            Core.gemm(globalR, scaledRelT, 1.0, Mat(), 0.0, worldStep)
            Core.add(globalT, worldStep, globalT)

            val newR = Mat()
            Core.gemm(relR, globalR, 1.0, Mat(), 0.0, newR)
            globalR.release()
            globalR = newR

            stepCount++
            scaledRelT.release()
            worldStep.release()
        }
        relT.release()
        relR.release()

        // --- Camera position ---
        val rT = globalR.t()
        val camPosMat = Mat()
        Core.gemm(rT, globalT, -1.0, Mat(), 0.0, camPosMat)
        val camPos = Point3(camPosMat.get(0, 0)[0], camPosMat.get(1, 0)[0], camPosMat.get(2, 0)[0])
        camPosMat.release(); rT.release()

        val trace = globalR.get(0, 0)[0] + globalR.get(1, 1)[0] + globalR.get(2, 2)[0]
        val cosAngle = ((trace - 1.0) / 2.0).coerceIn(-1.0, 1.0)
        val rotDeg = Math.toDegrees(acos(cosAngle))
        val inlierRatio = if (goodNextList.isNotEmpty()) inlierCount.toDouble() / goodNextList.size else 0.0

        frameCount++
        val poseFrame = PoseFrame(camPos, rotDeg, tNorm, inlierRatio)

        synchronized(this) {
            trajectoryHistory.add(camPos)
            if (trajectoryHistory.size > MAX_TRAJECTORY_POINTS) trajectoryHistory.removeAt(0)
            lastState = FullOdometryState(goodNextList.size, inlierCount, frameCount, stepCount, poseFrame)
        }

        // --- Triangulate and Update Landmarks --------------------------------
        if (inlierPrev.size >= 4) {
            triangulateAndUpdateLandmarks(inlierPrev, inlierNext, inlierIds, k, srcRgba)
        }

        // --- Prepare next iteration ------------------------------------------
        gray.copyTo(prevGray)
        goodNext.copyTo(prevPts)
        prevPointIds = goodIds.toIntArray()
        
        if (goodNextList.size < maxFeatures * 0.7) {
            detectNewFeatures(gray)
        }

        goodPrev.release(); goodNext.release(); nextPts.release()
    }

    // ---------------------------------------------------------------
    // Triangulation helper
    // ---------------------------------------------------------------

    private fun triangulateAndUpdateLandmarks(
        prevInliers: List<Point>,
        nextInliers: List<Point>,
        inlierIds: List<Int>,
        k: Mat,
        srcRgba: Mat
    ) {
        val p1 = buildProjectionMatrix(Mat.eye(3, 3, CvType.CV_64F), Mat.zeros(3, 1, CvType.CV_64F), k)
        val p2 = buildProjectionMatrix(globalR, globalT, k)

        val pts1 = MatOfPoint2f(*prevInliers.toTypedArray())
        val pts2 = MatOfPoint2f(*nextInliers.toTypedArray())
        val pts4d = Mat()
        Calib3d.triangulatePoints(p1, p2, pts1, pts2, pts4d)

        synchronized(this) {
            for (i in 0 until pts4d.cols()) {
                val w = pts4d.get(3, i)[0]
                if (abs(w) < MIN_HOMOGENEOUS_COORDINATE) continue
                
                val x = pts4d.get(0, i)[0] / w
                val y = pts4d.get(1, i)[0] / w
                val z = pts4d.get(2, i)[0] / w
                
                if (z > MIN_DEPTH && z < MAX_DEPTH) {
                    val pos = Point3(x, y, z)
                    val id = inlierIds[i]
                    
                    if (id != -1 && activeLandmarks.containsKey(id)) {
                        val landmark = activeLandmarks[id]!!
                        // Exponential moving average for position refinement
                        val alpha = 0.2
                        landmark.position = Point3(
                            landmark.position.x * (1 - alpha) + pos.x * alpha,
                            landmark.position.y * (1 - alpha) + pos.y * alpha,
                            landmark.position.z * (1 - alpha) + pos.z * alpha
                        )
                        landmark.lastSeenPoint = nextInliers[i]
                        landmark.observedCount++
                        
                        // Refine color? For now just keep first or update
                        val ix = nextInliers[i].x.toInt().coerceIn(0, srcRgba.cols() - 1)
                        val iy = nextInliers[i].y.toInt().coerceIn(0, srcRgba.rows() - 1)
                        val rgba = srcRgba.get(iy, ix)
                        landmark.color = Color.rgb(rgba[0].toInt(), rgba[1].toInt(), rgba[2].toInt())
                    } else if (id != -1) {
                        val ix = nextInliers[i].x.toInt().coerceIn(0, srcRgba.cols() - 1)
                        val iy = nextInliers[i].y.toInt().coerceIn(0, srcRgba.rows() - 1)
                        val rgba = srcRgba.get(iy, ix)
                        val color = Color.rgb(rgba[0].toInt(), rgba[1].toInt(), rgba[2].toInt())
                        activeLandmarks[id] = Landmark(pos, color, nextInliers[i])
                    }
                }
            }
            
            // Maintenance: remove old landmarks if too many
            if (activeLandmarks.size > MAX_MAP_POINTS) {
                val toRemove = activeLandmarks.size - MAX_MAP_POINTS
                val keysToRemove = activeLandmarks.keys.take(toRemove)
                keysToRemove.forEach { activeLandmarks.remove(it) }
            }
        }

        pts4d.release(); pts1.release(); pts2.release(); p1.release(); p2.release()
    }

    private fun buildProjectionMatrix(r: Mat, t: Mat, k: Mat): Mat {
        val rt = Mat(3, 4, CvType.CV_64F)
        for (row in 0 until 3) {
            for (col in 0 until 3) rt.put(row, col, r.get(row, col)[0])
            rt.put(row, 3, t.get(row, 0)[0])
        }
        val p = Mat()
        Core.gemm(k, rt, 1.0, Mat(), 0.0, p)
        rt.release()
        return p
    }

    /** Build a reasonable default K when camera calibration data is unavailable. */
    private fun buildDefaultK(width: Double, height: Double): Mat {
        val f = maxOf(width, height)
        val k = Mat.eye(3, 3, CvType.CV_64F)
        k.put(0, 0, f)
        k.put(1, 1, f)
        k.put(0, 2, width / 2.0)
        k.put(1, 2, height / 2.0)
        return k
    }

    // ---------------------------------------------------------------
    // Feature detection
    // ---------------------------------------------------------------

    private fun detectNewFeatures(gray: Mat) {
        val corners = MatOfPoint()
        Imgproc.goodFeaturesToTrack(gray, corners, maxFeatures, FEATURE_QUALITY_LEVEL, FEATURE_MIN_DISTANCE)
        if (!corners.empty()) {
            val cornersArr = corners.toArray()
            val newPtsList = mutableListOf<Point>()
            val newIds = mutableListOf<Int>()
            
            // Filter out points too close to existing points
            val existingPts = prevPts.toArray()
            
            for (p in cornersArr) {
                var tooClose = false
                for (ep in existingPts) {
                    if (hypot(p.x - ep.x, p.y - ep.y) < FEATURE_MIN_DISTANCE) {
                        tooClose = true; break
                    }
                }
                if (!tooClose) {
                    newPtsList.add(p)
                    newIds.add(nextLandmarkId++)
                }
            }
            
            if (newPtsList.isNotEmpty()) {
                val mergedPts = existingPts.toMutableList()
                mergedPts.addAll(newPtsList)
                
                val mergedIds = prevPointIds.toMutableList()
                mergedIds.addAll(newIds)
                
                prevPts.release()
                prevPts = MatOfPoint2f(*mergedPts.toTypedArray())
                prevPointIds = mergedIds.toIntArray()
            }
        }
        corners.release()
    }

    // ---------------------------------------------------------------
    // Reset
    // ---------------------------------------------------------------

    /** Reset all accumulated state (call when switching away from this tab). */
    fun reset() {
        synchronized(this) {
            prevGray.release()
            prevPts.release()
            prevGray = Mat()
            prevPts = MatOfPoint2f()
            globalR.release()
            globalT.release()
            globalR = Mat.eye(3, 3, CvType.CV_64F)
            globalT = Mat.zeros(3, 1, CvType.CV_64F)
            frameCount = 0
            stepCount = 0
            trajectoryHistory.clear()
            activeLandmarks.clear()
            nextLandmarkId = 0
            prevPointIds = IntArray(0)
            lastTracks.clear()
            lastState = null
        }
    }
}
