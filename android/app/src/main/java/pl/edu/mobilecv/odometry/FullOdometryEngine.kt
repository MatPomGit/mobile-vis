package pl.edu.mobilecv.odometry

import org.opencv.calib3d.Calib3d
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video
import pl.edu.mobilecv.vision.CameraCalibrator
import pl.edu.mobilecv.MarkerDetection
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.*
import org.opencv.features2d.ORB
import org.opencv.features2d.BFMatcher

/**
 * FullOdometryEngine: A robust SLAM-like odometry system that supports:
 * 1. Monocular visual odometry (VO).
 * 2. Marker-based global re-localization and anchoring.
 * 3. Map persistence (Save/Load).
 * 4. Loop closure detection and correction.
 */
class FullOdometryEngine {

    /** Current estimated pose relative to the world origin. */
    data class PoseFrame(
        val position: Point3,
        val rotationDeg: Double,
        val translationStep: Double,
        val inlierRatio: Double
    )

    /** Overall state summary for UI. */
    data class FullOdometryState(
        val tracksCount: Int,
        val inliersCount: Int,
        val frameCount: Int,
        val totalSteps: Int,
        val currentPose: PoseFrame?
    )

    /** Trajectory history for rendering. */
    data class TrajectoryState(
        val positions: List<Point3>,
        val currentPosition: Point3?
    )

    /** Sparse map data for persistence and rendering. */
    data class MapState(
        val points3d: List<Point3>,
        val colors: List<Int>,
        val cameraPosition: Point3?,
        val cameraRotation: Mat?,
        val markers: List<MarkerLandmark>
    )

    data class MarkerLandmark(
        val key: String,
        val position: Point3,
        val label: String
    )

    companion object {
        private const val TAG = "FullOdometryEngine"
        private const val MIN_TRACK_COUNT = 30
        private const val MAX_TRAJECTORY_POINTS = 5000
        private const val MAX_MAP_POINTS = 20000
        private const val MAX_DEPTH = 50.0
        private const val MIN_DEPTH = 0.1
        private const val FEATURE_QUALITY_LEVEL = 0.01
        private const val FEATURE_MIN_DISTANCE = 10.0
        private const val RANSAC_CONFIDENCE = 0.999
        private const val RANSAC_THRESHOLD = 1.0
        private const val MIN_HOMOGENEOUS_COORDINATE = 1e-6
        private const val MIN_TRANSLATION_NORM = 0.01
        private const val AUTO_SAVE_THRESHOLD = 100

        private fun vectorNorm(m: Mat): Double {
            if (m.empty()) return 0.0
            var sum = 0.0
            for (i in 0 until m.rows()) {
                val v = m.get(i, 0)[0]
                sum += v * v
            }
            return sqrt(sum)
        }
    }

    private var prevGray = Mat()
    private var prevPts = MatOfPoint2f()
    private var calibratorRef: CameraCalibrator? = null

    // Global Pose (World -> Camera)
    private var globalR = Mat.eye(3, 3, CvType.CV_64F)
    private var globalT = Mat.zeros(3, 1, CvType.CV_64F)
    
    private var scaleFactor = 1.0
    private var isScaleInitialized = false

    private var frameCount = 0
    private var stepCount = 0

    // Map & Landmarks
    data class Landmark(
        var position: Point3,
        val color: Int,
        var lastSeenPoint: Point,
        var observedCount: Int,
        var lastSeenFrame: Int
    )

    data class MarkerState(
        val key: String,
        var position: Point3,
        var orientation: Mat,
        val label: String,
        var lastSeenTs: Long,
        val isAnchored: Boolean
    )

    data class Keyframe(
        val id: Int,
        val poseR: Mat,
        val poseT: Mat,
        val descriptors: Mat,
        val keypoints: MatOfPoint2f,
        val associatedPoints3d: MatOfPoint3f // 3D points corresponding to descriptors
    )

    private val activeLandmarks = mutableMapOf<Int, Landmark>()
    private val worldMarkers = mutableMapOf<String, MarkerState>()
    private val keyframes = mutableListOf<Keyframe>()
    private val orbDetector = ORB.create(500)
    private val matcher = BFMatcher.create(Core.NORM_HAMMING, true)
    
    private var lastKfT = Mat.zeros(3, 1, CvType.CV_64F)
    private val KF_DISTANCE_THRESHOLD = 0.5 // Create KF every 50cm
    private var nextLandmarkId = 0
    private val trajectoryHistory = mutableListOf<Point3>()

    /** Information about feature tracking for visualization. */
    data class VisualTrack(
        val p1: Point,
        val p2: Point,
        val isInlier: Boolean
    )

    private var lastTracks = mutableListOf<VisualTrack>()
    private var lastState: FullOdometryState? = null

    /** Callback for automatic map export. */
    var onLargeMapDetected: ((MapState) -> Unit)? = null
    /** Callback for loop closure events. */
    var onLoopClosed: ((String) -> Unit)? = null
    private var lastAutoSaveCount = 0

    // Tracking indices between frames
    private var prevPointIds = IntArray(0)

    // ---------------------------------------------------------------
    // Settings (may be changed between frames)
    // ---------------------------------------------------------------

    var maxFeatures: Int = 800
    @Suppress("UNUSED_VARIABLE") var minParallax: Double = 0.5

    // ---------------------------------------------------------------
    // Public accessors
    // ---------------------------------------------------------------

    val currentTracks: List<VisualTrack>
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
                globalR.clone(),
                worldMarkers.values.map { MarkerLandmark(it.key, it.position, it.label) }
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
     * @param markers List of detected markers in current frame.
     */
    fun processFrameRgba(src: Mat, calib: CameraCalibrator? = null, markers: List<MarkerDetection> = emptyList()) {
        val gray = Mat()
        if (src.channels() > 1) {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        } else {
            src.copyTo(gray)
        }
        calibratorRef = calib
        
        // --- Marker Integration ---
        if (markers.isNotEmpty()) {
            integrateMarkers(markers)
        }

        processFrameInternal(gray, src)
        gray.release()
    }

    private fun integrateMarkers(markers: List<MarkerDetection>) {
        synchronized(this) {
            val predictedRs = mutableListOf<Mat>()
            val predictedTs = mutableListOf<Mat>()
            val weights = mutableListOf<Double>()

            for (m in markers) {
                val tvec = m.tvec ?: continue
                val rvec = m.rvec ?: continue
                
                // T_c_m: Marker in Camera frame
                val r_c_m = Mat()
                Calib3d.Rodrigues(MatOfDouble(rvec[0], rvec[1], rvec[2]), r_c_m)
                val t_c_m = Mat(3, 1, CvType.CV_64F)
                t_c_m.put(0, 0, tvec[0]); t_c_m.put(1, 0, tvec[1]); t_c_m.put(2, 0, tvec[2])
                
                val key = "${m.type}:${m.id}"
                val state = worldMarkers[key]
                
                if (state == null) {
                    // 1. Initial discovery: Anchor marker in World frame
                    val r_w_c = globalR.t()
                    val camPos = Mat()
                    Core.gemm(r_w_c, globalT, -1.0, Mat(), 0.0, camPos)
                    
                    val r_w_m = Mat()
                    Core.gemm(r_w_c, r_c_m, 1.0, Mat(), 0.0, r_w_m)
                    
                    val t_w_m = Mat()
                    Core.gemm(r_w_c, t_c_m, 1.0, camPos, 1.0, t_w_m)
                    
                    val pos = Point3(t_w_m.get(0, 0)[0], t_w_m.get(1, 0)[0], t_w_m.get(2, 0)[0])
                    worldMarkers[key] = MarkerState(key, pos, r_w_m, "${m.type}#${m.id}", System.currentTimeMillis(), isAnchored = true)
                    
                    r_w_c.release(); camPos.release(); t_w_m.release()
                } else {
                    // 2. SLAM Re-localization: Collect predicted camera poses
                    val r_m_w = state.orientation.t()
                    val t_w_m = Mat(3, 1, CvType.CV_64F)
                    t_w_m.put(0, 0, state.position.x); t_w_m.put(1, 0, state.position.y); t_w_m.put(2, 0, state.position.z)
                    val t_m_w = Mat()
                    Core.gemm(r_m_w, t_w_m, -1.0, Mat(), 0.0, t_m_w)
                    
                    val predR = Mat()
                    Core.gemm(r_c_m, r_m_w, 1.0, Mat(), 0.0, predR)
                    val predT = Mat()
                    Core.gemm(r_c_m, t_m_w, 1.0, t_c_m, 1.0, predT)
                    
                    predictedRs.add(predR)
                    predictedTs.add(predT)
                    
                    // Weight based on reprojection error or distance (closer markers are more reliable)
                    val dist = vectorNorm(t_c_m)
                    val confidence = m.quality.confidence?.coerceIn(0.1, 1.0) ?: 0.5
                    weights.add(confidence / (1.0 + dist))
                    
                    state.lastSeenTs = System.currentTimeMillis()
                    r_m_w.release(); t_w_m.release(); t_m_w.release()
                }
                r_c_m.release(); t_c_m.release()
            }

            if (predictedRs.isNotEmpty()) {
                // --- Scale Initialization/Update ---
                if (!isScaleInitialized && markers.size >= 1) {
                    val m = markers[0]
                    val tvec = m.tvec ?: return@synchronized
                    val markerDistCam = sqrt(tvec[0]*tvec[0] + tvec[1]*tvec[1] + tvec[2]*tvec[2])
                    
                    if (markerDistCam > 0.1) {
                        isScaleInitialized = true
                    }
                }

                // Average predicted poses
                val sumR = Mat.zeros(3, 3, CvType.CV_64F)
                val sumT = Mat.zeros(3, 1, CvType.CV_64F)
                var totalWeight = 0.0
                
                for (i in predictedRs.indices) {
                    val w = weights[i]
                    Core.addWeighted(sumR, 1.0, predictedRs[i], w, 0.0, sumR)
                    Core.addWeighted(sumT, 1.0, predictedTs[i], w, 0.0, sumT)
                    totalWeight += w
                }
                
                if (totalWeight > 0) {
                    val avgR = Mat()
                    val avgT = Mat()
                    Core.multiply(sumR, org.opencv.core.Scalar(1.0 / totalWeight), avgR)
                    Core.multiply(sumT, org.opencv.core.Scalar(1.0 / totalWeight), avgT)
                    
                    // Alpha-blending correction (EMA)
                    val alpha = 0.2
                    val correctedR = Mat()
                    Core.addWeighted(globalR, 1.0 - alpha, avgR, alpha, 0.0, correctedR)
                    
                    // Re-orthogonalize R
                    val w_svd = Mat(); val u_svd = Mat(); val vt_svd = Mat()
                    Core.SVDecomp(correctedR, w_svd, u_svd, vt_svd)
                    Core.gemm(u_svd, vt_svd, 1.0, Mat(), 0.0, globalR)
                    
                    Core.addWeighted(globalT, 1.0 - alpha, avgT, alpha, 0.0, globalT)
                    
                    avgR.release(); avgT.release(); correctedR.release()
                    w_svd.release(); u_svd.release(); vt_svd.release()
                }
                
                sumR.release(); sumT.release()
                predictedRs.forEach { it.release() }
                predictedTs.forEach { it.release() }
            }
        }
    }

    private fun processFrameInternal(gray: Mat, srcRgba: Mat) {
        if (prevGray.empty()) {
            gray.copyTo(prevGray)
            detectNewFeatures(gray)
            synchronized(this) {
                val origin = Point3(0.0, 0.0, 0.0)
                trajectoryHistory.add(origin)
            }
            return
        }

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
                    goodIds.add(-1)
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
            }
            return
        }

        val goodPrev = MatOfPoint2f(*goodPrevList.toTypedArray())
        val goodNext = MatOfPoint2f(*goodNextList.toTypedArray())

        val calibProfile = calibratorRef?.getCalibrationProfile(gray.size())
        val k = calibProfile?.calibration?.cameraMatrix
            ?: buildDefaultK(gray.cols().toDouble(), gray.rows().toDouble())

        val essential = try {
            Calib3d.findEssentialMat(goodPrev, goodNext, k, Calib3d.RANSAC, RANSAC_CONFIDENCE, RANSAC_THRESHOLD)
        } catch (e: CvException) {
            goodPrev.release(); goodNext.release(); nextPts.release()
            return
        }

        val relR = Mat()
        val relT = Mat()
        val poseMask = Mat()
        try {
            Calib3d.recoverPose(essential, goodPrev, goodNext, k, relR, relT, poseMask)
        } catch (e: CvException) {
            essential.release(); relR.release(); relT.release(); poseMask.release()
            goodPrev.release(); goodNext.release(); nextPts.release()
            return
        }
        essential.release()

        val maskArr = ByteArray(poseMask.rows() * poseMask.cols())
        if (!poseMask.empty()) poseMask.get(0, 0, maskArr)
        poseMask.release()

        val inlierPrev = mutableListOf<Point>()
        val inlierNext = mutableListOf<Point>()
        val inlierIds = mutableListOf<Int>()
        
        val newTracks = mutableListOf<VisualTrack>()

        for (i in goodNextList.indices) {
            val isInlier = i < maskArr.size && maskArr[i].toInt() != 0
            newTracks.add(VisualTrack(goodPrevList[i], goodNextList[i], isInlier))
            if (isInlier) {
                inlierPrev.add(goodPrevList[i])
                inlierNext.add(goodNextList[i])
                inlierIds.add(goodIds[i])
            }
        }

        val tNorm = vectorNorm(relT)
        if (tNorm > MIN_TRANSLATION_NORM) {
            synchronized(this) {
                val nextGlobalR = Mat()
                Core.gemm(relR, globalR, 1.0, Mat(), 0.0, nextGlobalR)
                
                val nextGlobalT = Mat()
                Core.gemm(relR, globalT, 1.0, relT, 1.0, nextGlobalT)
                
                nextGlobalR.copyTo(globalR)
                nextGlobalT.copyTo(globalT)
                
                val r_w_c = globalR.t()
                val camPos = Mat()
                Core.gemm(r_w_c, globalT, -1.0, Mat(), 0.0, camPos)
                val currentPos = Point3(camPos.get(0,0)[0], camPos.get(1,0)[0], camPos.get(2,0)[0])
                
                trajectoryHistory.add(currentPos)
                if (trajectoryHistory.size > MAX_TRAJECTORY_POINTS) trajectoryHistory.removeAt(0)
                
                triangulateAndUpdateMap(inlierPrev, inlierNext, inlierIds, k, relR, relT, srcRgba)
                
                checkKeyframeCreation(globalR, globalT, gray, k)
                
                stepCount++
                val rot = Mat()
                Calib3d.Rodrigues(globalR, rot)
                val yaw = Math.toDegrees(rot.get(1, 0)[0])
                
                lastState = FullOdometryState(
                    goodNextList.size,
                    inlierNext.size,
                    frameCount,
                    stepCount,
                    PoseFrame(currentPos, yaw, tNorm, inlierNext.size.toDouble() / goodNextList.size)
                )
                
                nextGlobalR.release(); nextGlobalT.release(); r_w_c.release(); camPos.release(); rot.release()
            }
        }

        gray.copyTo(prevGray)
        goodNext.copyTo(prevPts)
        prevPointIds = goodIds.toIntArray()
        
        synchronized(this) {
            lastTracks = newTracks
        }

        goodPrev.release(); goodNext.release(); nextPts.release()
        relR.release(); relT.release()
        frameCount++
    }

    private fun detectNewFeatures(gray: Mat) {
        val features = MatOfPoint()
        Imgproc.goodFeaturesToTrack(gray, features, maxFeatures, FEATURE_QUALITY_LEVEL, FEATURE_MIN_DISTANCE)
        val pts = features.toArray()
        
        prevPts.release()
        prevPts = MatOfPoint2f(*pts)
        
        prevPointIds = IntArray(pts.size) { 
            val id = nextLandmarkId++
            id
        }
        
        features.release()
    }

    private fun triangulateAndUpdateMap(prev: List<Point>, next: List<Point>, ids: List<Int>, k: Mat, r: Mat, t: Mat, src: Mat) {
        if (prev.isEmpty()) return

        val p1 = Mat.eye(3, 4, CvType.CV_64F)
        val p2 = Mat(3, 4, CvType.CV_64F)
        r.copyTo(p2.submat(0, 3, 0, 3))
        t.copyTo(p2.submat(0, 3, 3, 4))
        
        val kPrev = Mat()
        Core.gemm(k, p1, 1.0, Mat(), 0.0, kPrev)
        val kNext = Mat()
        Core.gemm(k, p2, 1.0, Mat(), 0.0, kNext)

        val pts1 = MatOfPoint2f(*prev.toTypedArray())
        val pts2 = MatOfPoint2f(*next.toTypedArray())
        val pts4d = Mat()
        
        Calib3d.triangulatePoints(kPrev, kNext, pts1, pts2, pts4d)

        for (i in 0 until pts4d.cols()) {
            val w = pts4d.get(3, i)[0]
            if (abs(w) > MIN_HOMOGENEOUS_COORDINATE) {
                val x = pts4d.get(0, i)[0] / w
                val y = pts4d.get(1, i)[0] / w
                val z = pts4d.get(2, i)[0] / w

                if (z in MIN_DEPTH..MAX_DEPTH) {
                    val ptCam = Mat(3, 1, CvType.CV_64F)
                    ptCam.put(0, 0, x); ptCam.put(1, 0, y); ptCam.put(2, 0, z)
                    
                    val r_w_c = globalR.t()
                    val ptWorld = Mat()
                    Core.gemm(r_w_c, ptCam, 1.0, globalT, -1.0, ptWorld)
                    Core.gemm(r_w_c, ptWorld, 1.0, Mat(), 0.0, ptWorld) 

                    val worldPos = Point3(ptWorld.get(0, 0)[0], ptWorld.get(1, 0)[0], ptWorld.get(2, 0)[0])
                    
                    val id = ids[i]
                    val pixel = next[i]
                    val color = if (pixel.x >= 0 && pixel.x < src.cols() && pixel.y >= 0 && pixel.y < src.rows()) {
                        val c = src.get(pixel.y.toInt(), pixel.x.toInt())
                        if (c != null && c.size >= 3) {
                            (255 shl 24) or (c[0].toInt() shl 16) or (c[1].toInt() shl 8) or c[2].toInt()
                        } else 0xFFFFFFFF.toInt()
                    } else 0xFFFFFFFF.toInt()

                    val existing = activeLandmarks[id]
                    if (existing == null) {
                        activeLandmarks[id] = Landmark(worldPos, color, pixel, 1, frameCount)
                    } else {
                        val alpha = 0.1
                        existing.position = Point3(
                            existing.position.x * (1 - alpha) + worldPos.x * alpha,
                            existing.position.y * (1 - alpha) + worldPos.y * alpha,
                            existing.position.z * (1 - alpha) + worldPos.z * alpha
                        )
                        existing.lastSeenPoint = pixel
                        existing.observedCount++
                        existing.lastSeenFrame = frameCount
                    }
                    ptCam.release(); ptWorld.release(); r_w_c.release()
                }
            }
        }
        
        if (activeLandmarks.size > MAX_MAP_POINTS) {
            val toRemove = activeLandmarks.keys.take(activeLandmarks.size - MAX_MAP_POINTS)
            toRemove.forEach { activeLandmarks.remove(it) }
        }

        if (activeLandmarks.size - lastAutoSaveCount > AUTO_SAVE_THRESHOLD) {
            onLargeMapDetected?.invoke(currentMap)
            lastAutoSaveCount = activeLandmarks.size
        }

        pts1.release(); pts2.release(); pts4d.release(); kPrev.release(); kNext.release(); p1.release(); p2.release()
    }

    private fun buildDefaultK(w: Double, h: Double): Mat {
        val f = max(w, h) * 0.8
        val k = Mat.eye(3, 3, CvType.CV_64F)
        k.put(0, 0, f)
        k.put(1, 1, f)
        k.put(0, 2, w / 2.0)
        k.put(1, 2, h / 2.0)
        return k
    }

    private fun checkKeyframeCreation(r: Mat, t: Mat, gray: Mat, k: Mat) {
        val dist = sqrt(
            (t.get(0, 0)[0] - lastKfT.get(0, 0)[0]).pow(2.0) +
            (t.get(1, 0)[0] - lastKfT.get(1, 0)[0]).pow(2.0) +
            (t.get(2, 0)[0] - lastKfT.get(2, 0)[0]).pow(2.0)
        )

        if (dist > KF_DISTANCE_THRESHOLD || keyframes.isEmpty()) {
            val keypoints = MatOfKeyPoint()
            val descriptors = Mat()
            orbDetector.detectAndCompute(gray, Mat(), keypoints, descriptors)
            
            if (!descriptors.empty()) {
                val kpArr = keypoints.toArray()
                val associated3d = mutableListOf<Point3>()
                val validIndices = mutableListOf<Int>()
                
                synchronized(this) {
                    val landmarks = activeLandmarks.values.toList()
                    for (i in kpArr.indices) {
                        val kp = kpArr[i].pt
                        val match = landmarks.find { hypot(it.lastSeenPoint.x - kp.x, it.lastSeenPoint.y - kp.y) < 4.0 }
                        if (match != null) {
                            associated3d.add(match.position)
                            validIndices.add(i)
                        }
                    }
                }

                if (associated3d.size > 20) {
                    val filteredDescriptors = Mat(validIndices.size, descriptors.cols(), descriptors.type())
                    val filteredKeypoints = mutableListOf<Point>()
                    for (i in validIndices.indices) {
                        val idx = validIndices[i]
                        descriptors.row(idx).copyTo(filteredDescriptors.row(i))
                        filteredKeypoints.add(kpArr[idx].pt)
                    }
                    
                    val kf = Keyframe(
                        keyframes.size, 
                        r.clone(), 
                        t.clone(), 
                        filteredDescriptors,
                        MatOfPoint2f(*filteredKeypoints.toTypedArray()),
                        MatOfPoint3f(*associated3d.toTypedArray())
                    )
                    
                    detectAndCorrectLoop(kf, k)
                    keyframes.add(kf)
                    t.copyTo(lastKfT)
                    filteredDescriptors.release()
                }
            }
            keypoints.release()
            descriptors.release()
        }
    }

    private fun detectAndCorrectLoop(currentKf: Keyframe, k: Mat) {
        if (keyframes.size < 15) return

        val candidates = mutableListOf<Int>()
        val totalKfs = keyframes.size
        
        val currentT = currentKf.poseT
        val currentX = currentT.get(0, 0)[0]
        val currentY = currentT.get(1, 0)[0]
        val currentZ = currentT.get(2, 0)[0]
        
        for (i in 0 until (totalKfs - 15)) {
            val kf = keyframes[i]
            val kfT = kf.poseT
            val dx = kfT.get(0, 0)[0] - currentX
            val dy = kfT.get(1, 0)[0] - currentY
            val dz = kfT.get(2, 0)[0] - currentZ
            val distSq = dx*dx + dy*dy + dz*dz
            
            if (distSq < 9.0) {
                candidates.add(i)
            }
        }
        
        val randomSampleCount = 5
        repeat(randomSampleCount) {
            val randIdx = (0 until (totalKfs - 15)).random()
            if (!candidates.contains(randIdx)) {
                candidates.add(randIdx)
            }
        }
        
        val sortedCandidates = candidates.distinct().sortedDescending().take(15)

        var bestMatchIdx = -1
        
        for (i in sortedCandidates) {
            val prevKf = keyframes[i]
            val matches = MatOfDMatch()
            
            matcher?.match(currentKf.descriptors, prevKf.descriptors, matches)
            
            val matchArr = matches.toArray()
            val goodMatches = matchArr.filter { it.distance < 45.0 }
            
            if (goodMatches.size > 30) {
                val objPointsList = mutableListOf<Point3>()
                val imgPointsList = mutableListOf<Point>()
                
                val prevPoints3d = prevKf.associatedPoints3d.toArray()
                val currPoints2f = currentKf.keypoints.toArray()
                
                for (m in goodMatches) {
                    if (m.trainIdx < prevPoints3d.size && m.queryIdx < currPoints2f.size) {
                        objPointsList.add(prevPoints3d[m.trainIdx])
                        imgPointsList.add(currPoints2f[m.queryIdx])
                    }
                }
                
                if (objPointsList.size > 20) {
                    val objPoints = MatOfPoint3f(*objPointsList.toTypedArray())
                    val imgPoints = MatOfPoint2f(*imgPointsList.toTypedArray())
                    val rvec = Mat()
                    val tvec = Mat()
                    val inliers = Mat()
                    
                    val success = try {
                        Calib3d.solvePnPRansac(objPoints, imgPoints, k, MatOfDouble(), rvec, tvec, false, 100, 8.0f, 0.99, inliers, Calib3d.SOLVEPNP_ITERATIVE)
                    } catch (e: Exception) {
                        false
                    }
                    
                    if (success && inliers.rows() > 25) {
                        bestMatchIdx = i
                        objPoints.release(); imgPoints.release(); rvec.release(); tvec.release(); inliers.release(); matches.release()
                        break 
                    }
                    objPoints.release(); imgPoints.release(); rvec.release(); tvec.release(); inliers.release()
                }
            }
            matches.release()
        }

        if (bestMatchIdx != -1) {
            val logMsg = "Loop Closure: KF#${keyframes.size} -> KF#$bestMatchIdx (Verified via PnP)"
            android.util.Log.i(TAG, logMsg)
            onLoopClosed?.invoke(logMsg)
            applyLoopCorrection(bestMatchIdx, currentKf)
        }
    }

    private fun applyLoopCorrection(matchIdx: Int, currentKf: Keyframe) {
        val targetT = keyframes[matchIdx].poseT
        val driftT = Mat()
        Core.subtract(currentKf.poseT, targetT, driftT)
        
        val driftNorm = vectorNorm(driftT)
        if (driftNorm > 0.05) {
            synchronized(this) {
                val correction = Mat()
                Core.multiply(driftT, org.opencv.core.Scalar(-0.8), correction)
                Core.add(globalT, correction, globalT)
                
                val corrPt = Point3(correction.get(0,0)[0], correction.get(1,0)[0], correction.get(2,0)[0])
                for (landmark in activeLandmarks.values) {
                    landmark.position = Point3(
                        landmark.position.x + corrPt.x,
                        landmark.position.y + corrPt.y,
                        landmark.position.z + corrPt.z
                    )
                }
                
                val windowSize = minOf(trajectoryHistory.size, 50)
                for (i in (trajectoryHistory.size - windowSize) until trajectoryHistory.size) {
                    val p = trajectoryHistory[i]
                    val weight = (i - (trajectoryHistory.size - windowSize)).toDouble() / windowSize
                    trajectoryHistory[i] = Point3(
                        p.x + corrPt.x * weight,
                        p.y + corrPt.y * weight,
                        p.z + corrPt.z * weight
                    )
                }
                correction.release()
            }
        }
        driftT.release()
    }

    fun importMap(state: MapState) {
        synchronized(this) {
            activeLandmarks.clear()
            trajectoryHistory.clear()
            
            for (m in state.markers) {
                worldMarkers[m.key] = MarkerState(
                    m.key,
                    m.position,
                    Mat.eye(3, 3, CvType.CV_64F),
                    m.label,
                    System.currentTimeMillis(),
                    isAnchored = true
                )
            }
            
            globalR.release(); globalT.release()
            globalR = Mat.eye(3, 3, CvType.CV_64F)
            globalT = Mat.zeros(3, 1, CvType.CV_64F)
            
            if (state.markers.isNotEmpty()) {
                val first = state.markers[0].position
                globalT.put(0, 0, -first.x)
                globalT.put(1, 0, -first.y)
                globalT.put(2, 0, -first.z)
            }
            
            isScaleInitialized = true
            frameCount = 0
            stepCount = 0
        }
    }

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
            scaleFactor = 1.0
            isScaleInitialized = false
            frameCount = 0
            stepCount = 0
            trajectoryHistory.clear()
            activeLandmarks.clear()
            worldMarkers.clear()
            nextLandmarkId = 0
            lastAutoSaveCount = 0
            prevPointIds = IntArray(0)
            lastTracks.clear()
            lastState = null
        }
    }
}
