package pl.edu.mobilecv

import kotlin.math.abs
import kotlin.math.min
import org.opencv.calib3d.Calib3d
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.imgproc.Imgproc
import org.opencv.imgproc.Subdiv2D
import org.opencv.video.Video

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
        val meanParallax: Double,
        val meshEdges: List<Pair<Point, Point>> = emptyList()
    )

    private var previousGray: Mat? = null
    private var previousFeatures: MatOfPoint2f? = null

    var maxFeatures: Int = MAX_FEATURES_DEFAULT
    var minParallax: Double = MIN_PARALLAX_DEFAULT
    var isMeshEnabled: Boolean = false

    fun reset() {
        previousFeatures?.release()
        previousFeatures = null
        previousGray?.release()
        previousGray = null
    }

    fun updateOdometry(rgba: Mat): OdometryState? {
        val gray = Mat()
        Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)

        if (previousGray == null) {
            previousGray = gray.clone()
            previousFeatures = detectFeatures(gray)
            gray.release()
            return null
        }

        val prevGrayLocal = previousGray!!
        val prevFeaturesLocal = previousFeatures!!

        val nextFeatures = MatOfPoint2f()
        val status = MatOfByte()
        val err = MatOfFloat()
        Video.calcOpticalFlowPyrLK(prevGrayLocal, gray, prevFeaturesLocal, nextFeatures, status, err)

        val trackedPairs = collectTrackedPairs(prevFeaturesLocal, nextFeatures, status)
        val prevTracked = trackedPairs.first
        val nextTracked = trackedPairs.second

        val state = if (prevTracked.rows() >= MIN_TRACKS_FOR_POSE) {
            estimateRelativePose(prevTracked, nextTracked, gray.cols().toDouble(), gray.rows().toDouble())
        } else null

        previousGray?.release()
        previousGray = gray.clone()
        previousFeatures?.release()
        previousFeatures = if (nextTracked.rows() >= MIN_FEATURES_TO_KEEP) nextTracked else detectFeatures(gray)

        gray.release()
        nextFeatures.release()
        status.release()
        err.release()
        if (previousFeatures !== nextTracked) nextTracked.release()
        prevTracked.release()

        return state
    }

    fun updatePointCloud(rgba: Mat): PointCloudState? {
        val gray = Mat()
        Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)

        if (previousGray == null || previousFeatures == null) {
            previousGray?.release()
            previousGray = gray.clone()
            previousFeatures?.release()
            previousFeatures = detectFeatures(gray)
            gray.release()
            return null
        }

        val nextFeatures = MatOfPoint2f()
        val status = MatOfByte()
        val err = MatOfFloat()
        Video.calcOpticalFlowPyrLK(previousGray!!, gray, previousFeatures!!, nextFeatures, status, err)

        val prevArr = previousFeatures!!.toArray()
        val nextArr = nextFeatures.toArray()
        val statusArr = status.toArray()

        val cloudPoints = ArrayList<Point>()
        var parallaxSum = 0.0
        var validCount = 0

        for (i in statusArr.indices) {
            if (statusArr[i].toInt() == 0 || i >= prevArr.size || i >= nextArr.size) continue
            val prevPt = prevArr[i]
            val nextPt = nextArr[i]
            val parallax = kotlin.math.sqrt((nextPt.x - prevPt.x) * (nextPt.x - prevPt.x) + (nextPt.y - prevPt.y) * (nextPt.y - prevPt.y))
            
            if (parallax < minParallax) continue
            parallaxSum += parallax
            validCount++

            val pseudoDepth = 1.0 / (parallax + EPSILON)
            val zScale = Math.log(1.0 + pseudoDepth * DEPTH_VISUAL_GAIN)
            cloudPoints.add(Point(nextPt.x, nextPt.y - zScale * PERSPECTIVE_FACTOR))
        }

        val meanParallax = if (validCount == 0) 0.0 else parallaxSum / validCount
        val meshEdges = mutableListOf<Pair<Point, Point>>()

        if (isMeshEnabled && cloudPoints.size >= 3) {
            try {
                var minX = Double.MAX_VALUE; var minY = Double.MAX_VALUE
                var maxX = Double.MIN_VALUE; var maxY = Double.MIN_VALUE
                for (p in cloudPoints) {
                    if (p.x < minX) minX = p.x; if (p.y < minY) minY = p.y
                    if (p.x > maxX) maxX = p.x; if (p.y > maxY) maxY = p.y
                }
                val rect = Rect((minX - 1).toInt(), (minY - 1).toInt(), (maxX - minX + 2).toInt(), (maxY - minY + 2).toInt())
                val subdiv = Subdiv2D(rect)
                for (p in cloudPoints) subdiv.insert(p)
                
                val edgeList = MatOfFloat()
                subdiv.getEdgeList(edgeList)
                val edges = edgeList.toArray()
                for (i in 0 until edges.size step 4) {
                    val p1 = Point(edges[i].toDouble(), edges[i+1].toDouble())
                    val p2 = Point(edges[i+2].toDouble(), edges[i+3].toDouble())
                    if (rect.contains(p1) && rect.contains(p2)) {
                        val d2 = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y)
                        if (d2 < MAX_MESH_EDGE_DIST_SQ) meshEdges.add(Pair(p1, p2))
                    }
                }
                edgeList.release()
            } catch (e: Exception) {}
        }

        previousGray?.release()
        previousGray = gray.clone()
        previousFeatures?.release()
        previousFeatures = if (cloudPoints.size >= MIN_FEATURES_TO_KEEP) {
            val tracked = nextArr.filterIndexed { i, _ -> statusArr[i].toInt() != 0 }.toTypedArray()
            MatOfPoint2f(*tracked)
        } else detectFeatures(gray)

        gray.release()
        nextFeatures.release()
        status.release()
        err.release()

        return PointCloudState(cloudPoints.take(MAX_CLOUD_POINTS), meanParallax, meshEdges)
    }
        
    private fun detectFeatures(gray: Mat): MatOfPoint2f {
        val cornersInt = MatOfPoint()
        Imgproc.goodFeaturesToTrack(gray, cornersInt, maxFeatures, QUALITY_LEVEL, MIN_DISTANCE, Mat(), 3, false, 0.04)
        val corners = MatOfPoint2f()
        cornersInt.convertTo(corners, CvType.CV_32F)
        cornersInt.release()
        return corners
    }

    private fun collectTrackedPairs(previous: MatOfPoint2f, current: MatOfPoint2f, status: MatOfByte): Pair<MatOfPoint2f, MatOfPoint2f> {
        val pArr = previous.toArray(); val cArr = current.toArray(); val sArr = status.toArray()
        val pList = ArrayList<Point>(); val cList = ArrayList<Point>()
        for (i in 0 until min(min(pArr.size, cArr.size), sArr.size)) {
            if (sArr[i].toInt() != 0) { pList.add(pArr[i]); cList.add(cArr[i]) }
        }
        return Pair(MatOfPoint2f(*pList.toTypedArray()), MatOfPoint2f(*cList.toTypedArray()))
    }

    private fun estimateRelativePose(previous: MatOfPoint2f, current: MatOfPoint2f, w: Double, h: Double): OdometryState? {
        val focal = maxOf(w, h); val pp = Point(w / 2.0, h / 2.0)
        val k = Mat.eye(3, 3, CvType.CV_64F)
        k.put(0, 0, focal); k.put(1, 1, focal); k.put(0, 2, pp.x); k.put(1, 2, pp.y)
        val mask = Mat()
        val e = Calib3d.findEssentialMat(previous, current, k, Calib3d.RANSAC, RANSAC_CONFIDENCE, RANSAC_THRESHOLD, RANSAC_MAX_ITERS, mask)
        k.release()
        if (e.empty()) { e.release(); mask.release(); return null }
        val r = Mat(); val t = Mat(); val recMask = Mat()
        Calib3d.recoverPose(e, previous, current, r, t, focal, pp, recMask)
        val inliers = countNonZeroSafe(recMask)
        val tNorm = kotlin.math.sqrt(t.get(0,0)[0]*t.get(0,0)[0] + t.get(1,0)[0]*t.get(1,0)[0] + t.get(2,0)[0]*t.get(2,0)[0])
        val trace = r.get(0,0)[0] + r.get(1,1)[0] + r.get(2,2)[0]
        val rDeg = Math.toDegrees(kotlin.math.acos(((trace - 1.0) / 2.0).coerceIn(-1.0, 1.0)))
        e.release(); mask.release(); recMask.release(); r.release(); t.release()
        return OdometryState(previous.rows(), inliers, tNorm, rDeg)
    }

    private fun countNonZeroSafe(mask: Mat): Int {
        if (mask.empty()) return 0
        var c = 0
        for (r in 0 until mask.rows()) { if (abs(mask.get(r, 0)[0]) > 0.0) c++ }
        return c
    }

    private companion object {
        private const val MAX_FEATURES_DEFAULT = 300
        private const val MIN_PARALLAX_DEFAULT = 1.0
        private const val QUALITY_LEVEL = 0.01
        private const val MIN_DISTANCE = 8.0
        private const val MIN_TRACKS_FOR_POSE = 8
        private const val MIN_FEATURES_TO_KEEP = 60
        private const val RANSAC_CONFIDENCE = 0.999
        private const val RANSAC_THRESHOLD = 1.0
        private const val RANSAC_MAX_ITERS = 1000
        private const val EPSILON = 1e-6
        private const val DEPTH_VISUAL_GAIN = 100.0
        private const val PERSPECTIVE_FACTOR = 40.0
        private const val MAX_CLOUD_POINTS = 300
        private const val MAX_MESH_EDGE_DIST_SQ = 150.0 * 150.0
    }
}
