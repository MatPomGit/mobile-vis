package pl.edu.mobilecv

import kotlin.math.min
import org.opencv.calib3d.Calib3d
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfFloat4
import org.opencv.core.MatOfPoint2f
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.imgproc.Imgproc
import org.opencv.imgproc.Subdiv2D
import org.opencv.video.Video
import kotlin.math.acos
import kotlin.math.sqrt

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
        val edges: List<Pair<Point, Point>>,
        val meanParallax: Double
    )

    companion object {
        private const val MAX_CORNERS = 100
        private const val MIN_TRACK_COUNT = 10
        private const val PERSPECTIVE_FACTOR = 0.5
        private const val MAX_MESH_EDGE_DIST_SQ = 50000.0
    }

    private var prevGray = Mat()
    private var prevPts = MatOfPoint2f()
    private var calibrator: CameraCalibrator? = null

    var maxFeatures = 50
    var minParallax = 2.0
    var isMeshEnabled = false

    fun updateOdometry(src: Mat): OdometryState? {
        val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        val (state, _) = processFrameInternal(gray)
        gray.release()
        return state
    }

    fun updatePointCloud(src: Mat): PointCloudState? {
        val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        val (_, cloud) = processFrameInternal(gray)
        gray.release()
        return cloud
    }

    fun processFrame(gray: Mat, calib: CameraCalibrator?): Pair<OdometryState?, PointCloudState?> {
        this.calibrator = calib
        return processFrameInternal(gray)
    }

    private fun processFrameInternal(gray: Mat): Pair<OdometryState?, PointCloudState?> {
        if (prevGray.empty()) {
            gray.copyTo(prevGray)
            detectNewFeatures(gray)
            return null to null
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

        for (i in statusArr.indices) {
            if (statusArr[i].toInt() == 1) {
                goodPrevList.add(prevPtsArr[i])
                goodNextList.add(nextPtsArr[i])
            }
        }

        status.release()
        err.release()

        if (goodNextList.size < MIN_TRACK_COUNT) {
            gray.copyTo(prevGray)
            detectNewFeatures(gray)
            nextPts.release()
            return null to null
        }

        val goodPrev = MatOfPoint2f(*goodPrevList.toTypedArray())
        val goodNext = MatOfPoint2f(*goodNextList.toTypedArray())

        val (state, points) = estimateMotionAndPoints(goodPrev, goodNext)

        gray.copyTo(prevGray)
        goodNext.copyTo(prevPts)

        if (goodNextList.size < maxFeatures / 2) {
            detectNewFeatures(gray)
        }

        nextPts.release()
        goodPrev.release()
        goodNext.release()

        return state to points
    }

    private fun detectNewFeatures(gray: Mat) {
        val corners = MatOfPoint()
        Imgproc.goodFeaturesToTrack(gray, corners, maxFeatures, 0.01, 10.0)
        if (!corners.empty()) {
            val corners2f = MatOfPoint2f(*corners.toArray())
            prevPts.release()
            prevPts = corners2f
        }
        corners.release()
    }

    private fun estimateMotionAndPoints(prev: MatOfPoint2f, next: MatOfPoint2f): Pair<OdometryState, PointCloudState> {
        val k = calibrator?.calibrationResult?.cameraMatrix ?: Mat.eye(3, 3, CvType.CV_64F)
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

        val transNorm = android.opengl.Matrix.length(t.get(0,0)[0].toFloat(), t.get(1,0)[0].toFloat(), t.get(2,0)[0].toFloat()).toDouble()
        
        // Simple rotation angle from matrix trace
        val trace = r.get(0,0)[0] + r.get(1,1)[0] + r.get(2,2)[0]
        val rotDeg = Math.toDegrees(acos(min(1.0, maxOf(-1.0, (trace - 1.0) / 2.0))))

        val state = OdometryState(prev.rows(), inlierCount, transNorm, rotDeg)

        val cloudPoints = mutableListOf<Point>()
        val prevArr = prev.toArray()
        val nextArr = next.toArray()
        var parallaxSum = 0.0
        var validCount = 0

        for (i in nextArr.indices) {
            val prevPt = prevArr[i]
            val nextPt = nextArr[i]
            val dx = nextPt.x - prevPt.x
            val dy = nextPt.y - prevPt.y
            val dist = sqrt(dx * dx + dy * dy)
            parallaxSum += dist
            validCount++

            val zScale = if (dist < minParallax) 0.0 else (dist - minParallax) / 10.0
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
                val subdivide = Subdiv2D(rect)
                for (p in cloudPoints) subdivide.insert(p)
                
                val edgeList = MatOfFloat4()
                subdivide.getEdgeList(edgeList)
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
            } catch (_: Exception) {}
        }

        essential.release()
        r.release()
        t.release()
        mask.release()

        return state to PointCloudState(cloudPoints, meshEdges, meanParallax)
    }

    fun reset() {
        prevGray.release()
        prevPts.release()
        prevGray = Mat()
        prevPts = MatOfPoint2f()
    }
}
