package pl.edu.mobilecv

import kotlin.math.abs
import kotlin.math.min
import org.opencv.calib3d.Calib3d
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video

/**
 * Tracks sparse visual odometry signals and creates a lightweight pseudo point cloud.
 *
 * This class stores the previous grayscale frame and tracked feature points.
 * It estimates relative camera motion from optical flow and decomposes it via
 * essential-matrix recovery.
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
    )

    private var previousGray: Mat? = null
    private var previousFeatures: MatOfPoint2f? = null

    /**
     * Reset all internal odometry buffers.
     */
    fun reset() {
        previousFeatures?.release()
        previousFeatures = null

        previousGray?.release()
        previousGray = null
    }

    /**
     * Update odometry with a single RGBA frame.
     *
     * @param rgba Input RGBA frame.
     * @return [OdometryState] when enough tracks are available, otherwise `null`.
     */
    fun updateOdometry(rgba: Mat): OdometryState? {
        val gray = Mat()
        Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)

        if (previousGray == null) {
            previousGray = gray.clone()
            previousFeatures = detectFeatures(gray)
            gray.release()
            return null
        }

        val prevGrayLocal = previousGray ?: run {
            gray.release()
            return null
        }
        val prevFeaturesLocal = previousFeatures ?: run {
            gray.release()
            return null
        }

        val nextFeatures = MatOfPoint2f()
        val status = MatOfByte()
        val err = MatOfFloat()
        Video.calcOpticalFlowPyrLK(
            prevGrayLocal,
            gray,
            prevFeaturesLocal,
            nextFeatures,
            status,
            err,
        )

        val trackedPairs = collectTrackedPairs(prevFeaturesLocal, nextFeatures, status)
        val prevTracked = trackedPairs.first
        val nextTracked = trackedPairs.second

        status.release()
        err.release()

        val state = if (prevTracked.rows() >= MIN_TRACKS_FOR_POSE) {
            estimateRelativePose(prevTracked, nextTracked, gray.cols().toDouble(), gray.rows().toDouble())
        } else {
            null
        }

        previousGray?.release()
        previousGray = gray.clone()

        previousFeatures?.release()
        previousFeatures = if (nextTracked.rows() >= MIN_FEATURES_TO_KEEP) {
            nextTracked
        } else {
            detectFeatures(gray)
        }

        gray.release()
        nextFeatures.release()
        if (previousFeatures !== nextTracked) {
            nextTracked.release()
        }
        prevTracked.release()

        return state
    }

    /**
     * Build a pseudo point cloud from tracked image-space points.
     *
     * The cloud is represented as 2D points with depth encoded in the y-offset.
     */
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

        val prevGrayLocal = previousGray ?: run {
            gray.release()
            return null
        }
        val prevFeaturesLocal = previousFeatures ?: run {
            gray.release()
            return null
        }

        val nextFeatures = MatOfPoint2f()
        val status = MatOfByte()
        val err = MatOfFloat()
        Video.calcOpticalFlowPyrLK(
            prevGrayLocal,
            gray,
            prevFeaturesLocal,
            nextFeatures,
            status,
            err,
        )

        val prevArr = prevFeaturesLocal.toArray()
        val nextArr = nextFeatures.toArray()
        val statusArr = status.toArray()

        val cloudPoints = ArrayList<Point>()
        var parallaxSum = 0.0

        for (i in statusArr.indices) {
            if (statusArr[i].toInt() == 0 || i >= prevArr.size || i >= nextArr.size) {
                continue
            }
            val prevPt = prevArr[i]
            val nextPt = nextArr[i]
            val dx = nextPt.x - prevPt.x
            val dy = nextPt.y - prevPt.y
            val parallax = kotlin.math.sqrt(dx * dx + dy * dy)
            parallaxSum += parallax

            val pseudoDepth = 1.0 / (parallax + EPSILON)
            val projectedY = nextPt.y - pseudoDepth * DEPTH_VISUAL_GAIN
            cloudPoints.add(Point(nextPt.x, projectedY))
        }

        val meanParallax = if (cloudPoints.isEmpty()) 0.0 else parallaxSum / cloudPoints.size

        previousGray?.release()
        previousGray = gray.clone()

        previousFeatures?.release()
        previousFeatures = if (cloudPoints.size >= MIN_FEATURES_TO_KEEP) {
            MatOfPoint2f(*nextArr)
        } else {
            detectFeatures(gray)
        }

        gray.release()
        nextFeatures.release()
        status.release()
        err.release()

        return PointCloudState(cloudPoints.take(MAX_CLOUD_POINTS), meanParallax)
    }

    private fun detectFeatures(gray: Mat): MatOfPoint2f {
        val corners = MatOfPoint2f()
        Imgproc.goodFeaturesToTrack(
            gray,
            corners,
            MAX_FEATURES,
            QUALITY_LEVEL,
            MIN_DISTANCE,
        )
        return corners
    }

    private fun collectTrackedPairs(
        previous: MatOfPoint2f,
        current: MatOfPoint2f,
        status: MatOfByte,
    ): Pair<MatOfPoint2f, MatOfPoint2f> {
        val previousPoints = previous.toArray()
        val currentPoints = current.toArray()
        val statusArray = status.toArray()

        val previousTracked = ArrayList<Point>()
        val currentTracked = ArrayList<Point>()

        val trackCount = min(min(previousPoints.size, currentPoints.size), statusArray.size)
        for (i in 0 until trackCount) {
            if (statusArray[i].toInt() != 0) {
                previousTracked.add(previousPoints[i])
                currentTracked.add(currentPoints[i])
            }
        }

        return Pair(
            MatOfPoint2f(*previousTracked.toTypedArray()),
            MatOfPoint2f(*currentTracked.toTypedArray()),
        )
    }

    private fun estimateRelativePose(
        previous: MatOfPoint2f,
        current: MatOfPoint2f,
        width: Double,
        height: Double,
    ): OdometryState? {
        val focal = maxOf(width, height)
        val principalPoint = Point(width / 2.0, height / 2.0)

        val essentialMask = Mat()
        val essential = Calib3d.findEssentialMat(
            previous,
            current,
            focal,
            principalPoint,
            Calib3d.RANSAC,
            RANSAC_CONFIDENCE,
            RANSAC_THRESHOLD,
            essentialMask,
        )

        if (essential.empty()) {
            essential.release()
            essentialMask.release()
            return null
        }

        val rotation = Mat()
        val translation = Mat()
        val recoverMask = Mat()
        Calib3d.recoverPose(
            essential,
            previous,
            current,
            rotation,
            translation,
            focal,
            principalPoint,
            recoverMask,
        )

        val inliers = CoreExt.countNonZeroSafe(recoverMask)
        val translationNorm = vectorNorm3(translation)
        val rotationDeg = rotationMagnitudeDegrees(rotation)

        essential.release()
        essentialMask.release()
        recoverMask.release()
        rotation.release()
        translation.release()

        return OdometryState(
            tracksCount = previous.rows(),
            inliersCount = inliers,
            translationNorm = translationNorm,
            rotationDeg = rotationDeg,
        )
    }

    private fun vectorNorm3(translation: Mat): Double {
        if (translation.rows() < 3 || translation.cols() != 1 || translation.type() != CvType.CV_64F) {
            return 0.0
        }
        val tx = translation.get(0, 0)?.firstOrNull() ?: 0.0
        val ty = translation.get(1, 0)?.firstOrNull() ?: 0.0
        val tz = translation.get(2, 0)?.firstOrNull() ?: 0.0
        return kotlin.math.sqrt(tx * tx + ty * ty + tz * tz)
    }

    private fun rotationMagnitudeDegrees(rotation: Mat): Double {
        if (rotation.rows() != 3 || rotation.cols() != 3) {
            return 0.0
        }
        val trace = (rotation.get(0, 0)?.firstOrNull() ?: 1.0) +
            (rotation.get(1, 1)?.firstOrNull() ?: 1.0) +
            (rotation.get(2, 2)?.firstOrNull() ?: 1.0)

        val cosTheta = ((trace - 1.0) / 2.0).coerceIn(-1.0, 1.0)
        return Math.toDegrees(kotlin.math.acos(cosTheta))
    }

    private object CoreExt {
        fun countNonZeroSafe(mask: Mat): Int {
            if (mask.empty()) {
                return 0
            }
            var count = 0
            val rows = mask.rows()
            for (r in 0 until rows) {
                val value = mask.get(r, 0)?.firstOrNull() ?: 0.0
                if (abs(value) > 0.0) {
                    count += 1
                }
            }
            return count
        }
    }

    private companion object {
        private const val MAX_FEATURES = 300
        private const val QUALITY_LEVEL = 0.01
        private const val MIN_DISTANCE = 8.0
        private const val MIN_TRACKS_FOR_POSE = 8
        private const val MIN_FEATURES_TO_KEEP = 60
        private const val RANSAC_CONFIDENCE = 0.999
        private const val RANSAC_THRESHOLD = 1.0
        private const val EPSILON = 1e-6
        private const val DEPTH_VISUAL_GAIN = 180.0
        private const val MAX_CLOUD_POINTS = 160
    }
}
