package pl.edu.mobilecv

import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.min

/**
 * Zarządza pipeline geometrii: linie -> VP -> hipotezy płaszczyzn -> tracking czasowy.
 */
class GeometryProcessor(private val exceptionLogger: (String, String, Throwable) -> Unit) {

    companion object {
        private const val OVERLAY_ALPHA = 0.18
        private const val PLANE_MIN_ANGLE_DIFF_DEG = 22.0
    }

    @Volatile
    var maxPlanes: Int = 3

    private val lineExtractor = LineExtractor()
    private val vpEstimator = VanishingPointEstimator()
    private val planeTracker = PlaneHypothesisTracker()

    /**
     * Wykrywa i śledzi płaszczyzny w nowym pipeline opartym o punkty zbieżności.
     */
    fun detectPlanes(src: Mat, res: Mat, labels: Map<String, String>): Int {
        return try {
            val extraction = lineExtractor.extract(src)
            val vanishingPoints = vpEstimator.estimate(extraction.clusters, src.size())
            val hypotheses = buildPlaneHypotheses(vanishingPoints, extraction.lines.size)
            val trackedPlanes = planeTracker.update(hypotheses, maxPlanes)

            drawTrackedPlanes(res, trackedPlanes)
            drawPlaneMetrics(res, extraction.lines.size, vanishingPoints.size, trackedPlanes, labels)
            trackedPlanes.size
        } catch (e: Exception) {
            exceptionLogger("plane_detection", "error", e)
            0
        }
    }

    /**
     * Wizualizuje punkty zbieżności korzystając z tej samej estymacji co detekcja płaszczyzn.
     */
    fun detectVanishingPoints(src: Mat, res: Mat, labels: Map<String, String>) {
        try {
            val extraction = lineExtractor.extract(src)
            val vanishingPoints = vpEstimator.estimate(extraction.clusters, src.size())
            val vpColors = arrayOf(
                Scalar(0.0, 255.0, 0.0),
                Scalar(0.0, 0.0, 255.0),
                Scalar(0.0, 165.0, 255.0),
                Scalar(255.0, 255.0, 0.0),
            )

            for ((index, vp) in vanishingPoints.take(4).withIndex()) {
                val color = vpColors[index % vpColors.size]
                for (segment in vp.cluster.lines) {
                    Imgproc.line(res, segment.p1, segment.p2, color, 1)
                }
                Imgproc.circle(res, vp.point, 10, color, -1)
                Imgproc.putText(
                    res,
                    "VP${index + 1} ${(vp.confidence * 100).toInt()}%",
                    Point(vp.point.x + 12, vp.point.y + 5),
                    Imgproc.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )
            }

            val labelNoLines = labels["noLines"] ?: "No lines"
            val labelNoVanishingPoints = labels["noVp"] ?: "No vanishing points"
            val labelLines = labels["lines"] ?: "Lines"
            val labelGroups = labels["groups"] ?: "Groups"

            when {
                extraction.lines.isEmpty() -> Imgproc.putText(res, labelNoLines, Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, Scalar(200.0, 200.0, 200.0), 2)
                vanishingPoints.isEmpty() -> Imgproc.putText(res, "$labelNoVanishingPoints ($labelLines: ${extraction.lines.size})", Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, Scalar(200.0, 200.0, 200.0), 2)
                else -> Imgproc.putText(res, "$labelLines: ${extraction.lines.size} | $labelGroups: ${extraction.clusters.size}", Point(30.0, 30.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255.0, 255.0, 255.0), 2)
            }
        } catch (e: Exception) {
            exceptionLogger("vanishing_points", "error", e)
            val labelVpError = labels["vpError"] ?: "VP Error"
            Imgproc.putText(res, "$labelVpError: ${e.message?.take(30)}", Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255.0, 100.0, 100.0), 2)
        }
    }

    /**
     * Buduje hipotezy płaszczyzn łącząc pary niezależnych punktów zbieżności.
     */
    private fun buildPlaneHypotheses(
        vanishingPoints: List<VanishingPointCandidate>,
        totalLineCount: Int,
    ): List<PlaneHypothesis> {
        if (vanishingPoints.size < 2 || totalLineCount <= 0) return emptyList()

        val hypotheses = mutableListOf<PlaneHypothesis>()

        for (i in vanishingPoints.indices) {
            for (j in i + 1 until vanishingPoints.size) {
                val first = vanishingPoints[i]
                val second = vanishingPoints[j]
                val angleDiff = angleDifference(first.cluster.meanAngleDeg, second.cluster.meanAngleDeg)
                if (angleDiff < PLANE_MIN_ANGLE_DIFF_DEG) continue

                val mergedLines = (first.cluster.lines + second.cluster.lines)
                if (mergedLines.size < 4) continue

                val centroid = computeCentroid(mergedLines)
                val normalAngle = computeNormalAngle(first.point, second.point)
                val supportRatio = mergedLines.size.toDouble() / totalLineCount
                val confidence = (
                    0.45 * supportRatio.coerceIn(0.0, 1.0) +
                        0.30 * first.confidence +
                        0.20 * second.confidence +
                        0.05 * (angleDiff / 90.0).coerceIn(0.0, 1.0)
                    ).coerceIn(0.0, 1.0)

                hypotheses.add(
                    PlaneHypothesis(
                        normalAngleDeg = normalAngle,
                        centroid = centroid,
                        lines = mergedLines,
                        supportLineCount = mergedLines.size,
                        confidence = confidence,
                        vanishingPair = first.point to second.point,
                    ),
                )
            }
        }

        return hypotheses.sortedByDescending { it.confidence }
    }

    private fun drawTrackedPlanes(res: Mat, planes: List<TrackedPlane>) {
        if (planes.isEmpty()) return

        val overlay = res.clone()
        for (plane in planes) {
            val color = getPlaneColor(plane.displayIndex)
            val allPoints = plane.lines.flatMap { listOf(it.p1, it.p2) }
            for (line in plane.lines) {
                Imgproc.line(res, line.p1, line.p2, color, 1)
            }
            drawHullOnMat(overlay, allPoints, color)
            Imgproc.putText(
                res,
                "P${plane.planeId}",
                Point(plane.centroid.x + 8, plane.centroid.y),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.58,
                color,
                2,
            )
        }

        Core.addWeighted(res, 1.0 - OVERLAY_ALPHA, overlay, OVERLAY_ALPHA, 0.0, res)
        overlay.release()
    }

    /**
     * Rysuje metryki jakości nowego modelu: linie, confidence, stability i jitter.
     */
    private fun drawPlaneMetrics(
        res: Mat,
        lineCount: Int,
        vpCount: Int,
        planes: List<TrackedPlane>,
        labels: Map<String, String>,
    ) {
        val labelLines = labels["lines"] ?: "Lines"
        val labelPlanes = labels["planes"] ?: "Planes"
        val labelConfidence = labels["confidence"] ?: "Confidence"
        val labelStability = labels["stability"] ?: "Stability"
        val labelJitter = labels["normalJitter"] ?: "Normal jitter"
        val labelNoPlanes = labels["noPlanes"] ?: "No planes"

        Imgproc.putText(
            res,
            "$labelLines: $lineCount | VP: $vpCount | $labelPlanes: ${planes.size}",
            Point(30.0, 30.0),
            Imgproc.FONT_HERSHEY_SIMPLEX,
            0.58,
            Scalar(255.0, 255.0, 255.0),
            2,
        )

        if (planes.isEmpty()) {
            Imgproc.putText(
                res,
                labelNoPlanes,
                Point(30.0, 55.0),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.64,
                Scalar(200.0, 200.0, 200.0),
                2,
            )
            return
        }

        var y = 58.0
        for (plane in planes) {
            Imgproc.putText(
                res,
                "P${plane.planeId} $labelConfidence ${(plane.confidence * 100).toInt()}% | $labelStability ${(plane.stability * 100).toInt()}% | $labelJitter ${"%.1f".format(plane.jitterDeg)}°",
                Point(30.0, y),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.50,
                getPlaneColor(plane.displayIndex),
                2,
            )
            y += 22.0
        }
    }

    private fun drawHullOnMat(dst: Mat, points: List<Point>, color: Scalar) {
        if (points.size < 3) return
        try {
            val contourMat = MatOfPoint().apply { fromArray(*points.toTypedArray()) }
            val hullIdx = MatOfInt()
            Imgproc.convexHull(contourMat, hullIdx)
            val hullPts = hullIdx.toArray().map { index -> points[index] }
            if (hullPts.size >= 3) {
                val hullMat = MatOfPoint().apply { fromArray(*hullPts.toTypedArray()) }
                Imgproc.fillConvexPoly(dst, hullMat, color)
                Imgproc.polylines(dst, listOf(hullMat), true, color, 2, Imgproc.LINE_AA)
                hullMat.release()
            }
            contourMat.release(); hullIdx.release()
        } catch (e: Exception) {
            exceptionLogger("plane_overlay", "opencv", e)
        }
    }

    private fun computeCentroid(lines: List<LineSegment>): Point {
        val points = lines.flatMap { listOf(it.p1, it.p2) }
        return Point(points.map { it.x }.average(), points.map { it.y }.average())
    }

    private fun computeNormalAngle(vp1: Point, vp2: Point): Double {
        val directionDeg = Math.toDegrees(atan2(vp2.y - vp1.y, vp2.x - vp1.x))
        val normal = (directionDeg + 90.0) % 180.0
        return if (normal < 0) normal + 180.0 else normal
    }

    private fun angleDifference(a: Double, b: Double): Double {
        val diff = abs(a - b)
        return min(diff, 180.0 - diff)
    }

    private fun getPlaneColor(index: Int): Scalar {
        val colors = arrayOf(
            Scalar(0.0, 255.0, 0.0),
            Scalar(255.0, 150.0, 0.0),
            Scalar(0.0, 180.0, 255.0),
            Scalar(255.0, 0.0, 255.0),
            Scalar(0.0, 255.0, 255.0),
        )
        return colors[index % colors.size]
    }
}
