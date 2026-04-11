package pl.edu.mobilecv

import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * Odpowiada za ekstrakcję linii z obrazu oraz segmentację kierunków.
 */
class LineExtractor {
    companion object {
        private const val CLUSTER_ANGLE_THRESHOLD_DEG = 8.0
        private const val CLUSTER_MAX_GAP_PX = 150.0
    }

    private val claheObj = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
    private val dilationKernel3x3 by lazy {
        Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
    }

    /**
     * Wykonuje pełny pipeline ekstrakcji linii i segmentacji kierunkowej.
     */
    fun extract(src: Mat): LineExtractionResult {
        val gray = Mat()
        val clahe = Mat()
        val blurred = Mat()
        val edges = Mat()
        val linesMat = Mat()

        return try {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
            claheObj.apply(gray, clahe)
            Imgproc.bilateralFilter(clahe, blurred, 9, 75.0, 75.0)

            val medianVal = medianIntensity(blurred)
            Imgproc.Canny(blurred, edges, max(0.0, 0.5 * medianVal), min(255.0, 1.2 * medianVal))
            Imgproc.dilate(edges, edges, dilationKernel3x3)

            val houghThreshold = (src.cols() * 0.08).toInt().coerceAtLeast(35)
            Imgproc.HoughLinesP(edges, linesMat, 1.0, Math.PI / 180.0, houghThreshold, 50.0, 20.0)

            val segments = decodeLineSegments(linesMat)
            val clusters = clusterLinesByDirection(segments)
            LineExtractionResult(segments, clusters)
        } finally {
            gray.release(); clahe.release(); blurred.release(); edges.release(); linesMat.release()
        }
    }

    private fun decodeLineSegments(lines: Mat): List<LineSegment> {
        val allSegments = mutableListOf<LineSegment>()
        for (i in 0 until lines.rows()) {
            val vec = lines.get(i, 0) ?: continue
            val x1 = vec[0]
            val y1 = vec[1]
            val x2 = vec[2]
            val y2 = vec[3]
            val length = hypot(x2 - x1, y2 - y1)
            val angle = Math.toDegrees(atan2(y2 - y1, x2 - x1))
                .let { if (it < 0) it + 180.0 else it } % 180.0
            allSegments.add(LineSegment(Point(x1, y1), Point(x2, y2), length, Math.toRadians(angle)))
        }
        return allSegments
    }

    private fun clusterLinesByDirection(segments: List<LineSegment>): List<DirectionCluster> {
        if (segments.isEmpty()) return emptyList()

        val clusters = mutableListOf<MutableList<LineSegment>>()
        val clusterStates = mutableListOf<Pair<Double, Rect>>()

        for (seg in segments) {
            var assigned = false
            for (k in clusters.indices) {
                val (clusterAngle, clusterRect) = clusterStates[k]
                var diff = abs(Math.toDegrees(seg.angleRad) - clusterAngle)
                diff = min(diff, 180.0 - diff)

                if (diff <= CLUSTER_ANGLE_THRESHOLD_DEG) {
                    val dist = min(distanceToRect(seg.p1, clusterRect), distanceToRect(seg.p2, clusterRect))
                    if (dist <= CLUSTER_MAX_GAP_PX) {
                        clusters[k].add(seg)
                        clusterStates[k] = weightedAngleMean(clusters[k]) to getClusterBoundingBox(clusters[k])
                        assigned = true
                        break
                    }
                }
            }
            if (!assigned) {
                clusters.add(mutableListOf(seg))
                clusterStates.add(Math.toDegrees(seg.angleRad) to getClusterBoundingBox(listOf(seg)))
            }
        }

        return clusters
            .map { cluster ->
                DirectionCluster(
                    lines = cluster,
                    totalWeight = cluster.sumOf { it.length },
                    meanAngleDeg = weightedAngleMean(cluster),
                    boundingBox = getClusterBoundingBox(cluster),
                )
            }
            .filter { it.lines.size >= 2 && it.totalWeight > 100.0 }
            .sortedByDescending { it.totalWeight }
    }

    private fun getClusterBoundingBox(cluster: List<LineSegment>): Rect {
        var minX = Double.MAX_VALUE
        var minY = Double.MAX_VALUE
        var maxX = Double.MIN_VALUE
        var maxY = Double.MIN_VALUE

        for (seg in cluster) {
            minX = min(minX, min(seg.p1.x, seg.p2.x))
            minY = min(minY, min(seg.p1.y, seg.p2.y))
            maxX = max(maxX, max(seg.p1.x, seg.p2.x))
            maxY = max(maxY, max(seg.p1.y, seg.p2.y))
        }

        return Rect(minX.toInt(), minY.toInt(), (maxX - minX).toInt(), (maxY - minY).toInt())
    }

    private fun distanceToRect(point: Point, rect: Rect): Double {
        val dx = max(0.0, max(rect.x.toDouble() - point.x, point.x - (rect.x + rect.width).toDouble()))
        val dy = max(0.0, max(rect.y.toDouble() - point.y, point.y - (rect.y + rect.height).toDouble()))
        return hypot(dx, dy)
    }

    private fun medianIntensity(mat: Mat): Double {
        val hist = Mat()
        Imgproc.calcHist(listOf(mat), org.opencv.core.MatOfInt(0), Mat(), hist, org.opencv.core.MatOfInt(256), org.opencv.core.MatOfFloat(0f, 256f))
        val total = mat.rows().toLong() * mat.cols().toLong()
        var cumulative = 0.0
        for (i in 0 until 256) {
            cumulative += hist.get(i, 0)[0]
            if (cumulative >= total / 2.0) {
                hist.release()
                return i.toDouble()
            }
        }
        hist.release()
        return 128.0
    }

    private fun weightedAngleMean(cluster: List<LineSegment>): Double {
        var sumSin = 0.0
        var sumCos = 0.0
        var totalWeight = 0.0

        for (seg in cluster) {
            sumSin += seg.length * kotlin.math.sin(2.0 * seg.angleRad)
            sumCos += seg.length * kotlin.math.cos(2.0 * seg.angleRad)
            totalWeight += seg.length
        }

        if (totalWeight == 0.0) return 0.0
        val angle = Math.toDegrees(kotlin.math.atan2(sumSin, sumCos) / 2.0)
        return (if (angle < 0) angle + 180.0 else angle) % 180.0
    }
}
