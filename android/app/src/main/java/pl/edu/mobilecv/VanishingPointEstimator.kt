package pl.edu.mobilecv

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Size
import kotlin.math.exp

/**
 * Estymuje punkty zbieżności na podstawie klastrów kierunków linii.
 */
class VanishingPointEstimator {
    companion object {
        private const val MAX_CLUSTER_COUNT = 6
    }

    /**
     * Zwraca listę punktów zbieżności wraz z confidence dla każdej grupy kierunkowej.
     */
    fun estimate(clusters: List<DirectionCluster>, frameSize: Size): List<VanishingPointCandidate> {
        if (clusters.isEmpty()) return emptyList()
        val maxWeight = clusters.maxOfOrNull { it.totalWeight } ?: 1.0

        return clusters
            .take(MAX_CLUSTER_COUNT)
            .mapNotNull { cluster ->
                val point = solveLeastSquaresIntersection(cluster.lines) ?: return@mapNotNull null
                val confidence = computeConfidence(cluster, point, frameSize, maxWeight)
                VanishingPointCandidate(point, cluster, confidence)
            }
            .sortedByDescending { it.confidence }
    }

    private fun computeConfidence(
        cluster: DirectionCluster,
        point: Point,
        frameSize: Size,
        maxWeight: Double,
    ): Double {
        val normalizedWeight = (cluster.totalWeight / maxWeight).coerceIn(0.0, 1.0)
        val clusterSizeScore = (cluster.lines.size / 12.0).coerceIn(0.0, 1.0)

        val centerX = frameSize.width / 2.0
        val centerY = frameSize.height / 2.0
        val distance = kotlin.math.hypot(point.x - centerX, point.y - centerY)
        val distanceScore = exp(-distance / (frameSize.width + frameSize.height).coerceAtLeast(1.0))

        return (0.5 * normalizedWeight + 0.3 * clusterSizeScore + 0.2 * distanceScore)
            .coerceIn(0.0, 1.0)
    }

    private fun solveLeastSquaresIntersection(lines: List<LineSegment>): Point? {
        if (lines.size < 2) return null

        val aMat = Mat(lines.size, 2, CvType.CV_64F)
        val bMat = Mat(lines.size, 1, CvType.CV_64F)
        return try {
            for (i in lines.indices) {
                val p1 = lines[i].p1
                val p2 = lines[i].p2
                val a = p1.y - p2.y
                val b = p2.x - p1.x
                val c = a * p1.x + b * p1.y
                aMat.put(i, 0, a)
                aMat.put(i, 1, b)
                bMat.put(i, 0, c)
            }

            val solution = Mat()
            try {
                val solved = Core.solve(aMat, bMat, solution, Core.DECOMP_SVD)
                if (solved && solution.rows() >= 2) {
                    Point(solution.get(0, 0)[0], solution.get(1, 0)[0])
                } else {
                    null
                }
            } finally {
                solution.release()
            }
        } finally {
            aMat.release(); bMat.release()
        }
    }
}
