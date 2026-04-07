package pl.edu.mobilecv

import android.util.Log
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.*

/**
 * Handles complex geometric computer vision tasks like plane detection and vanishing points.
 */
data class LineSegment(
    val p1: Point,
    val p2: Point,
    val length: Double,
    val angleRad: Double
)

data class PlaneData(
    val c1: Cluster,
    val c2: Cluster,
    val index: Int
)

data class Cluster(
    val lines: List<LineSegment>,
    val totalWeight: Double,
    val meanAngleDeg: Double
)

class GeometryProcessor(private val exceptionLogger: (String, String, Throwable) -> Unit) {

    companion object {
        private const val TAG = "GeometryProcessor"
        private const val PLANE_ANGLE_DIFF_MIN = 25.0
        private const val CLUSTER_ANGLE_THRESHOLD = 8.0
        private const val MIN_PLANE_LINES = 4
        private const val OVERLAY_ALPHA = 0.18
    }

    @Volatile
    var maxPlanes: Int = 3

    private val claheObj = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
    private val dilationKernel3x3 by lazy {
        Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
    }

    fun detectPlanes(src: Mat, res: Mat, labels: Map<String, String>): Int {
        val gray = Mat()
        val clahe = Mat()
        val blurred = Mat()
        val edges = Mat()
        val lines = Mat()
        var planeIdx = 0

        try {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
            claheObj.apply(gray, clahe)
            Imgproc.GaussianBlur(clahe, blurred, Size(5.0, 5.0), 0.0)

            val medianVal = medianIntensity(blurred)
            Imgproc.Canny(blurred, edges, maxOf(0.0, 0.67 * medianVal), minOf(255.0, 1.33 * medianVal))
            Imgproc.dilate(edges, edges, dilationKernel3x3)

            Imgproc.HoughLinesP(edges, lines, 1.0, Math.PI / 180.0, 45, 40.0, 15.0)

            val rawClusters = clusterLinesRaw(lines)
            val clusters = rawClusters.map { segs ->
                val totalWeight = segs.sumOf { it.length }
                val meanAngle = weightedAngleMean(segs)
                Cluster(segs, totalWeight, meanAngle)
            }

            val sortedClusters = clusters
                .filter { it.lines.size >= 2 }
                .sortedByDescending { it.totalWeight }
                .take(maxPlanes * 2)

            val usedClusters = mutableSetOf<Int>()
            val detectedPlanes = mutableListOf<PlaneData>()

            for (i in sortedClusters.indices) {
                if (planeIdx >= maxPlanes) break
                if (i in usedClusters) continue

                for (j in i + 1 until sortedClusters.size) {
                    if (j in usedClusters) continue

                    val c1 = sortedClusters[i]
                    val c2 = sortedClusters[j]

                    var angleDiff = abs(c1.meanAngleDeg - c2.meanAngleDeg)
                    angleDiff = minOf(angleDiff, 180.0 - angleDiff)

                    if (angleDiff >= PLANE_ANGLE_DIFF_MIN && areClustersSpatiallyRelated(c1.lines, c2.lines)) {
                        if (c1.lines.size + c2.lines.size >= MIN_PLANE_LINES) {
                            detectedPlanes.add(PlaneData(c1, c2, planeIdx))
                            usedClusters.add(i)
                            usedClusters.add(j)
                            planeIdx++
                            break
                        }
                    }
                }
            }

            drawAllPlanes(res, detectedPlanes, lines.rows())

        } catch (e: Exception) {
            exceptionLogger("plane_detection", "error", e)
        } finally {
            gray.release(); clahe.release(); blurred.release(); edges.release(); lines.release()
        }
        return planeIdx
    }

    fun detectVanishingPoints(src: Mat, res: Mat, labels: Map<String, String>) {
        val gray = Mat()
        val blurred = Mat()
        val edges = Mat()
        val lines = Mat()
        try {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
            Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
            Imgproc.Canny(blurred, edges, 50.0, 150.0)

            Imgproc.HoughLinesP(edges, lines, 1.0, Math.PI / 180.0, 40, 20.0, 8.0)

            val clusters = clusterLinesRaw(lines)
            val vpColors = arrayOf(
                Scalar(0.0, 255.0, 0.0),
                Scalar(0.0, 0.0, 255.0),
                Scalar(0.0, 165.0, 255.0),
                Scalar(255.0, 255.0, 0.0)
            )
            var foundVP = false

            val sortedClusters = clusters.sortedByDescending { it.size }.take(4)

            for (i in sortedClusters.indices) {
                val cluster = sortedClusters[i]
                if (cluster.size < 2) continue
                val color = vpColors[i % vpColors.size]
                for (seg in cluster) {
                    Imgproc.line(res, seg.p1, seg.p2, color, 1)
                }
                val vp = computeVanishingPoint(cluster)
                if (vp != null) {
                    Imgproc.circle(res, vp, 10, color, -1)
                    Imgproc.putText(res, "VP${i + 1}", Point(vp.x + 12, vp.y + 5), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    foundVP = true
                }
            }

            val labelNoLines = labels["noLines"] ?: "No lines"
            val labelNoVanishingPoints = labels["noVp"] ?: "No vanishing points"
            val labelLines = labels["lines"] ?: "Lines"
            val labelGroups = labels["groups"] ?: "Groups"

            when {
                lines.rows() == 0 -> Imgproc.putText(res, labelNoLines, Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, Scalar(200.0, 200.0, 200.0), 2)
                !foundVP -> Imgproc.putText(res, "$labelNoVanishingPoints ($labelLines: ${lines.rows()})", Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, Scalar(200.0, 200.0, 200.0), 2)
                else -> Imgproc.putText(res, "$labelLines: ${lines.rows()} | $labelGroups: ${clusters.size}", Point(30.0, 30.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255.0, 255.0, 255.0), 2)
            }
        } catch (e: Exception) {
            exceptionLogger("vanishing_points", "error", e)
            val labelVpError = labels["vpError"] ?: "VP Error"
            Imgproc.putText(res, "$labelVpError: ${e.message?.take(30)}", Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255.0, 100.0, 100.0), 2)
        } finally {
            gray.release(); blurred.release(); edges.release(); lines.release()
        }
    }

    private fun drawAllPlanes(res: Mat, planes: List<PlaneData>, totalLines: Int) {
        if (planes.isEmpty()) return

        val overlay = res.clone()
        for (plane in planes) {
            val color = getPlaneColor(plane.index)
            val allPoints = (plane.c1.lines + plane.c2.lines).flatMap { listOf(it.p1, it.p2) }

            for (line in plane.c1.lines + plane.c2.lines) {
                Imgproc.line(res, line.p1, line.p2, color, 1)
            }

            drawHullOnMat(overlay, allPoints, color)
            
            val cx = allPoints.map { it.x }.average()
            val cy = allPoints.map { it.y }.average()
            val planeLineCount = plane.c1.lines.size + plane.c2.lines.size
            val confidence = if (totalLines > 0) (planeLineCount * 100 / totalLines).coerceAtMost(100) else 0
            Imgproc.putText(res, "P${plane.index + 1} ($confidence%)", Point(cx + 8, cy), Imgproc.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        }

        Core.addWeighted(res, 1.0 - OVERLAY_ALPHA, overlay, OVERLAY_ALPHA, 0.0, res)
        overlay.release()
    }

    private fun drawHullOnMat(dst: Mat, points: List<Point>, color: Scalar) {
        if (points.size < 3) return
        try {
            val contourMat = MatOfPoint().apply { fromArray(*points.toTypedArray()) }
            val hullIdx = MatOfInt()
            Imgproc.convexHull(contourMat, hullIdx)
            val hullPts = hullIdx.toArray().map { i -> points[i] }
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

    private fun getPlaneColor(index: Int): Scalar {
        val colors = arrayOf(
            Scalar(0.0, 255.0, 0.0),
            Scalar(255.0, 150.0, 0.0),
            Scalar(0.0, 180.0, 255.0),
            Scalar(255.0, 0.0, 255.0),
            Scalar(0.0, 255.0, 255.0)
        )
        return colors[index % colors.size]
    }

    private fun areClustersSpatiallyRelated(c1: List<LineSegment>, c2: List<LineSegment>): Boolean {
        val r1 = getClusterBoundingBox(c1)
        val r2 = getClusterBoundingBox(c2)
        val padding = 50.0
        val inflatedR1 = Rect((r1.x - padding).toInt(), (r1.y - padding).toInt(), (r1.width + 2 * padding).toInt(), (r1.height + 2 * padding).toInt())
        return inflatedR1.x < r2.x + r2.width && inflatedR1.x + inflatedR1.width > r2.x && inflatedR1.y < r2.y + r2.height && inflatedR1.y + inflatedR1.height > r2.y
    }

    private fun getClusterBoundingBox(cluster: List<LineSegment>): Rect {
        var minX = Double.MAX_VALUE; var minY = Double.MAX_VALUE
        var maxX = Double.MIN_VALUE; var maxY = Double.MIN_VALUE
        for (seg in cluster) {
            minX = minOf(minX, seg.p1.x, seg.p2.x); maxX = maxOf(maxX, seg.p1.x, seg.p2.x)
            minY = minOf(minY, seg.p1.y, seg.p2.y); maxY = maxOf(maxY, seg.p1.y, seg.p2.y)
        }
        return Rect(minX.toInt(), minY.toInt(), (maxX - minX).toInt(), (maxY - minY).toInt())
    }

    private fun clusterLinesRaw(lines: Mat): List<List<LineSegment>> {
        val clusters = mutableListOf<MutableList<LineSegment>>()
        val clusterAngles = mutableListOf<Double>()
        for (i in 0 until lines.rows()) {
            val vec = lines.get(i, 0) ?: continue
            val x1 = vec[0]; val y1 = vec[1]; val x2 = vec[2]; val y2 = vec[3]
            val length = hypot(x2 - x1, y2 - y1)
            val angle = Math.toDegrees(atan2(y2 - y1, x2 - x1)).let { if (it < 0) it + 180.0 else it } % 180.0
            val seg = LineSegment(Point(x1, y1), Point(x2, y2), length, Math.toRadians(angle))
            
            var assigned = false
            for (k in clusterAngles.indices) {
                var diff = abs(angle - clusterAngles[k]); diff = minOf(diff, 180.0 - diff)
                if (diff <= CLUSTER_ANGLE_THRESHOLD) {
                    clusters[k].add(seg)
                    clusterAngles[k] = weightedAngleMean(clusters[k])
                    assigned = true; break
                }
            }

            if (!assigned) {
                clusters.add(mutableListOf(seg))
                clusterAngles.add(angle)
            }
        }
        return clusters
    }

    private fun medianIntensity(mat: Mat): Double {
        val hist = Mat()
        Imgproc.calcHist(listOf(mat), MatOfInt(0), Mat(), hist, MatOfInt(256), MatOfFloat(0f, 256f))
        val total = mat.rows().toLong() * mat.cols().toLong()
        var cumulative = 0.0
        for (i in 0 until 256) {
            cumulative += hist.get(i, 0)[0]
            if (cumulative >= total / 2.0) {
                hist.release(); return i.toDouble()
            }
        }
        hist.release(); return 128.0
    }

    private fun weightedAngleMean(cluster: List<LineSegment>): Double {
        var sumSin = 0.0; var sumCos = 0.0; var totalWeight = 0.0
        for (seg in cluster) {
            sumSin += seg.length * sin(2.0 * seg.angleRad)
            sumCos += seg.length * cos(2.0 * seg.angleRad)
            totalWeight += seg.length
        }
        return if (totalWeight == 0.0) 0.0 else Math.toDegrees(atan2(sumSin, sumCos) / 2.0).let { if (it < 0) it + 180.0 else it } % 180.0
    }

    private fun computeVanishingPoint(lines: List<LineSegment>): Point? {
        if (lines.size < 2) return null
        val aMat = Mat(lines.size, 2, CvType.CV_64F)
        val bMat = Mat(lines.size, 1, CvType.CV_64F)
        for (i in lines.indices) {
            val p1 = lines[i].p1; val p2 = lines[i].p2
            val a = p1.y - p2.y
            val b = p2.x - p1.x
            val c = a * p1.x + b * p1.y
            aMat.put(i, 0, a); aMat.put(i, 1, b)
            bMat.put(i, 0, c)
        }
        val solution = Mat()
        val solved = Core.solve(aMat, bMat, solution, Core.DECOMP_SVD)
        val res = if (solved && solution.rows() >= 2) Point(solution.get(0, 0)[0], solution.get(1, 0)[0]) else null
        aMat.release(); bMat.release(); solution.release()
        return res
    }
}
