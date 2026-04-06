package pl.edu.mobilecv

import android.util.Log
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.*

/**
 * Handles complex geometric computer vision tasks like plane detection and vanishing points.
 */
class GeometryProcessor(private val exceptionLogger: (String, String, Throwable) -> Unit) {

    companion object {
        private const val TAG = "GeometryProcessor"
        private const val CLUSTER_ANGLE_THRESHOLD_DEG = 8.0
    }

    @Volatile
    var maxPlanes: Int = 3

    private val dilationKernel3x3 by lazy {
        Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))
    }

    fun detectPlanes(
        src: Mat,
        res: Mat,
        labels: Map<String, String>
    ): Int {
        val gray = Mat(); val clahe = Mat(); val blurred = Mat(); val edges = Mat()
        var planeIdx = 0
        try {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
            val claheObj = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
            claheObj.apply(gray, clahe)
            Imgproc.GaussianBlur(clahe, blurred, Size(5.0, 5.0), 0.0)

            val medianVal = medianIntensity(blurred)
            Imgproc.Canny(blurred, edges, maxOf(0.0, 0.67 * medianVal), minOf(255.0, 1.33 * medianVal))
            Imgproc.dilate(edges, edges, dilationKernel3x3)

            val lines = Mat()
            // Higher thresholds for more robust line detection
            Imgproc.HoughLinesP(edges, lines, 1.0, Math.PI / 180.0, 45, 40.0, 15.0)

            val clusters = clusterLines(lines)
            val totalLines = lines.rows()
            val planeColors = arrayOf(
                Scalar(0.0, 255.0, 0.0),
                Scalar(255.0, 150.0, 0.0),
                Scalar(0.0, 180.0, 255.0),
                Scalar(255.0, 0.0, 255.0),
                Scalar(0.0, 255.0, 255.0)
            )
            
            // Filter and sort clusters by total weight (line length sum)
            val significantClusters = clusters.filter { it.size >= 2 }
            val sortedClusters = significantClusters.sortedByDescending { cluster ->
                cluster.sumOf { it[4] } 
            }.take(maxPlanes * 2)
            
            val usedClusters = mutableSetOf<Int>()

            for (i in sortedClusters.indices) {
                for (j in i + 1 until sortedClusters.size) {
                    if (planeIdx >= maxPlanes) break
                    if (i in usedClusters || j in usedClusters) continue
                    
                    val c1 = sortedClusters[i]
                    val c2 = sortedClusters[j]
                    
                    // Check angle difference between clusters
                    val a1 = weightedAngleMean(c1)
                    val a2 = weightedAngleMean(c2)
                    var angleDiff = abs(a1 - a2)
                    angleDiff = minOf(angleDiff, 180.0 - angleDiff)
                    
                    // Planes are usually formed by lines with significantly different directions (e.g. > 25 deg)
                    if (angleDiff < 25.0) continue
                    
                    // Check if clusters are somewhat spatially related (optional but good for quality)
                    if (!areClustersSpatiallyRelated(c1, c2)) continue

                    if (c1.size + c2.size < 4) continue
                    
                    val color = planeColors[planeIdx % planeColors.size]
                    drawPlane(res, c1, c2, color, totalLines, planeIdx)
                    
                    usedClusters.add(i); usedClusters.add(j)
                    planeIdx++
                }
            }
            lines.release()
        } catch (e: Exception) {
            exceptionLogger("plane_detection", "error", e)
        } finally {
            gray.release(); clahe.release(); blurred.release(); edges.release()
        }
        return planeIdx
    }

    fun detectVanishingPoints(
        src: Mat,
        res: Mat,
        labels: Map<String, String>
    ) {
        val gray = Mat(); val blurred = Mat(); val edges = Mat()
        try {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
            Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
            Imgproc.Canny(blurred, edges, 50.0, 150.0)

            val lines = Mat()
            Imgproc.HoughLinesP(edges, lines, 1.0, Math.PI / 180.0, 40, 20.0, 8.0)

            val clusters = clusterLines(lines)
            val vpColors = arrayOf(Scalar(0.0, 255.0, 0.0), Scalar(0.0, 0.0, 255.0), Scalar(0.0, 165.0, 255.0), Scalar(255.0, 255.0, 0.0))
            var foundVP = false
            
            val sortedClusters = clusters.sortedByDescending { it.size }.take(4)
            
            for (i in sortedClusters.indices) {
                val cluster = sortedClusters[i]
                if (cluster.size < 2) continue
                val color = vpColors[i % vpColors.size]
                for (seg in cluster) {
                    Imgproc.line(res, Point(seg[0], seg[1]), Point(seg[2], seg[3]), color, 1)
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
            lines.release()
        } catch (e: Exception) {
            exceptionLogger("vanishing_points", "error", e)
            val labelVpError = labels["vpError"] ?: "VP Error"
            Imgproc.putText(res, "$labelVpError: ${e.message?.take(30)}", Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255.0, 100.0, 100.0), 2)
        } finally {
            gray.release(); blurred.release(); edges.release()
        }
    }

    private fun areClustersSpatiallyRelated(c1: List<DoubleArray>, c2: List<DoubleArray>): Boolean {
        // Simple bounding box overlap check with padding
        val r1 = getClusterBoundingBox(c1)
        val r2 = getClusterBoundingBox(c2)
        
        // Inflate rects slightly to allow for nearby but not touching lines
        val padding = 50.0
        val inflatedR1 = Rect(
            (r1.x - padding).toInt(), (r1.y - padding).toInt(),
            (r1.width + 2 * padding).toInt(), (r1.height + 2 * padding).toInt()
        )
        
        return inflatedR1.x < r2.x + r2.width &&
               inflatedR1.x + inflatedR1.width > r2.x &&
               inflatedR1.y < r2.y + r2.height &&
               inflatedR1.y + inflatedR1.height > r2.y
    }

    private fun getClusterBoundingBox(cluster: List<DoubleArray>): Rect {
        var minX = Double.MAX_VALUE; var minY = Double.MAX_VALUE
        var maxX = Double.MIN_VALUE; var maxY = Double.MIN_VALUE
        for (seg in cluster) {
            minX = minOf(minX, seg[0], seg[2]); maxX = maxOf(maxX, seg[0], seg[2])
            minY = minOf(minY, seg[1], seg[3]); maxY = maxOf(maxY, seg[1], seg[3])
        }
        return Rect(minX.toInt(), minY.toInt(), (maxX - minX).toInt(), (maxY - minY).toInt())
    }

    private fun clusterLines(lines: Mat): ArrayList<ArrayList<DoubleArray>> {
        val clusters = ArrayList<ArrayList<DoubleArray>>()
        val clusterAngles = ArrayList<Double>()
        for (i in 0 until lines.rows()) {
            val seg = lines.get(i, 0).takeIf { it.isNotEmpty() } ?: continue
            val x1 = seg[0]; val y1 = seg[1]; val x2 = seg[2]; val y2 = seg[3]
            val length = hypot(x2 - x1, y2 - y1)
            val angle = Math.toDegrees(atan2(y2 - y1, x2 - x1)).let { if (it < 0) it + 180.0 else it } % 180.0
            var assigned = false
            for (k in clusterAngles.indices) {
                var diff = abs(angle - clusterAngles[k]); diff = minOf(diff, 180.0 - diff)
                if (diff <= CLUSTER_ANGLE_THRESHOLD_DEG) {
                    clusters[k].add(doubleArrayOf(x1, y1, x2, y2, length))
                    clusterAngles[k] = weightedAngleMean(clusters[k])
                    assigned = true; break
                }
            }
            if (!assigned) {
                clusters.add(arrayListOf(doubleArrayOf(x1, y1, x2, y2, length)))
                clusterAngles.add(angle)
            }
        }
        return clusters
    }

    private fun drawPlane(res: Mat, c1: List<DoubleArray>, c2: List<DoubleArray>, color: Scalar, totalLines: Int, idx: Int) {
        val planeLineCount = c1.size + c2.size
        val confidence = if (totalLines > 0) (planeLineCount * 100 / totalLines).coerceAtMost(100) else 0
        for (seg in c1) Imgproc.line(res, Point(seg[0], seg[1]), Point(seg[2], seg[3]), color, 1)
        for (seg in c2) Imgproc.line(res, Point(seg[0], seg[1]), Point(seg[2], seg[3]), color, 1)

        val allPoints = ArrayList<Point>()
        (c1 + c2).forEach { seg -> allPoints.add(Point(seg[0], seg[1])); allPoints.add(Point(seg[2], seg[3])) }
        
        drawPlaneOverlay(res, allPoints, color)
        val cx = allPoints.map { it.x }.average()
        val cy = allPoints.map { it.y }.average()
        Imgproc.putText(res, "P${idx + 1} ($confidence%)", Point(cx + 8, cy), Imgproc.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    }

    private fun drawPlaneOverlay(dst: Mat, points: List<Point>, color: Scalar) {
        if (points.size < 3) return
        try {
            val contourMat = MatOfPoint().apply { fromArray(*points.toTypedArray()) }
            val hullIdx = MatOfInt()
            Imgproc.convexHull(contourMat, hullIdx)
            val hullPts = hullIdx.toArray().map { i -> points[i] }
            if (hullPts.size >= 3) {
                val hullMat = MatOfPoint().apply { fromArray(*hullPts.toTypedArray()) }
                val overlay = dst.clone()
                // Use a more subtle transparency for the overlay
                Imgproc.fillConvexPoly(overlay, hullMat, color)
                Core.addWeighted(dst, 0.82, overlay, 0.18, 0.0, dst)
                
                // Draw a subtle border around the detected plane
                val hullList = listOf(hullMat)
                Imgproc.polylines(dst, hullList, true, color, 2, Imgproc.LINE_AA)

                overlay.release(); hullMat.release()
            }
            contourMat.release(); hullIdx.release()
        } catch (e: Exception) {
            exceptionLogger("plane_overlay", "opencv", e)
        }
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

    private fun weightedAngleMean(cluster: List<DoubleArray>): Double {
        var sumSin = 0.0; var sumCos = 0.0; var totalWeight = 0.0
        for (seg in cluster) {
            val angleRad = atan2(seg[3] - seg[1], seg[2] - seg[0])
            sumSin += seg[4] * sin(2.0 * angleRad)
            sumCos += seg[4] * cos(2.0 * angleRad)
            totalWeight += seg[4]
        }
        return if (totalWeight == 0.0) 0.0 else Math.toDegrees(atan2(sumSin, sumCos) / 2.0).let { if (it < 0) it + 180.0 else it } % 180.0
    }

    private fun computeVanishingPoint(lines: List<DoubleArray>): Point? {
        if (lines.size < 2) return null
        val aMat = Mat(lines.size, 2, CvType.CV_64F)
        val bMat = Mat(lines.size, 1, CvType.CV_64F)
        for (i in lines.indices) {
            val x1 = lines[i][0]; val y1 = lines[i][1]
            val x2 = lines[i][2]; val y2 = lines[i][3]
            val a = y1 - y2
            val b = x2 - x1
            val c = a * x1 + b * y1
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
