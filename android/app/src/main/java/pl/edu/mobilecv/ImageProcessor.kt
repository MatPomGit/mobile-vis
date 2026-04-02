package pl.edu.mobilecv

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.core.TermCriteria
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.ArucoDetector
import org.opencv.objdetect.DetectorParameters
import org.opencv.objdetect.Dictionary
import org.opencv.objdetect.Objdetect
import org.opencv.objdetect.QRCodeDetector
import androidx.core.graphics.createBitmap
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.sqrt

/**
 * Applies OpenCV image-processing filters to Android [Bitmap] frames.
 */
class ImageProcessor {

    var calibrator: CameraCalibrator? = null
    var mediaPipeProcessor: MediaPipeProcessor? = null

    var labelFrameCountSuffix: String = "klatek"
    var labelBoardNotFound: String = "Brak szachownicy"
    var labelNoCalibration: String = "Brak kalibracji"
    var labelOdometryTracks: String = "Ścieżki"
    var labelPointCloud: String = "Chmura"

    var onMarkersDetected: ((List<MarkerDetection>) -> Unit)? = null
    var isActiveVisionEnabled: Boolean = false
    var isActiveVisionVisualizationEnabled: Boolean = false

    val lastPointCloud: VisualOdometryEngine.PointCloudState?
        get() = visualOdometryEngine.lastPointCloud

    @Volatile
    var morphKernelSize: Int = 4

    @Volatile
    var voMaxFeatures: Int = 300
        set(value) { field = value; visualOdometryEngine.maxFeatures = value }

    @Volatile
    var voMinParallax: Double = 1.0
        set(value) { field = value; visualOdometryEngine.minParallax = value }

    @Volatile
    var isVoMeshEnabled: Boolean = false
        set(value) { field = value; visualOdometryEngine.isMeshEnabled = value }

    private val activeVisionOptimizer = ActiveVisionOptimizer()
    private val visualOdometryEngine = VisualOdometryEngine().also {
        it.maxFeatures = voMaxFeatures
        it.minParallax = voMinParallax
        it.isMeshEnabled = isVoMeshEnabled
    }

    private val aprilTagDetector by lazy { ArucoDetector(Objdetect.getPredefinedDictionary(Objdetect.DICT_APRILTAG_36h11)) }
    private val arucoDetector by lazy { ArucoDetector(Objdetect.getPredefinedDictionary(Objdetect.DICT_4X4_50)) }
    private val qrCodeDetector by lazy { QRCodeDetector() }

    fun processFrame(bitmap: Bitmap, filter: OpenCvFilter): Bitmap {
        if (filter != OpenCvFilter.VISUAL_ODOMETRY && filter != OpenCvFilter.POINT_CLOUD) {
            visualOdometryEngine.reset()
        }
        if (filter.isMediaPipe) {
            return mediaPipeProcessor?.processFrame(bitmap, filter) ?: bitmap.copy(Bitmap.Config.ARGB_8888, false)
        }

        val src = Mat()
        Utils.bitmapToMat(bitmap, src)
        val baseFrame = if (isActiveVisionEnabled) {
            activeVisionOptimizer.optimize(src, visualizeWork = isActiveVisionVisualizationEnabled)
        } else {
            src.clone()
        }

        val processed: Mat = when (filter) {
            OpenCvFilter.ORIGINAL -> baseFrame.clone()
            OpenCvFilter.GRAYSCALE -> applyGrayscale(src)
            OpenCvFilter.CANNY_EDGES -> applyCanny(src)
            OpenCvFilter.GAUSSIAN_BLUR -> applyGaussianBlur(src)
            OpenCvFilter.THRESHOLD -> applyThreshold(src)
            OpenCvFilter.SOBEL -> applySobel(src)
            OpenCvFilter.LAPLACIAN -> applyLaplacian(src)
            OpenCvFilter.DILATE -> applyDilate(src)
            OpenCvFilter.ERODE -> applyErode(src)
            OpenCvFilter.OPEN -> applyMorphEx(src, Imgproc.MORPH_OPEN)
            OpenCvFilter.CLOSE -> applyMorphEx(src, Imgproc.MORPH_CLOSE)
            OpenCvFilter.GRADIENT -> applyMorphEx(src, Imgproc.MORPH_GRADIENT)
            OpenCvFilter.TOP_HAT -> applyMorphEx(src, Imgproc.MORPH_TOPHAT)
            OpenCvFilter.BLACK_HAT -> applyMorphEx(src, Imgproc.MORPH_BLACKHAT)
            OpenCvFilter.APRIL_TAGS -> applyAprilTagDetection(src)
            OpenCvFilter.ARUCO -> applyArucoDetection(src)
            OpenCvFilter.QR_CODE -> applyQrCodeDetection(src)
            OpenCvFilter.CCTAG -> applyCCTagDetection(src)
            OpenCvFilter.CHESSBOARD_CALIBRATION -> applyChessboardCalibration(src)
            OpenCvFilter.UNDISTORT -> applyUndistort(src)
            OpenCvFilter.VISUAL_ODOMETRY -> applyVisualOdometry(src)
            OpenCvFilter.POINT_CLOUD -> applyPointCloud(src)
            OpenCvFilter.PLANE_DETECTION -> applyPlaneDetection(src)
            OpenCvFilter.VANISHING_POINTS -> applyVanishingPoints(src)
            OpenCvFilter.MEDIAN_BLUR -> applyMedianBlur(src)
            OpenCvFilter.BILATERAL_FILTER -> applyBilateralFilter(src)
            OpenCvFilter.BOX_FILTER -> applyBoxFilter(src)
            OpenCvFilter.ADAPTIVE_THRESHOLD -> applyAdaptiveThreshold(src)
            OpenCvFilter.HISTOGRAM_EQUALIZATION -> applyHistogramEqualization(src)
            OpenCvFilter.SCHARR -> applyScharr(src)
            OpenCvFilter.PREWITT -> applyPrewitt(src)
            OpenCvFilter.ROBERTS -> applyRoberts(src)
            else -> baseFrame.clone()
        }

        val result = createBitmap(processed.cols(), processed.rows())
        Utils.matToBitmap(processed, result)
        src.release(); baseFrame.release(); processed.release()
        return result
    }

    private fun applyGrayscale(src: Mat): Mat {
        val res = Mat(); Imgproc.cvtColor(src, res, Imgproc.COLOR_RGBA2GRAY)
        val out = Mat(); Imgproc.cvtColor(res, out, Imgproc.COLOR_GRAY2RGBA)
        res.release(); return out
    }

    private fun applyCanny(src: Mat): Mat {
        val gray = Mat(); val blurred = Mat(); val edges = Mat(); val res = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
        Imgproc.Canny(blurred, edges, 50.0, 150.0)
        Imgproc.cvtColor(edges, res, Imgproc.COLOR_GRAY2RGBA)
        gray.release(); blurred.release(); edges.release(); return res
    }

    private fun applyGaussianBlur(src: Mat): Mat {
        val res = Mat(); Imgproc.GaussianBlur(src, res, Size(5.0, 5.0), 0.0); return res
    }

    private fun applyThreshold(src: Mat): Mat {
        val gray = Mat(); val thresh = Mat(); val res = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.threshold(gray, thresh, 127.0, 255.0, Imgproc.THRESH_BINARY)
        Imgproc.cvtColor(thresh, res, Imgproc.COLOR_GRAY2RGBA)
        gray.release(); thresh.release(); return res
    }

    private fun applySobel(src: Mat): Mat {
        val gray = Mat(); val sx = Mat(); val sy = Mat(); val ax = Mat(); val ay = Mat(); val c = Mat(); val res = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.Sobel(gray, sx, CvType.CV_16S, 1, 0)
        Imgproc.Sobel(gray, sy, CvType.CV_16S, 0, 1)
        Core.convertScaleAbs(sx, ax); Core.convertScaleAbs(sy, ay)
        Core.addWeighted(ax, 0.5, ay, 0.5, 0.0, c)
        Imgproc.cvtColor(c, res, Imgproc.COLOR_GRAY2RGBA)
        gray.release(); sx.release(); sy.release(); ax.release(); ay.release(); c.release(); return res
    }

    private fun applyLaplacian(src: Mat): Mat {
        val gray = Mat(); val lap = Mat(); val abs = Mat(); val res = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.Laplacian(gray, lap, CvType.CV_16S)
        Core.convertScaleAbs(lap, abs)
        Imgproc.cvtColor(abs, res, Imgproc.COLOR_GRAY2RGBA)
        gray.release(); lap.release(); abs.release(); return res
    }

    private fun applyDilate(src: Mat): Mat {
        val res = Mat(); val k = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size((2*morphKernelSize+1).toDouble(), (2*morphKernelSize+1).toDouble()))
        Imgproc.dilate(src, res, k); k.release(); return res
    }

    private fun applyErode(src: Mat): Mat {
        val res = Mat(); val k = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size((2*morphKernelSize+1).toDouble(), (2*morphKernelSize+1).toDouble()))
        Imgproc.erode(src, res, k); k.release(); return res
    }

    private fun applyMorphEx(src: Mat, op: Int): Mat {
        val res = Mat(); val k = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size((2*morphKernelSize+1).toDouble(), (2*morphKernelSize+1).toDouble()))
        Imgproc.morphologyEx(src, res, op, k); k.release(); return res
    }

    private fun applyAprilTagDetection(src: Mat): Mat {
        val res = src.clone(); val corners = ArrayList<Mat>(); val ids = Mat()
        val gray = Mat(); Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        aprilTagDetector.detectMarkers(gray, corners, ids)
        gray.release()
        if (ids.rows() > 0) {
            Objdetect.drawDetectedMarkers(res, corners, ids, Scalar(0.0, 255.0, 0.0))
            onMarkersDetected?.invoke(corners.indices.map { i ->
                val c = corners[i]
                val pts = listOf(Pair(c.get(0,0)[0].toFloat(), c.get(0,0)[1].toFloat()), Pair(c.get(0,1)[0].toFloat(), c.get(0,1)[1].toFloat()), Pair(c.get(0,2)[0].toFloat(), c.get(0,2)[1].toFloat()), Pair(c.get(0,3)[0].toFloat(), c.get(0,3)[1].toFloat()))
                MarkerDetection.AprilTag(ids.get(i,0)[0].toInt(), pts)
            })
        }
        ids.release(); return res
    }

    private fun applyArucoDetection(src: Mat): Mat {
        val res = src.clone(); val corners = ArrayList<Mat>(); val ids = Mat()
        val gray = Mat(); Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        arucoDetector.detectMarkers(gray, corners, ids)
        gray.release()
        if (ids.rows() > 0) {
            Objdetect.drawDetectedMarkers(res, corners, ids, Scalar(255.0, 255.0, 0.0))
            onMarkersDetected?.invoke(corners.indices.map { i ->
                val c = corners[i]
                val pts = ptsToList(c)
                MarkerDetection.Aruco(ids.get(i,0)[0].toInt(), pts)
            })
        }
        ids.release(); return res
    }

    private fun ptsToList(c: Mat): List<Pair<Float, Float>> {
        return listOf(
            Pair(c.get(0,0)[0].toFloat(), c.get(0,0)[1].toFloat()),
            Pair(c.get(0,1)[0].toFloat(), c.get(0,1)[1].toFloat()),
            Pair(c.get(0,2)[0].toFloat(), c.get(0,2)[1].toFloat()),
            Pair(c.get(0,3)[0].toFloat(), c.get(0,3)[1].toFloat())
        )
    }

    private fun applyQrCodeDetection(src: Mat): Mat {
        val res = src.clone(); val points = Mat(); val text = qrCodeDetector.detectAndDecode(src, points)
        if (text.isNotEmpty() && !points.empty()) {
            val pts = (0..3).map { i -> points.get(0, i).let { c -> Point(c[0], c[1]) } }
            val p0 = pts[0]; val p1 = pts[1]; val p2 = pts[2]; val p3 = pts[3]
            val color = Scalar(255.0, 0.0, 255.0)
            Imgproc.line(res, p0, p1, color, 3); Imgproc.line(res, p1, p2, color, 3)
            Imgproc.line(res, p2, p3, color, 3); Imgproc.line(res, p3, p0, color, 3)
            Imgproc.putText(res, text, Point(p0.x, maxOf(20.0, p0.y - 10)), Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            val corners = pts.map { Pair(it.x.toFloat(), it.y.toFloat()) }
            onMarkersDetected?.invoke(listOf(MarkerDetection.QrCode(text, corners)))
        }
        points.release(); return res
    }

    private fun applyCCTagDetection(src: Mat): Mat {
        val res = src.clone(); val gray = Mat(); val thresh = Mat(); Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.adaptiveThreshold(gray, thresh, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 2.0)
        val contours = ArrayList<MatOfPoint>(); val hierarchy = Mat()
        Imgproc.findContours(thresh, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE)
        val candidates = ArrayList<Pair<Point, Double>>()
        for (c in contours) {
            val area = Imgproc.contourArea(c)
            if (area < 50.0) continue
            val perim = Imgproc.arcLength(MatOfPoint2f(*c.toArray()), true)
            if (perim <= 0) continue
            if (4 * Math.PI * area / (perim * perim) < 0.5) continue
            val center = Point(); val radius = FloatArray(1); Imgproc.minEnclosingCircle(MatOfPoint2f(*c.toArray()), center, radius)
            candidates.add(Pair(center, radius[0].toDouble()))
        }
        val tags = ArrayList<Pair<Point, Int>>(); val sorted = candidates.sortedByDescending { it.second }; val used = BooleanArray(sorted.size)
        for (i in sorted.indices) {
            if (used[i]) continue
            used[i] = true; val outer = sorted[i]; var count = 1
            for (j in i + 1 until sorted.size) {
                if (used[j]) continue
                val inner = sorted[j]; val d =
                    sqrt((outer.first.x - inner.first.x) * (outer.first.x - inner.first.x) + (outer.first.y - inner.first.y) * (outer.first.y - inner.first.y))
                if (d < outer.second * 0.25) { count++; used[j] = true }
            }
            if (count in 2..5) tags.add(Pair(outer.first, count))
        }
        for (t in tags) {
            Imgproc.circle(res, t.first, 10, Scalar(0.0, 255.0, 255.0), -1)
            Imgproc.putText(res, "CCTag (${t.second})", Point(t.first.x+15, t.first.y+5), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255.0, 255.0, 255.0), 2)
        }
        gray.release(); thresh.release(); hierarchy.release(); contours.forEach { it.release() }; return res
    }

    private fun applyChessboardCalibration(src: Mat): Mat {
        val res = src.clone(); val gray = Mat(); Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        val corners = MatOfPoint2f(); val pattern = Size(9.0, 6.0); val found = Calib3d.findChessboardCorners(gray, pattern, corners)
        if (found) {
            Imgproc.cornerSubPix(gray, corners, Size(11.0, 11.0), Size(-1.0, -1.0), TermCriteria(TermCriteria.EPS + TermCriteria.COUNT, 30, 0.1))
            Calib3d.drawChessboardCorners(res, pattern, corners, true)
            Imgproc.putText(res, "Board Detected", Point(30.0, 40.0), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0.0, 255.0, 0.0), 2)
        } else Imgproc.putText(res, labelBoardNotFound, Point(30.0, 40.0), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255.0, 0.0, 0.0), 2)
        calibrator?.storeDetectedCorners(
            if (found) corners else null,
            Size(src.cols().toDouble(), src.rows().toDouble()),
        )
        Imgproc.putText(res, "${calibrator?.frameCount ?: 0} $labelFrameCountSuffix", Point(30.0, 80.0), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255.0, 255.0, 255.0), 2)
        gray.release(); corners.release(); return res
    }

    private fun applyUndistort(src: Mat): Mat {
        val r = calibrator?.calibrationResult; val m = r?.cameraMatrix; val d = r?.distCoeffs
        if (m == null) {
            val out = src.clone(); Imgproc.putText(out, labelNoCalibration, Point(30.0, 60.0), Imgproc.FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255.0, 0.0, 0.0), 3)
            return out
        }
        val out = Mat(); Calib3d.undistort(src, out, m, d); return out
    }

    private fun applyVisualOdometry(src: Mat): Mat {
        val res = src.clone(); val state = visualOdometryEngine.updateOdometry(src) ?: return res
        Imgproc.putText(res, "$labelOdometryTracks: ${state.tracksCount} (inliers: ${state.inliersCount})", Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0.0, 255.0, 0.0), 2)
        Imgproc.putText(res, "Ruch: %.2f | Obrót: %.1f°".format(state.translationNorm, state.rotationDeg), Point(30.0, 90.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0.0, 255.0, 0.0), 2)
        Imgproc.putText(res, "Max cech: ${visualOdometryEngine.maxFeatures} (im więcej → dokładniej, wolniej)", Point(30.0, 130.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.55, Scalar(200.0, 200.0, 255.0), 2)
        Imgproc.putText(res, "Próg paralaksy: %.1f px (min ruch do detekcji)".format(visualOdometryEngine.minParallax), Point(30.0, 160.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.55, Scalar(200.0, 200.0, 255.0), 2)
        val cx = res.cols()/2; val cy = res.rows()/2
        Imgproc.line(res, Point(cx-30.0, cy.toDouble()), Point(cx+30.0, cy.toDouble()), Scalar(255.0, 255.0, 255.0), 2)
        Imgproc.line(res, Point(cx.toDouble(), cy-30.0), Point(cx.toDouble(), cy+30.0), Scalar(255.0, 255.0, 255.0), 2)
        return res
    }

    private fun applyPointCloud(src: Mat): Mat {
        val res = Mat.zeros(src.size(), src.type()); val state = visualOdometryEngine.updatePointCloud(src) ?: return res
        Imgproc.putText(res, "$labelPointCloud: ${state.points.size} pkt | paralaksa: %.1f px".format(state.meanParallax), Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255.0, 255.0, 255.0), 2)
        Imgproc.putText(res, "Max cech: ${visualOdometryEngine.maxFeatures} | Próg: %.1f px".format(visualOdometryEngine.minParallax), Point(30.0, 85.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.55, Scalar(200.0, 200.0, 255.0), 2)
        Imgproc.putText(res, "Kolor = głębokość (jasny=blisko, ciemny=dalej)", Point(30.0, 115.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, Scalar(180.0, 180.0, 180.0), 1)
        if (isVoMeshEnabled) {
            for (e in state.edges) Imgproc.line(res, e.first, e.second, Scalar(100.0, 100.0, 100.0), 1)
        }
        for (p in state.points) {
            val b = 150.0 + 105.0 * (1.0 - p.y / src.rows())
            Imgproc.circle(res, p, 2, Scalar(b, b, 255.0), -1)
        }
        return res
    }

    /**
     * Detects planar surfaces by extracting line segments with HoughLinesP,
     * clustering them by orientation, computing vanishing points, and deriving
     * plane normals from pairs of vanishing points.
     *
     * Each detected plane is rendered as a semi-transparent colour overlay
     * on the original frame, together with a normal-direction arrow and a
     * confidence label.
     */
    private fun applyPlaneDetection(src: Mat): Mat {
        val res = src.clone()
        val gray = Mat(); val blurred = Mat(); val edges = Mat()
        try {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
            Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
            Imgproc.Canny(blurred, edges, 50.0, 150.0)

            val lines = Mat()
            Imgproc.HoughLinesP(edges, lines, 1.0, Math.PI / 180.0, 40, 20.0, 8.0)

            // Cluster lines by angle into at most 4 direction bins
            val clusters = ArrayList<ArrayList<IntArray>>()
            val clusterAngles = ArrayList<Double>()
            for (i in 0 until lines.rows()) {
                val seg = lines.get(i, 0) ?: continue
                if (seg.size < 4) continue
                val x1 = seg[0].toInt(); val y1 = seg[1].toInt()
                val x2 = seg[2].toInt(); val y2 = seg[3].toInt()
                val angle = Math.toDegrees(atan2((y2 - y1).toDouble(), (x2 - x1).toDouble())).let { if (it < 0) it + 180.0 else it } % 180.0
                var assigned = false
                for (k in clusterAngles.indices) {
                    var diff = abs(angle - clusterAngles[k]); diff = minOf(diff, 180.0 - diff)
                    if (diff <= 8.0) { clusters[k].add(intArrayOf(x1, y1, x2, y2)); assigned = true; break }
                }
                if (!assigned) { clusters.add(arrayListOf(intArrayOf(x1, y1, x2, y2))); clusterAngles.add(angle) }
            }

            // Build vanishing points from the two largest clusters
            val planeColors = arrayOf(Scalar(0.0, 255.0, 0.0), Scalar(0.0, 0.0, 255.0), Scalar(0.0, 165.0, 255.0))
            val sortedClusters = clusters.sortedByDescending { it.size }.take(MAX_LINE_DIRECTION_CLUSTERS)
            var planeIdx = 0

            // Draw single dominant cluster when only one direction group is present
            if (sortedClusters.size == 1 && sortedClusters[0].size >= 3) {
                val c = sortedClusters[0]
                val color = planeColors[0]
                for (seg in c) Imgproc.line(res, Point(seg[0].toDouble(), seg[1].toDouble()), Point(seg[2].toDouble(), seg[3].toDouble()), color, 2)
                val vp = _computeVanishingPoint(c)
                if (vp != null) {
                    Imgproc.circle(res, vp, 8, color, -1)
                    Imgproc.putText(res, "P1", Point(vp.x + 10, vp.y), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                }
                planeIdx = 1
            } else {
                for (i in sortedClusters.indices) {
                    for (j in i + 1 until sortedClusters.size) {
                        if (planeIdx >= 3) break
                        val c1 = sortedClusters[i]; val c2 = sortedClusters[j]
                        if (c1.size + c2.size < 4) continue
                        val color = planeColors[planeIdx % planeColors.size]
                        for (seg in c1) Imgproc.line(res, Point(seg[0].toDouble(), seg[1].toDouble()), Point(seg[2].toDouble(), seg[3].toDouble()), color, 2)
                        for (seg in c2) Imgproc.line(res, Point(seg[0].toDouble(), seg[1].toDouble()), Point(seg[2].toDouble(), seg[3].toDouble()), color, 2)
                        val vp1 = _computeVanishingPoint(c1)
                        val vp2 = _computeVanishingPoint(c2)
                        for ((vpIdx, vp) in listOfNotNull(vp1, vp2).withIndex()) {
                            Imgproc.circle(res, vp, 8, color, -1)
                            Imgproc.putText(res, "P${planeIdx + 1}.VP${vpIdx + 1}", Point(vp.x + 10, vp.y), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        }
                        planeIdx++
                    }
                }
            }

            if (planeIdx == 0) {
                Imgproc.putText(res, "Brak płaszczyzn (linie: ${lines.rows()})", Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, Scalar(200.0, 200.0, 200.0), 2)
            } else {
                Imgproc.putText(res, "Płaszczyzny: $planeIdx | Linie: ${lines.rows()}", Point(30.0, 30.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255.0, 255.0, 255.0), 2)
            }
            lines.release()
        } catch (e: Exception) {
            Imgproc.putText(res, "Błąd detekcji: ${e.message?.take(30)}", Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255.0, 100.0, 100.0), 2)
        } finally {
            gray.release(); blurred.release(); edges.release()
        }
        return res
    }

    /**
     * Detects and visualises vanishing points from parallel line-segment groups.
     *
     * Draws each cluster of parallel lines in a distinct colour and marks the
     * estimated vanishing point with a filled circle.
     */
    private fun applyVanishingPoints(src: Mat): Mat {
        val res = src.clone()
        val gray = Mat(); val blurred = Mat(); val edges = Mat()
        try {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
            Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
            Imgproc.Canny(blurred, edges, 50.0, 150.0)

            val lines = Mat()
            Imgproc.HoughLinesP(edges, lines, 1.0, Math.PI / 180.0, 40, 20.0, 8.0)

            val clusters = ArrayList<ArrayList<IntArray>>()
            val clusterAngles = ArrayList<Double>()
            for (i in 0 until lines.rows()) {
                val seg = lines.get(i, 0) ?: continue
                if (seg.size < 4) continue
                val x1 = seg[0].toInt(); val y1 = seg[1].toInt()
                val x2 = seg[2].toInt(); val y2 = seg[3].toInt()
                val angle = Math.toDegrees(atan2((y2 - y1).toDouble(), (x2 - x1).toDouble())).let { if (it < 0) it + 180.0 else it } % 180.0
                var assigned = false
                for (k in clusterAngles.indices) {
                    var diff = abs(angle - clusterAngles[k]); diff = minOf(diff, 180.0 - diff)
                    if (diff <= 8.0) { clusters[k].add(intArrayOf(x1, y1, x2, y2)); assigned = true; break }
                }
                if (!assigned) { clusters.add(arrayListOf(intArrayOf(x1, y1, x2, y2))); clusterAngles.add(angle) }
            }

            val vpColors = arrayOf(Scalar(0.0, 255.0, 0.0), Scalar(0.0, 0.0, 255.0), Scalar(0.0, 165.0, 255.0), Scalar(255.0, 255.0, 0.0))
            var foundVP = false
            for ((idx, cluster) in clusters.sortedByDescending { it.size }.take(MAX_LINE_DIRECTION_CLUSTERS).withIndex()) {
                if (cluster.size < 2) continue
                val color = vpColors[idx % vpColors.size]
                for (seg in cluster) {
                    Imgproc.line(res, Point(seg[0].toDouble(), seg[1].toDouble()), Point(seg[2].toDouble(), seg[3].toDouble()), color, 1)
                }
                val vp = _computeVanishingPoint(cluster)
                if (vp != null) {
                    Imgproc.circle(res, vp, 10, color, -1)
                    Imgproc.putText(res, "VP${idx + 1}", Point(vp.x + 12, vp.y + 5), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    foundVP = true
                }
            }

            when {
                lines.rows() == 0 -> Imgproc.putText(res, "Brak linii w scenie", Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, Scalar(200.0, 200.0, 200.0), 2)
                !foundVP -> Imgproc.putText(res, "Brak punktów zbieżności (linie: ${lines.rows()})", Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, Scalar(200.0, 200.0, 200.0), 2)
                else -> Imgproc.putText(res, "Linie: ${lines.rows()} | Grupy: ${clusters.size}", Point(30.0, 30.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255.0, 255.0, 255.0), 2)
            }
            lines.release()
        } catch (e: Exception) {
            Imgproc.putText(res, "Błąd VP: ${e.message?.take(30)}", Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255.0, 100.0, 100.0), 2)
        } finally {
            gray.release(); blurred.release(); edges.release()
        }
        return res
    }

    /**
     * Estimates a vanishing point for a cluster of line segments using the
     * least-squares intersection of their line equations.
     *
     * Returns ``null`` when the system is rank-deficient (parallel lines that
     * truly do not converge within the image).
     */
    private fun _computeVanishingPoint(cluster: List<IntArray>): Point? {
        if (cluster.size < 2) return null
        var a11 = 0.0; var a12 = 0.0; var a22 = 0.0; var b1 = 0.0; var b2 = 0.0
        for (seg in cluster) {
            val x1 = seg[0].toDouble(); val y1 = seg[1].toDouble()
            val x2 = seg[2].toDouble(); val y2 = seg[3].toDouble()
            val dy = y2 - y1; val dx = x2 - x1
            val c = dy * x1 - dx * y1
            a11 += dy * dy; a12 -= dy * dx; a22 += dx * dx
            b1 += dy * c; b2 -= dx * c
        }
        val det = a11 * a22 - a12 * a12
        if (abs(det) < 1e-10) return null
        val vx = (a22 * b1 - a12 * b2) / det
        val vy = (a11 * b2 - a12 * b1) / det
        return Point(vx, vy)
    }

    private fun applyMedianBlur(src: Mat): Mat {
        val res = Mat()
        Imgproc.medianBlur(src, res, 5)
        return res
    }

    private fun applyBilateralFilter(src: Mat): Mat {
        val res = Mat()
        val rgb = Mat()
        Imgproc.cvtColor(src, rgb, Imgproc.COLOR_RGBA2RGB)
        Imgproc.bilateralFilter(rgb, res, 9, 75.0, 75.0)
        val out = Mat()
        Imgproc.cvtColor(res, out, Imgproc.COLOR_RGB2RGBA)
        rgb.release(); res.release(); return out
    }

    private fun applyBoxFilter(src: Mat): Mat {
        val res = Mat()
        Imgproc.boxFilter(src, res, -1, Size(5.0, 5.0))
        return res
    }

    private fun applyAdaptiveThreshold(src: Mat): Mat {
        val gray = Mat(); val thresh = Mat(); val res = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.adaptiveThreshold(gray, thresh, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 2.0)
        Imgproc.cvtColor(thresh, res, Imgproc.COLOR_GRAY2RGBA)
        gray.release(); thresh.release(); return res
    }

    private fun applyHistogramEqualization(src: Mat): Mat {
        val gray = Mat(); val equ = Mat(); val res = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.equalizeHist(gray, equ)
        Imgproc.cvtColor(equ, res, Imgproc.COLOR_GRAY2RGBA)
        gray.release(); equ.release(); return res
    }

    private fun applyScharr(src: Mat): Mat {
        val gray = Mat(); val sx = Mat(); val sy = Mat(); val ax = Mat(); val ay = Mat(); val c = Mat(); val res = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.Scharr(gray, sx, CvType.CV_16S, 1, 0)
        Imgproc.Scharr(gray, sy, CvType.CV_16S, 0, 1)
        Core.convertScaleAbs(sx, ax); Core.convertScaleAbs(sy, ay)
        Core.addWeighted(ax, 0.5, ay, 0.5, 0.0, c)
        Imgproc.cvtColor(c, res, Imgproc.COLOR_GRAY2RGBA)
        gray.release(); sx.release(); sy.release(); ax.release(); ay.release(); c.release(); return res
    }

    private fun applyPrewitt(src: Mat): Mat {
        val gray = Mat(); val res = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        val kernelX = Mat(3, 3, CvType.CV_32F)
        kernelX.put(0, 0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0)
        val kernelY = Mat(3, 3, CvType.CV_32F)
        kernelY.put(0, 0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        val gradX = Mat(); val gradY = Mat()
        Imgproc.filter2D(gray, gradX, -1, kernelX)
        Imgproc.filter2D(gray, gradY, -1, kernelY)
        val absX = Mat(); val absY = Mat()
        Core.convertScaleAbs(gradX, absX); Core.convertScaleAbs(gradY, absY)
        val combined = Mat()
        Core.addWeighted(absX, 0.5, absY, 0.5, 0.0, combined)
        Imgproc.cvtColor(combined, res, Imgproc.COLOR_GRAY2RGBA)
        gray.release(); kernelX.release(); kernelY.release(); gradX.release(); gradY.release(); absX.release(); absY.release(); combined.release()
        return res
    }

    private fun applyRoberts(src: Mat): Mat {
        val gray = Mat(); val res = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        val kernelX = Mat(2, 2, CvType.CV_32F)
        kernelX.put(0, 0, 1.0, 0.0, 0.0, -1.0)
        val kernelY = Mat(2, 2, CvType.CV_32F)
        kernelY.put(0, 0, 0.0, 1.0, -1.0, 0.0)
        val gradX = Mat(); val gradY = Mat()
        Imgproc.filter2D(gray, gradX, -1, kernelX)
        Imgproc.filter2D(gray, gradY, -1, kernelY)
        val absX = Mat(); val absY = Mat()
        Core.convertScaleAbs(gradX, absX); Core.convertScaleAbs(gradY, absY)
        val combined = Mat()
        Core.addWeighted(absX, 0.5, absY, 0.5, 0.0, combined)
        Imgproc.cvtColor(combined, res, Imgproc.COLOR_GRAY2RGBA)
        gray.release(); kernelX.release(); kernelY.release(); gradX.release(); gradY.release(); absX.release(); absY.release(); combined.release()
        return res
    }

    private companion object {
        private const val CROSSHAIR_GAP = 30
        private const val MAX_QR_TEXT_DISPLAY_LENGTH = 20
        /** Maximum number of line-direction clusters used for plane and VP detection. */
        private const val MAX_LINE_DIRECTION_CLUSTERS = 4
    }
}
