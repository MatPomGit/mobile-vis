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

    private val detectorParameters by lazy {
        DetectorParameters().apply {
            set_adaptiveThreshWinSizeMin(3)
            set_adaptiveThreshWinSizeMax(23)
            set_adaptiveThreshWinSizeStep(10)
        }
    }

    private val aprilTagDetector by lazy { 
        ArucoDetector(Objdetect.getPredefinedDictionary(Objdetect.DICT_APRILTAG_36h11), detectorParameters) 
    }
    private val arucoDetector by lazy { 
        ArucoDetector(Objdetect.getPredefinedDictionary(Objdetect.DICT_4X4_50), detectorParameters) 
    }
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
            OpenCvFilter.GRAYSCALE -> applyGrayscale(baseFrame)
            OpenCvFilter.CANNY_EDGES -> applyCanny(baseFrame)
            OpenCvFilter.GAUSSIAN_BLUR -> applyGaussianBlur(baseFrame)
            OpenCvFilter.THRESHOLD -> applyThreshold(baseFrame)
            OpenCvFilter.SOBEL -> applySobel(baseFrame)
            OpenCvFilter.LAPLACIAN -> applyLaplacian(baseFrame)
            OpenCvFilter.DILATE -> applyDilate(baseFrame)
            OpenCvFilter.ERODE -> applyErode(baseFrame)
            OpenCvFilter.OPEN -> applyMorphEx(baseFrame, Imgproc.MORPH_OPEN)
            OpenCvFilter.CLOSE -> applyMorphEx(baseFrame, Imgproc.MORPH_CLOSE)
            OpenCvFilter.GRADIENT -> applyMorphEx(baseFrame, Imgproc.MORPH_GRADIENT)
            OpenCvFilter.TOP_HAT -> applyMorphEx(baseFrame, Imgproc.MORPH_TOPHAT)
            OpenCvFilter.BLACK_HAT -> applyMorphEx(baseFrame, Imgproc.MORPH_BLACKHAT)
            OpenCvFilter.APRIL_TAGS -> applyAprilTagDetection(baseFrame)
            OpenCvFilter.ARUCO -> applyArucoDetection(baseFrame)
            OpenCvFilter.QR_CODE -> applyQrCodeDetection(baseFrame)
            OpenCvFilter.CCTAG -> applyCCTagDetection(baseFrame)
            OpenCvFilter.CHESSBOARD_CALIBRATION -> applyChessboardCalibration(baseFrame)
            OpenCvFilter.UNDISTORT -> applyUndistort(baseFrame)
            OpenCvFilter.VISUAL_ODOMETRY -> applyVisualOdometry(baseFrame)
            OpenCvFilter.POINT_CLOUD -> applyPointCloud(baseFrame)
            OpenCvFilter.PLANE_DETECTION -> applyPlaneDetection(baseFrame)
            OpenCvFilter.VANISHING_POINTS -> applyVanishingPoints(baseFrame)
            OpenCvFilter.MEDIAN_BLUR -> applyMedianBlur(baseFrame)
            OpenCvFilter.BILATERAL_FILTER -> applyBilateralFilter(baseFrame)
            OpenCvFilter.BOX_FILTER -> applyBoxFilter(baseFrame)
            OpenCvFilter.ADAPTIVE_THRESHOLD -> applyAdaptiveThreshold(baseFrame)
            OpenCvFilter.HISTOGRAM_EQUALIZATION -> applyHistogramEqualization(baseFrame)
            OpenCvFilter.SCHARR -> applyScharr(baseFrame)
            OpenCvFilter.PREWITT -> applyPrewitt(baseFrame)
            OpenCvFilter.ROBERTS -> applyRoberts(baseFrame)
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
            Objdetect.drawDetectedMarkers(res, corners, ids, Scalar(0.0, 255.0, 0.0, 255.0))
            onMarkersDetected?.invoke(corners.indices.map { i ->
                val c = corners[i]
                val pts = listOf(
                    Pair(c.get(0,0)[0].toFloat(), c.get(0,0)[1].toFloat()),
                    Pair(c.get(0,1)[0].toFloat(), c.get(0,1)[1].toFloat()),
                    Pair(c.get(0,2)[0].toFloat(), c.get(0,2)[1].toFloat()),
                    Pair(c.get(0,3)[0].toFloat(), c.get(0,3)[1].toFloat())
                )
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
            Objdetect.drawDetectedMarkers(res, corners, ids, Scalar(255.0, 255.0, 0.0, 255.0))
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
        val res = src.clone()
        val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        val points = Mat()
        val text = qrCodeDetector.detectAndDecode(gray, points)
        if (!points.empty()) {
            val pts = (0 until 4).map { i -> 
                val data = points.get(0, i)
                if (data != null) Point(data[0], data[1]) else Point(0.0, 0.0)
            }
            val color = Scalar(255.0, 0.0, 255.0, 255.0)
            for (i in 0 until 4) {
                Imgproc.line(res, pts[i], pts[(i + 1) % 4], color, 3)
            }
            if (text.isNotEmpty()) {
                Imgproc.putText(res, text.take(MAX_QR_TEXT_DISPLAY_LENGTH), Point(pts[0].x, maxOf(20.0, pts[0].y - 10)), Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                val corners = pts.map { Pair(it.x.toFloat(), it.y.toFloat()) }
                onMarkersDetected?.invoke(listOf(MarkerDetection.QrCode(text, corners)))
            }
        }
        gray.release(); points.release(); return res
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
        val detections = ArrayList<MarkerDetection>()
        for (i in sorted.indices) {
            if (used[i]) continue
            used[i] = true; val outer = sorted[i]; var count = 1
            for (j in i + 1 until sorted.size) {
                if (used[j]) continue
                val inner = sorted[j]; val d =
                    sqrt((outer.first.x - inner.first.x) * (outer.first.x - inner.first.x) + (outer.first.y - inner.first.y) * (outer.first.y - inner.first.y))
                if (d < outer.second * 0.25) { count++; used[j] = true }
            }
            if (count in 2..5) {
                tags.add(Pair(outer.first, count))
                val r = outer.second.toFloat()
                val corners = listOf(
                    Pair(outer.first.x.toFloat() - r, outer.first.y.toFloat() - r),
                    Pair(outer.first.x.toFloat() + r, outer.first.y.toFloat() - r),
                    Pair(outer.first.x.toFloat() + r, outer.first.y.toFloat() + r),
                    Pair(outer.first.x.toFloat() - r, outer.first.y.toFloat() + r)
                )
                detections.add(MarkerDetection.CCTag(count, Pair(outer.first.x.toFloat(), outer.first.y.toFloat()), r, corners))
            }
        }
        for (t in tags) {
            Imgproc.circle(res, t.first, 10, Scalar(0.0, 255.0, 255.0, 255.0), -1)
            Imgproc.putText(res, "CCTag (${t.second})", Point(t.first.x+15, t.first.y+5), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255.0, 255.0, 255.0, 255.0), 2)
        }
        if (detections.isNotEmpty()) onMarkersDetected?.invoke(detections)
        gray.release(); thresh.release(); hierarchy.release(); contours.forEach { it.release() }; return res
    }

    private fun applyChessboardCalibration(src: Mat): Mat {
        val res = src.clone(); val gray = Mat(); Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        val corners = MatOfPoint2f(); val pattern = Size(9.0, 6.0); val found = Calib3d.findChessboardCorners(gray, pattern, corners)
        if (found) {
            Imgproc.cornerSubPix(gray, corners, Size(11.0, 11.0), Size(-1.0, -1.0), TermCriteria(TermCriteria.EPS + TermCriteria.COUNT, 30, 0.1))
            Calib3d.drawChessboardCorners(res, pattern, corners, true)
            Imgproc.putText(res, "Board Detected", Point(30.0, 40.0), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0.0, 255.0, 0.0, 255.0), 2)
        } else Imgproc.putText(res, labelBoardNotFound, Point(30.0, 40.0), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255.0, 0.0, 0.0, 255.0), 2)
        calibrator?.storeDetectedCorners(
            if (found) corners else null,
            Size(src.cols().toDouble(), src.rows().toDouble()),
        )
        Imgproc.putText(res, "${calibrator?.frameCount ?: 0} $labelFrameCountSuffix", Point(30.0, 80.0), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255.0, 255.0, 255.0, 255.0), 2)
        gray.release(); corners.release(); return res
    }

    private fun applyUndistort(src: Mat): Mat {
        val r = calibrator?.calibrationResult; val m = r?.cameraMatrix; val d = r?.distCoeffs
        if (m == null) {
            val out = src.clone(); Imgproc.putText(out, labelNoCalibration, Point(30.0, 60.0), Imgproc.FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255.0, 0.0, 0.0, 255.0), 3)
            return out
        }
        val out = Mat(); Calib3d.undistort(src, out, m, d); return out
    }

    private fun applyVisualOdometry(src: Mat): Mat {
        val res = src.clone(); val state = visualOdometryEngine.updateOdometry(src) ?: return res
        Imgproc.putText(res, "$labelOdometryTracks: ${state.tracksCount} (inliers: ${state.inliersCount})", Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0.0, 255.0, 0.0, 255.0), 2)
        Imgproc.putText(res, "Move: %.2f | Rot: %.1f deg".format(state.translationNorm, state.rotationDeg), Point(30.0, 90.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0.0, 255.0, 0.0, 255.0), 2)
        val cx = res.cols()/2; val cy = res.rows()/2
        Imgproc.line(res, Point(cx-30.0, cy.toDouble()), Point(cx+30.0, cy.toDouble()), Scalar(255.0, 255.0, 255.0, 255.0), 2)
        Imgproc.line(res, Point(cx.toDouble(), cy-30.0), Point(cx.toDouble(), cy+30.0), Scalar(255.0, 255.0, 255.0, 255.0), 2)
        return res
    }

    private fun applyPointCloud(src: Mat): Mat {
        val res = Mat.zeros(src.size(), src.type()); val state = visualOdometryEngine.updatePointCloud(src) ?: return res
        Imgproc.putText(res, "$labelPointCloud: ${state.points.size} (parallax: %.1f)".format(state.meanParallax), Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255.0, 255.0, 255.0, 255.0), 2)
        if (isVoMeshEnabled) {
            for (e in state.edges) Imgproc.line(res, e.first, e.second, Scalar(100.0, 100.0, 100.0, 255.0), 1)
        }
        for (p in state.points) {
            val b = 150.0 + 105.0 * (1.0 - p.y / src.rows())
            Imgproc.circle(res, p, 2, Scalar(b, b, 255.0, 255.0), -1)
        }
        return res
    }

    private fun applyPlaneDetection(src: Mat): Mat {
        val res = src.clone()
        val gray = Mat(); val blurred = Mat(); val edges = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
        Imgproc.Canny(blurred, edges, 50.0, 150.0)

        val lines = Mat()
        Imgproc.HoughLinesP(edges, lines, 1.0, Math.PI / 180.0, 50, 30.0, 10.0)

        val clusters = ArrayList<ArrayList<IntArray>>()
        val clusterAngles = ArrayList<Double>()
        for (i in 0 until lines.rows()) {
            val seg = lines.get(i, 0)
            val x1 = seg[0].toInt(); val y1 = seg[1].toInt()
            val x2 = seg[2].toInt(); val y2 = seg[3].toInt()
            val angle = Math.toDegrees(atan2((y2 - y1).toDouble(), (x2 - x1).toDouble())).let { if (it < 0) it + 180.0 else it } % 180.0
            var assigned = false
            for (k in clusterAngles.indices) {
                var diff = abs(angle - clusterAngles[k]); diff = minOf(diff, 180.0 - diff)
                if (diff <= 5.0) { clusters[k].add(intArrayOf(x1, y1, x2, y2)); assigned = true; break }
            }
            if (!assigned) { clusters.add(arrayListOf(intArrayOf(x1, y1, x2, y2))); clusterAngles.add(angle) }
        }

        val planeColors = arrayOf(Scalar(0.0, 255.0, 0.0, 255.0), Scalar(0.0, 0.0, 255.0, 255.0), Scalar(0.0, 165.0, 255.0, 255.0))
        val sortedClusters = clusters.sortedByDescending { it.size }.take(MAX_LINE_DIRECTION_CLUSTERS)
        var planeIdx = 0
        for (i in sortedClusters.indices) {
            for (j in i + 1 until sortedClusters.size) {
                if (planeIdx >= 3) break
                val c1 = sortedClusters[i]; val c2 = sortedClusters[j]
                if (c1.size + c2.size < 5) continue
                val color = planeColors[planeIdx % planeColors.size]
                for (seg in c1) Imgproc.line(res, Point(seg[0].toDouble(), seg[1].toDouble()), Point(seg[2].toDouble(), seg[3].toDouble()), color, 2)
                for (seg in c2) Imgproc.line(res, Point(seg[0].toDouble(), seg[1].toDouble()), Point(seg[2].toDouble(), seg[3].toDouble()), color, 2)
                val vp = _computeVanishingPoint(c1)
                if (vp != null) {
                    Imgproc.circle(res, vp, 8, color, -1)
                    val label = "P${planeIdx + 1}"
                    Imgproc.putText(res, label, Point(vp.x + 10, vp.y), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                }
                planeIdx++
            }
        }
        if (planeIdx == 0) {
            Imgproc.putText(res, "Brak płaszczyzn", Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, Scalar(200.0, 200.0, 200.0, 255.0), 2)
        }
        gray.release(); blurred.release(); edges.release(); lines.release()
        return res
    }

    private fun applyVanishingPoints(src: Mat): Mat {
        val res = src.clone()
        val gray = Mat(); val blurred = Mat(); val edges = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
        Imgproc.Canny(blurred, edges, 50.0, 150.0)

        val lines = Mat()
        Imgproc.HoughLinesP(edges, lines, 1.0, Math.PI / 180.0, 50, 30.0, 10.0)

        val clusters = ArrayList<ArrayList<IntArray>>()
        val clusterAngles = ArrayList<Double>()
        for (i in 0 until lines.rows()) {
            val seg = lines.get(i, 0)
            val x1 = seg[0].toInt(); val y1 = seg[1].toInt()
            val x2 = seg[2].toInt(); val y2 = seg[3].toInt()
            val angle = Math.toDegrees(atan2((y2 - y1).toDouble(), (x2 - x1).toDouble())).let { if (it < 0) it + 180.0 else it } % 180.0
            var assigned = false
            for (k in clusterAngles.indices) {
                var diff = abs(angle - clusterAngles[k]); diff = minOf(diff, 180.0 - diff)
                if (diff <= 5.0) { clusters[k].add(intArrayOf(x1, y1, x2, y2)); assigned = true; break }
            }
            if (!assigned) { clusters.add(arrayListOf(intArrayOf(x1, y1, x2, y2))); clusterAngles.add(angle) }
        }

        val vpColors = arrayOf(Scalar(0.0, 255.0, 0.0, 255.0), Scalar(0.0, 0.0, 255.0, 255.0), Scalar(0.0, 165.0, 255.0, 255.0), Scalar(255.0, 255.0, 0.0, 255.0))
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
            }
        }
        if (lines.rows() == 0) {
            Imgproc.putText(res, "Brak linii", Point(30.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, Scalar(200.0, 200.0, 200.0, 255.0), 2)
        }
        gray.release(); blurred.release(); edges.release(); lines.release()
        return res
    }

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
