package pl.edu.mobilecv

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfRect2d
import org.opencv.core.Rect2d
import org.opencv.core.Size
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc
import androidx.core.graphics.createBitmap
import kotlin.math.cos
import kotlin.math.sin

/**
 * Applies RTMDet-based detection filters to Android [Bitmap] frames.
 *
 * Inference is performed on-device using the OpenCV DNN module loading
 * RTMDet-nano ONNX models exported via mmdeploy.  Two modes are supported,
 * each driven by [OpenCvFilter]:
 * - [OpenCvFilter.RTMDET_DETECT]  – 80-class COCO axis-aligned detection.
 * - [OpenCvFilter.RTMDET_ROTATED] – oriented bounding-box detection (RTMDet-R).
 *
 * The mmdeploy end2end export format is assumed: the ONNX model produces two
 * output tensors named ``dets`` and ``labels``.
 * - ``dets``: shape [1, K, 5] (axis-aligned: x1, y1, x2, y2, score) or
 *             shape [1, K, 6] (rotated: cx, cy, w, h, angle_rad, score).
 * - ``labels``: shape [1, K] with integer class indices.
 *
 * Call [initialize] once after construction and [close] when the processor is
 * no longer needed.  If a required model file has not been downloaded yet, the
 * frame is returned with an informational overlay and detection is skipped
 * gracefully.
 *
 * This class is **not thread-safe**; call all methods from the same thread or
 * synchronise externally.
 */
class RtmDetProcessor(private val context: Context) {

    companion object {
        private const val TAG = "RtmDetProcessor"

        /** ONNX model filenames stored in internal storage by [ModelDownloadManager]. */
        const val MODEL_DETECT = "rtmdet_nano_det.onnx"
        const val MODEL_ROTATED = "rtmdet_nano_rotated.onnx"

        /** RTMDet inference input size (square). */
        private const val INPUT_SIZE = 640

        /** Minimum detection confidence. */
        private const val CONFIDENCE_THRESHOLD = 0.3f

        /** IoU threshold for Non-Maximum Suppression (axis-aligned only). */
        private const val NMS_THRESHOLD = 0.45f

        /** Number of object classes in the COCO vocabulary. */
        private const val NUM_CLASSES = 80

        // Drawing constants
        private const val BOX_THICKNESS = 4f
        private const val TEXT_SIZE = 36f
        private const val LABEL_BG_ALPHA = 200

        /**
         * Deterministic colour palette for 80 COCO classes (ARGB).
         * Colours wrap around via modulo when class_id >= palette size.
         */
        private val CLASS_COLORS = intArrayOf(
            Color.rgb(56, 56, 255),
            Color.rgb(151, 157, 255),
            Color.rgb(31, 112, 255),
            Color.rgb(29, 178, 255),
            Color.rgb(49, 210, 207),
            Color.rgb(10, 249, 72),
            Color.rgb(23, 204, 146),
            Color.rgb(134, 219, 61),
            Color.rgb(52, 147, 26),
            Color.rgb(187, 212, 0),
            Color.rgb(168, 153, 44),
            Color.rgb(255, 194, 0),
            Color.rgb(255, 152, 0),
            Color.rgb(255, 87, 34),
            Color.rgb(244, 54, 45),
            Color.rgb(255, 48, 112),
            Color.rgb(221, 0, 255),
            Color.rgb(124, 77, 255),
            Color.rgb(157, 148, 255),
            Color.rgb(189, 151, 255),
        )

        /** 80 COCO object class names in canonical order. */
        val COCO_CLASSES = arrayOf(
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush",
        )
    }

    // -------------------------------------------------------------------------
    // Internal state
    // -------------------------------------------------------------------------

    private var netDetect: Net? = null
    private var netRotated: Net? = null

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    /**
     * Load all available RTMDet ONNX models from internal storage.
     *
     * Models that have not been downloaded yet are silently skipped; the
     * corresponding filter will display an informational overlay instead of
     * detection results when invoked.
     *
     * Always call this method from a **background thread** to avoid blocking
     * the main thread during file I/O.
     */
    fun initialize() {
        netDetect = tryLoadNet(MODEL_DETECT)
        netRotated = tryLoadNet(MODEL_ROTATED)
    }

    /**
     * Release all loaded networks and free associated native memory.
     *
     * After calling this method the processor must not be used until
     * [initialize] is called again.
     */
    fun close() {
        netDetect = null
        netRotated = null
        Log.d(TAG, "RtmDetProcessor closed")
    }

    // -------------------------------------------------------------------------
    // Frame processing
    // -------------------------------------------------------------------------

    /**
     * Apply the RTMDet filter specified by [filter] to [bitmap] and return the
     * annotated result.
     *
     * @param bitmap Input camera frame (any config, will be converted to ARGB_8888).
     * @param filter One of [OpenCvFilter.RTMDET_DETECT] or
     *               [OpenCvFilter.RTMDET_ROTATED].
     * @param onDetections Optional callback invoked with RTMDet detections for
     *                     ROS publishing.  Called only when detections are non-empty.
     * @return Annotated [Bitmap] in ARGB_8888 format.
     */
    fun processFrame(
        bitmap: Bitmap,
        filter: OpenCvFilter,
        onDetections: ((List<MarkerDetection>) -> Unit)? = null,
    ): Bitmap {
        return when (filter) {
            OpenCvFilter.RTMDET_DETECT -> applyDetection(bitmap, onDetections)
            OpenCvFilter.RTMDET_ROTATED -> applyRotatedDetection(bitmap, onDetections)
            else -> bitmap.copy(Bitmap.Config.ARGB_8888, false)
        }
    }

    // -------------------------------------------------------------------------
    // Private – filter implementations
    // -------------------------------------------------------------------------

    /**
     * Axis-aligned object detection using RTMDet-nano.
     *
     * Expected ONNX output (mmdeploy end2end export):
     * - ``dets``: shape [1, K, 5] where each row is (x1, y1, x2, y2, score).
     * - ``labels``: shape [1, K] with class indices.
     */
    private fun applyDetection(
        bitmap: Bitmap,
        onDetections: ((List<MarkerDetection>) -> Unit)?,
    ): Bitmap {
        val net = netDetect ?: return drawModelMissing(bitmap, MODEL_DETECT)
        val (src, scaleX, scaleY) = bitmapToSquareMat(bitmap)

        val blob = Dnn.blobFromImage(
            src, 1.0 / 255.0, Size(INPUT_SIZE.toDouble(), INPUT_SIZE.toDouble()),
            org.opencv.core.Scalar(0.0, 0.0, 0.0), true, false,
        )

        val outputNames = net.getUnconnectedOutLayersNames()
        val outputs = ArrayList<Mat>()
        for (ignored in outputNames) outputs.add(Mat())
        net.setInput(blob)
        net.forward(outputs, outputNames)
        src.release()
        blob.release()

        // Locate dets and labels tensors by iterating output names
        val dets = findTensor(outputs, outputNames, "dets") ?: outputs.getOrNull(0)
        val labels = findTensor(outputs, outputNames, "labels") ?: outputs.getOrNull(1)

        val result = ensureArgb8888(bitmap)
        val canvas = Canvas(result)
        val detections = ArrayList<MarkerDetection>()

        if (dets != null && labels != null) {
            val numDets = dets.size(1)
            val numAttribs = dets.size(2)  // 5 for axis-aligned

            val boxes = ArrayList<Rect2d>()
            val scores = ArrayList<Float>()
            val classIds = ArrayList<Int>()

            for (i in 0 until numDets) {
                // dets[0, i] returns float array of length numAttribs
                val attribs = dets.get(0, i)
                if (attribs == null || attribs.size < numAttribs) continue

                val score = attribs[numAttribs - 1].toFloat()
                if (score < CONFIDENCE_THRESHOLD) continue

                val x1 = attribs[0].toDouble()
                val y1 = attribs[1].toDouble()
                val x2 = attribs[2].toDouble()
                val y2 = attribs[3].toDouble()
                boxes.add(Rect2d(x1, y1, x2 - x1, y2 - y1))
                scores.add(score)

                val labelArray = labels.get(0, i)
                classIds.add(if (labelArray != null && labelArray.isNotEmpty()) labelArray[0].toInt() else 0)
            }

            val kept = applyNms(boxes, scores)
            for (idx in kept) {
                val box = boxes[idx]
                val classId = classIds[idx].coerceIn(0, NUM_CLASSES - 1)
                val score = scores[idx]
                val label = COCO_CLASSES.getOrElse(classId) { classId.toString() }
                val color = CLASS_COLORS[classId % CLASS_COLORS.size]

                val rx1 = (box.x * scaleX).toFloat()
                val ry1 = (box.y * scaleY).toFloat()
                val rx2 = ((box.x + box.width) * scaleX).toFloat()
                val ry2 = ((box.y + box.height) * scaleY).toFloat()
                val rectF = RectF(rx1, ry1, rx2, ry2)

                val detection = MarkerDetection.RtmDetObject(label, classId, score, rectF)
                drawBox(canvas, rectF, label, score, color)
                logMarkerDiagnostics(detection)
                detections.add(detection)
            }
        }

        outputs.forEach { it.release() }
        if (detections.isNotEmpty()) onDetections?.invoke(detections)
        return result
    }

    /**
     * Rotated bounding-box detection using RTMDet-nano-r (RTMDet-Rotated).
     *
     * Expected ONNX output (mmdeploy end2end export):
     * - ``dets``: shape [1, K, 6] where each row is
     *             (cx, cy, w, h, angle_rad, score).
     * - ``labels``: shape [1, K] with class indices.
     *
     * Rotated boxes are rendered as oriented rectangles on the canvas.
     */
    private fun applyRotatedDetection(
        bitmap: Bitmap,
        onDetections: ((List<MarkerDetection>) -> Unit)?,
    ): Bitmap {
        val net = netRotated ?: return drawModelMissing(bitmap, MODEL_ROTATED)
        val (src, scaleX, scaleY) = bitmapToSquareMat(bitmap)

        val blob = Dnn.blobFromImage(
            src, 1.0 / 255.0, Size(INPUT_SIZE.toDouble(), INPUT_SIZE.toDouble()),
            org.opencv.core.Scalar(0.0, 0.0, 0.0), true, false,
        )

        val outputNames = net.getUnconnectedOutLayersNames()
        val outputs = ArrayList<Mat>()
        for (ignored in outputNames) outputs.add(Mat())
        net.setInput(blob)
        net.forward(outputs, outputNames)
        src.release()
        blob.release()

        val dets = findTensor(outputs, outputNames, "dets") ?: outputs.getOrNull(0)
        val labels = findTensor(outputs, outputNames, "labels") ?: outputs.getOrNull(1)

        val result = ensureArgb8888(bitmap)
        val canvas = Canvas(result)
        val detections = ArrayList<MarkerDetection>()

        if (dets != null && labels != null) {
            val numDets = dets.size(1)
            val numAttribs = dets.size(2)  // 6 for rotated

            for (i in 0 until numDets) {
                val attribs = dets.get(0, i)
                if (attribs == null || attribs.size < numAttribs) continue

                val score = attribs[numAttribs - 1].toFloat()
                if (score < CONFIDENCE_THRESHOLD) continue

                val cx = (attribs[0] * scaleX).toFloat()
                val cy = (attribs[1] * scaleY).toFloat()
                val w = (attribs[2] * scaleX).toFloat()
                val h = (attribs[3] * scaleY).toFloat()
                val angleRad = attribs[4].toFloat()
                val angleDeg = Math.toDegrees(angleRad.toDouble()).toFloat()

                val labelArray = labels.get(0, i)
                val classId = (if (labelArray != null && labelArray.isNotEmpty()) labelArray[0].toInt() else 0)
                    .coerceIn(0, NUM_CLASSES - 1)
                val label = COCO_CLASSES.getOrElse(classId) { classId.toString() }
                val color = CLASS_COLORS[classId % CLASS_COLORS.size]

                val bboxForDetection = RectF(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                val detection = MarkerDetection.RtmDetObject(
                    label, classId, score, bboxForDetection, angleDeg
                )
                drawRotatedBox(canvas, cx, cy, w, h, angleDeg, label, score, color)
                logMarkerDiagnostics(detection)
                detections.add(detection)
            }
        }

        outputs.forEach { it.release() }
        if (detections.isNotEmpty()) onDetections?.invoke(detections)
        return result
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /**
     * Load an RTMDet ONNX model from internal storage.
     *
     * Returns ``null`` when the file has not been downloaded yet, allowing the
     * caller to display a "model missing" overlay gracefully.
     */
    private fun tryLoadNet(filename: String): Net? {
        val path = ModelDownloadManager.getRtmDetModelPath(context, filename) ?: run {
            Log.d(TAG, "RTMDet model not available: $filename")
            return null
        }
        return try {
            val net = Dnn.readNetFromONNX(path)
            net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(Dnn.DNN_TARGET_CPU)
            Log.i(TAG, "Loaded RTMDet model: $filename")
            net
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load RTMDet model: $filename", e)
            null
        }
    }

    /**
     * Convert [bitmap] to a square 640×640 OpenCV [Mat] (RGB) and return it
     * together with the x/y scale factors needed to map detections back to the
     * original bitmap dimensions.
     */
    private fun bitmapToSquareMat(bitmap: Bitmap): Triple<Mat, Double, Double> {
        val src = Mat()
        val argb = ensureArgb8888(bitmap)
        Utils.bitmapToMat(argb, src)

        val rgb = Mat()
        Imgproc.cvtColor(src, rgb, Imgproc.COLOR_RGBA2RGB)
        src.release()

        val scaleX = argb.width.toDouble() / INPUT_SIZE
        val scaleY = argb.height.toDouble() / INPUT_SIZE

        val resized = Mat()
        Imgproc.resize(rgb, resized, Size(INPUT_SIZE.toDouble(), INPUT_SIZE.toDouble()))
        rgb.release()

        return Triple(resized, scaleX, scaleY)
    }

    /**
     * Find an output tensor by name in the list of output Mats.
     *
     * @param outputs List of output Mats returned by [Net.forward].
     * @param outputNames List of output layer names in the same order.
     * @param name Name to search for (case-insensitive prefix match).
     * @return The matching [Mat], or ``null`` if not found.
     */
    private fun findTensor(
        outputs: List<Mat>,
        outputNames: List<String>,
        name: String,
    ): Mat? {
        val idx = outputNames.indexOfFirst { it.contains(name, ignoreCase = true) }
        return if (idx >= 0) outputs.getOrNull(idx) else null
    }

    /** Apply NMS using OpenCV and return the list of kept indices. */
    private fun applyNms(boxes: List<Rect2d>, scores: List<Float>): List<Int> {
        if (boxes.isEmpty()) return emptyList()
        val matBoxes = MatOfRect2d(*boxes.toTypedArray())
        val matScores = MatOfFloat(*scores.toFloatArray())
        val indices = MatOfInt()
        Dnn.NMSBoxes(matBoxes, matScores, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices)
        val result = indices.toArray().toList()
        matBoxes.release()
        matScores.release()
        indices.release()
        return result
    }

    /**
     * Draw an axis-aligned bounding-box rectangle, filled label background and
     * text onto [canvas].
     */
    private fun drawBox(
        canvas: Canvas,
        rect: RectF,
        label: String,
        confidence: Float,
        color: Int,
    ) {
        val boxPaint = Paint().apply {
            this.color = color
            strokeWidth = BOX_THICKNESS
            style = Paint.Style.STROKE
            isAntiAlias = true
        }
        canvas.drawRect(rect, boxPaint)

        val textPaint = Paint().apply {
            this.color = Color.WHITE
            textSize = TEXT_SIZE
            isAntiAlias = true
        }
        val labelText = "$label ${"%.0f".format(confidence * 100)}%"
        val textW = textPaint.measureText(labelText)
        val textH = TEXT_SIZE

        val bgPaint = Paint().apply {
            this.color = color
            alpha = LABEL_BG_ALPHA
            style = Paint.Style.FILL
        }
        val labelTop = (rect.top - textH - 4f).coerceAtLeast(0f)
        canvas.drawRect(rect.left, labelTop, rect.left + textW + 8f, labelTop + textH + 4f, bgPaint)
        canvas.drawText(labelText, rect.left + 4f, labelTop + textH, textPaint)
    }

    /**
     * Draw a rotated bounding box, label and rotation angle onto [canvas].
     *
     * The box is rendered as a quadrilateral by rotating four corner points
     * around the centre (cx, cy) by [angleDeg] degrees.
     *
     * @param canvas Target canvas.
     * @param cx Centre x in screen pixels.
     * @param cy Centre y in screen pixels.
     * @param w Box width in screen pixels.
     * @param h Box height in screen pixels.
     * @param angleDeg Rotation angle in degrees (counter-clockwise).
     * @param label Class label text.
     * @param confidence Detection confidence in [0, 1].
     * @param color ARGB box colour.
     */
    private fun drawRotatedBox(
        canvas: Canvas,
        cx: Float,
        cy: Float,
        w: Float,
        h: Float,
        angleDeg: Float,
        label: String,
        confidence: Float,
        color: Int,
    ) {
        val hw = w / 2f
        val hh = h / 2f
        val rad = Math.toRadians(angleDeg.toDouble())
        val cosA = cos(rad).toFloat()
        val sinA = sin(rad).toFloat()

        // Four corners before rotation (relative to centre)
        val cornersRel = arrayOf(
            Pair(-hw, -hh),
            Pair(hw, -hh),
            Pair(hw, hh),
            Pair(-hw, hh),
        )

        // Rotate and translate to screen coordinates
        val pts = cornersRel.map { (dx, dy) ->
            val rx = dx * cosA - dy * sinA + cx
            val ry = dx * sinA + dy * cosA + cy
            Pair(rx, ry)
        }

        val boxPaint = Paint().apply {
            this.color = color
            strokeWidth = BOX_THICKNESS
            style = Paint.Style.STROKE
            isAntiAlias = true
        }

        val path = Path().apply {
            moveTo(pts[0].first, pts[0].second)
            lineTo(pts[1].first, pts[1].second)
            lineTo(pts[2].first, pts[2].second)
            lineTo(pts[3].first, pts[3].second)
            close()
        }
        canvas.drawPath(path, boxPaint)

        // Draw label at the top-left corner of the rotated box
        val textPaint = Paint().apply {
            this.color = Color.WHITE
            textSize = TEXT_SIZE
            isAntiAlias = true
        }
        val labelText = "$label ${"%.0f".format(confidence * 100)}% ${angleDeg.toInt()}°"
        val textW = textPaint.measureText(labelText)
        val textH = TEXT_SIZE

        val bgPaint = Paint().apply {
            this.color = color
            alpha = LABEL_BG_ALPHA
            style = Paint.Style.FILL
        }
        // Place the label at the rotated top-left corner
        val lx = pts[0].first
        val ly = (pts[0].second - textH - 4f).coerceAtLeast(0f)
        canvas.drawRect(lx, ly, lx + textW + 8f, ly + textH + 4f, bgPaint)
        canvas.drawText(labelText, lx + 4f, ly + textH, textPaint)
    }

    /** Overlay a "model missing" message and return an ARGB copy of [bitmap]. */
    private fun drawModelMissing(bitmap: Bitmap, modelName: String): Bitmap {
        val result = ensureArgb8888(bitmap)
        val canvas = Canvas(result)
        val paint = Paint().apply {
            color = Color.RED
            textSize = 40f
            isAntiAlias = true
        }
        canvas.drawText("Model missing: $modelName", 30f, 60f, paint)
        canvas.drawText("Select RTMDet tab to download", 30f, 110f, paint)
        return result
    }

    /** Return an ARGB_8888 copy of [bitmap], converting if necessary. */
    private fun ensureArgb8888(bitmap: Bitmap): Bitmap =
        if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap.copy(Bitmap.Config.ARGB_8888, true)
        else bitmap.copy(Bitmap.Config.ARGB_8888, false)

    private fun logMarkerDiagnostics(detection: MarkerDetection) {
        Log.d(TAG, "marker_detection ${detection.toDiagnosticSummary()}")
    }
}
