package pl.edu.mobilecv

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.Core
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

/**
 * Applies YOLO-based detection, segmentation, pose estimation, image classification and
 * oriented-bounding-box detection filters to Android [Bitmap] frames.
 *
 * Inference is performed on-device using the OpenCV DNN module loading YOLOv8-nano
 * ONNX models.  Five modes are supported, each driven by [OpenCvFilter]:
 * - [OpenCvFilter.YOLO_DETECT]   – 80-class COCO object detection (bounding boxes).
 * - [OpenCvFilter.YOLO_SEGMENT]  – instance segmentation (bounding boxes only on CPU).
 * - [OpenCvFilter.YOLO_POSE]     – 17-keypoint human pose estimation.
 * - [OpenCvFilter.YOLO_CLASSIFY] – ImageNet-1000 image classification (top-5 overlay).
 * - [OpenCvFilter.YOLO_OBB]      – DOTAv1 oriented bounding boxes (15 aerial classes).
 *
 * Call [initialize] once after construction and [close] when the processor is no longer
 * needed.  If a required model file has not been downloaded yet, the frame is returned
 * with an informational overlay and detection is skipped gracefully.
 *
 * This class is **not thread-safe**; call all methods from the same thread or synchronise
 * externally.
 */
class YoloProcessor(private val context: Context) {

    companion object {
        private const val TAG = "YoloProcessor"

        /** ONNX model filenames stored in internal storage by [ModelDownloadManager]. */
        const val MODEL_DETECT = "yolov8n_det.onnx"
        const val MODEL_SEGMENT = "yolov8n_seg.onnx"
        const val MODEL_POSE = "yolov8n_pose.onnx"
        const val MODEL_CLASSIFY = "yolov8n_cls.onnx"
        const val MODEL_OBB = "yolov8n_obb.onnx"

        /** YOLOv8 inference input size (square) for detection/segmentation/pose/OBB. */
        private const val INPUT_SIZE = 640

        /** YOLOv8-classify inference input size (square). */
        private const val CLS_INPUT_SIZE = 224

        /** Minimum detection confidence. */
        private const val CONFIDENCE_THRESHOLD = 0.5f

        /** IoU threshold for Non-Maximum Suppression. */
        private const val NMS_THRESHOLD = 0.45f

        /** Number of object classes in the COCO vocabulary. */
        private const val NUM_CLASSES = 80

        /** Number of output classes for YOLOv8-nano classify (ImageNet-1000). */
        private const val NUM_IMAGENET_CLASSES = 1000

        /** Number of classes in the DOTAv1 dataset used by the YOLOv8-nano OBB model. */
        private const val NUM_DOTA_CLASSES = 15

        /** Number of keypoints in YOLOv8-pose output (COCO-person: 17 joints). */
        private const val NUM_KEYPOINTS = 17

        /** Minimum keypoint visibility score to draw a joint. */
        private const val KEYPOINT_VISIBILITY_THRESHOLD = 0.5f

        // Drawing constants
        private const val BOX_THICKNESS = 4f
        private const val TEXT_SIZE = 36f
        private const val KEYPOINT_RADIUS = 6f
        private const val SKELETON_THICKNESS = 3f
        private const val LABEL_BG_ALPHA = 200

        /** COCO keypoint skeleton connections (0-indexed). */
        private val SKELETON = listOf(
            0 to 1, 0 to 2, 1 to 3, 2 to 4,       // head
            5 to 6,                                  // shoulders
            5 to 7, 7 to 9,                          // left arm
            6 to 8, 8 to 10,                         // right arm
            5 to 11, 6 to 12,                        // torso
            11 to 12,                                // hips
            11 to 13, 13 to 15,                      // left leg
            12 to 14, 14 to 16,                      // right leg
        )

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

        /** 15 DOTAv1 class names in canonical order (used by YOLOv8-nano OBB). */
        private val DOTA_CLASSES = arrayOf(
            "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
            "basketball-court", "ground-track-field", "harbor", "bridge", "large-vehicle",
            "small-vehicle", "helicopter", "roundabout", "soccer-ball-field", "swimming-pool",
        )

        /** Number of top-N predictions shown in classify mode. */
        private const val CLASSIFY_TOP_N = 5

        /** Font size for the classification result overlay text. */
        private const val CLASSIFY_TEXT_SIZE = 42f

        /** Number of features per proposal in the OBB output tensor (4 xywh + 1 angle + classes). */
        private const val OBB_BBOX_DIM = 5
    }

    // -------------------------------------------------------------------------
    // Internal state
    // -------------------------------------------------------------------------

    private var netDetect: Net? = null
    private var netSegment: Net? = null
    private var netPose: Net? = null
    private var netClassify: Net? = null
    private var netObb: Net? = null

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    /**
     * Load all available YOLO ONNX models from internal storage.
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
        netSegment = tryLoadNet(MODEL_SEGMENT)
        netPose = tryLoadNet(MODEL_POSE)
        netClassify = tryLoadNet(MODEL_CLASSIFY)
        netObb = tryLoadNet(MODEL_OBB)
    }

    /**
     * Release all loaded networks and free associated native memory.
     *
     * After calling this method the processor must not be used until
     * [initialize] is called again.
     */
    fun close() {
        netDetect = null
        netSegment = null
        netPose = null
        netClassify = null
        netObb = null
        Log.d(TAG, "YoloProcessor closed")
    }

    // -------------------------------------------------------------------------
    // Frame processing
    // -------------------------------------------------------------------------

    /**
     * Apply the YOLO filter specified by [filter] to [bitmap] and return the
     * annotated result.
     *
     * @param bitmap Input camera frame (any config, will be converted to ARGB_8888).
     * @param filter One of [OpenCvFilter.YOLO_DETECT], [OpenCvFilter.YOLO_SEGMENT],
     *               [OpenCvFilter.YOLO_POSE], [OpenCvFilter.YOLO_CLASSIFY] or
     *               [OpenCvFilter.YOLO_OBB].
     * @param onDetections Optional callback invoked with YOLO detections for ROS
     *                     publishing.  Called only when detections are non-empty.
     * @return Annotated [Bitmap] in ARGB_8888 format.
     */
    fun processFrame(
        bitmap: Bitmap,
        filter: OpenCvFilter,
        onDetections: ((List<MarkerDetection>) -> Unit)? = null,
    ): Bitmap {
        return when (filter) {
            OpenCvFilter.YOLO_DETECT -> applyDetection(bitmap, onDetections)
            OpenCvFilter.YOLO_SEGMENT -> applySegmentation(bitmap, onDetections)
            OpenCvFilter.YOLO_POSE -> applyPose(bitmap, onDetections)
            OpenCvFilter.YOLO_CLASSIFY -> applyClassify(bitmap)
            OpenCvFilter.YOLO_OBB -> applyObb(bitmap, onDetections)
            else -> bitmap.copy(Bitmap.Config.ARGB_8888, false)
        }
    }

    // -------------------------------------------------------------------------
    // Private – filter implementations
    // -------------------------------------------------------------------------

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
        net.setInput(blob)
        val output = net.forward()
        src.release(); blob.release()

        // YOLOv8-detect output shape: [1, 84, 8400]
        // Reshape to [8400, 84] then transpose is not needed; iterate columns.
        val numBoxes = output.size(2)
        val numAttribs = output.size(1) // 4 + NUM_CLASSES

        val boxes = ArrayList<Rect2d>()
        val scores = ArrayList<Float>()
        val classIds = ArrayList<Int>()

        for (i in 0 until numBoxes) {
            val cx = output.get(0, 0)[i]
            val cy = output.get(0, 1)[i]
            val w = output.get(0, 2)[i]
            val h = output.get(0, 3)[i]

            var maxScore = 0f
            var maxClassId = 0
            for (c in 0 until numAttribs - 4) {
                val score = output.get(0, 4 + c)[i].toFloat()
                if (score > maxScore) { maxScore = score; maxClassId = c }
            }

            if (maxScore < CONFIDENCE_THRESHOLD) continue

            val x1 = (cx - w / 2.0)
            val y1 = (cy - h / 2.0)
            boxes.add(Rect2d(x1, y1, w, h))
            scores.add(maxScore)
            classIds.add(maxClassId)
        }

        output.release()
        val kept = applyNms(boxes, scores)

        val result = ensureArgb8888(bitmap)
        val canvas = Canvas(result)
        val detections = ArrayList<MarkerDetection>()

        for (idx in kept) {
            val box = boxes[idx]
            val classId = classIds[idx]
            val score = scores[idx]
            val label = COCO_CLASSES.getOrElse(classId) { classId.toString() }
            val color = CLASS_COLORS[classId % CLASS_COLORS.size]

            val rx1 = (box.x * scaleX).toFloat()
            val ry1 = (box.y * scaleY).toFloat()
            val rx2 = ((box.x + box.width) * scaleX).toFloat()
            val ry2 = ((box.y + box.height) * scaleY).toFloat()
            val rectF = RectF(rx1, ry1, rx2, ry2)

            val detection = MarkerDetection.YoloObject(label, classId, score, rectF)
            drawBox(canvas, rectF, "${detection.type}:${detection.id}", score, color)
            logMarkerDiagnostics(detection)
            detections.add(detection)
        }

        if (detections.isNotEmpty()) onDetections?.invoke(detections)
        return result
    }

    private fun applySegmentation(
        bitmap: Bitmap,
        onDetections: ((List<MarkerDetection>) -> Unit)?,
    ): Bitmap {
        val net = netSegment ?: return drawModelMissing(bitmap, MODEL_SEGMENT)
        val (src, scaleX, scaleY) = bitmapToSquareMat(bitmap)

        val blob = Dnn.blobFromImage(
            src, 1.0 / 255.0, Size(INPUT_SIZE.toDouble(), INPUT_SIZE.toDouble()),
            org.opencv.core.Scalar(0.0, 0.0, 0.0), true, false,
        )
        net.setInput(blob)

        // YOLOv8-seg outputs: [output0: 1×116×8400, output1: 1×32×160×160]
        // For CPU performance, we only use output0 for bounding boxes.
        val outputNames = net.getUnconnectedOutLayersNames()
        val outputs = ArrayList<Mat>()
        for (name in outputNames) outputs.add(Mat())
        net.forward(outputs, outputNames)
        src.release(); blob.release()

        val output = outputs.firstOrNull() ?: return ensureArgb8888(bitmap)
        val numBoxes = output.size(2)
        val numAttribs = output.size(1) // 4 + NUM_CLASSES + 32 mask coefficients

        val boxes = ArrayList<Rect2d>()
        val scores = ArrayList<Float>()
        val classIds = ArrayList<Int>()

        for (i in 0 until numBoxes) {
            val cx = output.get(0, 0)[i]
            val cy = output.get(0, 1)[i]
            val w = output.get(0, 2)[i]
            val h = output.get(0, 3)[i]

            var maxScore = 0f
            var maxClassId = 0
            for (c in 0 until minOf(NUM_CLASSES, numAttribs - 4)) {
                val score = output.get(0, 4 + c)[i].toFloat()
                if (score > maxScore) { maxScore = score; maxClassId = c }
            }

            if (maxScore < CONFIDENCE_THRESHOLD) continue
            boxes.add(Rect2d(cx - w / 2.0, cy - h / 2.0, w, h))
            scores.add(maxScore)
            classIds.add(maxClassId)
        }

        outputs.forEach { it.release() }
        val kept = applyNms(boxes, scores)

        val result = ensureArgb8888(bitmap)
        val canvas = Canvas(result)
        val detections = ArrayList<MarkerDetection>()

        for (idx in kept) {
            val box = boxes[idx]
            val classId = classIds[idx]
            val score = scores[idx]
            val label = COCO_CLASSES.getOrElse(classId) { classId.toString() }
            val color = CLASS_COLORS[classId % CLASS_COLORS.size]

            val rx1 = (box.x * scaleX).toFloat()
            val ry1 = (box.y * scaleY).toFloat()
            val rx2 = ((box.x + box.width) * scaleX).toFloat()
            val ry2 = ((box.y + box.height) * scaleY).toFloat()
            val rectF = RectF(rx1, ry1, rx2, ry2)

            // Semi-transparent fill to indicate segmentation mode
            val fillPaint = Paint().apply {
                this.color = CLASS_COLORS[classId % CLASS_COLORS.size]
                alpha = 60
                style = Paint.Style.FILL
            }
            canvas.drawRect(rectF, fillPaint)
            val detection = MarkerDetection.YoloObject(label, classId, score, rectF)
            drawBox(canvas, rectF, "${detection.type}:${detection.id}[seg]", score, color)
            logMarkerDiagnostics(detection)
            detections.add(detection)
        }

        if (detections.isNotEmpty()) onDetections?.invoke(detections)
        return result
    }

    private fun applyPose(
        bitmap: Bitmap,
        onDetections: ((List<MarkerDetection>) -> Unit)?,
    ): Bitmap {
        val net = netPose ?: return drawModelMissing(bitmap, MODEL_POSE)
        val (src, scaleX, scaleY) = bitmapToSquareMat(bitmap)

        val blob = Dnn.blobFromImage(
            src, 1.0 / 255.0, Size(INPUT_SIZE.toDouble(), INPUT_SIZE.toDouble()),
            org.opencv.core.Scalar(0.0, 0.0, 0.0), true, false,
        )
        net.setInput(blob)
        val output = net.forward()
        src.release(); blob.release()

        // YOLOv8-pose output shape: [1, 56, 8400]
        // 56 = 4 (bbox) + 1 (person conf) + 51 (17 keypoints × 3: x, y, visibility)
        val numBoxes = output.size(2)

        val boxes = ArrayList<Rect2d>()
        val scores = ArrayList<Float>()
        val keypointSets = ArrayList<FloatArray>() // flat: [kp0x, kp0y, kp0v, ..., kp16x, kp16y, kp16v]

        for (i in 0 until numBoxes) {
            val cx = output.get(0, 0)[i]
            val cy = output.get(0, 1)[i]
            val w = output.get(0, 2)[i]
            val h = output.get(0, 3)[i]
            val conf = output.get(0, 4)[i].toFloat()

            if (conf < CONFIDENCE_THRESHOLD) continue

            boxes.add(Rect2d(cx - w / 2.0, cy - h / 2.0, w, h))
            scores.add(conf)

            val kps = FloatArray(NUM_KEYPOINTS * 3)
            for (k in 0 until NUM_KEYPOINTS) {
                kps[k * 3 + 0] = output.get(0, 5 + k * 3)[i].toFloat()
                kps[k * 3 + 1] = output.get(0, 5 + k * 3 + 1)[i].toFloat()
                kps[k * 3 + 2] = output.get(0, 5 + k * 3 + 2)[i].toFloat()
            }
            keypointSets.add(kps)
        }

        output.release()
        val kept = applyNms(boxes, scores)

        val result = ensureArgb8888(bitmap)
        val canvas = Canvas(result)
        val detections = ArrayList<MarkerDetection>()

        val kpPaint = Paint().apply { isAntiAlias = true; strokeWidth = KEYPOINT_RADIUS }
        val bonePaint = Paint().apply {
            isAntiAlias = true; strokeWidth = SKELETON_THICKNESS; style = Paint.Style.STROKE
        }

        for (idx in kept) {
            val box = boxes[idx]
            val score = scores[idx]
            val kps = keypointSets[idx]

            val rx1 = (box.x * scaleX).toFloat()
            val ry1 = (box.y * scaleY).toFloat()
            val rx2 = ((box.x + box.width) * scaleX).toFloat()
            val ry2 = ((box.y + box.height) * scaleY).toFloat()
            val rectF = RectF(rx1, ry1, rx2, ry2)

            val detection = MarkerDetection.YoloObject("person", 0, score, rectF)
            drawBox(canvas, rectF, "${detection.type}:${detection.id}", score, Color.GREEN)

            // Draw skeleton connections
            bonePaint.color = Color.GREEN
            for ((a, b) in SKELETON) {
                val vA = kps[a * 3 + 2]
                val vB = kps[b * 3 + 2]
                if (vA < KEYPOINT_VISIBILITY_THRESHOLD || vB < KEYPOINT_VISIBILITY_THRESHOLD) continue
                val xA = kps[a * 3] * scaleX.toFloat()
                val yA = kps[a * 3 + 1] * scaleY.toFloat()
                val xB = kps[b * 3] * scaleX.toFloat()
                val yB = kps[b * 3 + 1] * scaleY.toFloat()
                canvas.drawLine(xA, yA, xB, yB, bonePaint)
            }

            // Draw keypoint circles
            for (k in 0 until NUM_KEYPOINTS) {
                val vis = kps[k * 3 + 2]
                if (vis < KEYPOINT_VISIBILITY_THRESHOLD) continue
                val kx = kps[k * 3] * scaleX.toFloat()
                val ky = kps[k * 3 + 1] * scaleY.toFloat()
                kpPaint.color = if (k < 5) Color.YELLOW else Color.CYAN
                kpPaint.style = Paint.Style.FILL
                canvas.drawCircle(kx, ky, KEYPOINT_RADIUS, kpPaint)
            }

            logMarkerDiagnostics(detection)
            detections.add(detection)
        }

        if (detections.isNotEmpty()) onDetections?.invoke(detections)
        return result
    }

    private fun applyClassify(bitmap: Bitmap): Bitmap {
        val net = netClassify ?: return drawModelMissing(bitmap, MODEL_CLASSIFY)

        // Resize to the classification input size (224×224) instead of the detection size.
        val argb = ensureArgb8888(bitmap)
        val src = Mat()
        Utils.bitmapToMat(argb, src)
        val rgb = Mat()
        Imgproc.cvtColor(src, rgb, Imgproc.COLOR_RGBA2RGB)
        src.release()

        val blob = Dnn.blobFromImage(
            rgb, 1.0 / 255.0, Size(CLS_INPUT_SIZE.toDouble(), CLS_INPUT_SIZE.toDouble()),
            org.opencv.core.Scalar(0.0, 0.0, 0.0), true, false,
        )
        rgb.release()
        net.setInput(blob)
        val output = net.forward()
        blob.release()

        // Output shape is [1, NUM_IMAGENET_CLASSES] – read all class logits.
        val numClasses = minOf(NUM_IMAGENET_CLASSES, output.total().toInt())
        val logits = FloatArray(numClasses) { i -> output.get(0, i)[0].toFloat() }
        output.release()

        // Numerical-stable softmax
        val maxLogit = logits.maxOrNull() ?: 0f
        val expLogits = FloatArray(numClasses) { i -> Math.exp((logits[i] - maxLogit).toDouble()).toFloat() }
        val sumExp = expLogits.sum()
        val probs = FloatArray(numClasses) { i -> expLogits[i] / sumExp }

        // Pick top-N by descending probability
        val topIndices = probs.indices.sortedByDescending { probs[it] }.take(CLASSIFY_TOP_N)

        val result = ensureArgb8888(bitmap)
        val canvas = Canvas(result)

        val bgPaint = Paint().apply {
            color = Color.argb(180, 0, 0, 0)
            style = Paint.Style.FILL
        }
        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = CLASSIFY_TEXT_SIZE
            isAntiAlias = true
        }
        val headerPaint = Paint().apply {
            color = Color.YELLOW
            textSize = CLASSIFY_TEXT_SIZE
            isFakeBoldText = true
            isAntiAlias = true
        }

        val lineH = CLASSIFY_TEXT_SIZE + 8f
        val boxW = result.width * 0.85f
        val boxH = lineH * (CLASSIFY_TOP_N + 1) + 16f
        canvas.drawRect(16f, 16f, 16f + boxW, 16f + boxH, bgPaint)
        canvas.drawText("Top-$CLASSIFY_TOP_N (ImageNet-1000):", 24f, 16f + lineH, headerPaint)

        topIndices.forEachIndexed { rank, classIdx ->
            val pct = "%.1f".format(probs[classIdx] * 100)
            canvas.drawText(
                "${rank + 1}. class $classIdx  $pct%",
                24f, 16f + lineH * (rank + 2), textPaint,
            )
        }

        return result
    }

    private fun applyObb(
        bitmap: Bitmap,
        onDetections: ((List<MarkerDetection>) -> Unit)?,
    ): Bitmap {
        val net = netObb ?: return drawModelMissing(bitmap, MODEL_OBB)
        val (src, scaleX, scaleY) = bitmapToSquareMat(bitmap)

        val blob = Dnn.blobFromImage(
            src, 1.0 / 255.0, Size(INPUT_SIZE.toDouble(), INPUT_SIZE.toDouble()),
            org.opencv.core.Scalar(0.0, 0.0, 0.0), true, false,
        )
        net.setInput(blob)
        val output = net.forward()
        src.release(); blob.release()

        // YOLOv8-obb output shape: [1, OBB_BBOX_DIM + NUM_DOTA_CLASSES, num_proposals]
        // Channels: [cx, cy, w, h, angle, cls_0 … cls_14]
        val numProposals = output.size(2)
        val numFeatures = output.size(1)

        val boxes = ArrayList<Rect2d>()
        val scores = ArrayList<Float>()
        val classIds = ArrayList<Int>()
        val angles = ArrayList<Float>()

        for (i in 0 until numProposals) {
            val cx = output.get(0, 0)[i]
            val cy = output.get(0, 1)[i]
            val w = output.get(0, 2)[i]
            val h = output.get(0, 3)[i]
            val angle = output.get(0, 4)[i].toFloat()

            var maxScore = 0f
            var maxClassId = 0
            for (c in 0 until minOf(NUM_DOTA_CLASSES, numFeatures - OBB_BBOX_DIM)) {
                val score = output.get(0, OBB_BBOX_DIM + c)[i].toFloat()
                if (score > maxScore) { maxScore = score; maxClassId = c }
            }

            if (maxScore < CONFIDENCE_THRESHOLD) continue

            boxes.add(Rect2d(cx - w / 2.0, cy - h / 2.0, w, h))
            scores.add(maxScore)
            classIds.add(maxClassId)
            angles.add(angle)
        }

        output.release()
        val kept = applyNms(boxes, scores)

        val result = ensureArgb8888(bitmap)
        val canvas = Canvas(result)
        val detections = ArrayList<MarkerDetection>()

        for (idx in kept) {
            val box = boxes[idx]
            val classId = classIds[idx]
            val score = scores[idx]
            val angleDeg = Math.toDegrees(angles[idx].toDouble()).toFloat()
            val label = DOTA_CLASSES.getOrElse(classId) { classId.toString() }
            val color = CLASS_COLORS[classId % CLASS_COLORS.size]

            val cx = ((box.x + box.width / 2.0) * scaleX).toFloat()
            val cy = ((box.y + box.height / 2.0) * scaleY).toFloat()
            val rw = (box.width * scaleX).toFloat()
            val rh = (box.height * scaleY).toFloat()

            val rectF = RectF(cx - rw / 2f, cy - rh / 2f, cx + rw / 2f, cy + rh / 2f)

            canvas.save()
            canvas.rotate(angleDeg, cx, cy)
            val boxPaint = Paint().apply {
                this.color = color
                strokeWidth = BOX_THICKNESS
                style = Paint.Style.STROKE
                isAntiAlias = true
            }
            canvas.drawRect(rectF, boxPaint)
            canvas.restore()

            // Draw label at (non-rotated) bounding box top-left for readability
            val detection = MarkerDetection.YoloObject(label, classId, score, rectF)
            drawBox(canvas, rectF, "${detection.type}:${detection.id}[obb]", score, color)
            logMarkerDiagnostics(detection)
            detections.add(detection)
        }

        if (detections.isNotEmpty()) onDetections?.invoke(detections)
        return result
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /**
     * Load an ONNX model from internal storage.
     *
     * Returns ``null`` when the file has not been downloaded yet, allowing the
     * caller to display a "model missing" overlay gracefully.
     */
    private fun tryLoadNet(filename: String): Net? {
        val path = ModelDownloadManager.getYoloModelPath(context, filename) ?: run {
            Log.d(TAG, "YOLO model not available: $filename")
            return null
        }
        return try {
            val net = Dnn.readNetFromONNX(path)
            net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(Dnn.DNN_TARGET_CPU)
            Log.i(TAG, "Loaded YOLO model: $filename")
            net
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load YOLO model: $filename", e)
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

    /** Apply NMS using OpenCV and return the list of kept indices. */
    private fun applyNms(boxes: List<Rect2d>, scores: List<Float>): List<Int> {
        if (boxes.isEmpty()) return emptyList()
        val matBoxes = MatOfRect2d(*boxes.toTypedArray())
        val matScores = MatOfFloat(*scores.toFloatArray())
        val indices = MatOfInt()
        Dnn.NMSBoxes(matBoxes, matScores, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices)
        val result = indices.toArray().toList()
        matBoxes.release(); matScores.release(); indices.release()
        return result
    }

    /**
     * Draw a bounding-box rectangle, filled label background and text onto [canvas].
     */
    private fun drawBox(canvas: Canvas, rect: RectF, label: String, confidence: Float, color: Int) {
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
        canvas.drawText("Select YOLO tab to download", 30f, 110f, paint)
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
