package pl.edu.mobilecv

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.Log
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfRect2d
import org.opencv.core.Rect2d
import org.opencv.dnn.Dnn
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import pl.edu.mobilecv.util.BBoxKalmanFilter
import java.util.Locale
import kotlin.math.exp

/**
 * Applies YOLO-based detection, segmentation, pose estimation, image classification and
 * oriented-bounding-box detection filters to Android [Bitmap] frames.
 */
class YoloProcessor(private val context: Context) {

    companion object {
        private const val TAG = "YoloProcessor"

        const val MODEL_DETECT = "yolov8n.torchscript"
        const val MODEL_SEGMENT = "yolov8n-seg.torchscript"
        const val MODEL_POSE = "yolov8n-pose.torchscript"
        const val MODEL_CLASSIFY = "yolov8n-cls.torchscript"
        const val MODEL_OBB = "yolov8n-obb.torchscript"

        private const val INPUT_SIZE = 640
        private const val CLS_INPUT_SIZE = 224

        private const val CONFIDENCE_THRESHOLD = 0.45f
        private const val NMS_THRESHOLD = 0.45f

        private const val NUM_CLASSES = 80
        private const val NUM_IMAGENET_CLASSES = 1000
        private const val NUM_DOTA_CLASSES = 15
        private const val NUM_KEYPOINTS = 17
        private const val KEYPOINT_VISIBILITY_THRESHOLD = 0.5f

        private const val BOX_THICKNESS = 3f
        private const val TEXT_SIZE = 36f
        private const val KEYPOINT_RADIUS = 5f
        private const val SKELETON_THICKNESS = 4f
        private const val LABEL_BG_ALPHA = 200

        private val SKELETON = listOf(
            Pair(16, 14), Pair(14, 12), Pair(17, 15), Pair(15, 13), Pair(12, 13),
            Pair(6, 12), Pair(7, 13), Pair(6, 7), Pair(6, 8), Pair(7, 9),
            Pair(8, 10), Pair(9, 11), Pair(2, 3), Pair(1, 2), Pair(1, 3),
            Pair(2, 4), Pair(3, 5), Pair(4, 6), Pair(5, 7)
        )

        private val CLASS_COLORS = intArrayOf(
            Color.RED, Color.GREEN, Color.BLUE, Color.CYAN, Color.MAGENTA, Color.YELLOW,
            Color.rgb(255, 128, 0), Color.rgb(128, 0, 255), Color.rgb(0, 255, 128),
            Color.rgb(255, 0, 128), Color.rgb(128, 255, 0), Color.rgb(0, 128, 255)
        )

        private val COCO_CLASSES = arrayOf(
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "Kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        )

        private val DOTA_CLASSES = arrayOf(
            "plane", "ship", "storage tank", "baseball diamond", "tennis court",
            "basketball court", "ground track field", "harbor", "bridge", "large vehicle",
            "small vehicle", "helicopter", "roundabout", "soccer ball field", "swimming pool"
        )

        private const val CLASSIFY_TOP_N = 5
        private const val CLASSIFY_TEXT_SIZE = 48f

        private const val OBB_BBOX_DIM = 5 // cx, cy, w, h, angle
    }

    // -------------------------------------------------------------------------
    // Internal state
    // -------------------------------------------------------------------------

    private var netDetect: Module? = null
    private var netSegment: Module? = null
    private var netPose: Module? = null
    private var netClassify: Module? = null
    private var netObb: Module? = null

    // Tracking for YOLO_KALMAN mode
    private data class TrackedObject(
        val classId: Int,
        val label: String,
        val color: Int,
        var score: Float,
        val kalman: BBoxKalmanFilter,
        var rect: RectF,
        var framesSinceSeen: Int = 0
    )

    private val activeTracks = mutableListOf<TrackedObject>()
    private var lastFilter: OpenCvFilter? = null

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    fun initialize() {
        netDetect = tryLoadNet(MODEL_DETECT)
        netSegment = tryLoadNet(MODEL_SEGMENT)
        netPose = tryLoadNet(MODEL_POSE)
        netClassify = tryLoadNet(MODEL_CLASSIFY)
        netObb = tryLoadNet(MODEL_OBB)
    }

    fun close() {
        netDetect = null
        netSegment = null
        netPose = null
        netClassify = null
        netObb = null
        activeTracks.clear()
        Log.d(TAG, "YoloProcessor closed")
    }

    // -------------------------------------------------------------------------
    // Frame processing
    // -------------------------------------------------------------------------

    fun processFrame(bitmap: Bitmap, filter: OpenCvFilter): Bitmap {
        if (filter != OpenCvFilter.YOLO_KALMAN && lastFilter == OpenCvFilter.YOLO_KALMAN) {
            activeTracks.clear()
        }
        lastFilter = filter

        return try {
            when (filter) {
                OpenCvFilter.YOLO_DETECT, OpenCvFilter.YOLO_KALMAN ->
                    applyDetection(bitmap, filter == OpenCvFilter.YOLO_KALMAN)
                OpenCvFilter.YOLO_SEGMENT -> applySegmentation(bitmap)
                OpenCvFilter.YOLO_POSE -> applyPose(bitmap)
                OpenCvFilter.YOLO_CLASSIFY -> applyClassify(bitmap)
                OpenCvFilter.YOLO_OBB -> applyObb(bitmap)
                else -> bitmap.copy(Bitmap.Config.ARGB_8888, false)
            }
        } catch (e: Exception) {
            Log.e(TAG, "YOLO processing failed for filter=${filter.name}", e)
            drawModuleError(bitmap, "YOLO error: ${filter.displayName}")
        }
    }

    private fun applyDetection(bitmap: Bitmap, useKalman: Boolean = false): Bitmap {
        val net = netDetect ?: return drawModelMissing(bitmap, MODEL_DETECT)
        val (inputTensor, scaleX, scaleY) = bitmapToInputTensor(bitmap, INPUT_SIZE)

        val outputTensor = runForward(net, inputTensor)
        val outputData = outputTensor.dataAsFloatArray
        val shape = outputTensor.shape()
        val numBoxes = shape[2].toInt()
        val numAttribs = shape[1].toInt()

        val boxes = ArrayList<Rect2d>()
        val scores = ArrayList<Float>()
        val classIds = ArrayList<Int>()

        for (i in 0 until numBoxes) {
            val cx = outputData[0 * numBoxes + i].toDouble()
            val cy = outputData[1 * numBoxes + i].toDouble()
            val w = outputData[2 * numBoxes + i].toDouble()
            val h = outputData[3 * numBoxes + i].toDouble()

            var maxScore = 0f
            var maxClassId = 0
            for (c in 0 until numAttribs - 4) {
                val score = outputData[(4 + c) * numBoxes + i]
                if (score > maxScore) { maxScore = score; maxClassId = c }
            }

            if (maxScore < CONFIDENCE_THRESHOLD) continue

            boxes.add(Rect2d(cx - w / 2.0, cy - h / 2.0, w, h))
            scores.add(maxScore)
            classIds.add(maxClassId)
        }

        val kept = applyNms(boxes, scores)
        val result = ensureArgb8888(bitmap)
        val canvas = Canvas(result)

        if (!useKalman) {
            for (idx in kept) {
                val box = boxes[idx]
                val classId = classIds[idx]
                val score = scores[idx]
                val label = COCO_CLASSES.getOrElse(classId) { classId.toString() }
                val color = CLASS_COLORS[classId % CLASS_COLORS.size]
                val rectF = RectF((box.x * scaleX).toFloat(), (box.y * scaleY).toFloat(),
                                  ((box.x + box.width) * scaleX).toFloat(), ((box.y + box.height) * scaleY).toFloat())
                drawBox(canvas, rectF, label, score, color)
            }
        } else {
            applyKalmanTracking(canvas, boxes, scores, classIds, kept, scaleX, scaleY)
        }

        return result
    }

    private fun applyKalmanTracking(canvas: Canvas, boxes: List<Rect2d>, scores: List<Float>,
                                    classIds: List<Int>, kept: List<Int>, scaleX: Double, scaleY: Double) {
        activeTracks.forEach { it.kalman.predict() }

        val currentDetections = kept.map { idx ->
            val box = boxes[idx]
            val rect = RectF(box.x.toFloat(), box.y.toFloat(),
                            (box.x + box.width).toFloat(), (box.y + box.height).toFloat())
            Triple(classIds[idx], scores[idx], rect)
        }

        val matchedDetections = mutableSetOf<Int>()
        for (track in activeTracks) {
            var bestIou = 0.3f
            var bestIdx = -1
            for (i in currentDetections.indices) {
                if (i in matchedDetections) continue
                if (currentDetections[i].first != track.classId) continue
                val iou = calculateIou(track.rect, currentDetections[i].third)
                if (iou > bestIou) { bestIou = iou; bestIdx = i }
            }

            if (bestIdx != -1) {
                matchedDetections.add(bestIdx)
                val det = currentDetections[bestIdx]
                val updated = track.kalman.update(det.third.left, det.third.top, det.third.width(), det.third.height())
                track.rect = RectF(updated[0], updated[1], updated[0] + updated[2], updated[1] + updated[3])
                track.score = det.second
                track.framesSinceSeen = 0
            } else {
                track.framesSinceSeen++
                val pred = track.kalman.predict()
                track.rect = RectF(pred[0], pred[1], pred[0] + pred[2], pred[1] + pred[3])
            }
        }

        for (i in currentDetections.indices) {
            if (i !in matchedDetections) {
                val det = currentDetections[i]
                val track = TrackedObject(det.first, COCO_CLASSES.getOrElse(det.first) { det.first.toString() },
                    CLASS_COLORS[det.first % CLASS_COLORS.size], det.second, BBoxKalmanFilter(), det.third)
                track.kalman.update(det.third.left, det.third.top, det.third.width(), det.third.height())
                activeTracks.add(track)
            }
        }

        activeTracks.removeAll { it.framesSinceSeen > 15 }
        for (track in activeTracks) {
            val rectF = RectF(track.rect.left * scaleX.toFloat(), track.rect.top * scaleY.toFloat(),
                              track.rect.right * scaleX.toFloat(), track.rect.bottom * scaleY.toFloat())
            drawBox(canvas, rectF, track.label, track.score, track.color)
        }
    }

    private fun calculateIou(r1: RectF, r2: RectF): Float {
        val intersection = RectF()
        if (!intersection.setIntersect(r1, r2)) return 0f
        val interArea = intersection.width() * intersection.height()
        val unionArea = r1.width() * r1.height() + r2.width() * r2.height() - interArea
        return if (unionArea > 0) interArea / unionArea else 0f
    }

    private fun applySegmentation(bitmap: Bitmap): Bitmap {
        val net = netSegment ?: return drawModelMissing(bitmap, MODEL_SEGMENT)
        val (inputTensor, scaleX, scaleY) = bitmapToInputTensor(bitmap, INPUT_SIZE)
        val outputTensor = runForward(net, inputTensor)
        val outputData = outputTensor.dataAsFloatArray
        val shape = outputTensor.shape()
        val numBoxes = shape[2].toInt()
        val numAttribs = shape[1].toInt()

        val result = ensureArgb8888(bitmap)
        val canvas = Canvas(result)

        for (i in 0 until numBoxes) {
            val cx = outputData[0 * numBoxes + i].toDouble()
            val cy = outputData[1 * numBoxes + i].toDouble()
            val w = outputData[2 * numBoxes + i].toDouble()
            val h = outputData[3 * numBoxes + i].toDouble()
            var maxScore = 0f
            var maxClassId = 0
            for (c in 0 until numAttribs - 4 - 32) {
                val score = outputData[(4 + c) * numBoxes + i]
                if (score > maxScore) { maxScore = score; maxClassId = c }
            }
            if (maxScore < CONFIDENCE_THRESHOLD) continue
            val rectF = RectF(((cx - w / 2.0) * scaleX).toFloat(), ((cy - h / 2.0) * scaleY).toFloat(),
                              ((cx + w / 2.0) * scaleX).toFloat(), ((cy + h / 2.0) * scaleY).toFloat())
            drawBox(canvas, rectF, COCO_CLASSES[maxClassId], maxScore, CLASS_COLORS[maxClassId % CLASS_COLORS.size])
        }
        return result
    }

    private fun applyPose(bitmap: Bitmap): Bitmap {
        val net = netPose ?: return drawModelMissing(bitmap, MODEL_POSE)
        val (inputTensor, scaleX, scaleY) = bitmapToInputTensor(bitmap, INPUT_SIZE)
        val outputTensor = runForward(net, inputTensor)
        val outputData = outputTensor.dataAsFloatArray
        val shape = outputTensor.shape()
        val numBoxes = shape[2].toInt()

        val result = ensureArgb8888(bitmap)
        val canvas = Canvas(result)
        val paint = Paint().apply { strokeWidth = SKELETON_THICKNESS; style = Paint.Style.STROKE }

        for (i in 0 until numBoxes) {
            val score = outputData[4 * numBoxes + i]
            if (score < CONFIDENCE_THRESHOLD) continue
            val cx = outputData[0 * numBoxes + i].toDouble()
            val cy = outputData[1 * numBoxes + i].toDouble()
            val w = outputData[2 * numBoxes + i].toDouble()
            val h = outputData[3 * numBoxes + i].toDouble()
            drawBox(canvas, RectF(((cx - w / 2.0) * scaleX).toFloat(), ((cy - h / 2.0) * scaleY).toFloat(),
                                  ((cx + w / 2.0) * scaleX).toFloat(), ((cy + h / 2.0) * scaleY).toFloat()), "person", score, Color.RED)

            val kpts = FloatArray(NUM_KEYPOINTS * 3)
            for (k in 0 until NUM_KEYPOINTS) {
                kpts[k * 3] = (outputData[(5 + k * 3) * numBoxes + i] * scaleX).toFloat()
                kpts[k * 3 + 1] = (outputData[(6 + k * 3) * numBoxes + i] * scaleY).toFloat()
                kpts[k * 3 + 2] = outputData[(7 + k * 3) * numBoxes + i]
            }
            for (pair in SKELETON) {
                val p1 = pair.first - 1
                val p2 = pair.second - 1
                if (kpts[p1 * 3 + 2] > KEYPOINT_VISIBILITY_THRESHOLD && kpts[p2 * 3 + 2] > KEYPOINT_VISIBILITY_THRESHOLD) {
                    paint.color = Color.RED
                    canvas.drawLine(kpts[p1 * 3], kpts[p1 * 3 + 1], kpts[p2 * 3], kpts[p2 * 3 + 1], paint)
                }
            }
        }
        return result
    }

    private fun applyClassify(bitmap: Bitmap): Bitmap {
        val net = netClassify ?: return drawModelMissing(bitmap, MODEL_CLASSIFY)
        val (inputTensor, _, _) = bitmapToInputTensor(bitmap, CLS_INPUT_SIZE)
        val outputTensor = runForward(net, inputTensor)
        val outputData = outputTensor.dataAsFloatArray
        val scores = outputData.map { exp(it) }
        val sum = scores.sum()
        val prob = scores.map { it / sum }
        val topIndices = prob.indices.sortedByDescending { prob[it] }.take(CLASSIFY_TOP_N)

        val result = ensureArgb8888(bitmap)
        val canvas = Canvas(result)
        val paint = Paint().apply { color = Color.RED; textSize = CLASSIFY_TEXT_SIZE; style = Paint.Style.FILL; isFakeBoldText = true }
        topIndices.forEachIndexed { i, idx ->
            canvas.drawText("${i + 1}. Class $idx: ${"%.2f".format(prob[idx])}", 50f, 100f + i * 60f, paint)
        }
        return result
    }

    private fun applyObb(bitmap: Bitmap): Bitmap {
        val net = netObb ?: return drawModelMissing(bitmap, MODEL_OBB)
        val (inputTensor, scaleX, scaleY) = bitmapToInputTensor(bitmap, INPUT_SIZE)
        val outputTensor = runForward(net, inputTensor)
        val outputData = outputTensor.dataAsFloatArray
        val shape = outputTensor.shape()
        val numBoxes = shape[2].toInt()
        val numAttribs = shape[1].toInt()

        val result = ensureArgb8888(bitmap)
        val canvas = Canvas(result)
        val paint = Paint().apply { color = Color.RED; strokeWidth = BOX_THICKNESS; style = Paint.Style.STROKE }

        for (i in 0 until numBoxes) {
            var maxScore = 0f
            var maxClassId = 0
            for (c in 0 until NUM_DOTA_CLASSES) {
                val score = outputData[(OBB_BBOX_DIM + c) * numBoxes + i]
                if (score > maxScore) { maxScore = score; maxClassId = c }
            }
            if (maxScore < CONFIDENCE_THRESHOLD) continue
            val cx = (outputData[0 * numBoxes + i] * scaleX).toFloat()
            val cy = (outputData[1 * numBoxes + i] * scaleY).toFloat()
            val w = (outputData[2 * numBoxes + i] * scaleX).toFloat()
            val h = (outputData[3 * numBoxes + i] * scaleY).toFloat()
            val angle = outputData[4 * numBoxes + i]

            canvas.save()
            canvas.rotate(Math.toDegrees(angle.toDouble()).toFloat(), cx, cy)
            canvas.drawRect(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, paint)
            canvas.restore()
        }
        return result
    }

    private fun tryLoadNet(path: String): Module? = try {
        Module.load(context.filesDir.absolutePath + "/" + path)
    } catch (e: Exception) {
        Log.e(TAG, "Error loading model $path: ${e.message}"); null
    }

    private fun runForward(net: Module, input: Tensor): Tensor = net.forward(IValue.from(input)).toTensor()

    private fun bitmapToInputTensor(bitmap: Bitmap, size: Int): Triple<Tensor, Double, Double> {
        val argb = if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap else bitmap.copy(Bitmap.Config.ARGB_8888, false)
        val resized = Bitmap.createScaledBitmap(argb, size, size, true)
        val pixels = IntArray(size * size)
        resized.getPixels(pixels, 0, size, 0, 0, size, size)

        val floatData = FloatArray(3 * size * size)
        for (i in 0 until size * size) {
            val p = pixels[i]
            floatData[i] = ((p shr 16) and 0xFF) / 255.0f
            floatData[size * size + i] = ((p shr 8) and 0xFF) / 255.0f
            floatData[2 * size * size + i] = (p and 0xFF) / 255.0f
        }

        val tensor = Tensor.fromBlob(floatData, longArrayOf(1, 3, size.toLong(), size.toLong()))
        return Triple(tensor, bitmap.width.toDouble() / size, bitmap.height.toDouble() / size)
    }

    private fun applyNms(boxes: List<Rect2d>, scores: List<Float>): List<Int> {
        val matBoxes = MatOfRect2d(*boxes.toTypedArray())
        val matScores = MatOfFloat(*scores.toFloatArray())
        val indices = MatOfInt()
        Dnn.NMSBoxes(matBoxes, matScores, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices)
        return indices.toList()
    }

    private fun drawBox(canvas: Canvas, rect: RectF, label: String, score: Float, color: Int) {
        val paint = Paint().apply { this.color = color; strokeWidth = BOX_THICKNESS; style = Paint.Style.STROKE }
        canvas.drawRect(rect, paint)
        val text = "$label ${"%.2f".format(score)}"
        paint.style = Paint.Style.FILL
        paint.textSize = TEXT_SIZE
        val textWidth = paint.measureText(text)
        canvas.drawRect(rect.left, rect.top - TEXT_SIZE, rect.left + textWidth, rect.top, paint)
        paint.color = Color.BLACK
        canvas.drawText(text, rect.left, rect.top - 5f, paint)
    }

    private fun drawModelMissing(bitmap: Bitmap, model: String): Bitmap {
        val result = ensureArgb8888(bitmap)
        val canvas = Canvas(result)
        val paint = Paint().apply { color = Color.RED; textSize = 40f; isFakeBoldText = true }
        canvas.drawText("Model missing: $model", 50f, 100f, paint)
        return result
    }

    private fun drawModuleError(bitmap: Bitmap, message: String): Bitmap {
        val result = ensureArgb8888(bitmap)
        val canvas = Canvas(result)
        val paint = Paint().apply { color = Color.RED; textSize = 40f; isFakeBoldText = true }
        canvas.drawText(message, 50f, 100f, paint)
        return result
    }

    private fun ensureArgb8888(bitmap: Bitmap): Bitmap =
        if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap.copy(Bitmap.Config.ARGB_8888, true)
        else bitmap.copy(Bitmap.Config.ARGB_8888, true)

    private fun logMarkerDiagnostics(detection: MarkerDetection) {
        // Log.d(TAG, detection.toDiagnosticSummary())
    }
}
