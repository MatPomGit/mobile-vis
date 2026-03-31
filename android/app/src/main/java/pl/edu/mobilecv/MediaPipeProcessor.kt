package pl.edu.mobilecv

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import kotlin.math.sqrt

/**
 * Applies MediaPipe-based detection and tracking filters to Android [Bitmap] frames.
 *
 * Supports four detection modes driven by [OpenCvFilter]:
 * - [OpenCvFilter.HOLISTIC_BODY]  – 33 full-body pose landmarks via [PoseLandmarker].
 * - [OpenCvFilter.HOLISTIC_HANDS] – up to 42 hand landmarks (21 per hand) via [HandLandmarker].
 * - [OpenCvFilter.HOLISTIC_FACE]  – 468 face-mesh landmarks via [FaceLandmarker].
 * - [OpenCvFilter.IRIS]           – 478 refined face landmarks including iris via [FaceLandmarker].
 *
 * Call [initialize] once after construction and [close] when the processor is no longer needed.
 * If a required model file has not been downloaded yet, the corresponding detector will be
 * unavailable and the frame is returned with an informational overlay.
 *
 * This class is **not thread-safe**; call all methods from the same thread or synchronise
 * externally.
 */
class MediaPipeProcessor(private val context: Context) {

    companion object {
        private const val TAG = "MediaPipeProcessor"

        /** Model asset filenames stored in internal storage by [ModelDownloadManager]. */
        const val MODEL_POSE = "pose_landmarker_lite.task"
        const val MODEL_HAND = "hand_landmarker.task"
        const val MODEL_FACE = "face_landmarker.task"

        // ------------------------------------------------------------------
        // Drawing constants
        // ------------------------------------------------------------------

        private const val LANDMARK_RADIUS = 6f
        private const val CONNECTION_STROKE = 4f

        private val POSE_COLOR = Color.rgb(0, 220, 0)
        private val LEFT_HAND_COLOR = Color.rgb(255, 140, 0)
        private val RIGHT_HAND_COLOR = Color.rgb(0, 140, 255)
        private val FACE_COLOR = Color.rgb(180, 180, 180)
        private val IRIS_COLOR_LEFT = Color.rgb(0, 220, 0)
        private val IRIS_COLOR_RIGHT = Color.rgb(220, 0, 0)

        /** MediaPipe pose landmark connection pairs (indices into the 33-landmark list). */
        private val POSE_CONNECTIONS = listOf(
            0 to 1, 1 to 2, 2 to 3, 3 to 7, 0 to 4, 4 to 5, 5 to 6, 6 to 8,
            9 to 10, 11 to 12, 11 to 13, 13 to 15, 15 to 17, 15 to 19, 15 to 21,
            17 to 19, 12 to 14, 14 to 16, 16 to 18, 16 to 20, 16 to 22, 18 to 20,
            11 to 23, 12 to 24, 23 to 24, 23 to 25, 24 to 26, 25 to 27, 26 to 28,
            27 to 29, 28 to 30, 29 to 31, 30 to 32, 27 to 31, 28 to 32,
        )

        /** MediaPipe hand landmark connection pairs (indices into the 21-landmark list). */
        private val HAND_CONNECTIONS = listOf(
            0 to 1, 1 to 2, 2 to 3, 3 to 4,
            0 to 5, 5 to 6, 6 to 7, 7 to 8,
            0 to 9, 9 to 10, 10 to 11, 11 to 12,
            0 to 13, 13 to 14, 14 to 15, 15 to 16,
            0 to 17, 17 to 18, 18 to 19, 19 to 20,
            5 to 9, 9 to 13, 13 to 17,
        )

        /** Iris landmark index range in the refined (478) face-landmark set. */
        private val LEFT_IRIS_INDICES = 468..472
        private val RIGHT_IRIS_INDICES = 473..477
    }

    // ------------------------------------------------------------------
    // Lazy detector instances
    // ------------------------------------------------------------------

    private var poseLandmarker: PoseLandmarker? = null
    private var handLandmarker: HandLandmarker? = null
    private var faceLandmarker: FaceLandmarker? = null
    private var faceLandmarkerIris: FaceLandmarker? = null

    // ------------------------------------------------------------------
    // Paint objects (reused across frames)
    // ------------------------------------------------------------------

    private val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = CONNECTION_STROKE
    }
    private val overlayPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(255, 80, 0)
        textSize = 48f
    }

    /**
     * Initialise all available MediaPipe detectors.
     *
     * Detectors whose model file is absent (not yet downloaded) are silently skipped;
     * [processFrame] will display an info overlay for that filter instead.
     *
     * Call this method once, on the analysis executor thread, after [ModelDownloadManager]
     * has finished downloading model files.
     */
    fun initialize() {
        poseLandmarker = tryCreatePoseLandmarker()
        handLandmarker = tryCreateHandLandmarker()
        faceLandmarker = tryCreateFaceLandmarker(refineIris = false)
        faceLandmarkerIris = tryCreateFaceLandmarker(refineIris = true)
    }

    /**
     * Release all native MediaPipe resources.
     *
     * Must be called from the same thread as [initialize] and [processFrame].
     */
    fun close() {
        poseLandmarker?.close()
        handLandmarker?.close()
        faceLandmarker?.close()
        faceLandmarkerIris?.close()
        poseLandmarker = null
        handLandmarker = null
        faceLandmarker = null
        faceLandmarkerIris = null
    }

    /**
     * Process a single [bitmap] frame with the given MediaPipe [filter].
     *
     * @param bitmap ARGB_8888 bitmap to process.
     * @param filter One of the MediaPipe-specific [OpenCvFilter] values.
     * @return New ARGB_8888 bitmap with landmarks overlaid.
     */
    fun processFrame(bitmap: Bitmap, filter: OpenCvFilter): Bitmap {
        return when (filter) {
            OpenCvFilter.HOLISTIC_BODY -> applyPoseLandmarker(bitmap)
            OpenCvFilter.HOLISTIC_HANDS -> applyHandLandmarker(bitmap)
            OpenCvFilter.HOLISTIC_FACE -> applyFaceLandmarker(bitmap, iris = false)
            OpenCvFilter.IRIS -> applyFaceLandmarker(bitmap, iris = true)
            else -> bitmap.copy(Bitmap.Config.ARGB_8888, false)
        }
    }

    // ------------------------------------------------------------------
    // Private processing methods
    // ------------------------------------------------------------------

    private fun applyPoseLandmarker(bitmap: Bitmap): Bitmap {
        val detector = poseLandmarker
        if (detector == null) {
            return overlayMissingModel(bitmap, context.getString(R.string.mediapipe_model_missing_pose))
        }

        val mpImage = BitmapImageBuilder(ensureArgb8888(bitmap)).build()
        val result: PoseLandmarkerResult = detector.detect(mpImage)

        val output = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)

        for (person in result.landmarks()) {
            drawConnections(canvas, person, output.width, output.height, POSE_COLOR, POSE_CONNECTIONS)
            drawDots(canvas, person, output.width, output.height, POSE_COLOR)
        }

        return output
    }

    private fun applyHandLandmarker(bitmap: Bitmap): Bitmap {
        val detector = handLandmarker
        if (detector == null) {
            return overlayMissingModel(bitmap, context.getString(R.string.mediapipe_model_missing_hands))
        }

        val mpImage = BitmapImageBuilder(ensureArgb8888(bitmap)).build()
        val result: HandLandmarkerResult = detector.detect(mpImage)

        val output = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)

        for ((index, hand) in result.landmarks().withIndex()) {
            val handedness = result.handedness().getOrNull(index)
            val isLeft = handedness?.firstOrNull()?.categoryName()
                ?.equals("Left", ignoreCase = true) == true
            val color = if (isLeft) LEFT_HAND_COLOR else RIGHT_HAND_COLOR
            drawConnections(canvas, hand, output.width, output.height, color, HAND_CONNECTIONS)
            drawDots(canvas, hand, output.width, output.height, color)
        }

        return output
    }

    private fun applyFaceLandmarker(bitmap: Bitmap, iris: Boolean): Bitmap {
        val detector = if (iris) faceLandmarkerIris else faceLandmarker
        val missingMsg = if (iris) {
            context.getString(R.string.mediapipe_model_missing_face)
        } else {
            context.getString(R.string.mediapipe_model_missing_face)
        }

        if (detector == null) {
            return overlayMissingModel(bitmap, missingMsg)
        }

        val mpImage = BitmapImageBuilder(ensureArgb8888(bitmap)).build()
        val result: FaceLandmarkerResult = detector.detect(mpImage)

        val output = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)

        for (face in result.faceLandmarks()) {
            val baseCount = minOf(face.size, 468)
            // Draw the base 468 face-mesh landmarks as small dots.
            dotPaint.color = FACE_COLOR
            for (i in 0 until baseCount) {
                val lm = face[i]
                val px = lm.x() * output.width
                val py = lm.y() * output.height
                canvas.drawCircle(px, py, 2f, dotPaint)
            }

            if (iris && face.size >= 478) {
                drawIrisCircle(canvas, face, output.width, output.height, LEFT_IRIS_INDICES, IRIS_COLOR_LEFT)
                drawIrisCircle(canvas, face, output.width, output.height, RIGHT_IRIS_INDICES, IRIS_COLOR_RIGHT)
            }
        }

        return output
    }

    // ------------------------------------------------------------------
    // Drawing helpers
    // ------------------------------------------------------------------

    private fun drawDots(
        canvas: Canvas,
        landmarks: List<NormalizedLandmark>,
        width: Int,
        height: Int,
        color: Int,
    ) {
        dotPaint.color = color
        for (lm in landmarks) {
            canvas.drawCircle(lm.x() * width, lm.y() * height, LANDMARK_RADIUS, dotPaint)
        }
    }

    private fun drawConnections(
        canvas: Canvas,
        landmarks: List<NormalizedLandmark>,
        width: Int,
        height: Int,
        color: Int,
        connections: List<Pair<Int, Int>>,
    ) {
        linePaint.color = color
        for ((start, end) in connections) {
            if (start >= landmarks.size || end >= landmarks.size) continue
            val s = landmarks[start]
            val e = landmarks[end]
            canvas.drawLine(
                s.x() * width, s.y() * height,
                e.x() * width, e.y() * height,
                linePaint,
            )
        }
    }

    /**
     * Draw an iris circle using the 5 iris landmarks (centre + 4 contour points).
     *
     * The radius is estimated as the mean distance from the centre landmark to each
     * of the four contour landmarks.
     */
    private fun drawIrisCircle(
        canvas: Canvas,
        landmarks: List<NormalizedLandmark>,
        width: Int,
        height: Int,
        indices: IntRange,
        color: Int,
    ) {
        val center = landmarks[indices.first]
        val cx = center.x() * width
        val cy = center.y() * height

        var totalRadius = 0f
        var count = 0
        for (i in (indices.first + 1)..indices.last) {
            val pt = landmarks[i]
            val dx = pt.x() * width - cx
            val dy = pt.y() * height - cy
            totalRadius += sqrt(dx * dx + dy * dy)
            count++
        }
        val radius = if (count > 0) maxOf(4f, totalRadius / count) else 10f

        linePaint.color = color
        canvas.drawCircle(cx, cy, radius, linePaint)
        dotPaint.color = color
        canvas.drawCircle(cx, cy, LANDMARK_RADIUS, dotPaint)
    }

    /**
     * Return a copy of [bitmap] with a warning overlay when a model file is absent.
     */
    private fun overlayMissingModel(bitmap: Bitmap, message: String): Bitmap {
        val output = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)
        canvas.drawText(message, 16f, 60f, overlayPaint)
        return output
    }

    /**
     * Ensure [bitmap] uses [Bitmap.Config.ARGB_8888] as required by [BitmapImageBuilder].
     */
    private fun ensureArgb8888(bitmap: Bitmap): Bitmap =
        if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap
        else bitmap.copy(Bitmap.Config.ARGB_8888, false)

    // ------------------------------------------------------------------
    // Factory helpers
    // ------------------------------------------------------------------

    private fun tryCreatePoseLandmarker(): PoseLandmarker? {
        val modelPath = ModelDownloadManager.getModelPath(context, MODEL_POSE)
            ?: return null.also { Log.d(TAG, "Pose model not available") }
        return try {
            val options = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath(modelPath).build())
                .setRunningMode(RunningMode.IMAGE)
                .setNumPoses(1)
                .build()
            PoseLandmarker.createFromOptions(context, options)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create PoseLandmarker", e)
            null
        }
    }

    private fun tryCreateHandLandmarker(): HandLandmarker? {
        val modelPath = ModelDownloadManager.getModelPath(context, MODEL_HAND)
            ?: return null.also { Log.d(TAG, "Hand model not available") }
        return try {
            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath(modelPath).build())
                .setRunningMode(RunningMode.IMAGE)
                .setNumHands(2)
                .build()
            HandLandmarker.createFromOptions(context, options)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create HandLandmarker", e)
            null
        }
    }

    private fun tryCreateFaceLandmarker(refineIris: Boolean): FaceLandmarker? {
        val modelPath = ModelDownloadManager.getModelPath(context, MODEL_FACE)
            ?: return null.also { Log.d(TAG, "Face model not available") }
        return try {
            val options = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath(modelPath).build())
                .setRunningMode(RunningMode.IMAGE)
                .setNumFaces(1)
                .setOutputFaceBlendshapes(false)
                .build()
            FaceLandmarker.createFromOptions(context, options)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create FaceLandmarker (iris=$refineIris)", e)
            null
        }
    }
}
