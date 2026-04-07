package pl.edu.mobilecv

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.Rect
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.google.mediapipe.tasks.vision.gesturerecognizer.GestureRecognizer
import com.google.mediapipe.tasks.vision.gesturerecognizer.GestureRecognizerResult
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetectorResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import java.util.Locale
import kotlin.math.sqrt

/**
 * Applies MediaPipe-based detection and tracking filters to Android [Bitmap] frames.
 *
 * Supports five detection modes driven by [OpenCvFilter]:
 * - [OpenCvFilter.HOLISTIC_BODY]  – 33 full-body pose landmarks via [PoseLandmarker].
 * - [OpenCvFilter.HOLISTIC_HANDS] – up to 42 hand landmarks (21 per hand) via [HandLandmarker].
 * - [OpenCvFilter.HOLISTIC_FACE]  – 468 face-mesh landmarks via [FaceLandmarker].
 * - [OpenCvFilter.IRIS]           – 478 refined face landmarks including iris via [FaceLandmarker].
 * - [OpenCvFilter.HOLOGRAM_3D]    – 3-D wireframe cube rotated by the viewer's face position.
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
        const val MODEL_FACE_DETECTOR = "face_detector.task"
        const val MODEL_OBJECTRON = "object_detector_3d_shoe.task"
        const val MODEL_GESTURE = "gesture_recognizer.task"

        // ------------------------------------------------------------------
        // Drawing constants
        // ------------------------------------------------------------------

        private const val POSE_COLOR = Color.RED
        private const val LEFT_HAND_COLOR = Color.RED
        private const val RIGHT_HAND_COLOR = Color.RED
        private const val FACE_COLOR = Color.RED
        private const val IRIS_COLOR_LEFT = Color.RED
        private const val IRIS_COLOR_RIGHT = Color.RED

        private const val LANDMARK_RADIUS = 4f
        private const val LINE_WIDTH = 3f

        /** Jaw open blendshape score thresholds for mouth-state detection. */
        private const val MOUTH_OPEN_THRESHOLD = 0.15f
        private const val MOUTH_TALKING_THRESHOLD = 0.30f

        /**
         * Multiplier applied to the normalised iris-vs-eye-centre offset when drawing gaze lines.
         * A value of 8 makes the line 8× longer than the raw offset, improving visibility.
         */
        private const val GAZE_LINE_SCALE = 8f
        private const val GAZE_LINE_WIDTH = 5f
        private const val INFO_TEXT_SIZE = 42f
        private const val INFO_BADGE_MARGIN = 8f
        private const val INFO_BADGE_PADDING = 10f

        // ------------------------------------------------------------------
        // Named eye / iris landmark indices (MediaPipe 478-point face mesh)
        // ------------------------------------------------------------------

        /** Lateral (temporal) corner of the left eye. */
        private const val LEFT_EYE_OUTER = 33

        /** Medial (nasal) corner of the left eye. */
        private const val LEFT_EYE_INNER = 133

        /** Lateral (temporal) corner of the right eye. */
        private const val RIGHT_EYE_OUTER = 263

        /** Medial (nasal) corner of the right eye. */
        private const val RIGHT_EYE_INNER = 362

        /** Centre landmark of the left iris (index 468 in the 478-point mesh). */
        private const val LEFT_IRIS_CENTER = 468

        /** Centre landmark of the right iris (index 473 in the 478-point mesh). */
        private const val RIGHT_IRIS_CENTER = 473

        /** MediaPipe Pose landmarks connections (subset for visualization). */
        private val POSE_CONNECTIONS = listOf(
            0 to 1, 1 to 2, 2 to 3, 3 to 7, 0 to 4, 4 to 5, 5 to 6, 6 to 8,
            9 to 10, 11 to 12, 11 to 13, 13 to 15, 12 to 14, 14 to 16,
            11 to 23, 12 to 24, 23 to 24, 23 to 25, 25 to 27, 27 to 29, 29 to 31,
            24 to 26, 26 to 28, 28 to 30, 30 to 32, 27 to 31, 28 to 32,
        )

        /** MediaPipe Hand landmarks connections. */
        private val HAND_CONNECTIONS = listOf(
            0 to 1, 1 to 2, 2 to 3, 3 to 4,
            0 to 5, 5 to 6, 6 to 7, 7 to 8,
            5 to 9, 9 to 10, 10 to 11, 11 to 12,
            9 to 13, 13 to 14, 14 to 15, 15 to 16,
            13 to 17, 17 to 18, 18 to 19, 19 to 20,
            0 to 17,
        )

        /** Indices for iris landmarks in the 478 face mesh. */
        private val LEFT_IRIS_INDICES = 468..472
        private val RIGHT_IRIS_INDICES = 473..477

        // ------------------------------------------------------------------
        // Hologram 3-D rendering constants
        // ------------------------------------------------------------------

        /**
         * MediaPipe face-mesh nose-tip landmark index.
         * Used to derive the viewer's gaze direction relative to the screen centre.
         */
        private const val HOLOGRAM_NOSE_TIP = 4

        /**
         * Hologram object size as a fraction of the shorter bitmap dimension.
         * Controls how large the projected cube appears on screen.
         */
        private const val HOLOGRAM_SIZE_FRACTION = 0.22f

        /**
         * Maximum yaw rotation angle in degrees applied when the nose is at the
         * horizontal screen edge (normalised offset ±0.5 × HOLOGRAM_ORIENT_SCALE).
         */
        private const val HOLOGRAM_MAX_YAW = 60f

        /**
         * Maximum pitch rotation angle in degrees applied when the nose is at the
         * vertical screen edge.
         */
        private const val HOLOGRAM_MAX_PITCH = 45f

        /**
         * Multiplier applied to the normalised face-centre offset before clamping.
         * A value of 2.0 means an offset of ±0.25 already saturates the rotation.
         */
        private const val HOLOGRAM_ORIENT_SCALE = 2.0f

        /** Teal-cyan ARGB colour (R=0, G=255, B=200) for hologram edges and vertex dots. */
        private const val HOLOGRAM_EDGE_COLOR = 0xFF00FFC8.toInt()

        /** Semi-transparent dark-teal for hologram face fills. */
        private const val HOLOGRAM_FILL_COLOR = 0x4000C880.toInt()

        /** Stroke width for hologram wireframe edges (pixels). */
        private const val HOLOGRAM_EDGE_WIDTH = 3f

        /**
         * 8 × 3 array of unit-cube vertices centred at the origin.
         * Rows: [x, y, z] with each component ∈ {-1, +1}.
         */
        private val HOLOGRAM_CUBE_VERTICES = arrayOf(
            floatArrayOf(-1f, -1f, -1f),
            floatArrayOf(+1f, -1f, -1f),
            floatArrayOf(+1f, +1f, -1f),
            floatArrayOf(-1f, +1f, -1f),
            floatArrayOf(-1f, -1f, +1f),
            floatArrayOf(+1f, -1f, +1f),
            floatArrayOf(+1f, +1f, +1f),
            floatArrayOf(-1f, +1f, +1f),
        )

        /** 12 edges of the cube as pairs of vertex indices. */
        private val HOLOGRAM_CUBE_EDGES = arrayOf(
            0 to 1, 1 to 2, 2 to 3, 3 to 0,  // back face
            4 to 5, 5 to 6, 6 to 7, 7 to 4,  // front face
            0 to 4, 1 to 5, 2 to 6, 3 to 7,  // connecting edges
        )

        /** 6 faces of the cube as groups of 4 vertex indices (for fill rendering). */
        private val HOLOGRAM_CUBE_FACES = arrayOf(
            intArrayOf(0, 1, 2, 3),
            intArrayOf(4, 5, 6, 7),
            intArrayOf(0, 1, 5, 4),
            intArrayOf(2, 3, 7, 6),
            intArrayOf(0, 3, 7, 4),
            intArrayOf(1, 2, 6, 5),
        )

        /** Text size for the hologram orientation HUD overlay. */
        private const val HOLOGRAM_HUD_TEXT_SIZE = 36f

        // ------------------------------------------------------------------
        // Eye Tracking Calibration
        // ------------------------------------------------------------------

        /** Number of calibration points (e.g., 5-point calibration: corners + center). */
        private const val CALIBRATION_POINTS_COUNT = 5

        /** Frames to collect per calibration point. */
        private const val SAMPLES_PER_POINT = 45

        /** Minimum face size (as fraction of frame) for reliable tracking. */
        private const val MIN_FACE_SIZE = 0.15f
    }

    private var poseLandmarker: PoseLandmarker? = null
    private var handLandmarker: HandLandmarker? = null
    private var faceLandmarker: FaceLandmarker? = null
    private var objectDetector: ObjectDetector? = null
    private var gestureRecognizer: GestureRecognizer? = null

    private var frameCounter = 0
    private var lastPoseResult: PoseLandmarkerResult? = null
    private var lastHandResult: HandLandmarkerResult? = null
    private var lastFaceResult: FaceLandmarkerResult? = null
    private var lastObjectResult: ObjectDetectorResult? = null
    private var lastGestureResult: GestureRecognizerResult? = null

    // ------------------------------------------------------------------
    // Eye Tracking & Calibration State
    // ------------------------------------------------------------------

    private var isCalibrating = false
    private var calibrationPointIndex = 0
    private var samplesCollectedCount = 0

    /** Storage for iris-to-screen training data. Keys: (ScreenX, ScreenY). Values: List of (IrisOffsetX, IrisOffsetY). */
    private val calibrationSamples = mutableListOf<CalibrationSample>()

    /** Linear regression weights for mapping iris offset to screen coordinates. */
    private var weightX = floatArrayOf(0f, 0f, 0f) // [const, dx, dy]
    private var weightY = floatArrayOf(0f, 0f, 0f) // [const, dx, dy]
    private var isCalibrated = false

    private data class CalibrationSample(
        val screenX: Float, // Normalised 0..1
        val screenY: Float, // Normalised 0..1
        val irisOffsetX: Float,
        val irisOffsetY: Float
    )

    private val detectionInterval = 3 // Run detection every 3 frames to save CPU

    private val dotPaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    private val linePaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = LINE_WIDTH
        isAntiAlias = true
    }
    private val overlayPaint = Paint().apply {
        color = Color.RED
        textSize = 40f
        isFakeBoldText = true
        setShadowLayer(5f, 0f, 0f, Color.BLACK)
    }

    /** Semi-transparent paint for info text backgrounds. */
    private val infoBgPaint = Paint().apply {
        color = 0x99000000.toInt()
        style = Paint.Style.FILL
    }

    /** Paint for person-detection and mouth-status overlay text. */
    private val infoPaint = Paint().apply {
        textSize = INFO_TEXT_SIZE
        isFakeBoldText = true
        isAntiAlias = true
        setShadowLayer(6f, 0f, 0f, Color.BLACK)
    }

    /** Paint for gaze-direction lines drawn from iris centres. */
    private val gazePaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = GAZE_LINE_WIDTH
        isAntiAlias = true
        strokeCap = Paint.Cap.ROUND
    }

    /** Paint for hologram wireframe edges. */
    private val hologramEdgePaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = HOLOGRAM_EDGE_WIDTH
        color = HOLOGRAM_EDGE_COLOR
        isAntiAlias = true
        strokeCap = Paint.Cap.ROUND
    }

    /** Paint for hologram vertex dots (filled circles at cube corners). */
    private val hologramDotPaint = Paint().apply {
        style = Paint.Style.FILL
        color = HOLOGRAM_EDGE_COLOR
        isAntiAlias = true
    }

    /** Paint for hologram face fills (semi-transparent). */
    private val hologramFillPaint = Paint().apply {
        style = Paint.Style.FILL
        color = HOLOGRAM_FILL_COLOR
    }

    /** Paint for hologram HUD text (yaw / pitch readout). */
    private val hologramHudPaint = Paint().apply {
        textSize = HOLOGRAM_HUD_TEXT_SIZE
        color = HOLOGRAM_EDGE_COLOR
        isAntiAlias = true
        isFakeBoldText = true
        setShadowLayer(4f, 0f, 0f, Color.BLACK)
    }

    /** Paint for calibration target (red circle with white stroke). */
    private val calibrationTargetPaint = Paint().apply {
        style = Paint.Style.FILL
        color = Color.RED
        isAntiAlias = true
    }

    private val calibrationStrokePaint = Paint().apply {
        style = Paint.Style.STROKE
        color = Color.WHITE
        strokeWidth = 4f
        isAntiAlias = true
    }

    /** Paint for gaze reticle on the screen. */
    private val gazeReticlePaint = Paint().apply {
        style = Paint.Style.STROKE
        color = 0xFF00FF00.toInt() // Green
        strokeWidth = 6f
        isAntiAlias = true
    }

    /** Reusable Rect used for text-bounds measurement to avoid per-frame allocations. */
    private val textBoundsRect = Rect()

    /**
     * Initialize all required MediaPipe detectors.
     *
     * Should be called from a background thread to avoid blocking the UI.
     */
    fun initialize() {
        // No-op: detectors are now initialized lazily in apply methods.
        // This avoids invalidation logs and unnecessary resource allocation at startup.
    }

    /**
     * Release all detector resources.
     */
    fun close() {
        poseLandmarker?.close()
        handLandmarker?.close()
        faceLandmarker?.close()
        objectDetector?.close()
        gestureRecognizer?.close()
        poseLandmarker = null
        handLandmarker = null
        faceLandmarker = null
        objectDetector = null
        gestureRecognizer = null
    }

    /**
     * Process the given [bitmap] according to the specified [filter] mode.
     *
     * Returns a new [Bitmap] containing the visualized landmarks.
     */
    fun processFrame(bitmap: Bitmap, filter: OpenCvFilter): Bitmap {
        frameCounter++
        return try {
            when (filter) {
                OpenCvFilter.HOLISTIC_BODY -> applyPoseLandmarker(bitmap)
                OpenCvFilter.HOLISTIC_HANDS -> applyHandLandmarker(bitmap)
                OpenCvFilter.HOLISTIC_FACE -> applyFaceLandmarker(bitmap, false)
                OpenCvFilter.IRIS -> applyFaceLandmarker(bitmap, true)
                OpenCvFilter.EYE_TRACKING -> applyEyeTracking(bitmap)
                OpenCvFilter.HOLOGRAM_3D -> applyHologram3D(bitmap)
                OpenCvFilter.OBJECTRON -> applyObjectron(bitmap)
                OpenCvFilter.GESTURE_RECOGNIZER -> applyGestureRecognizer(bitmap)
                OpenCvFilter.FACE_DETECTION_BLAZE -> applyFaceDetector(bitmap)
                else -> bitmap.copy(Bitmap.Config.ARGB_8888, false)
            }
        } catch (error: Throwable) {
            Log.e(TAG, "MediaPipe processing failed for filter=${filter.name}", error)
            drawModuleError(bitmap, "MediaPipe error: ${filter.displayName}")
        }
    }

    private fun applyGestureRecognizer(bitmap: Bitmap): Bitmap {
        val recognizer = gestureRecognizer ?: tryCreateGestureRecognizer().also { gestureRecognizer = it }
        if (recognizer == null) return overlayMissingModel(
            bitmap,
            context.getString(R.string.mediapipe_model_missing_gesture)
        )

        val argbBitmap = ensureArgb8888(bitmap)
        val result = recognizer.recognize(BitmapImageBuilder(argbBitmap).build())
        lastGestureResult = result

        val output = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)
        val w = output.width
        val h = output.height

        // Draw hand landmarks if detected
        for (landmarks in result.landmarks()) {
            drawConnections(canvas, landmarks, w, h, Color.WHITE, HAND_CONNECTIONS)
            drawDots(canvas, landmarks, w, h, Color.CYAN)
        }

        // Display recognized gestures
        val gestures = result.gestures()
        if (gestures.isNotEmpty()) {
            val topGesture = gestures[0][0]
            val text = context.getString(R.string.mediapipe_gesture_detected, topGesture.categoryName())
            
            infoPaint.color = Color.CYAN
            infoPaint.getTextBounds(text, 0, text.length, textBoundsRect)
            val topY = 20f
            canvas.drawRect(
                16f,
                topY,
                textBoundsRect.width() + 16f + 20f,
                topY + textBoundsRect.height() + 20f,
                infoBgPaint
            )
            canvas.drawText(text, 26f, topY + 10f + textBoundsRect.height(), infoPaint)
        }

        return output
    }

    private fun applyObjectron(bitmap: Bitmap): Bitmap {
        val detector = objectDetector ?: tryCreateObjectDetector().also { objectDetector = it }
        if (detector == null) return overlayMissingModel(
            bitmap,
            context.getString(R.string.mediapipe_model_missing_objectron)
        )

        val argbBitmap = ensureArgb8888(bitmap)
        if (frameCounter % detectionInterval == 0 || lastObjectResult == null) {
            lastObjectResult = detector.detect(BitmapImageBuilder(argbBitmap).build())
        }
        val result = lastObjectResult!!

        val output = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)

        val detections = result.detections()
        for (detection in detections) {
            // Draw 2D bounding box (optional, usually provides context)
            val box = detection.boundingBox()
            linePaint.color = Color.GREEN
            linePaint.strokeWidth = 2f
            canvas.drawRect(box, linePaint)

            // Draw category label
            val category = detection.categories().firstOrNull()
            category?.let {
                val label = context.getString(R.string.mediapipe_object_detected, it.categoryName())
                canvas.drawText(label, box.left, box.top - 10f, overlayPaint)
            }

            // Objectron models often store 3D box corners as keypoints in the detection.
            // There are typically 9 keypoints: 1 center + 8 corners.
            val keypoints = detection.keypoints().orElse(emptyList())
            if (keypoints.size >= 9) {
                drawObjectronBox(canvas, keypoints, output.width, output.height)
            }
        }

        return output
    }

    private fun applyFaceDetector(bitmap: Bitmap): Bitmap {
        val detector = faceLandmarker ?: tryCreateFaceLandmarker().also { faceLandmarker = it }
        if (detector == null) return overlayMissingModel(
            bitmap,
            context.getString(R.string.mediapipe_model_missing_face)
        )

        val argbBitmap = ensureArgb8888(bitmap)
        if (frameCounter % detectionInterval == 0 || lastFaceResult == null) {
            lastFaceResult = detector.detect(BitmapImageBuilder(argbBitmap).build())
        }
        val result = lastFaceResult!!

        val output = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)

        for (face in result.faceLandmarks()) {
            // In BlazeFace mode, we just draw the bounding box or key landmarks
            // Using the first 6 keypoints which are typically: left eye, right eye, nose tip, mouth center, left ear, right ear
            val keypoints = listOf(33, 263, 4, 13, 234, 454) 
            dotPaint.color = Color.YELLOW
            for (idx in keypoints) {
                if (idx < face.size) {
                    val lm = face[idx]
                    canvas.drawCircle(lm.x() * output.width, lm.y() * output.height, 6f, dotPaint)
                }
            }
        }
        
        drawPersonInfo(canvas, result.faceLandmarks().size)

        return output
    }

    /**
     * Draw a 3D wireframe box from Objectron keypoints.
     * Keypoints order: 0: center, 1-4: front face, 5-8: back face.
     */
    private fun drawObjectronBox(
        canvas: Canvas,
        keypoints: List<com.google.mediapipe.tasks.components.containers.NormalizedKeypoint>,
        width: Int,
        height: Int
    ) {
        val pts = keypoints.map { pt ->
            (pt.x() * width) to (pt.y() * height)
        }

        linePaint.color = Color.CYAN
        linePaint.strokeWidth = 4f

        // Front face (1-4)
        canvas.drawLine(pts[1].first, pts[1].second, pts[2].first, pts[2].second, linePaint)
        canvas.drawLine(pts[2].first, pts[2].second, pts[3].first, pts[3].second, linePaint)
        canvas.drawLine(pts[3].first, pts[3].second, pts[4].first, pts[4].second, linePaint)
        canvas.drawLine(pts[4].first, pts[4].second, pts[1].first, pts[1].second, linePaint)

        // Back face (5-8)
        canvas.drawLine(pts[5].first, pts[5].second, pts[6].first, pts[6].second, linePaint)
        canvas.drawLine(pts[6].first, pts[6].second, pts[7].first, pts[7].second, linePaint)
        canvas.drawLine(pts[7].first, pts[7].second, pts[8].first, pts[8].second, linePaint)
        canvas.drawLine(pts[8].first, pts[8].second, pts[5].first, pts[5].second, linePaint)

        // Connecting lines
        canvas.drawLine(pts[1].first, pts[1].second, pts[5].first, pts[5].second, linePaint)
        canvas.drawLine(pts[2].first, pts[2].second, pts[6].first, pts[6].second, linePaint)
        canvas.drawLine(pts[3].first, pts[3].second, pts[7].first, pts[7].second, linePaint)
        canvas.drawLine(pts[4].first, pts[4].second, pts[8].first, pts[8].second, linePaint)

        // Center dot
        dotPaint.color = Color.YELLOW
        canvas.drawCircle(pts[0].first, pts[0].second, 6f, dotPaint)
    }

    private fun drawModuleError(bitmap: Bitmap, message: String): Bitmap {
        val output = ensureArgb8888(bitmap)
        val canvas = Canvas(output)
        val paint = Paint().apply {
            color = Color.RED
            textSize = 40f
            isFakeBoldText = true
            isAntiAlias = true
        }
        canvas.drawText(message, 30f, 60f, paint)
        return output
    }

    private fun applyPoseLandmarker(bitmap: Bitmap): Bitmap {
        val detector = poseLandmarker ?: tryCreatePoseLandmarker().also { poseLandmarker = it }
        if (detector == null) return overlayMissingModel(
            bitmap,
            context.getString(R.string.mediapipe_model_missing_pose)
        )

        val argbBitmap = ensureArgb8888(bitmap)
        
        if (frameCounter % detectionInterval == 0 || lastPoseResult == null) {
            lastPoseResult = detector.detect(BitmapImageBuilder(argbBitmap).build())
        }
        val result = lastPoseResult!!

        val output = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)

        val personCount = result.landmarks().size

        for (person in result.landmarks()) {
            drawConnections(canvas, person, output.width, output.height, POSE_COLOR, POSE_CONNECTIONS)
            drawDots(canvas, person, output.width, output.height, POSE_COLOR)
        }

        // -- Human detection info overlay --
        val personBadgeBottom = drawPersonInfo(canvas, personCount)

        // -- Mouth / speaking detection (runs FaceLandmarker on the same frame) --
        if (personCount > 0) {
            val faceDetector = faceLandmarker ?: tryCreateFaceLandmarker().also { faceLandmarker = it }
            faceDetector?.let { fd ->
                runCatching {
                    val faceResult: FaceLandmarkerResult =
                        fd.detect(BitmapImageBuilder(argbBitmap).build())
                    // Only examine the first detected face to avoid overlapping status badges.
                    if (faceResult.faceLandmarks().isNotEmpty()) {
                        val jawOpen = extractJawOpen(faceResult, 0)
                        drawMouthStatus(canvas, jawOpen, personBadgeBottom)
                    }
                }.onFailure { e -> Log.w(TAG, "Face detection in pose mode failed", e) }
            }
        }

        return output
    }

    private fun applyHandLandmarker(bitmap: Bitmap): Bitmap {
        val detector = handLandmarker ?: tryCreateHandLandmarker().also { handLandmarker = it }
        if (detector == null) return overlayMissingModel(
            bitmap,
            context.getString(R.string.mediapipe_model_missing_hands)
        )

        val argbBitmap = ensureArgb8888(bitmap)
        if (frameCounter % detectionInterval == 0 || lastHandResult == null) {
            lastHandResult = detector.detect(BitmapImageBuilder(argbBitmap).build())
        }
        val result = lastHandResult!!

        val output = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)

        for ((index, hand) in result.landmarks().withIndex()) {
            val handednesses = result.handednesses()
            val handedness = handednesses.getOrNull(index)
            val isLeft = handedness?.firstOrNull()?.categoryName()
                ?.equals("Left", ignoreCase = true) == true
            val color = if (isLeft) LEFT_HAND_COLOR else RIGHT_HAND_COLOR
            drawConnections(canvas, hand, output.width, output.height, color, HAND_CONNECTIONS)
            drawDots(canvas, hand, output.width, output.height, color)
        }

        return output
    }

    private fun applyFaceLandmarker(bitmap: Bitmap, iris: Boolean): Bitmap {
        val detector = faceLandmarker ?: tryCreateFaceLandmarker().also { faceLandmarker = it }
        val missingMsg = context.getString(R.string.mediapipe_model_missing_face)

        if (detector == null) {
            return overlayMissingModel(bitmap, missingMsg)
        }

        val argbBitmap = ensureArgb8888(bitmap)
        if (frameCounter % detectionInterval == 0 || lastFaceResult == null) {
            lastFaceResult = detector.detect(BitmapImageBuilder(argbBitmap).build())
        }
        val result = lastFaceResult!!

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
                // Draw gaze-direction lines extending from each iris centre.
                drawGazeLines(canvas, face, output.width, output.height)
            }
        }

        return output
    }

    /**
     * Start/Reset eye tracking calibration.
     */
    fun startEyeTrackingCalibration() {
        isCalibrating = true
        calibrationPointIndex = 0
        samplesCollectedCount = 0
        calibrationSamples.clear()
        isCalibrated = false
    }

    private fun applyEyeTracking(bitmap: Bitmap): Bitmap {
        val detector = faceLandmarker ?: tryCreateFaceLandmarker().also { faceLandmarker = it }
        if (detector == null) return overlayMissingModel(
            bitmap,
            context.getString(R.string.mediapipe_model_missing_face)
        )

        val argbBitmap = ensureArgb8888(bitmap)
        val result = detector.detect(BitmapImageBuilder(argbBitmap).build())
        val output = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)
        val w = output.width.toFloat()
        val h = output.height.toFloat()

        if (result.faceLandmarks().isNotEmpty()) {
            val face = result.faceLandmarks()[0]
            if (face.size >= 478) {
                // Get iris and eye corner landmarks.
                val lIris = face[LEFT_IRIS_CENTER]
                val rIris = face[RIGHT_IRIS_CENTER]
                val lOuter = face[LEFT_EYE_OUTER]
                val lInner = face[LEFT_EYE_INNER]
                val rOuter = face[RIGHT_EYE_OUTER]
                val rInner = face[RIGHT_EYE_INNER]

                // Eye midpoints (normalised).
                val lEyeMidX = (lOuter.x() + lInner.x()) / 2f
                val lEyeMidY = (lOuter.y() + lInner.y()) / 2f
                val rEyeMidX = (rOuter.x() + rInner.x()) / 2f
                val rEyeMidY = (rOuter.y() + rInner.y()) / 2f

                // Iris offset from eye midpoint (normalised).
                val dx = ((lIris.x() - lEyeMidX) + (rIris.x() - rEyeMidX)) / 2f
                val dy = ((lIris.y() - lEyeMidY) + (rIris.y() - rEyeMidY)) / 2f

                if (isCalibrating) {
                    processCalibrationFrame(canvas, w, h, dx, dy)
                } else if (isCalibrated) {
                    // Map iris offset (dx, dy) to screen (sx, sy) using linear regression weights.
                    val sx = (weightX[0] + weightX[1] * dx + weightX[2] * dy).coerceIn(0f, 1f)
                    val sy = (weightY[0] + weightY[1] * dx + weightY[2] * dy).coerceIn(0f, 1f)
                    val px = sx * w
                    val py = sy * h
                    drawGazeReticle(canvas, px, py)

                    // Display gaze convergence coordinates
                    val coordText = String.format(Locale.US, "Gaze: (%.0f, %.0f)", px, py)
                    canvas.drawText(coordText, px + 40f, py - 40f, hologramHudPaint)
                }
                
                // Still draw basic iris dots for visual feedback.
                drawIrisCircle(canvas, face, output.width, output.height, LEFT_IRIS_INDICES, IRIS_COLOR_LEFT)
                drawIrisCircle(canvas, face, output.width, output.height, RIGHT_IRIS_INDICES, IRIS_COLOR_RIGHT)
            }
        }

        // HUD overlay.
        val hudText = when {
            isCalibrating -> context.getString(R.string.eye_tracking_calibration_point)
            isCalibrated -> context.getString(R.string.eye_tracking_status_calibrated)
            else -> context.getString(R.string.eye_tracking_status_not_calibrated)
        }
        canvas.drawText(hudText, 16f, h - 20f, hologramHudPaint)

        return output
    }

    private fun processCalibrationFrame(canvas: Canvas, w: Float, h: Float, dx: Float, dy: Float) {
        val targetX: Float
        val targetY: Float

        // Calibration point sequence (0.02..0.98 to maximize distance).
        when (calibrationPointIndex) {
            0 -> { targetX = 0.5f; targetY = 0.5f } // Center
            1 -> { targetX = 0.02f; targetY = 0.02f } // Top-left
            2 -> { targetX = 0.98f; targetY = 0.02f } // Top-right
            3 -> { targetX = 0.02f; targetY = 0.98f } // Bottom-left
            4 -> { targetX = 0.98f; targetY = 0.98f } // Bottom-right
            else -> return
        }

        // Draw target on screen.
        val tx = targetX * w
        val ty = targetY * h
        canvas.drawCircle(tx, ty, 40f, calibrationTargetPaint)
        canvas.drawCircle(tx, ty, 40f, calibrationStrokePaint)
        canvas.drawCircle(tx, ty, 10f, calibrationStrokePaint)

        // Collect samples.
        samplesCollectedCount++
        calibrationSamples.add(CalibrationSample(targetX, targetY, dx, dy))

        if (samplesCollectedCount >= SAMPLES_PER_POINT) {
            samplesCollectedCount = 0
            calibrationPointIndex++
            if (calibrationPointIndex >= CALIBRATION_POINTS_COUNT) {
                finishCalibration()
            }
        }
    }

    private fun finishCalibration() {
        isCalibrating = false
        // Perform simple multi-variable linear regression for sx = f(dx, dy) and sy = f(dx, dy).
        // Since we have fixed calibration points, we could use a more robust solver,
        // but for now, we'll use a simplified mean-based mapping for demonstration.
        // For a proper solution, one would use Least Squares.
        
        // sx = a + b*dx + c*dy
        // sy = d + e*dx + f*dy
        
        // Simplified least-squares estimate for the weights.
        val n = calibrationSamples.size.toDouble()
        if (n < 5) return

        var sumX = 0.0; var sumY = 0.0; var sumDX = 0.0; var sumDY = 0.0
        var sumDX2 = 0.0; var sumDY2 = 0.0; var sumDXDY = 0.0
        var sumXDX = 0.0; var sumXDY = 0.0; var sumYDX = 0.0; var sumYDY = 0.0

        for (s in calibrationSamples) {
            val sx = s.screenX.toDouble(); val sy = s.screenY.toDouble()
            val dx = s.irisOffsetX.toDouble(); val dy = s.irisOffsetY.toDouble()
            sumX += sx; sumY += sy; sumDX += dx; sumDY += dy
            sumDX2 += dx * dx; sumDY2 += dy * dy; sumDXDY += dx * dy
            sumXDX += sx * dx; sumXDY += sx * dy
            sumYDX += sy * dx; sumYDY += sy * dy
        }

        // Solve systems:
        // [ n     sumDX  sumDY  ] [ a ]   [ sumX   ]
        // [ sumDX sumDX2 sumDXDY] [ b ] = [ sumXDX ]
        // [ sumDY sumDXDY sumDY2] [ c ]   [ sumXDY ]
        
        weightX = solve3x3(n, sumDX, sumDY, sumDX2, sumDXDY, sumDY2, sumX, sumXDX, sumXDY)
        weightY = solve3x3(n, sumDX, sumDY, sumDX2, sumDXDY, sumDY2, sumY, sumYDX, sumYDY)
        
        isCalibrated = true
    }

    private fun solve3x3(
        n: Double, sd: Double, se: Double,
        sd2: Double, sde: Double, se2: Double,
        sz: Double, szd: Double, sze: Double
    ): FloatArray {
        // Determinant using Sarrus rule.
        val det = n * (sd2 * se2 - sde * sde) - sd * (sd * se2 - sde * se) + se * (sd * sde - sd2 * se)
        if (abs(det) < 1e-9) return floatArrayOf(0.5f, 0f, 0f)

        val a = (sz * (sd2 * se2 - sde * sde) - sd * (szd * se2 - sze * sde) + se * (szd * sde - sze * sd2)) / det
        val b = (n * (szd * se2 - sze * sde) - sz * (sd * se2 - se * sde) + se * (sd * sze - szd * se)) / det
        val c = (n * (sd2 * sze - sde * szd) - sd * (sd * sze - se * szd) + sz * (sd * sde - sd2 * se)) / det
        
        return floatArrayOf(a.toFloat(), b.toFloat(), c.toFloat())
    }

    private fun drawGazeReticle(canvas: Canvas, x: Float, y: Float) {
        canvas.drawCircle(x, y, 30f, gazeReticlePaint)
        canvas.drawLine(x - 40, y, x + 40, y, gazeReticlePaint)
        canvas.drawLine(x, y - 40, x, y + 40, gazeReticlePaint)
    }

    /**
     * Render a 3D hologram (wireframe cube) whose rotation is driven by the viewer's face
     * position relative to the screen centre.
     *
     * The nose-tip landmark (index 4) is used to estimate horizontal (yaw) and vertical (pitch)
     * offsets relative to the normalised frame centre (0.5, 0.5).  These offsets are scaled and
     * clamped to derive rotation angles, which are applied to the cube vertices before perspective
     * projection onto the canvas.
     *
     * If the face-landmarker model is unavailable, a fallback overlay is shown.  If no face is
     * detected in the current frame, a static (unrotated) hologram is drawn along with a
     * "Brak twarzy" indicator.
     */
    private fun applyHologram3D(bitmap: Bitmap): Bitmap {
        val detector = faceLandmarker ?: tryCreateFaceLandmarker().also { faceLandmarker = it }
        if (detector == null) return overlayMissingModel(
            bitmap,
            context.getString(R.string.mediapipe_model_missing_face)
        )

        val argbBitmap = ensureArgb8888(bitmap)
        val result: FaceLandmarkerResult = detector.detect(BitmapImageBuilder(argbBitmap).build())

        val output = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)

        val w = output.width
        val h = output.height
        val cx = w / 2f
        val cy = h / 2f
        val size = minOf(w, h) * HOLOGRAM_SIZE_FRACTION

        // Compute yaw/pitch from nose-tip position (or use zeros if no face).
        var yawDeg = 0f
        var pitchDeg = 0f
        val faceDetected = result.faceLandmarks().isNotEmpty()

        if (faceDetected) {
            val face = result.faceLandmarks()[0]
            if (face.size > HOLOGRAM_NOSE_TIP) {
                val nose = face[HOLOGRAM_NOSE_TIP]
                val offsetX = (nose.x() - 0.5f) * HOLOGRAM_ORIENT_SCALE
                val offsetY = (nose.y() - 0.5f) * HOLOGRAM_ORIENT_SCALE
                yawDeg = offsetX.coerceIn(-1f, 1f) * HOLOGRAM_MAX_YAW
                pitchDeg = offsetY.coerceIn(-1f, 1f) * HOLOGRAM_MAX_PITCH
            }
        }

        // Build rotation matrices.
        val yawRad = Math.toRadians(yawDeg.toDouble()).toFloat()
        val pitchRad = Math.toRadians(pitchDeg.toDouble()).toFloat()

        // Project cube vertices onto 2-D canvas.
        val focal = w * 1.5f
        val camZ = size * 4f
        val pts2d = Array(8) { 0f to 0f }
        for ((i, v) in HOLOGRAM_CUBE_VERTICES.withIndex()) {
            val rv = rotateVertex(v, yawRad, pitchRad)
            val vx = rv[0] * size
            val vy = rv[1] * size
            val vz = rv[2] * size
            val z = camZ + vz
            val safeZ = if (abs(z) < 1e-6f) 1e-6f else z
            pts2d[i] = (vx * focal / safeZ + cx) to (vy * focal / safeZ + cy)
        }

        // Draw semi-transparent face fills first.
        for (faceIndices in HOLOGRAM_CUBE_FACES) {
            val path = Path()
            val (fx0, fy0) = pts2d[faceIndices[0]]
            path.moveTo(fx0, fy0)
            for (k in 1 until faceIndices.size) {
                val (fx, fy) = pts2d[faceIndices[k]]
                path.lineTo(fx, fy)
            }
            path.close()
            canvas.drawPath(path, hologramFillPaint)
        }

        // Draw wireframe edges.
        for ((startIdx, endIdx) in HOLOGRAM_CUBE_EDGES) {
            val (x1, y1) = pts2d[startIdx]
            val (x2, y2) = pts2d[endIdx]
            canvas.drawLine(x1, y1, x2, y2, hologramEdgePaint)
        }

        // Draw vertex dots.
        for ((px, py) in pts2d) {
            canvas.drawCircle(px, py, 5f, hologramDotPaint)
        }

        // HUD: yaw / pitch readout or "no face" message.
        val hudText = if (faceDetected) {
            context.getString(R.string.hologram_hud_angles, yawDeg, pitchDeg)
        } else {
            context.getString(R.string.hologram_no_face)
        }
        canvas.drawText(hudText, 16f, h - 20f, hologramHudPaint)

        return output
    }

    /**
     * Rotate a 3-D vertex [v] by [yawRad] around the Y axis then [pitchRad] around the X axis.
     *
     * @param v     Float array ``[x, y, z]``.
     * @param yawRad   Rotation angle around Y in radians.
     * @param pitchRad Rotation angle around X in radians.
     * @return New rotated ``[x, y, z]`` float array.
     */
    private fun rotateVertex(v: FloatArray, yawRad: Float, pitchRad: Float): FloatArray {
        // Yaw (around Y axis): x' = x·cos(y) + z·sin(y), y' = y, z' = -x·sin(y) + z·cos(y)
        val cy = cos(yawRad)
        val sy = sin(yawRad)
        val x1 = v[0] * cy + v[2] * sy
        val y1 = v[1]
        val z1 = -v[0] * sy + v[2] * cy

        // Pitch (around X axis): x'' = x', y'' = y'·cos(p) - z'·sin(p), z'' = y'·sin(p) + z'·cos(p)
        val cp = cos(pitchRad)
        val sp = sin(pitchRad)
        return floatArrayOf(x1, y1 * cp - z1 * sp, y1 * sp + z1 * cp)
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
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(modelPath)
                .setDelegate(Delegate.GPU)
                .build()
            val options = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .setNumPoses(3) // Detect up to 3 people
                .build()
            PoseLandmarker.createFromOptions(context, options).also {
                Log.i(TAG, "PoseLandmarker created on GPU")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create PoseLandmarker on GPU, falling back to CPU", e)
            tryCreatePoseLandmarkerCpu(modelPath)
        }
    }

    private fun tryCreatePoseLandmarkerCpu(modelPath: String): PoseLandmarker? {
        return try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(modelPath)
                .setDelegate(Delegate.CPU)
                .build()
            val options = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .setNumPoses(3)
                .build()
            PoseLandmarker.createFromOptions(context, options)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create PoseLandmarker on CPU", e)
            null
        }
    }

    private fun tryCreateHandLandmarker(): HandLandmarker? {
        val modelPath = ModelDownloadManager.getModelPath(context, MODEL_HAND)
            ?: return null.also { Log.d(TAG, "Hand model not available") }
        return try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(modelPath)
                .setDelegate(Delegate.GPU)
                .build()
            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .setNumHands(4) // Detect up to 4 hands
                .build()
            HandLandmarker.createFromOptions(context, options).also {
                Log.i(TAG, "HandLandmarker created on GPU")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create HandLandmarker on GPU, falling back to CPU", e)
            tryCreateHandLandmarkerCpu(modelPath)
        }
    }

    private fun tryCreateHandLandmarkerCpu(modelPath: String): HandLandmarker? {
        return try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(modelPath)
                .setDelegate(Delegate.CPU)
                .build()
            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .setNumHands(4)
                .build()
            HandLandmarker.createFromOptions(context, options)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create HandLandmarker on CPU", e)
            null
        }
    }

    private fun tryCreateFaceLandmarker(): FaceLandmarker? {
        val modelPath = ModelDownloadManager.getModelPath(context, MODEL_FACE)
            ?: return null.also { Log.d(TAG, "Face model not available") }
        return try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(modelPath)
                .setDelegate(Delegate.GPU)
                .build()
            val options = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .setNumFaces(3) // Detect up to 3 faces
                .setOutputFaceBlendshapes(true)
                .build()
            FaceLandmarker.createFromOptions(context, options).also {
                Log.i(TAG, "FaceLandmarker created on GPU")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create FaceLandmarker on GPU, falling back to CPU", e)
            tryCreateFaceLandmarkerCpu(modelPath)
        }
    }

    private fun tryCreateFaceLandmarkerCpu(modelPath: String): FaceLandmarker? {
        return try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(modelPath)
                .setDelegate(Delegate.CPU)
                .build()
            val options = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .setNumFaces(3)
                .setOutputFaceBlendshapes(true)
                .build()
            FaceLandmarker.createFromOptions(context, options)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create FaceLandmarker on CPU", e)
            null
        }
    }

    private fun tryCreateGestureRecognizer(): GestureRecognizer? {
        val modelPath = ModelDownloadManager.getModelPath(context, MODEL_GESTURE)
            ?: return null.also { Log.d(TAG, "Gesture model not available") }
        return try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(modelPath)
                .setDelegate(Delegate.GPU)
                .build()
            val options = GestureRecognizer.GestureRecognizerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .build()
            GestureRecognizer.createFromOptions(context, options).also {
                Log.i(TAG, "GestureRecognizer created on GPU")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create GestureRecognizer on GPU, falling back to CPU", e)
            tryCreateGestureRecognizerCpu(modelPath)
        }
    }

    private fun tryCreateGestureRecognizerCpu(modelPath: String): GestureRecognizer? {
        return try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(modelPath)
                .setDelegate(Delegate.CPU)
                .build()
            val options = GestureRecognizer.GestureRecognizerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .build()
            GestureRecognizer.createFromOptions(context, options)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create GestureRecognizer on CPU", e)
            null
        }
    }

    private fun tryCreateObjectDetector(): ObjectDetector? {
        val modelPath = ModelDownloadManager.getModelPath(context, MODEL_OBJECTRON)
            ?: return null.also { Log.d(TAG, "Objectron model not available") }
        return try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(modelPath)
                .setDelegate(Delegate.GPU)
                .build()
            val options = ObjectDetector.ObjectDetectorOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .setScoreThreshold(0.3f)
                .build()
            ObjectDetector.createFromOptions(context, options).also {
                Log.i(TAG, "ObjectDetector created on GPU")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create ObjectDetector on GPU, falling back to CPU", e)
            tryCreateObjectDetectorCpu(modelPath)
        }
    }

    private fun tryCreateObjectDetectorCpu(modelPath: String): ObjectDetector? {
        return try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(modelPath)
                .setDelegate(Delegate.CPU)
                .build()
            val options = ObjectDetector.ObjectDetectorOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .setScoreThreshold(0.3f)
                .build()
            ObjectDetector.createFromOptions(context, options)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create ObjectDetector on CPU", e)
            null
        }
    }

    // ------------------------------------------------------------------
    // Info / gaze overlay helpers
    // ------------------------------------------------------------------

    /**
     * Draw a person-count badge at the top-left of the canvas.
     *
     * @return The bottom Y coordinate of the drawn badge (for stacking subsequent labels below it).
     */
    private fun drawPersonInfo(canvas: Canvas, personCount: Int): Float {
        val text = if (personCount > 0) {
            context.getString(R.string.mediapipe_person_detected, personCount)
        } else {
            context.getString(R.string.mediapipe_no_person)
        }
        infoPaint.color = Color.RED
        infoPaint.getTextBounds(text, 0, text.length, textBoundsRect)
        val badgeBottom = textBoundsRect.height() + INFO_BADGE_MARGIN + INFO_BADGE_PADDING * 2
        canvas.drawRect(
            INFO_BADGE_MARGIN,
            INFO_BADGE_MARGIN,
            textBoundsRect.width() + INFO_BADGE_MARGIN + INFO_BADGE_PADDING * 2,
            badgeBottom,
            infoBgPaint,
        )
        canvas.drawText(
            text,
            INFO_BADGE_MARGIN + INFO_BADGE_PADDING,
            INFO_BADGE_MARGIN + INFO_BADGE_PADDING + textBoundsRect.height(),
            infoPaint,
        )
        return badgeBottom
    }

    /**
     * Draw mouth-state label ("Usta otwarte" / "Mówi!") when [jawOpen] exceeds a threshold.
     *
     * @param personBadgeBottom Bottom Y coordinate returned by [drawPersonInfo]; the label is
     *   placed below that position so the two badges do not overlap.
     */
    private fun drawMouthStatus(canvas: Canvas, jawOpen: Float, personBadgeBottom: Float) {
        val (text, color) = when {
            jawOpen > MOUTH_TALKING_THRESHOLD ->
                context.getString(R.string.mediapipe_speaking) to Color.RED
            jawOpen > MOUTH_OPEN_THRESHOLD ->
                context.getString(R.string.mediapipe_mouth_open) to Color.RED
            else -> return
        }
        infoPaint.color = color
        infoPaint.getTextBounds(text, 0, text.length, textBoundsRect)
        val topY = personBadgeBottom + INFO_BADGE_MARGIN
        canvas.drawRect(
            INFO_BADGE_MARGIN,
            topY,
            textBoundsRect.width() + INFO_BADGE_MARGIN + INFO_BADGE_PADDING * 2,
            topY + textBoundsRect.height() + INFO_BADGE_PADDING * 2,
            infoBgPaint,
        )
        canvas.drawText(
            text,
            INFO_BADGE_MARGIN + INFO_BADGE_PADDING,
            topY + INFO_BADGE_PADDING + textBoundsRect.height(),
            infoPaint,
        )
    }

    /**
     * Extract the `jawOpen` blendshape score from a [FaceLandmarkerResult] for the face at
     * [faceIndex].  Returns 0 if blendshapes are unavailable or the category is not found.
     */
    private fun extractJawOpen(result: FaceLandmarkerResult, faceIndex: Int): Float {
        val shapesOpt = result.faceBlendshapes()
        if (!shapesOpt.isPresent) return 0f
        val shapes = shapesOpt.get()
        if (faceIndex >= shapes.size) return 0f
        return shapes[faceIndex].firstOrNull { it.categoryName() == "jawOpen" }?.score() ?: 0f
    }

    /**
     * Draw gaze-direction lines from each iris centre in the given [landmarks] list.
     *
     * The direction is estimated as the vector from the eye-corner midpoint to the iris centre,
     * scaled by [GAZE_LINE_SCALE] to produce a visible arrow.
     *
     * Left-eye gaze uses [IRIS_COLOR_LEFT], right-eye gaze uses [IRIS_COLOR_RIGHT].
     *
     * Requires at least 478 landmarks (iris indices 468–477 must be present).
     */
    private fun drawGazeLines(
        canvas: Canvas,
        landmarks: List<NormalizedLandmark>,
        width: Int,
        height: Int,
    ) {
        if (landmarks.size < 478) return
        // Left eye: outer corner = LEFT_EYE_OUTER, inner corner = LEFT_EYE_INNER, iris centre = LEFT_IRIS_CENTER
        drawSingleGazeLine(canvas, landmarks, width, height, LEFT_EYE_OUTER, LEFT_EYE_INNER, LEFT_IRIS_CENTER, IRIS_COLOR_LEFT)
        // Right eye: outer corner = RIGHT_EYE_OUTER, inner corner = RIGHT_EYE_INNER, iris centre = RIGHT_IRIS_CENTER
        drawSingleGazeLine(canvas, landmarks, width, height, RIGHT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_IRIS_CENTER, IRIS_COLOR_RIGHT)
    }

    /**
     * Draw a single gaze line for one eye.
     *
     * The eye centre is the midpoint of [outerIdx] and [innerIdx] corners.
     * The gaze vector is (iris_centre − eye_centre) × [GAZE_LINE_SCALE].
     */
    private fun drawSingleGazeLine(
        canvas: Canvas,
        landmarks: List<NormalizedLandmark>,
        width: Int,
        height: Int,
        outerIdx: Int,
        innerIdx: Int,
        irisIdx: Int,
        color: Int,
    ) {
        val outer = landmarks[outerIdx]
        val inner = landmarks[innerIdx]
        val iris = landmarks[irisIdx]

        val eyeCenterX = (outer.x() + inner.x()) / 2f * width
        val eyeCenterY = (outer.y() + inner.y()) / 2f * height
        val irisPx = iris.x() * width
        val irisPy = iris.y() * height

        val gazeX = (irisPx - eyeCenterX) * GAZE_LINE_SCALE
        val gazeY = (irisPy - eyeCenterY) * GAZE_LINE_SCALE

        gazePaint.color = color
        canvas.drawLine(irisPx, irisPy, irisPx + gazeX, irisPy + gazeY, gazePaint)
        // Small filled dot at iris centre for clarity.
        dotPaint.color = color
        canvas.drawCircle(irisPx, irisPy, LANDMARK_RADIUS + 2f, dotPaint)
    }
}
