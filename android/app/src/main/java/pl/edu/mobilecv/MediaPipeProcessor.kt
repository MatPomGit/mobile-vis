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
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
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

        // ------------------------------------------------------------------
        // Drawing constants
        // ------------------------------------------------------------------

        private const val POSE_COLOR = Color.GREEN
        private const val LEFT_HAND_COLOR = Color.YELLOW
        private const val RIGHT_HAND_COLOR = Color.CYAN
        private const val FACE_COLOR = Color.WHITE
        private const val IRIS_COLOR_LEFT = Color.RED
        private const val IRIS_COLOR_RIGHT = Color.BLUE

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
    }

    private var poseLandmarker: PoseLandmarker? = null
    private var handLandmarker: HandLandmarker? = null
    private var faceLandmarker: FaceLandmarker? = null
    private var faceLandmarkerIris: FaceLandmarker? = null

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

    /** Reusable Rect used for text-bounds measurement to avoid per-frame allocations. */
    private val textBoundsRect = Rect()

    /**
     * Initialize all required MediaPipe detectors.
     *
     * Should be called from a background thread to avoid blocking the UI.
     */
    fun initialize() {
        poseLandmarker = tryCreatePoseLandmarker()
        handLandmarker = tryCreateHandLandmarker()
        faceLandmarker = tryCreateFaceLandmarker(false)
        faceLandmarkerIris = tryCreateFaceLandmarker(true)
    }

    /**
     * Release all detector resources.
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
     * Process the given [bitmap] according to the specified [filter] mode.
     *
     * Returns a new [Bitmap] containing the visualized landmarks.
     */
    fun processFrame(bitmap: Bitmap, filter: OpenCvFilter): Bitmap {
        return when (filter) {
            OpenCvFilter.HOLISTIC_BODY -> applyPoseLandmarker(bitmap)
            OpenCvFilter.HOLISTIC_HANDS -> applyHandLandmarker(bitmap)
            OpenCvFilter.HOLISTIC_FACE -> applyFaceLandmarker(bitmap, false)
            OpenCvFilter.IRIS -> applyFaceLandmarker(bitmap, true)
            OpenCvFilter.HOLOGRAM_3D -> applyHologram3D(bitmap)
            else -> bitmap
        }
    }

    private fun applyPoseLandmarker(bitmap: Bitmap): Bitmap {
        val detector = poseLandmarker ?: return overlayMissingModel(
            bitmap,
            context.getString(R.string.mediapipe_model_missing_pose)
        )

        val argbBitmap = ensureArgb8888(bitmap)
        val result: PoseLandmarkerResult = detector.detect(BitmapImageBuilder(argbBitmap).build())

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
            faceLandmarker?.let { faceDetector ->
                runCatching {
                    val faceResult: FaceLandmarkerResult =
                        faceDetector.detect(BitmapImageBuilder(argbBitmap).build())
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
        val detector = handLandmarker ?: return overlayMissingModel(
            bitmap,
            context.getString(R.string.mediapipe_model_missing_hands)
        )

        val mpImage = BitmapImageBuilder(ensureArgb8888(bitmap)).build()
        val result: HandLandmarkerResult = detector.detect(mpImage)

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
        val detector = if (iris) faceLandmarkerIris else faceLandmarker
        val missingMsg = context.getString(R.string.mediapipe_model_missing_face)

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
                // Draw gaze-direction lines extending from each iris centre.
                drawGazeLines(canvas, face, output.width, output.height)
            }
        }

        return output
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
        val detector = faceLandmarker ?: return overlayMissingModel(
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
                .setOutputFaceBlendshapes(true)
                .build()
            FaceLandmarker.createFromOptions(context, options)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create FaceLandmarker (iris=$refineIris)", e)
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
        infoPaint.color = if (personCount > 0) Color.GREEN else Color.RED
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
                context.getString(R.string.mediapipe_speaking) to Color.YELLOW
            jawOpen > MOUTH_OPEN_THRESHOLD ->
                context.getString(R.string.mediapipe_mouth_open) to Color.WHITE
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
