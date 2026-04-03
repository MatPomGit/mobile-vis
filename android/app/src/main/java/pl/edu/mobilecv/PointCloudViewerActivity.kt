package pl.edu.mobilecv

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.util.Log
import android.view.MotionEvent
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin

/**
 * Activity for loading and visualizing a saved point cloud (CSV or PLY).
 *
 * Supports the CSV format (x,y,z) and PLY ASCII format (x y z per vertex after header)
 * exported by [MainActivity]. The point cloud can be rotated in 3D by dragging on the screen.
 */
class PointCloudViewerActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "PointCloudViewer"
    }

    private lateinit var cloudView: PointCloudView
    private lateinit var textStatus: TextView
    private lateinit var btnLoad: Button
    private lateinit var btnBack: Button
    private lateinit var btnResetRotation: Button

    private val filePicker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) loadPointCloud(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_point_cloud_viewer)

        cloudView = findViewById(R.id.pointCloudView)
        textStatus = findViewById(R.id.textViewCloudStatus)
        btnLoad = findViewById(R.id.btnLoadPointCloud)
        btnBack = findViewById(R.id.btnBackFromViewer)
        btnResetRotation = findViewById(R.id.btnResetRotation)

        btnLoad.setOnClickListener {
            // Use "*/*" to support both .csv (text/csv) and .ply (application/octet-stream)
            // as some file managers don't map .ply to a specific MIME type.
            filePicker.launch("*/*")
        }

        btnBack.setOnClickListener {
            finish()
        }

        btnResetRotation.setOnClickListener {
            cloudView.resetRotation()
        }

        textStatus.text = getString(R.string.point_cloud_viewer_empty)
    }

    /**
     * Returns the display name of [uri] by querying the ContentResolver.
     * Content URIs (e.g. from DocumentProvider) don't expose the original filename
     * in [Uri.lastPathSegment], so this query is necessary for reliable extension detection.
     */
    private fun getDisplayName(uri: Uri): String {
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val col = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    if (col >= 0) return cursor.getString(col)
                }
            }
        return uri.lastPathSegment ?: ""
    }

    private fun loadPointCloud(uri: Uri) {
        try {
            val points = mutableListOf<Triple<Float, Float, Float>>()
            contentResolver.openInputStream(uri)?.bufferedReader()?.use { reader ->
                val filename = getDisplayName(uri)
                val isPly = filename.endsWith(".ply", ignoreCase = true)

                if (isPly) {
                    var headerDone = false
                    for (line in reader.lineSequence()) {
                        if (!headerDone) {
                            if (line.trim() == "end_header") headerDone = true
                            continue
                        }
                        val parts = line.trim().split(" ")
                        if (parts.size >= 2) {
                            val x = parts[0].toFloatOrNull() ?: continue
                            val y = parts[1].toFloatOrNull() ?: continue
                            val z = parts.getOrNull(2)?.toFloatOrNull() ?: 0f
                            points.add(Triple(x, y, z))
                        }
                    }
                } else {
                    // CSV format: x,y[,z] per line; skip header and comment lines
                    for (line in reader.lineSequence()) {
                        val trimmed = line.trim()
                        if (trimmed.startsWith("#") || trimmed.isEmpty()) continue
                        // Skip header line (x,y or x,y,z)
                        if (trimmed.equals("x,y", ignoreCase = true) ||
                            trimmed.equals("x,y,z", ignoreCase = true)) continue
                        val parts = trimmed.split(",")
                        if (parts.size >= 2) {
                            val x = parts[0].toFloatOrNull() ?: continue
                            val y = parts[1].toFloatOrNull() ?: continue
                            val z = parts.getOrNull(2)?.toFloatOrNull() ?: 0f
                            points.add(Triple(x, y, z))
                        }
                    }
                }
            }

            if (points.isEmpty()) {
                Toast.makeText(
                    this,
                    getString(R.string.point_cloud_viewer_error, "Brak punktów w pliku"),
                    Toast.LENGTH_SHORT,
                ).show()
                return
            }

            cloudView.setPoints(points)
            textStatus.text = getString(R.string.point_cloud_viewer_loaded, points.size)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load point cloud", e)
            Toast.makeText(
                this,
                getString(R.string.point_cloud_viewer_error, e.message ?: "?"),
                Toast.LENGTH_LONG,
            ).show()
        }
    }
}

/**
 * Custom view that renders a point cloud using an orthographic projection.
 * The cloud can be rotated in 3D by dragging with one finger.
 *
 * Rotation is implemented as a Y-then-X Euler rotation applied to centred point coordinates
 * before 2D projection. Depth (Z after rotation) is encoded as colour brightness.
 * The painter's algorithm (back-to-front sort) is used for correct depth ordering.
 *
 * Performance: the cloud centroid is cached on [setPoints]. The rotated/sorted list is
 * cached between frames and only recomputed when the rotation angles or point data changes.
 */
class PointCloudView(context: Context, attrs: android.util.AttributeSet?) : View(context, attrs) {

    companion object {
        /** Radians of rotation per screen pixel dragged. */
        private const val ROTATION_SENSITIVITY = 0.005f
        private const val POINT_RADIUS = 3f
        private const val HINT_TEXT_ALPHA = 160
        private const val HINT_TEXT_SIZE = 36f
        /** Distance from the bottom edge at which the drag hint is drawn, in pixels. */
        private const val HINT_BOTTOM_OFFSET = 60f
    }

    private var points: List<Triple<Float, Float, Float>> = emptyList()

    /** Cached centroid of [points], recomputed only in [setPoints]. */
    private var centroidX = 0f
    private var centroidY = 0f
    private var centroidZ = 0f

    /**
     * Pre-computed rotated and depth-sorted list.
     * Recomputed only when [projectedDirty] is true.
     */
    private var projected: MutableList<Triple<Float, Float, Float>> = mutableListOf()
    private var minX = 0f; private var maxX = 0f
    private var minY = 0f; private var maxY = 0f
    private var minZ = 0f; private var maxZ = 0f
    private var projectedDirty = true

    private val paint = Paint().apply { isAntiAlias = true }
    private val hintPaint = Paint().apply {
        isAntiAlias = true
        color = Color.argb(HINT_TEXT_ALPHA, 255, 255, 255)
        textSize = HINT_TEXT_SIZE
        textAlign = Paint.Align.CENTER
    }

    /** Elevation angle (rotation around the X axis), in radians. */
    private var rotX = 0f

    /** Azimuth angle (rotation around the Y axis), in radians. */
    private var rotY = 0f

    private var lastTouchX = 0f
    private var lastTouchY = 0f
    private var isDragging = false
    private var everRotated = false

    /** Replaces the displayed point cloud and resets rotation to the default view. */
    fun setPoints(pts: List<Triple<Float, Float, Float>>) {
        points = pts
        if (pts.isNotEmpty()) {
            // Compute centroid in a single pass to avoid three separate iterations
            var sumX = 0.0; var sumY = 0.0; var sumZ = 0.0
            for ((x, y, z) in pts) {
                sumX += x
                sumY += y
                sumZ += z
            }
            val n = pts.size.toDouble()
            centroidX = (sumX / n).toFloat()
            centroidY = (sumY / n).toFloat()
            centroidZ = (sumZ / n).toFloat()
        }
        projectedDirty = true
        resetRotation()
    }

    /** Resets the rotation back to the initial orientation. */
    fun resetRotation() {
        rotX = 0f
        rotY = 0f
        everRotated = false
        projectedDirty = true
        invalidate()
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (points.isEmpty()) return false
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                lastTouchX = event.x
                lastTouchY = event.y
                isDragging = true
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                if (isDragging) {
                    val dx = event.x - lastTouchX
                    val dy = event.y - lastTouchY
                    rotY += dx * ROTATION_SENSITIVITY
                    rotX += dy * ROTATION_SENSITIVITY
                    lastTouchX = event.x
                    lastTouchY = event.y
                    everRotated = true
                    projectedDirty = true
                    invalidate()
                }
                return true
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                isDragging = false
                return true
            }
        }
        return super.onTouchEvent(event)
    }

    /** Recomputes [projected] (rotation + sort) only when [projectedDirty]. */
    private fun ensureProjected() {
        if (!projectedDirty) return
        projectedDirty = false

        val cosX = cos(rotX)
        val sinX = sin(rotX)
        val cosY = cos(rotY)
        val sinY = sin(rotY)

        if (projected.size != points.size) {
            projected = ArrayList(points.size)
            repeat(points.size) { projected.add(Triple(0f, 0f, 0f)) }
        }

        var pMinX = Float.MAX_VALUE; var pMaxX = -Float.MAX_VALUE
        var pMinY = Float.MAX_VALUE; var pMaxY = -Float.MAX_VALUE
        var pMinZ = Float.MAX_VALUE; var pMaxZ = -Float.MAX_VALUE

        for (i in points.indices) {
            val (x, y, z) = points[i]
            val tx = x - centroidX
            val ty = y - centroidY
            val tz = z - centroidZ

            // Rotate around Y axis (horizontal drag → azimuth)
            val rx1 = tx * cosY + tz * sinY
            val ry1 = ty
            val rz1 = -tx * sinY + tz * cosY

            // Rotate around X axis (vertical drag → elevation)
            val rx2 = rx1
            val ry2 = ry1 * cosX - rz1 * sinX
            val rz2 = ry1 * sinX + rz1 * cosX

            projected[i] = Triple(rx2, ry2, rz2)

            if (rx2 < pMinX) pMinX = rx2
            if (rx2 > pMaxX) pMaxX = rx2
            if (ry2 < pMinY) pMinY = ry2
            if (ry2 > pMaxY) pMaxY = ry2
            if (rz2 < pMinZ) pMinZ = rz2
            if (rz2 > pMaxZ) pMaxZ = rz2
        }

        minX = pMinX; maxX = pMaxX
        minY = pMinY; maxY = pMaxY
        minZ = pMinZ; maxZ = pMaxZ

        // Painter's algorithm: sort back-to-front so near points appear on top
        projected.sortBy { it.third }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawColor(Color.BLACK)

        if (points.isEmpty()) return

        ensureProjected()

        val rangeX = max(1f, maxX - minX)
        val rangeY = max(1f, maxY - minY)
        val rangeZ = max(1f, maxZ - minZ)

        val padX = width * 0.05f
        val padY = height * 0.05f
        val drawW = width - 2 * padX
        val drawH = height - 2 * padY

        for ((rx, ry, rz) in projected) {
            val px = padX + (rx - minX) / rangeX * drawW
            val py = padY + (ry - minY) / rangeY * drawH
            val brightness = ((rz - minZ) / rangeZ * 205 + 50).toInt().coerceIn(50, 255)
            paint.color = Color.rgb(brightness, brightness, 255)
            canvas.drawCircle(px, py, POINT_RADIUS, paint)
        }

        // Show a one-time drag hint until the user rotates the cloud
        if (!everRotated) {
            canvas.drawText(
                context.getString(R.string.point_cloud_rotate_hint),
                width / 2f,
                height - HINT_BOTTOM_OFFSET,
                hintPaint,
            )
        }
    }
}
