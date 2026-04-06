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
            val colors = mutableListOf<Int>()
            contentResolver.openInputStream(uri)?.bufferedReader()?.use { reader ->
                val filename = getDisplayName(uri)
                val isPly = filename.endsWith(".ply", ignoreCase = true)

                if (isPly) {
                    var headerDone = false
                    var vertexCount = 0
                    var propertyXIdx = -1; var propertyYIdx = -1; var propertyZIdx = -1
                    var propertyRIdx = -1; var propertyGIdx = -1; var propertyBIdx = -1
                    var currentPropertyIdx = 0

                    val lines = reader.readLines()
                    var lineIdx = 0

                    while (lineIdx < lines.size && !headerDone) {
                        val line = lines[lineIdx].trim()
                        lineIdx++
                        if (line.startsWith("element vertex")) {
                            vertexCount = line.split(" ").lastOrNull()?.toIntOrNull() ?: 0
                        } else if (line.startsWith("property float") || line.startsWith("property uchar")) {
                            val propName = line.split(" ").lastOrNull()
                            when (propName) {
                                "x" -> propertyXIdx = currentPropertyIdx
                                "y" -> propertyYIdx = currentPropertyIdx
                                "z" -> propertyZIdx = currentPropertyIdx
                                "red" -> propertyRIdx = currentPropertyIdx
                                "green" -> propertyGIdx = currentPropertyIdx
                                "blue" -> propertyBIdx = currentPropertyIdx
                            }
                            currentPropertyIdx++
                        } else if (line == "end_header") {
                            headerDone = true
                        }
                    }

                    while (lineIdx < lines.size && points.size < vertexCount) {
                        val line = lines[lineIdx].trim()
                        lineIdx++
                        if (line.isEmpty()) continue
                        val parts = line.split(Regex("\\s+"))
                        if (parts.size >= currentPropertyIdx) {
                            val x = if (propertyXIdx != -1 && propertyXIdx < parts.size) parts[propertyXIdx].toFloatOrNull() ?: 0f else 0f
                            val y = if (propertyYIdx != -1 && propertyYIdx < parts.size) parts[propertyYIdx].toFloatOrNull() ?: 0f else 0f
                            val z = if (propertyZIdx != -1 && propertyZIdx < parts.size) parts[propertyZIdx].toFloatOrNull() ?: 0f else 0f
                            points.add(Triple(x, y, z))
                            
                            if (propertyRIdx != -1 && propertyGIdx != -1 && propertyBIdx != -1) {
                                val r = parts[propertyRIdx].toIntOrNull() ?: 255
                                val g = parts[propertyGIdx].toIntOrNull() ?: 255
                                val b = parts[propertyBIdx].toIntOrNull() ?: 255
                                colors.add(Color.rgb(r, g, b))
                            } else {
                                colors.add(Color.WHITE)
                            }
                        }
                    }
                } else {
                    // CSV format: x,y,z,r,g,b per line; skip header and comment lines
                    for (line in reader.lineSequence()) {
                        val trimmed = line.trim()
                        if (trimmed.startsWith("#") || trimmed.isEmpty()) continue
                        if (trimmed.startsWith("x,y", ignoreCase = true)) continue
                        
                        val parts = trimmed.split(",")
                        if (parts.size >= 2) {
                            val x = parts[0].toFloatOrNull() ?: continue
                            val y = parts[1].toFloatOrNull() ?: continue
                            val z = parts.getOrNull(2)?.toFloatOrNull() ?: 0f
                            points.add(Triple(x, y, z))
                            
                            if (parts.size >= 6) {
                                val r = parts[3].toIntOrNull() ?: 255
                                val g = parts[4].toIntOrNull() ?: 255
                                val b = parts[5].toIntOrNull() ?: 255
                                colors.add(Color.rgb(r, g, b))
                            } else {
                                colors.add(Color.WHITE)
                            }
                        }
                    }
                }
            }

            if (points.isEmpty()) {
                Toast.makeText(this, getString(R.string.point_cloud_viewer_error, "Brak punktów"), Toast.LENGTH_SHORT).show()
                return
            }

            cloudView.setPoints(points, colors)
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

    private data class PointData(
        val x: Float, val y: Float, val z: Float,
        val color: Int
    )

    companion object {
        private const val ROTATION_SENSITIVITY = 0.005f
        private const val POINT_RADIUS = 3f
        private const val HINT_TEXT_ALPHA = 160
        private const val HINT_TEXT_SIZE = 36f
        private const val HINT_BOTTOM_OFFSET = 60f
    }

    private var points: List<PointData> = emptyList()

    private var centroidX = 0f
    private var centroidY = 0f
    private var centroidZ = 0f

    private var projected: MutableList<PointData> = mutableListOf()
    private var vertexBuffer: FloatArray = FloatArray(0)
    private var colorBuffer: IntArray = IntArray(0)
    
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

    private var rotX = 0f
    private var rotY = 0f

    private var lastTouchX = 0f
    private var lastTouchY = 0f
    private var isDragging = false
    private var everRotated = false

    fun setPoints(pts: List<Triple<Float, Float, Float>>, colors: List<Int>? = null) {
        points = pts.mapIndexed { i, triple ->
            PointData(triple.first, triple.second, triple.third, colors?.getOrNull(i) ?: Color.WHITE)
        }
        if (points.isNotEmpty()) {
            var sumX = 0.0; var sumY = 0.0; var sumZ = 0.0
            for (p in points) {
                sumX += p.x; sumY += p.y; sumZ += p.z
            }
            val n = points.size.toDouble()
            centroidX = (sumX / n).toFloat()
            centroidY = (sumY / n).toFloat()
            centroidZ = (sumZ / n).toFloat()
        }
        
        if (projected.size != points.size) {
            projected = ArrayList(points.size)
            repeat(points.size) { projected.add(PointData(0f, 0f, 0f, 0)) }
            vertexBuffer = FloatArray(points.size * 2)
            colorBuffer = IntArray(points.size)
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

        var pMinX = Float.MAX_VALUE; var pMaxX = -Float.MAX_VALUE
        var pMinY = Float.MAX_VALUE; var pMaxY = -Float.MAX_VALUE
        var pMinZ = Float.MAX_VALUE; var pMaxZ = -Float.MAX_VALUE

        for (i in points.indices) {
            val p = points[i]
            val tx = p.x - centroidX
            val ty = p.y - centroidY
            val tz = p.z - centroidZ

            // Rotate around Y axis
            val rx1 = tx * cosY + tz * sinY
            val ry1 = ty
            val rz1 = -tx * sinY + tz * cosY

            // Rotate around X axis
            val rx2 = rx1
            val ry2 = ry1 * cosX - rz1 * sinX
            val rz2 = ry1 * sinX + rz1 * cosX

            projected[i] = PointData(rx2, ry2, rz2, p.color)

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

        // Sort back-to-front
        projected.sortBy { it.z }

        val rangeX = max(1f, maxX - minX)
        val rangeY = max(1f, maxY - minY)
        val rangeZ = max(1f, maxZ - minZ)

        val padX = width * 0.05f
        val padY = height * 0.05f
        val drawW = width - 2 * padX
        val drawH = height - 2 * padY

        for (i in projected.indices) {
            val p = projected[i]
            val px = padX + (p.x - minX) / rangeX * drawW
            val py = padY + (p.y - minY) / rangeY * drawH
            
            vertexBuffer[i * 2] = px
            vertexBuffer[i * 2 + 1] = py

            // Adjust brightness based on depth
            val zFactor = ((p.z - minZ) / rangeZ * 0.5 + 0.5).toFloat()
            val r = (Color.red(p.color) * zFactor).toInt()
            val g = (Color.green(p.color) * zFactor).toInt()
            val b = (Color.blue(p.color) * zFactor).toInt()
            colorBuffer[i] = Color.rgb(r, g, b)
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawColor(Color.BLACK)

        if (points.isEmpty()) return

        ensureProjected()

        paint.strokeWidth = POINT_RADIUS * 2
        paint.strokeCap = Paint.Cap.ROUND
        canvas.drawPoints(vertexBuffer, 0, projected.size * 2, paint)
        
        // Unfortunately drawPoints doesn't support per-point colors easily without GL.
        // We'll stick to individual drawCircle for now but the buffers are ready
        // if we move to a custom shader or many drawPoint calls.
        // Actually, let's optimize the loop by avoiding object allocations.
        for (i in projected.indices) {
            paint.color = colorBuffer[i]
            canvas.drawPoint(vertexBuffer[i * 2], vertexBuffer[i * 2 + 1], paint)
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
