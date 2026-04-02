package pl.edu.mobilecv

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import kotlin.math.max
import kotlin.math.min

/**
 * Activity for loading and visualizing a saved point cloud (CSV or PLY).
 *
 * Supports the CSV format (x,y) and PLY format (x y z per line after header)
 * exported by [MainActivity].
 */
class PointCloudViewerActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "PointCloudViewer"
    }

    private lateinit var cloudView: PointCloudView
    private lateinit var textStatus: TextView
    private lateinit var btnLoad: Button
    private lateinit var btnBack: Button

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

        btnLoad.setOnClickListener {
            filePicker.launch("*/*")
        }

        btnBack.setOnClickListener {
            finish()
        }

        textStatus.text = getString(R.string.point_cloud_viewer_empty)
    }

    private fun loadPointCloud(uri: Uri) {
        try {
            val points = mutableListOf<Triple<Float, Float, Float>>()
            contentResolver.openInputStream(uri)?.bufferedReader()?.use { reader ->
                val filename = uri.lastPathSegment ?: ""
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
                    // CSV format: x,y[,z] per line; skip comment lines
                    for (line in reader.lineSequence()) {
                        if (line.startsWith("#") || line.startsWith("x")) continue
                        val parts = line.trim().split(",")
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
                Toast.makeText(this, getString(R.string.point_cloud_viewer_error, "Brak punktów w pliku"), Toast.LENGTH_SHORT).show()
                return
            }

            cloudView.setPoints(points)
            textStatus.text = getString(R.string.point_cloud_viewer_loaded, points.size)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load point cloud", e)
            Toast.makeText(this, getString(R.string.point_cloud_viewer_error, e.message ?: "?"), Toast.LENGTH_LONG).show()
        }
    }
}

/**
 * Custom view that renders a point cloud as a 2D scatter plot
 * with depth (z) encoded as colour brightness.
 */
class PointCloudView(context: Context, attrs: android.util.AttributeSet?) : View(context, attrs) {

    private var points: List<Triple<Float, Float, Float>> = emptyList()
    private val paint = Paint().apply { isAntiAlias = true }

    fun setPoints(pts: List<Triple<Float, Float, Float>>) {
        points = pts
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (points.isEmpty()) return

        canvas.drawColor(Color.BLACK)

        val xs = points.map { it.first }
        val ys = points.map { it.second }
        val zs = points.map { it.third }

        val minX = xs.min(); val maxX = xs.max()
        val minY = ys.min(); val maxY = ys.max()
        val minZ = zs.min(); val maxZ = zs.max()
        val rangeX = max(1f, maxX - minX)
        val rangeY = max(1f, maxY - minY)
        val rangeZ = max(1f, maxZ - minZ)

        val padX = width * 0.05f
        val padY = height * 0.05f
        val drawW = width - 2 * padX
        val drawH = height - 2 * padY

        for ((x, y, z) in points) {
            val px = padX + (x - minX) / rangeX * drawW
            val py = padY + (y - minY) / rangeY * drawH
            val brightness = ((z - minZ) / rangeZ * 205 + 50).toInt().coerceIn(50, 255)
            paint.color = Color.rgb(brightness, brightness, 255)
            canvas.drawCircle(px, py, 3f, paint)
        }
    }
}
