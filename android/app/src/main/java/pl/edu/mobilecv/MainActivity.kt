package pl.edu.mobilecv

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.google.android.material.chip.Chip
import org.opencv.android.OpenCVLoader
import pl.edu.mobilecv.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Main (and only) activity of the MobileCV application.
 *
 * Responsibilities:
 * - Requests the [Manifest.permission.CAMERA] runtime permission.
 * - Initialises the OpenCV library via [OpenCVLoader.initLocal].
 * - Binds a CameraX [ImageAnalysis] use-case to the lifecycle so that
 *   every new frame is processed by [ImageProcessor] with the filter
 *   chosen by the user.
 * - Renders the processed [Bitmap] into the full-screen [ImageView].
 * - Provides a [FloatingActionButton] to toggle front ↔ back camera.
 * - Provides a horizontal [ChipGroup] to select the active [OpenCvFilter].
 */
class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MobileCV"
    }

    private lateinit var binding: ActivityMainBinding

    // CameraX
    private var cameraProvider: ProcessCameraProvider? = null
    private var lensFacing = CameraSelector.LENS_FACING_BACK

    // OpenCV
    private val imageProcessor = ImageProcessor()
    private var currentFilter = OpenCvFilter.ORIGINAL

    // Single-threaded executor so that frame processing is serialised and
    // we never accumulate a backlog of pending frames.
    private lateinit var analysisExecutor: ExecutorService

    // ---------------------------------------------------------------------------
    // Permission launcher
    // ---------------------------------------------------------------------------

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            startCamera()
        } else {
            Toast.makeText(this, getString(R.string.camera_permission_denied), Toast.LENGTH_LONG)
                .show()
            finish()
        }
    }

    // ---------------------------------------------------------------------------
    // Lifecycle
    // ---------------------------------------------------------------------------

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        analysisExecutor = Executors.newSingleThreadExecutor()

        initOpenCv()
        setupFilterChips()
        setupCameraSwitchButton()
        requestCameraOrStart()
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
    }

    // ---------------------------------------------------------------------------
    // Initialisation helpers
    // ---------------------------------------------------------------------------

    /**
     * Initialise the OpenCV native library bundled inside the APK.
     *
     * [OpenCVLoader.initLocal] loads the `.so` that was packaged during the
     * build; no separate OpenCV Manager app is required.
     */
    private fun initOpenCv() {
        if (!OpenCVLoader.initLocal()) {
            Log.e(TAG, "OpenCV initialisation failed")
            Toast.makeText(this, getString(R.string.opencv_init_error), Toast.LENGTH_LONG).show()
        } else {
            Log.i(TAG, "OpenCV initialised successfully")
        }
    }

    /**
     * Inflate one [Chip] per [OpenCvFilter] entry into the [ChipGroup].
     *
     * The chip group is configured for single selection; tapping a chip
     * updates [currentFilter] and the overlay label.
     */
    private fun setupFilterChips() {
        binding.chipGroupFilters.isSingleSelection = true
        binding.chipGroupFilters.isSelectionRequired = true

        OpenCvFilter.entries.forEach { filter ->
            val chip = Chip(this).apply {
                id = View.generateViewId()
                text = filter.displayName
                isCheckable = true
                isChecked = (filter == currentFilter)
                setOnCheckedChangeListener { _, checked ->
                    if (checked) {
                        currentFilter = filter
                        binding.textViewCurrentFilter.text = filter.displayName
                    }
                }
            }
            binding.chipGroupFilters.addView(chip)
        }

        binding.textViewCurrentFilter.text = currentFilter.displayName
    }

    /** Toggle the lens direction and re-bind the camera. */
    private fun setupCameraSwitchButton() {
        binding.fabSwitchCamera.setOnClickListener {
            lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
                CameraSelector.LENS_FACING_FRONT
            } else {
                CameraSelector.LENS_FACING_BACK
            }
            startCamera()
        }
    }

    // ---------------------------------------------------------------------------
    // Camera permission + startup
    // ---------------------------------------------------------------------------

    private fun requestCameraOrStart() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    // ---------------------------------------------------------------------------
    // CameraX
    // ---------------------------------------------------------------------------

    /**
     * Obtain a [ProcessCameraProvider] and bind the [ImageAnalysis] use-case.
     *
     * Called on first launch and whenever the user switches cameras.
     */
    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            cameraProvider = future.get()
            bindAnalysisUseCase()
        }, ContextCompat.getMainExecutor(this))
    }

    /**
     * Unbind all existing use-cases and bind a fresh [ImageAnalysis] for the
     * currently selected lens.
     *
     * Frames are delivered in [ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888] so
     * that [ImageProxy.toBitmap] yields an ARGB_8888 [Bitmap] directly,
     * avoiding a manual YUV→RGB conversion step.
     */
    private fun bindAnalysisUseCase() {
        val provider = cameraProvider ?: return

        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(lensFacing)
            .build()

        val imageAnalysis = ImageAnalysis.Builder()
            .setTargetResolution(Size(640, 480))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            .also { analysis ->
                analysis.setAnalyzer(analysisExecutor) { imageProxy ->
                    processFrame(imageProxy)
                }
            }

        try {
            provider.unbindAll()
            provider.bindToLifecycle(this, cameraSelector, imageAnalysis)
        } catch (e: Exception) {
            Log.e(TAG, "Camera binding failed", e)
        }
    }

    // ---------------------------------------------------------------------------
    // Frame processing
    // ---------------------------------------------------------------------------

    /**
     * Convert [imageProxy] to a correctly-oriented [Bitmap], apply the
     * current [OpenCvFilter] via [ImageProcessor], and post the result to
     * the UI thread for display.
     *
     * **Must be called on the [analysisExecutor] thread.**
     *
     * @param imageProxy Frame delivered by the [ImageAnalysis] analyser.
     *                   Closed at the end of this method.
     */
    private fun processFrame(imageProxy: ImageProxy) {
        try {
            // toBitmap() returns ARGB_8888 when OUTPUT_IMAGE_FORMAT_RGBA_8888 is set.
            val bitmap: Bitmap = imageProxy.toBitmap()
            val rotation: Int = imageProxy.imageInfo.rotationDegrees

            val oriented: Bitmap = orientBitmap(bitmap, rotation, lensFacing)
            val processed: Bitmap = imageProcessor.processFrame(oriented, currentFilter)

            runOnUiThread {
                binding.imageViewPreview.setImageBitmap(processed)
            }
        } finally {
            imageProxy.close()
        }
    }

    /**
     * Rotate [bitmap] by [rotationDegrees] and, when using the front camera,
     * mirror it horizontally so the preview is not laterally inverted.
     *
     * @param bitmap Source bitmap (ARGB_8888).
     * @param rotationDegrees CW rotation required to make the image upright.
     * @param lensFacing [CameraSelector.LENS_FACING_FRONT] or BACK.
     * @return Correctly oriented bitmap (may be the original object if no
     *         transformation is needed).
     */
    private fun orientBitmap(bitmap: Bitmap, rotationDegrees: Int, lensFacing: Int): Bitmap {
        val isFront = lensFacing == CameraSelector.LENS_FACING_FRONT
        if (rotationDegrees == 0 && !isFront) return bitmap

        val matrix = Matrix().apply {
            if (rotationDegrees != 0) postRotate(rotationDegrees.toFloat())
            if (isFront) postScale(-1f, 1f, bitmap.width / 2f, bitmap.height / 2f)
        }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
}
