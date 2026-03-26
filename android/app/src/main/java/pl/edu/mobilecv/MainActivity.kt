package pl.edu.mobilecv

import android.Manifest
import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.util.Size
import android.view.View
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.MediaStoreOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.core.content.ContextCompat
import com.google.android.material.chip.Chip
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.android.material.tabs.TabLayout
import org.opencv.android.OpenCVLoader
import pl.edu.mobilecv.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Main (and only) activity of the MobileCV application.
 *
 * Responsibilities:
 * - Requests [Manifest.permission.CAMERA], [Manifest.permission.RECORD_AUDIO]
 *   and (on API < 29) [Manifest.permission.WRITE_EXTERNAL_STORAGE] runtime permissions.
 * - Initialises the OpenCV library via [OpenCVLoader.initLocal].
 * - Shows a [TabLayout] at the top to switch between [AnalysisMode] groups;
 *   the chip group updates accordingly.
 * - Binds CameraX [ImageAnalysis], [ImageCapture] and [VideoCapture] use-cases
 *   so the user can watch a live processed preview, take photos, and record videos.
 * - Renders processed [Bitmap] frames into the full-screen [ImageView].
 * - Provides a camera-switch [FloatingActionButton] and a shutter/capture button.
 */
class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MobileCV"
    }

    private lateinit var binding: ActivityMainBinding

    // CameraX
    private var cameraProvider: ProcessCameraProvider? = null
    @Volatile
    private var lensFacing = CameraSelector.LENS_FACING_BACK
    private var imageCapture: ImageCapture? = null
    private var videoCapture: VideoCapture<Recorder>? = null
    private var activeRecording: Recording? = null
    private var isRecording = false

    // OpenCV
    private val imageProcessor = ImageProcessor()

    @Volatile
    private var currentFilter = OpenCvFilter.ORIGINAL

    // Calibration
    val cameraCalibrator = CameraCalibrator()

    // To prevent OOM, we keep track of the last bitmap set to the ImageView to recycle it.
    private var lastProcessedBitmap: Bitmap? = null

    // Single-threaded executor so that frame processing is serialised and
    // we never accumulate a backlog of pending frames.
    private lateinit var analysisExecutor: ExecutorService

    private val permissionsLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { results ->
        if (results[Manifest.permission.CAMERA] == true) {
            startCamera()
        } else {
            Toast.makeText(this, getString(R.string.camera_permission_denied), Toast.LENGTH_LONG)
                .show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        analysisExecutor = Executors.newSingleThreadExecutor()

        imageProcessor.calibrator = cameraCalibrator
        imageProcessor.labelFrameCountSuffix = getString(R.string.calibration_overlay_frames_suffix)
        imageProcessor.labelBoardNotFound = getString(R.string.calibration_overlay_board_not_found)
        imageProcessor.labelNoCalibration = getString(R.string.calibration_overlay_no_calibration)
        initOpenCv()
        setupAnalysisTabs()
        setupCameraSwitchButton()
        setupCaptureButton()
        setupCalibrationFab()
        requestPermissionsOrStart()
    }

    override fun onDestroy() {
        super.onDestroy()
        activeRecording?.stop()
        analysisExecutor.shutdown()
        lastProcessedBitmap?.recycle()
        lastProcessedBitmap = null
    }

    private fun initOpenCv() {
        if (!OpenCVLoader.initLocal()) {
            Log.e(TAG, "OpenCV initialisation failed")
            Toast.makeText(this, getString(R.string.opencv_init_error), Toast.LENGTH_LONG).show()
        }
    }

    /**
     * Populate the [TabLayout] with one tab per [AnalysisMode].
     *
     * Selecting a tab calls [updateFilterChips] to rebuild the chip group
     * with the filters that belong to that mode.
     */
    private fun setupAnalysisTabs() {
        AnalysisMode.entries.forEach { mode ->
            binding.tabLayoutModes.addTab(
                binding.tabLayoutModes.newTab().setText(mode.displayName)
            )
        }

        binding.tabLayoutModes.addOnTabSelectedListener(object : TabLayout.OnTabSelectedListener {
            override fun onTabSelected(tab: TabLayout.Tab) {
                val mode = AnalysisMode.entries[tab.position]
                updateFilterChips(mode)
            }

            override fun onTabUnselected(tab: TabLayout.Tab) = Unit
            override fun onTabReselected(tab: TabLayout.Tab) = Unit
        })

        // Initialise chip group for the first (pre-selected) tab.
        updateFilterChips(AnalysisMode.entries.first())
    }

    /**
     * Rebuild the chip group to show only the filters belonging to [mode].
     *
     * The first filter in the mode is automatically selected.
     * Precondition: [AnalysisMode.filters] must be non-empty for every enum entry.
     */
    private fun updateFilterChips(mode: AnalysisMode) {
        binding.chipGroupFilters.removeAllViews()

        currentFilter = mode.filters.firstOrNull() ?: return
        binding.textViewCurrentFilter.text = currentFilter.displayName

        binding.chipGroupFilters.isSingleSelection = true
        binding.chipGroupFilters.isSelectionRequired = true

        mode.filters.forEach { filter ->
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

        // Show calibration FAB only in CALIBRATION mode.
        binding.fabCalibrationMenu.visibility =
            if (mode == AnalysisMode.CALIBRATION) View.VISIBLE else View.GONE
    }

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

    /**
     * Configure the shutter/capture button.
     *
     * - Short tap while idle   → take a photo.
     * - Long press while idle  → start video recording.
     * - Short tap while recording → stop video recording.
     */
    private fun setupCaptureButton() {
        binding.btnCapture.setOnClickListener {
            if (isRecording) {
                stopVideoRecording()
            } else {
                takePhoto()
            }
        }

        binding.btnCapture.setOnLongClickListener {
            if (!isRecording) {
                startVideoRecording()
            }
            true
        }

        // Show a tooltip hinting at the long-press action on API 26+.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            binding.btnCapture.tooltipText = getString(R.string.long_press_hint)
        }
    }

    /**
     * Configure the calibration menu FAB.
     *
     * Tapping it opens the [CalibrationBottomSheet] with callbacks wired
     * to [cameraCalibrator].
     */
    private fun setupCalibrationFab() {
        binding.fabCalibrationMenu.setOnClickListener {
            openCalibrationMenu()
        }
    }

    /**
     * Show the calibration bottom sheet and wire its action callbacks.
     */
    private fun openCalibrationMenu() {
        val sheet = CalibrationBottomSheet().apply {
            onCollectFrame = {
                val collected = cameraCalibrator.collectLastFrame()
                if (collected) {
                    val count = cameraCalibrator.frameCount
                    runOnUiThread {
                        Toast.makeText(
                            this@MainActivity,
                            getString(
                                R.string.calibration_frame_collected,
                                count,
                                CameraCalibrator.MIN_FRAMES,
                            ),
                            Toast.LENGTH_SHORT,
                        ).show()
                    }
                } else {
                    runOnUiThread {
                        Toast.makeText(
                            this@MainActivity,
                            getString(R.string.calibration_board_not_visible),
                            Toast.LENGTH_SHORT,
                        ).show()
                    }
                }
                collected
            }
            onCalibrate = {
                val result = cameraCalibrator.calibrate()
                runOnUiThread {
                    val msgRes = if (result != null) R.string.calibration_success
                    else R.string.calibration_failed
                    Toast.makeText(this@MainActivity, getString(msgRes), Toast.LENGTH_SHORT).show()
                }
                result
            }
            onReset = {
                cameraCalibrator.reset()
            }
        }
        sheet.show(supportFragmentManager, CalibrationBottomSheet.TAG)
    }

    // ---------------------------------------------------------------------------
    // Camera permission + startup
    // ---------------------------------------------------------------------------

    private fun requiredPermissions(): Array<String> {
        val perms = mutableListOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            perms.add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
        }
        return perms.toTypedArray()
    }

    private fun allPermissionsGranted(): Boolean =
        requiredPermissions().all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }

    private fun hasAudioPermission(): Boolean =
        ContextCompat.checkSelfPermission(
            this, Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED

    private fun requestPermissionsOrStart() {
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            permissionsLauncher.launch(requiredPermissions())
        }
    }

    // ---------------------------------------------------------------------------
    // CameraX
    // ---------------------------------------------------------------------------

    /**
     * Obtain a [ProcessCameraProvider] and bind the camera use-cases.
     *
     * Called on first launch and whenever the user switches cameras.
     */
    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            cameraProvider = try {
                future.get()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to get camera provider", e)
                null
            }
            bindUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    /**
     * Unbind all existing use-cases and bind [ImageAnalysis], [ImageCapture]
     * and [VideoCapture] for the currently selected lens.
     *
     * [ImageAnalysis] drives the live processed preview.
     * [ImageCapture] enables single-frame photo capture.
     * [VideoCapture] enables video recording.
     */
    private fun bindUseCases() {
        val provider = cameraProvider ?: return

        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(lensFacing)
            .build()

        val resolutionSelector = ResolutionSelector.Builder()
            .setResolutionStrategy(
                ResolutionStrategy(
                    Size(640, 480),
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                )
            )
            .build()

        val imageAnalysis = ImageAnalysis.Builder()
            .setResolutionSelector(resolutionSelector)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            .also { analysis ->
                analysis.setAnalyzer(analysisExecutor) { imageProxy ->
                    processFrame(imageProxy)
                }
            }

        imageCapture = ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .build()

        val recorder = Recorder.Builder()
            .setQualitySelector(QualitySelector.from(Quality.HIGHEST))
            .build()
        videoCapture = VideoCapture.withOutput(recorder)

        try {
            provider.unbindAll()
            val capture = imageCapture ?: return
            val video = videoCapture ?: return
            provider.bindToLifecycle(
                this, cameraSelector,
                imageAnalysis, capture, video
            )
        } catch (e: Exception) {
            Log.e(TAG, "Camera binding failed", e)
        }
    }

    // ---------------------------------------------------------------------------
    // Photo capture
    // ---------------------------------------------------------------------------

    /**
     * Capture a still image and save it to the device's picture gallery
     * using [MediaStore].
     */
    private fun takePhoto() {
        val capture = imageCapture ?: return

        val name = "MobileCV_${System.currentTimeMillis()}"
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/MobileCV")
            }
        }

        val outputOptions = ImageCapture.OutputFileOptions.Builder(
            contentResolver,
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            contentValues
        ).build()

        capture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    Toast.makeText(
                        this@MainActivity,
                        getString(R.string.photo_saved),
                        Toast.LENGTH_SHORT
                    ).show()
                }

                override fun onError(exc: ImageCaptureException) {
                    Log.e(TAG, "Photo capture failed", exc)
                    Toast.makeText(
                        this@MainActivity,
                        getString(R.string.photo_error),
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }
        )
    }

    // ---------------------------------------------------------------------------
    // Video recording
    // ---------------------------------------------------------------------------

    /**
     * Begin recording a video and save it to the device's movies gallery
     * via [MediaStore].
     *
     * Audio is included only when [Manifest.permission.RECORD_AUDIO] is granted;
     * if the permission is absent the video is recorded silently.
     * The [SuppressLint] annotation is safe here because [hasAudioPermission]
     * gates the `withAudioEnabled` call at runtime.
     *
     * The button state is updated to show the recording indicator.
     */
    @SuppressLint("MissingPermission")
    private fun startVideoRecording() {
        val vc = videoCapture ?: return

        isRecording = true
        updateCaptureButtonState()

        val name = "MobileCV_${System.currentTimeMillis()}"
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/MobileCV")
            }
        }

        val mediaStoreOutput = MediaStoreOutputOptions.Builder(
            contentResolver,
            MediaStore.Video.Media.EXTERNAL_CONTENT_URI
        ).setContentValues(contentValues).build()

        val pendingRecording = vc.output.prepareRecording(this, mediaStoreOutput)
        if (hasAudioPermission()) {
            pendingRecording.withAudioEnabled()
        }

        activeRecording = pendingRecording.start(
            ContextCompat.getMainExecutor(this)
        ) { event ->
            if (event is VideoRecordEvent.Finalize) {
                isRecording = false
                activeRecording = null
                updateCaptureButtonState()
                if (event.hasError()) {
                    Log.e(TAG, "Video recording error: ${event.error}")
                    Toast.makeText(this, getString(R.string.video_error), Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this, getString(R.string.video_saved), Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    /** Stop an ongoing video recording and let the finalise callback update the UI. */
    private fun stopVideoRecording() {
        activeRecording?.stop()
    }

    /**
     * Sync the capture button's visual state with [isRecording]:
     * - activated (red ring)  when recording
     * - normal   (white ring) when idle
     */
    private fun updateCaptureButtonState() {
        binding.btnCapture.isActivated = isRecording
        binding.btnCapture.contentDescription = if (isRecording) {
            getString(R.string.stop_recording_description)
        } else {
            getString(R.string.capture_button_description)
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
            // imageProxy.toBitmap() creates a new Bitmap instance.
            val bitmap: Bitmap = imageProxy.toBitmap()
            val rotation: Int = imageProxy.imageInfo.rotationDegrees

            // Rotate and mirror (if front camera) the bitmap.
            val oriented: Bitmap = orientBitmap(bitmap, rotation, lensFacing)
            
            // Apply OpenCV filters.
            val processed: Bitmap = imageProcessor.processFrame(oriented, currentFilter)

            runOnUiThread {
                val toRecycle = lastProcessedBitmap
                binding.imageViewPreview.setImageBitmap(processed)
                lastProcessedBitmap = processed
                // Recycle the previous processed bitmap to free memory.
                toRecycle?.recycle()
            }

            // Cleanup intermediate bitmaps.
            if (oriented !== bitmap) {
                bitmap.recycle()
            }
            oriented.recycle()
        } catch (e: Exception) {
            Log.e(TAG, "Error processing frame", e)
        } finally {
            imageProxy.close()
        }
    }

    private fun orientBitmap(bitmap: Bitmap, rotationDegrees: Int, lensFacing: Int): Bitmap {
        val isFront = lensFacing == CameraSelector.LENS_FACING_FRONT
        if (rotationDegrees == 0 && !isFront) return bitmap

        val matrix = Matrix()
        if (rotationDegrees != 0) {
            matrix.postRotate(rotationDegrees.toFloat())
        }
        if (isFront) {
            // Mirror horizontally for front-facing camera.
            matrix.postScale(-1f, 1f)
        }
        
        // Bitmap.createBitmap handles the matrix and fits the result into a new bitmap.
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
}
