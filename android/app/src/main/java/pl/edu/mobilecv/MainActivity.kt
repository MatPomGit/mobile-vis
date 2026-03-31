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
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import com.google.android.material.chip.Chip
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.android.material.tabs.TabLayout
import org.opencv.android.OpenCVLoader
import pl.edu.mobilecv.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

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

        private const val PREFS_NAME = "mobilecv_prefs"
        private const val PREF_ROBOT_HOST = "robot_host"
        private const val PREF_ROBOT_PORT = "robot_port"
        private const val PREF_CAMERA_RESOLUTION = "camera_resolution"
    }

    private lateinit var binding: ActivityMainBinding

    // CameraX
    private var cameraProvider: ProcessCameraProvider? = null
    @Volatile
    private var lensFacing = CameraSelector.LENS_FACING_BACK
    @Volatile
    private var currentResolution: CameraResolution = CameraResolution.DEFAULT
    private var imageCapture: ImageCapture? = null
    private var videoCapture: VideoCapture<Recorder>? = null
    private var activeRecording: Recording? = null
    private var isRecording = false

    // OpenCV + MediaPipe
    private val imageProcessor = ImageProcessor()
    private val mediaPipeProcessor: MediaPipeProcessor by lazy { MediaPipeProcessor(this) }

    /** Guards against concurrent model downloads when POSE tab is tapped rapidly. */
    @Volatile
    private var mediaPipeDownloadInProgress = false

    @Volatile
    private var currentFilter = OpenCvFilter.ORIGINAL

    @Volatile
    private var isActiveVisionEnabled = false

    // Calibration
    val cameraCalibrator = CameraCalibrator()

    // ROS2 / ROSBridge
    private val rosBridgeClient = RosBridgeClient()
    private var lastRobotHost: String = "192.168.1.100"
    private var lastRobotPort: Int = RosBridgeClient.DEFAULT_PORT

    // FPS and diagnostics
    private val fpsCounter = FpsCounter()

    /** Last processed-frame width in pixels (updated on the analysis executor). */
    @Volatile
    private var frameWidth: Int = 0

    /** Last processed-frame height in pixels (updated on the analysis executor). */
    @Volatile
    private var frameHeight: Int = 0

    /** Processing time for the last frame in milliseconds (updated on the analysis executor). */
    @Volatile
    private var lastProcessingTimeMs: Long = 0

    // Bitmap double-buffer: we keep two references so we can safely recycle the one that
    // was displayed TWO frames ago.  Recycling one frame earlier risks a RenderThread race
    // where the GPU is still uploading the bitmap to a texture when we mark it recycled.
    private var lastProcessedBitmap: Bitmap? = null
    private var pendingRecycleBitmap: Bitmap? = null

    /**
     * Prevents stacking redundant UI-update runnables on the main thread when
     * frame processing keeps up faster than vsync.  [AtomicBoolean.compareAndSet]
     * atomically flips the flag from `false` to `true` so only one UI post is
     * outstanding at a time; cleared back to `false` inside the posted runnable.
     */
    private val uiUpdatePending = AtomicBoolean(false)

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

        // Restore persisted robot connection settings.
        val prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
        lastRobotHost = prefs.getString(PREF_ROBOT_HOST, lastRobotHost) ?: lastRobotHost
        lastRobotPort = prefs.getInt(PREF_ROBOT_PORT, lastRobotPort)
        currentResolution = CameraResolution.entries.find {
            it.name == prefs.getString(PREF_CAMERA_RESOLUTION, null)
        } ?: run {
            val saved = prefs.getString(PREF_CAMERA_RESOLUTION, null)
            if (saved != null) {
                Log.w(TAG, "Unrecognised resolution preference '$saved', falling back to default")
            }
            CameraResolution.DEFAULT
        }

        imageProcessor.calibrator = cameraCalibrator
        imageProcessor.labelFrameCountSuffix = getString(R.string.calibration_overlay_frames_suffix)
        imageProcessor.labelBoardNotFound = getString(R.string.calibration_overlay_board_not_found)
        imageProcessor.labelNoCalibration = getString(R.string.calibration_overlay_no_calibration)
        imageProcessor.labelOdometryTracks = getString(R.string.vo_overlay_tracks)
        imageProcessor.labelPointCloud = getString(R.string.vo_overlay_point_cloud)
        imageProcessor.onMarkersDetected = { detections ->
            rosBridgeClient.publishMarkers(detections)
        }

        // Initialise MediaPipe processor on the analysis executor so that native
        // resources are created on the same thread they will be used on.
        analysisExecutor.execute {
            mediaPipeProcessor.initialize()
            imageProcessor.mediaPipeProcessor = mediaPipeProcessor
        }

        rosBridgeClient.onStateChanged = { state ->
            if (!isDestroyed && !isFinishing) {
                runOnUiThread { onRosBridgeStateChanged(state) }
            }
        }

        initOpenCv()
        setupAnalysisTabs()
        setupActiveVisionToggle()
        setupCameraSwitchButton()
        setupCaptureButton()
        setupCalibrationFab()
        setupRobotConnectionFab()
        setupResolutionFab()
        requestPermissionsOrStart()
    }


    override fun onResume() {
        super.onResume()
        enableImmersiveFullscreen()
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) {
            enableImmersiveFullscreen()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        activeRecording?.stop()
        analysisExecutor.execute { mediaPipeProcessor.close() }
        analysisExecutor.shutdown()
        pendingRecycleBitmap?.recycle()
        pendingRecycleBitmap = null
        lastProcessedBitmap?.recycle()
        lastProcessedBitmap = null
        rosBridgeClient.shutdown()
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

    private fun setupActiveVisionToggle() {
        binding.switchActiveVision.setOnCheckedChangeListener { _, isChecked ->
            isActiveVisionEnabled = isChecked
            imageProcessor.isActiveVisionEnabled = isChecked
            val messageRes = if (isChecked) {
                R.string.active_vision_enabled
            } else {
                R.string.active_vision_disabled
            }
            Toast.makeText(this, getString(messageRes), Toast.LENGTH_SHORT).show()
        }
    }

    private fun enableImmersiveFullscreen() {
        WindowCompat.setDecorFitsSystemWindows(window, false)
        WindowInsetsControllerCompat(window, window.decorView).let { controller ->
            controller.hide(WindowInsetsCompat.Type.systemBars())
            controller.systemBarsBehavior =
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        }
    }

    /**
     * Rebuild the chip group to show only the filters belonging to [mode].
     *
     * The first filter in the mode is automatically selected.
     * When [AnalysisMode.POSE] is selected and MediaPipe models have not yet
     * been downloaded, a background download is started automatically.
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

        // Show calibration FAB only in CALIBRATION mode;
        // robot FAB is hidden in CALIBRATION mode to avoid visual clutter.
        binding.fabCalibrationMenu.visibility =
            if (mode == AnalysisMode.CALIBRATION) View.VISIBLE else View.GONE
        binding.fabRobotConnection.visibility =
            if (mode == AnalysisMode.CALIBRATION) View.GONE else View.VISIBLE

        // Trigger background model download when the user first opens the POSE tab.
        if (mode == AnalysisMode.POSE && !ModelDownloadManager.areAllModelsReady(this)) {
            startMediaPipeModelDownload()
        }
    }

    /**
     * Start downloading MediaPipe model files in the background.
     *
     * A volatile flag prevents concurrent duplicate downloads if the user switches
     * tabs rapidly.  Shows a toast when download starts, and re-initialises
     * [MediaPipeProcessor] once all files are available.
     */
    private fun startMediaPipeModelDownload() {
        if (mediaPipeDownloadInProgress) return
        mediaPipeDownloadInProgress = true

        Toast.makeText(this, getString(R.string.mediapipe_models_downloading), Toast.LENGTH_LONG)
            .show()

        analysisExecutor.execute {
            try {
                val success = ModelDownloadManager.downloadMissingModels(this)
                if (success) {
                    // Re-initialise the processor now that model files are present.
                    mediaPipeProcessor.close()
                    mediaPipeProcessor.initialize()
                    imageProcessor.mediaPipeProcessor = mediaPipeProcessor
                    runOnUiThread {
                        Toast.makeText(
                            this,
                            getString(R.string.mediapipe_models_ready),
                            Toast.LENGTH_SHORT,
                        ).show()
                    }
                } else {
                    runOnUiThread {
                        Toast.makeText(
                            this,
                            getString(R.string.mediapipe_models_download_failed),
                            Toast.LENGTH_LONG,
                        ).show()
                    }
                }
            } finally {
                mediaPipeDownloadInProgress = false
            }
        }
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
     * Configure the robot connection FAB.
     *
     * Tapping it opens [RobotConnectionSheet] pre-filled with the last used
     * host/port and the current [RosBridgeClient] state.
     */
    private fun setupRobotConnectionFab() {
        binding.fabRobotConnection.setOnClickListener {
            openRobotConnectionMenu()
        }
    }

    /**
     * Configure the resolution picker FAB.
     *
     * Tapping it opens [ResolutionBottomSheet] so the user can choose a lower
     * resolution for better real-time performance or a higher one for quality.
     */
    private fun setupResolutionFab() {
        binding.fabResolution.setOnClickListener {
            openResolutionMenu()
        }
    }

    /**
     * Open the robot connection bottom sheet.
     */
    private fun openRobotConnectionMenu() {
        RobotConnectionSheet().apply {
            currentState = rosBridgeClient.state
            lastHost = this@MainActivity.lastRobotHost
            lastPort = this@MainActivity.lastRobotPort
            onConnect = { host, port ->
                this@MainActivity.lastRobotHost = host
                this@MainActivity.lastRobotPort = port
                // Persist settings across app restarts.
                getSharedPreferences(PREFS_NAME, MODE_PRIVATE).edit()
                    .putString(PREF_ROBOT_HOST, host)
                    .putInt(PREF_ROBOT_PORT, port)
                    .apply()
                rosBridgeClient.connect(host, port)
            }
            onDisconnect = {
                rosBridgeClient.disconnect()
            }
        }.show(supportFragmentManager, RobotConnectionSheet.TAG)
    }

    /**
     * Open the resolution picker bottom sheet.
     *
     * When the user confirms a new resolution, it is persisted to SharedPreferences
     * and the camera use-cases are rebound with the new target size.
     */
    private fun openResolutionMenu() {
        ResolutionBottomSheet().apply {
            currentResolution = this@MainActivity.currentResolution
            onResolutionSelected = { resolution ->
                this@MainActivity.currentResolution = resolution
                getSharedPreferences(PREFS_NAME, MODE_PRIVATE).edit()
                    .putString(PREF_CAMERA_RESOLUTION, resolution.name)
                    .apply()
                startCamera()
                Toast.makeText(
                    this@MainActivity,
                    getString(R.string.resolution_applied, resolution.displayName),
                    Toast.LENGTH_SHORT,
                ).show()
            }
        }.show(supportFragmentManager, ResolutionBottomSheet.TAG)
    }

    /**
     * React to [RosBridgeClient] state changes: update the FAB tint and show a toast.
     *
     * Must be called on the main thread.
     */
    private fun onRosBridgeStateChanged(state: RosBridgeClient.State) {
        val (toastRes, colorRes) = when (state) {
            RosBridgeClient.State.CONNECTED ->
                Pair(R.string.robot_toast_connected, R.color.robot_connected)
            RosBridgeClient.State.DISCONNECTED ->
                Pair(R.string.robot_toast_disconnected, R.color.button_robot_default)
            RosBridgeClient.State.ERROR ->
                Pair(R.string.robot_toast_error, R.color.robot_error)
            RosBridgeClient.State.CONNECTING -> return
        }
        Toast.makeText(this, getString(toastRes), Toast.LENGTH_SHORT).show()
        binding.fabRobotConnection.backgroundTintList =
            android.content.res.ColorStateList.valueOf(
                ContextCompat.getColor(this, colorRes)
            )
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
     * [ImageAnalysis] drives the live processed preview.  The resolution is
     * controlled by [currentResolution] so the user can trade image quality
     * for lower per-frame processing latency.
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
                    currentResolution.size,
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
            fpsCounter.reset()
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
     * If a UI update is already pending (i.e., the main thread has not yet
     * consumed the previous frame), the processed result is discarded to
     * avoid stacking redundant runnables.  Combined with
     * [ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST] this keeps end-to-end latency
     * as low as possible while the UI remains fully responsive.
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

            // Skip the UI post if the main thread is still rendering the previous frame.
            // This prevents runOnUiThread runnables from accumulating under load and
            // reduces perceived jitter without dropping any analysis work.
            // compareAndSet atomically transitions false → true, ensuring at most one
            // UI update runnable is queued at any given time.
            if (uiUpdatePending.compareAndSet(false, true)) {
                runOnUiThread {
                    // Recycle the bitmap from two frames ago – by then the RenderThread has had
                    // at least one full vsync cycle to upload the previous bitmap to a GPU texture
                    // and no longer reads from its CPU pixel buffer.
                    pendingRecycleBitmap?.recycle()
                    pendingRecycleBitmap = lastProcessedBitmap
                    binding.imageViewPreview.setImageBitmap(processed)
                    lastProcessedBitmap = processed
                    uiUpdatePending.set(false)
                }
            } else {
                processed.recycle()
            // Measure filter processing time.
            val frameStart = System.nanoTime()

            // Apply OpenCV filters.
            val processed: Bitmap = imageProcessor.processFrame(oriented, currentFilter)

            lastProcessingTimeMs = (System.nanoTime() - frameStart) / 1_000_000L
            frameWidth = processed.width
            frameHeight = processed.height
            fpsCounter.onFrame()

            runOnUiThread {
                // Recycle the bitmap from two frames ago – by then the RenderThread has had
                // at least one full vsync cycle to upload the previous bitmap to a GPU texture
                // and no longer reads from its CPU pixel buffer.
                pendingRecycleBitmap?.recycle()
                pendingRecycleBitmap = lastProcessedBitmap
                binding.imageViewPreview.setImageBitmap(processed)
                lastProcessedBitmap = processed

                updateDiagnosticsOverlay(
                    fps = fpsCounter.fps,
                    width = frameWidth,
                    height = frameHeight,
                    processingMs = lastProcessingTimeMs,
                    filter = currentFilter,
                    isFrontCamera = lensFacing == CameraSelector.LENS_FACING_FRONT,
                    activeVisionEnabled = isActiveVisionEnabled,
                )
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

    /**
     * Update the on-screen diagnostics overlay with current performance metrics.
     *
     * Must be called on the main thread.
     *
     * @param fps            Current frames per second.
     * @param width          Processed frame width in pixels.
     * @param height         Processed frame height in pixels.
     * @param processingMs   Time spent processing the last frame (milliseconds).
     * @param filter         Currently active [OpenCvFilter].
     * @param isFrontCamera  `true` if the front-facing camera is active.
     */
    private fun updateDiagnosticsOverlay(
        fps: Double,
        width: Int,
        height: Int,
        processingMs: Long,
        filter: OpenCvFilter,
        isFrontCamera: Boolean,
        activeVisionEnabled: Boolean,
    ) {
        val cameraLabel = if (isFrontCamera) {
            getString(R.string.diagnostics_camera_front)
        } else {
            getString(R.string.diagnostics_camera_back)
        }
        val text = buildString {
            appendLine(getString(R.string.diagnostics_fps, fps))
            if (width > 0 && height > 0) {
                appendLine(getString(R.string.diagnostics_resolution, width, height))
            }
            appendLine(getString(R.string.diagnostics_processing_time, processingMs))
            appendLine(getString(R.string.diagnostics_filter, filter.displayName))
            appendLine(getString(R.string.diagnostics_active_vision, activeVisionEnabled))
            append(cameraLabel)
        }
        binding.textViewDiagnostics.text = text
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
