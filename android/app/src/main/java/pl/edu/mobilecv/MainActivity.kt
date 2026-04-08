package pl.edu.mobilecv

import android.graphics.Color
import android.net.Uri
import android.provider.OpenableColumns
import org.opencv.core.Point3
import android.Manifest
import android.annotation.SuppressLint
import android.app.AlertDialog
import android.content.ContentValues
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.SystemClock
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.SeekBar
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

import androidx.core.content.ContextCompat
import androidx.core.content.edit
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.core.view.children
import com.google.android.material.chip.Chip
import com.google.android.material.tabs.TabLayout
import org.opencv.android.OpenCVLoader
import pl.edu.mobilecv.databinding.ActivityMainBinding
import pl.edu.mobilecv.vision.CameraCalibrator
import pl.edu.mobilecv.odometry.VisualOdometryEngine
import pl.edu.mobilecv.odometry.FullOdometryEngine
import java.io.IOException
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.*
import kotlin.getValue

/**
 * Main (and only) activity of the MobileCV application.
 */
class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MobileCV"
        private const val PREFS_NAME = "mobilecv_prefs"
        private const val PREF_CAMERA_RESOLUTION = "camera_resolution"
        private const val RECORDING_TIMER_FORMAT = "%02d:%02d"
        private const val UI_UPDATE_MIN_INTERVAL_NS = 33_000_000L
    }

    private lateinit var binding: ActivityMainBinding
    private val prefs by lazy { getSharedPreferences(PREFS_NAME, MODE_PRIVATE) }

    // CameraX
    private var cameraProvider: ProcessCameraProvider? = null
    @Volatile private var lensFacing = CameraSelector.LENS_FACING_BACK
    @Volatile private var currentResolution: CameraResolution = CameraResolution.DEFAULT
    private var imageCapture: ImageCapture? = null
    private val processedVideoRecorder by lazy { ProcessedVideoRecorder(this) }
    private var isRecording = false
    private var recordingStartTimeMs: Long = 0
    private val recordingTimerHandler = android.os.Handler(android.os.Looper.getMainLooper())
    private var recordingTimerRunnable: Runnable? = null

    // OpenCV + MediaPipe
    private val imageProcessor by lazy { ImageProcessor() }
    private val mediaPipeProcessor: MediaPipeProcessor by lazy { MediaPipeProcessor(this) }
    private val yoloProcessor: YoloProcessor by lazy { YoloProcessor(this) }
    private val rtmDetProcessor: RtmDetProcessor by lazy { RtmDetProcessor(this) }
    private val tfliteProcessor: TfliteProcessor by lazy { TfliteProcessor(this) }

    @Volatile private var mediaPipeDownloadInProgress = false
    private val yoloDownloadInProgress = AtomicBoolean(false)
    private val rtmDetDownloadInProgress = AtomicBoolean(false)
    private val mobilintDownloadInProgress = AtomicBoolean(false)
    private val tfliteDownloadInProgress = AtomicBoolean(false)
    @Volatile private var currentFilter = OpenCvFilter.ORIGINAL
    @Volatile private var currentMode: AnalysisMode = AnalysisMode.entries.first()
    @Volatile private var isActiveVisionEnabled = false
    @Volatile private var isActiveVisionVisualizationEnabled = false

    // Calibration
    val cameraCalibrator = CameraCalibrator()

    // FPS and diagnostics
    private val fpsCounter = FpsCounter()
    @Volatile private var frameWidth: Int = 0
    @Volatile private var frameHeight: Int = 0
    @Volatile private var lastProcessingTimeMs: Long = 0

    private var lastProcessedBitmap: Bitmap? = null
    private var pendingRecycleBitmap: Bitmap? = null
    private val uiUpdatePending = AtomicBoolean(false)
    @Volatile private var lastUiUpdateNs: Long = 0
    private lateinit var cameraAnalysisExecutor: ExecutorService
    private lateinit var backgroundExecutor: ExecutorService
    @Volatile private var cameraStartTimeMs: Long = 0
    private val firstFrameRenderedLogged = AtomicBoolean(false)
    private val exceptionTelemetry = ConcurrentHashMap<String, AtomicInteger>()
    private lateinit var dataCollectionCache: DataCollectionCacheDataStore
    private val telemetryScope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    private fun logExceptionTelemetry(scope: String, category: String, error: Throwable) {
        val key = "$scope:$category"
        val count = exceptionTelemetry.getOrPut(key) { AtomicInteger(0) }.incrementAndGet()
        
        telemetryScope.launch {
            try {
                dataCollectionCache.incrementErrorCount(scope, category)
            } catch (e: Exception) {
                Log.e(TAG, "Persistence failure for telemetry: $key", e)
            }
        }

        val ranking = exceptionTelemetry.entries
            .sortedByDescending { it.value.get() }
            .take(3)
            .joinToString { "${it.key}=${it.value.get()}" }
            .ifBlank { "n/a" }
        Log.i(
            TAG,
            "Exception telemetry key=$key count=$count type=${error::class.java.simpleName} top=$ranking",
        )
    }

    private val permissionsLauncher = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { results ->
        if (results[Manifest.permission.CAMERA] == true) startCamera()
        else { Toast.makeText(this, getString(R.string.camera_permission_denied), Toast.LENGTH_LONG).show(); finish() }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Initialize DataStore and schedule background sync
        dataCollectionCache = DataCollectionCacheDataStore(this)
        AssistantDailyDataSyncWorker.schedule(this)

        // Initialize OpenCV first before any components that might use it are created.
        initOpenCv()

        cameraAnalysisExecutor = Executors.newSingleThreadExecutor()
        backgroundExecutor = Executors.newSingleThreadExecutor()

        currentResolution = CameraResolution.entries.find { it.name == prefs.getString(PREF_CAMERA_RESOLUTION, null) } ?: CameraResolution.DEFAULT

        imageProcessor.calibrator = cameraCalibrator
        imageProcessor.labelFrameCountSuffix = getString(R.string.calibration_overlay_frames_suffix)
        imageProcessor.labelBoardNotFound = getString(R.string.calibration_overlay_board_not_found)
        imageProcessor.labelNoCalibration = getString(R.string.calibration_overlay_no_calibration)
        imageProcessor.labelOdometryTracks = getString(R.string.vo_overlay_tracks)
        imageProcessor.labelPointCloud = getString(R.string.vo_overlay_point_cloud)
        imageProcessor.labelVoMaxFeaturesDesc = getString(R.string.overlay_vo_max_features_desc)
        imageProcessor.labelVoMinParallaxDesc = getString(R.string.overlay_vo_min_parallax_desc)
        imageProcessor.labelVoColorDepthDesc = getString(R.string.overlay_vo_color_depth_desc)
        imageProcessor.labelNoPlanes = getString(R.string.overlay_no_planes)
        imageProcessor.labelNoVanishingPoints = getString(R.string.overlay_no_vp)
        imageProcessor.labelNoLines = getString(R.string.overlay_no_lines)
        imageProcessor.labelPlanes = getString(R.string.overlay_planes)
        imageProcessor.labelLines = getString(R.string.overlay_lines)
        imageProcessor.labelGroups = getString(R.string.overlay_groups)
        imageProcessor.labelGeometryError = getString(R.string.overlay_geometry_error)
        imageProcessor.labelVpError = getString(R.string.overlay_vp_error)

        imageProcessor.onLargeMapDetected = { map ->
            backgroundExecutor.execute {
                saveSlamMap(map, isAutoSave = true)
            }
        }

        imageProcessor.onLoopClosed = { message ->
            runOnUiThread {
                Toast.makeText(this, message, Toast.LENGTH_LONG).show()
            }
        }

        backgroundExecutor.execute {
            try {
                mediaPipeProcessor.initialize()
                imageProcessor.mediaPipeProcessor = mediaPipeProcessor
            } catch (error: Throwable) {
                logExceptionTelemetry("startup_module_init", "mediapipe", error)
                Log.e(TAG, "MediaPipe initialization failed. Other modules remain available.", error)
            }
            try {
                yoloProcessor.initialize()
                imageProcessor.yoloProcessor = yoloProcessor
            } catch (error: Throwable) {
                logExceptionTelemetry("startup_module_init", "yolo", error)
                Log.e(TAG, "YOLO initialization failed. Other modules remain available.", error)
            }
            try {
                rtmDetProcessor.initialize()
                imageProcessor.rtmDetProcessor = rtmDetProcessor
            } catch (error: Throwable) {
                logExceptionTelemetry("startup_module_init", "rtmdet", error)
                Log.e(TAG, "RTMDet initialization failed. Other modules remain available.", error)
            }
            try {
                tfliteProcessor.initialize()
                imageProcessor.tfliteProcessor = tfliteProcessor
            } catch (error: Throwable) {
                logExceptionTelemetry("startup_module_init", "tflite", error)
                Log.e(TAG, "TFLite initialization failed. Other modules remain available.", error)
            }

            // Automatically download missing YOLO models in the background at startup
            // so they are ready regardless of which tab the user opens first.
            if (!ModelDownloadManager.areYoloModelsReady(this) && yoloDownloadInProgress.compareAndSet(false, true)) {
                runOnUiThread {
                    if (!isDestroyed && !isFinishing)
                        Toast.makeText(this, getString(R.string.yolo_models_downloading), Toast.LENGTH_LONG).show()
                }
                try {
                    if (ModelDownloadManager.downloadMissingYoloModels(this)) {
                        yoloProcessor.close()
                        yoloProcessor.initialize()
                        imageProcessor.yoloProcessor = yoloProcessor
                        runOnUiThread {
                            if (!isDestroyed && !isFinishing)
                                Toast.makeText(this, getString(R.string.yolo_models_ready), Toast.LENGTH_SHORT).show()
                        }
                    } else {
                        runOnUiThread {
                            if (!isDestroyed && !isFinishing)
                                Toast.makeText(this, getString(R.string.yolo_models_download_failed), Toast.LENGTH_LONG).show()
                        }
                    }
                } catch (e: Exception) {
                    logExceptionTelemetry("startup_yolo_download", "unexpected", e)
                    Log.e(TAG, "Startup YOLO download failed", e)
                } finally {
                    yoloDownloadInProgress.set(false)
                }
            }

            // Automatically download missing RTMDet models in the background at startup.
            if (!ModelDownloadManager.areRtmDetModelsReady(this) && rtmDetDownloadInProgress.compareAndSet(false, true)) {
                try {
                    if (ModelDownloadManager.downloadMissingRtmDetModels(this)) {
                        rtmDetProcessor.close()
                        rtmDetProcessor.initialize()
                        imageProcessor.rtmDetProcessor = rtmDetProcessor
                    }
                } catch (e: Exception) {
                    logExceptionTelemetry("startup_rtmdet_download", "unexpected", e)
                    Log.e(TAG, "Startup RTMDet download failed", e)
                } finally {
                    rtmDetDownloadInProgress.set(false)
                }
            }
        }

        setupAnalysisTabs()
        applyInitialModeFromIntent()
        setupSliders()
        setupActiveVisionToggle()
        setupActiveVisionVisualizationToggle()
        setupMeshToggle()
        setupCameraSwitchButton()
        setupCaptureButton()
        setupCalibrationFab()
        setupEyeTrackingCalibrationFab()
        setupResolutionFab()
        setupSavePointCloudFab()
        setupBackToMenuButton()
        requestPermissionsOrStart()
    }

    override fun onResume() { super.onResume(); enableImmersiveFullscreen() }
    override fun onWindowFocusChanged(hasFocus: Boolean) { super.onWindowFocusChanged(hasFocus); if (hasFocus) enableImmersiveFullscreen() }

    override fun onDestroy() {
        super.onDestroy()
        telemetryScope.cancel()
        if (isRecording) {
            isRecording = false
            backgroundExecutor.execute {
                processedVideoRecorder.finalize { success ->
                    Log.d(TAG, "Recording finalized on destroy, success=$success")
                }
            }
        }
        stopRecordingTimer()
        backgroundExecutor.execute { 
            mediaPipeProcessor.close()
            yoloProcessor.close()
            imageProcessor.release()
        }
        cameraAnalysisExecutor.shutdown()
        backgroundExecutor.shutdown()
        pendingRecycleBitmap?.recycle(); lastProcessedBitmap?.recycle()
    }

    private fun initOpenCv() { if (!OpenCVLoader.initLocal()) Toast.makeText(this, getString(R.string.opencv_init_error), Toast.LENGTH_LONG).show() }

    @SuppressLint("SetTextI18n")
    private fun setupSliders() {
        // Morphology
        binding.seekBarKernelSize.progress = imageProcessor.morphKernelSize - 1
        updateKernelSizeLabel(imageProcessor.morphKernelSize)
        binding.seekBarKernelSize.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(s: SeekBar, p: Int, f: Boolean) { val half = p + 1; imageProcessor.morphKernelSize = half; updateKernelSizeLabel(half) }
            override fun onStartTrackingTouch(s: SeekBar) {}
            override fun onStopTrackingTouch(s: SeekBar) {}
        })

        // VO Max Features
        binding.seekBarVoMaxFeatures.progress = imageProcessor.voMaxFeatures
        binding.textViewVoMaxFeatures.text = imageProcessor.voMaxFeatures.toString()
        binding.seekBarVoMaxFeatures.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(s: SeekBar, p: Int, f: Boolean) { val v = maxOf(10, p); imageProcessor.voMaxFeatures = v; binding.textViewVoMaxFeatures.text = v.toString() }
            override fun onStartTrackingTouch(s: SeekBar) {}
            override fun onStopTrackingTouch(s: SeekBar) {}
        })

        // VO Min Parallax
        binding.seekBarVoMinParallax.progress = (imageProcessor.voMinParallax * 10).toInt()
        binding.textViewVoMinParallax.text = "%.1f".format(imageProcessor.voMinParallax)
        binding.seekBarVoMinParallax.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(s: SeekBar, p: Int, f: Boolean) { val v = p / 10.0; imageProcessor.voMinParallax = v; binding.textViewVoMinParallax.text = "%.1f".format(v) }
            override fun onStartTrackingTouch(s: SeekBar) {}
            override fun onStopTrackingTouch(s: SeekBar) {}
        })

        binding.switchPoseSmoothing.isChecked = imageProcessor.poseSmoothingEnabled
        binding.switchPoseSmoothing.setOnCheckedChangeListener { _, isChecked ->
            imageProcessor.poseSmoothingEnabled = isChecked
            if (!binding.switchPoseRawVsSmoothed.isChecked) {
                imageProcessor.poseOutputMode = if (isChecked) PoseOutputMode.SMOOTHED else PoseOutputMode.RAW
            }
        }
        binding.switchPoseOneEuro.isChecked = imageProcessor.poseTemporalFilterType == PoseTemporalFilterType.ONE_EURO
        binding.switchPoseOneEuro.setOnCheckedChangeListener { _, isChecked ->
            imageProcessor.poseTemporalFilterType = if (isChecked) {
                PoseTemporalFilterType.ONE_EURO
            } else {
                PoseTemporalFilterType.EMA
            }
        }
        binding.switchPoseRawVsSmoothed.setOnCheckedChangeListener { _, isChecked ->
            imageProcessor.poseOutputMode = if (isChecked) {
                PoseOutputMode.RAW_VS_SMOOTHED
            } else if (imageProcessor.poseSmoothingEnabled) {
                PoseOutputMode.SMOOTHED
            } else {
                PoseOutputMode.RAW
            }
        }
        binding.seekBarPoseEmaAlpha.progress = (imageProcessor.poseEmaAlpha * 100.0).toInt().coerceIn(5, 95)
        binding.textViewPoseEmaAlpha.text = "%.2f".format(imageProcessor.poseEmaAlpha)
        binding.seekBarPoseEmaAlpha.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(s: SeekBar, p: Int, f: Boolean) {
                val v = (p.coerceIn(5, 95)) / 100.0
                imageProcessor.poseEmaAlpha = v
                binding.textViewPoseEmaAlpha.text = "%.2f".format(v)
            }
            override fun onStartTrackingTouch(s: SeekBar) {}
            override fun onStopTrackingTouch(s: SeekBar) {}
        })

        // Geometry Max Planes
        binding.seekBarGeometryMaxPlanes.progress = imageProcessor.geometryMaxPlanes - 1
        binding.textViewGeometryMaxPlanes.text = imageProcessor.geometryMaxPlanes.toString()
        binding.seekBarGeometryMaxPlanes.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(s: SeekBar, p: Int, f: Boolean) {
                val v = p + 1
                imageProcessor.geometryMaxPlanes = v
                binding.textViewGeometryMaxPlanes.text = v.toString()
            }
            override fun onStartTrackingTouch(s: SeekBar) {}
            override fun onStopTrackingTouch(s: SeekBar) {}
        })
    }

    private fun updateKernelSizeLabel(half: Int) {
        val side = 2 * half + 1
        binding.textViewKernelSize.text = getString(R.string.morphology_kernel_size_value, side, side)
    }

    private fun setupMeshToggle() {
        binding.switchVoMesh.setOnCheckedChangeListener { _, isChecked ->
            imageProcessor.isVoMeshEnabled = isChecked
        }
    }

    private fun setupAnalysisTabs() {
        AnalysisMode.entries.forEach { binding.tabLayoutModes.addTab(binding.tabLayoutModes.newTab().setText(it.displayName)) }
        binding.tabLayoutModes.addOnTabSelectedListener(object : TabLayout.OnTabSelectedListener {
            override fun onTabSelected(tab: TabLayout.Tab) { updateFilterChips(AnalysisMode.entries[tab.position]) }
            override fun onTabUnselected(tab: TabLayout.Tab) {}
            override fun onTabReselected(tab: TabLayout.Tab) {}
        })
        updateFilterChips(AnalysisMode.entries.first())
    }

    /**
     * Reads the [MenuActivity.EXTRA_MODE] extra from the launching intent and
     * selects the corresponding tab, overriding the default first-tab selection.
     */
    private fun applyInitialModeFromIntent() {
        val modeName = intent.getStringExtra(MenuActivity.EXTRA_MODE) ?: return
        val index = AnalysisMode.entries.indexOfFirst { it.name == modeName }
        if (index >= 0) binding.tabLayoutModes.getTabAt(index)?.select()
        else Log.w(TAG, "Unknown analysis mode received from intent: $modeName")
    }

    private fun setupActiveVisionToggle() {
        binding.switchActiveVision.setOnCheckedChangeListener { _, isChecked ->
            isActiveVisionEnabled = isChecked
            imageProcessor.isActiveVisionEnabled = isChecked
            updateContextualControls()
        }
    }

    private fun setupActiveVisionVisualizationToggle() {
        binding.switchActiveVisionVisualization.setOnCheckedChangeListener { _, isChecked ->
            isActiveVisionVisualizationEnabled = isChecked
            imageProcessor.isActiveVisionVisualizationEnabled = isChecked
        }
    }

    private fun updateFilterChips(mode: AnalysisMode) {
        currentMode = mode
        binding.chipGroupFilters.removeAllViews()
        val firstFilter = mode.filters.firstOrNull() ?: run { updateContextualControls(); return }
        currentFilter = firstFilter
        binding.textViewCurrentFilter.text = currentFilter.displayName

        mode.filters.forEach { filter ->
            val chip = Chip(this).apply {
                text = filter.displayName; isCheckable = true; isChecked = (filter == currentFilter)
                setOnClickListener {
                    if (isChecked) {
                        currentFilter = filter
                        binding.textViewCurrentFilter.text = filter.displayName
                        binding.chipGroupFilters.children.filterIsInstance<Chip>().forEach { c ->
                            if (c !== this) c.isChecked = false
                        }
                    } else {
                        val defaultFilter = mode.filters.first()
                        currentFilter = defaultFilter
                        binding.textViewCurrentFilter.text = defaultFilter.displayName
                        if (defaultFilter != filter) {
                            (binding.chipGroupFilters.getChildAt(0) as? Chip)?.isChecked = true
                        } else {
                            isChecked = true
                        }
                    }
                    updateContextualControls()
                }
            }
            binding.chipGroupFilters.addView(chip)
        }

        binding.layoutKernelSize.visibility = if (mode == AnalysisMode.MORPHOLOGY) View.VISIBLE else View.GONE
        binding.layoutGeometryMaxPlanes.visibility = if (mode == AnalysisMode.GEOMETRY && currentFilter == OpenCvFilter.PLANE_DETECTION) View.VISIBLE else View.GONE

        val isOdometry = mode == AnalysisMode.ODOMETRY
        binding.layoutVoMaxFeatures.visibility = if (isOdometry) View.VISIBLE else View.GONE
        binding.layoutVoMinParallax.visibility = if (isOdometry) View.VISIBLE else View.GONE
        binding.layoutVoMesh.visibility = if (isOdometry) View.VISIBLE else View.GONE
        binding.layoutPoseTemporalControls.visibility = if (mode == AnalysisMode.MARKERS) View.VISIBLE else View.GONE

        binding.fabCalibrationMenu.visibility = if (mode == AnalysisMode.CALIBRATION) View.VISIBLE else View.GONE
        binding.fabEyeTrackingCalibration.visibility = if (mode == AnalysisMode.POSE && currentFilter == OpenCvFilter.EYE_TRACKING) View.VISIBLE else View.GONE

        if (mode == AnalysisMode.POSE && !ModelDownloadManager.areAllModelsReady(this)) startMediaPipeModelDownload()
        if ((mode == AnalysisMode.YOLO || mode == AnalysisMode.TRACKING) && !ModelDownloadManager.areYoloModelsReady(this)) startYoloModelDownload()
        if (mode == AnalysisMode.RTMDET && !ModelDownloadManager.areRtmDetModelsReady(this)) startRtmDetModelDownload()
        if (mode == AnalysisMode.MOBILINT && !ModelDownloadManager.areMobilintModelsReady(this)) startMobilintModelDownload()
        if (mode == AnalysisMode.TFLITE && !ModelDownloadManager.areTfliteModelsReady(this)) startTfliteModelDownload()

        updateContextualControls()
    }

    private fun updateContextualControls() {
        val isFiltersMode = currentMode == AnalysisMode.FILTERS
        val isGeometryMode = currentMode == AnalysisMode.GEOMETRY
        
        binding.switchActiveVision.visibility = if (isFiltersMode) View.VISIBLE else View.GONE
        binding.switchActiveVisionVisualization.visibility =
            if (isFiltersMode && isActiveVisionEnabled) View.VISIBLE else View.GONE
        binding.fabSavePointCloud.visibility =
            if (currentFilter == OpenCvFilter.POINT_CLOUD) View.VISIBLE else View.GONE
        binding.fabSaveSlamMap.visibility =
            if (currentFilter.isFullOdometry) View.VISIBLE else View.GONE
        binding.fabLoadSlamMap.visibility =
            if (currentFilter.isFullOdometry) View.VISIBLE else View.GONE
        binding.fabEyeTrackingCalibration.visibility =
            if (currentMode == AnalysisMode.POSE && currentFilter == OpenCvFilter.EYE_TRACKING) View.VISIBLE else View.GONE
        
        binding.layoutGeometryMaxPlanes.visibility = if (isGeometryMode && currentFilter == OpenCvFilter.PLANE_DETECTION) View.VISIBLE else View.GONE
    }

    private fun startMediaPipeModelDownload() {
        if (mediaPipeDownloadInProgress) return
        mediaPipeDownloadInProgress = true
        Toast.makeText(this, getString(R.string.mediapipe_models_downloading), Toast.LENGTH_LONG).show()
        backgroundExecutor.execute {
            try {
                if (ModelDownloadManager.downloadMissingModels(this)) {
                    mediaPipeProcessor.close(); mediaPipeProcessor.initialize()
                    runOnUiThread { Toast.makeText(this, getString(R.string.mediapipe_models_ready), Toast.LENGTH_SHORT).show() }
                }
            } finally { mediaPipeDownloadInProgress = false }
        }
    }

    private fun startYoloModelDownload() {
        if (!yoloDownloadInProgress.compareAndSet(false, true)) return
        Toast.makeText(this, getString(R.string.yolo_models_downloading), Toast.LENGTH_LONG).show()
        backgroundExecutor.execute {
            try {
                if (ModelDownloadManager.downloadMissingYoloModels(this)) {
                    yoloProcessor.close(); yoloProcessor.initialize()
                    imageProcessor.yoloProcessor = yoloProcessor
                    runOnUiThread {
                        if (!isDestroyed && !isFinishing)
                            Toast.makeText(this, getString(R.string.yolo_models_ready), Toast.LENGTH_SHORT).show()
                    }
                } else {
                    runOnUiThread {
                        if (!isDestroyed && !isFinishing)
                            Toast.makeText(this, getString(R.string.yolo_models_download_failed), Toast.LENGTH_LONG).show()
                    }
                }
            } catch (e: IOException) {
                logExceptionTelemetry("yolo_download", "model_io", e)
                Log.e(TAG, "YOLO model download failed: model I/O error", e)
                runOnUiThread {
                    if (!isDestroyed && !isFinishing)
                        Toast.makeText(this, getString(R.string.yolo_models_download_failed), Toast.LENGTH_LONG).show()
                }
            } catch (e: IllegalStateException) {
                logExceptionTelemetry("yolo_download", "state", e)
                Log.e(TAG, "YOLO model download failed: invalid processor state", e)
                runOnUiThread {
                    if (!isDestroyed && !isFinishing)
                        Toast.makeText(this, getString(R.string.yolo_models_download_failed), Toast.LENGTH_LONG).show()
                }
            } catch (e: Exception) {
                logExceptionTelemetry("yolo_download", "unexpected", e)
                Log.e(
                    TAG,
                    "Unhandled YOLO model download error type=${e::class.java.name} message=${e.message}",
                    e,
                )
                runOnUiThread {
                    if (!isDestroyed && !isFinishing)
                        Toast.makeText(this, getString(R.string.yolo_models_download_failed), Toast.LENGTH_LONG).show()
                }
            } finally { yoloDownloadInProgress.set(false) }
        }
    }

    private fun startRtmDetModelDownload() {
        if (!rtmDetDownloadInProgress.compareAndSet(false, true)) return
        runOnUiThread { Toast.makeText(this, getString(R.string.rtmdet_models_downloading), Toast.LENGTH_LONG).show() }
        backgroundExecutor.execute {
            try {
                if (ModelDownloadManager.downloadMissingRtmDetModels(this)) {
                    rtmDetProcessor.close()
                    rtmDetProcessor.initialize()
                    imageProcessor.rtmDetProcessor = rtmDetProcessor
                    runOnUiThread {
                        if (!isDestroyed && !isFinishing)
                            Toast.makeText(this, getString(R.string.rtmdet_models_ready), Toast.LENGTH_SHORT).show()
                    }
                } else {
                    runOnUiThread {
                        if (!isDestroyed && !isFinishing)
                            Toast.makeText(this, getString(R.string.rtmdet_models_download_failed), Toast.LENGTH_LONG).show()
                    }
                }
            } catch (e: Exception) {
                logExceptionTelemetry("rtmdet_download", "unexpected", e)
                Log.e(TAG, "RTMDet model download failed", e)
                runOnUiThread {
                    if (!isDestroyed && !isFinishing)
                        Toast.makeText(this, getString(R.string.rtmdet_models_download_failed), Toast.LENGTH_LONG).show()
                }
            } finally {
                rtmDetDownloadInProgress.set(false)
            }
        }
    }

    private fun startMobilintModelDownload() {
        if (!mobilintDownloadInProgress.compareAndSet(false, true)) return
        runOnUiThread { Toast.makeText(this, getString(R.string.mobilint_models_downloading), Toast.LENGTH_LONG).show() }
        backgroundExecutor.execute {
            try {
                if (ModelDownloadManager.downloadMissingMobilintModels(this)) {
                    // Mobilint-specific processor initialization would go here
                    runOnUiThread {
                        if (!isDestroyed && !isFinishing)
                            Toast.makeText(this, getString(R.string.mobilint_models_ready), Toast.LENGTH_SHORT).show()
                    }
                } else {
                    runOnUiThread {
                        if (!isDestroyed && !isFinishing)
                            Toast.makeText(this, getString(R.string.mobilint_models_download_failed), Toast.LENGTH_LONG).show()
                    }
                }
            } catch (e: Exception) {
                logExceptionTelemetry("mobilint_download", "unexpected", e)
                Log.e(TAG, "Mobilint model download failed", e)
                runOnUiThread {
                    if (!isDestroyed && !isFinishing)
                        Toast.makeText(this, getString(R.string.mobilint_models_download_failed), Toast.LENGTH_LONG).show()
                    }
            } finally {
                mobilintDownloadInProgress.set(false)
            }
        }
    }

    private fun startTfliteModelDownload() {
        if (!tfliteDownloadInProgress.compareAndSet(false, true)) return
        runOnUiThread { Toast.makeText(this, getString(R.string.tflite_models_downloading), Toast.LENGTH_LONG).show() }
        backgroundExecutor.execute {
            try {
                if (ModelDownloadManager.downloadMissingTfliteModels(this)) {
                    tfliteProcessor.close()
                    tfliteProcessor.initialize()
                    imageProcessor.tfliteProcessor = tfliteProcessor
                    runOnUiThread {
                        if (!isDestroyed && !isFinishing)
                            Toast.makeText(this, getString(R.string.tflite_models_ready), Toast.LENGTH_SHORT).show()
                    }
                } else {
                    runOnUiThread {
                        if (!isDestroyed && !isFinishing)
                            Toast.makeText(this, getString(R.string.tflite_models_download_failed), Toast.LENGTH_LONG).show()
                    }
                }
            } catch (e: Exception) {
                logExceptionTelemetry("tflite_download", "unexpected", e)
                Log.e(TAG, "TFLite model download failed", e)
                runOnUiThread {
                    if (!isDestroyed && !isFinishing)
                        Toast.makeText(this, getString(R.string.tflite_models_download_failed), Toast.LENGTH_LONG).show()
                }
            } finally {
                tfliteDownloadInProgress.set(false)
            }
        }
    }

    private fun setupEyeTrackingCalibrationFab() {
        binding.fabEyeTrackingCalibration.setOnClickListener {
            mediaPipeProcessor.startEyeTrackingCalibration()
            Toast.makeText(this, getString(R.string.eye_tracking_calibration_started), Toast.LENGTH_SHORT).show()
        }
    }

    private fun setupCameraSwitchButton() {
        binding.fabSwitchCamera.setOnClickListener { lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) CameraSelector.LENS_FACING_FRONT else CameraSelector.LENS_FACING_BACK; startCamera() }
    }

    private fun setupCaptureButton() {
        binding.btnCapture.setOnClickListener { if (isRecording) stopVideoRecording() else takePhoto() }
        binding.btnCapture.setOnLongClickListener { if (!isRecording) startVideoRecording(); true }
    }

    private fun setupCalibrationFab() = binding.fabCalibrationMenu.setOnClickListener { openCalibrationMenu() }
    private fun setupResolutionFab() = binding.fabResolution.setOnClickListener { openResolutionMenu() }
    private val filePickerSlam = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) loadSlamMap(uri)
    }

    private fun setupSavePointCloudFab() {
        binding.fabSavePointCloud.setOnClickListener {
            if (currentFilter.isFullOdometry) {
                showSaveSlamMapDialog()
            } else {
                showSavePointCloudDialog()
            }
        }
        binding.fabSaveSlamMap.setOnClickListener {
            showSaveSlamMapDialog()
        }
        binding.fabLoadSlamMap.setOnClickListener {
            filePickerSlam.launch("*/*")
        }
    }

    private fun setupBackToMenuButton() {
        binding.fabBackToMenu.setOnClickListener {
            startActivity(Intent(this, MenuActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP
            })
            finish()
        }
    }

    private fun showSavePointCloudDialog() {
        val cloud = imageProcessor.lastPointCloud
        if (cloud == null || cloud.points.isEmpty()) {
            Toast.makeText(this, getString(R.string.point_cloud_empty), Toast.LENGTH_SHORT).show()
            return
        }
        val formats = arrayOf(
            getString(R.string.point_cloud_format_csv),
            getString(R.string.point_cloud_format_ply),
        )
        AlertDialog.Builder(this)
            .setTitle(getString(R.string.point_cloud_format_title))
            .setItems(formats) { _, which ->
                when (which) {
                    0 -> savePointCloud(cloud, usePly = false)
                    1 -> savePointCloud(cloud, usePly = true)
                }
            }
            .show()
    }

    private fun showSaveSlamMapDialog() {
        val map = imageProcessor.currentSlamMap
        if (map.points3d.isEmpty() && map.markers.isEmpty()) {
            Toast.makeText(this, getString(R.string.point_cloud_empty), Toast.LENGTH_SHORT).show()
            return
        }
        val formats = arrayOf(
            getString(R.string.point_cloud_format_ply),
        )
        AlertDialog.Builder(this)
            .setTitle(getString(R.string.point_cloud_format_title))
            .setItems(formats) { _, which ->
                when (which) {
                    0 -> saveSlamMap(map)
                }
            }
            .show()
    }

    private fun saveSlamMap(map: FullOdometryEngine.MapState, isAutoSave: Boolean = false) {
        try {
            val timestamp = System.currentTimeMillis()
            val filename = if (isAutoSave) "autosave_slam_map_$timestamp.csv" else "slam_map_$timestamp.ply"
            
            val content = if (isAutoSave) {
                // CSV for autosave (Excel-friendly)
                buildString {
                    appendLine("x,y,z,r,g,b,label")
                    map.points3d.forEachIndexed { i, p ->
                        val colorInt = map.colors[i]
                        val r = Color.red(colorInt)
                        val g = Color.green(colorInt)
                        val b = Color.blue(colorInt)
                        appendLine("${p.x},${p.y},${p.z},$r,$g,$b,landmark")
                    }
                    for (m in map.markers) {
                        appendLine("${m.position.x},${m.position.y},${m.position.z},0,255,255,${m.label}")
                    }
                }
            } else {
                // PLY for manual save (MeshLab/CloudCompare friendly)
                buildString {
                    appendLine("ply")
                    appendLine("format ascii 1.0")
                    appendLine("comment MobileCV SLAM Sparse Map")
                    appendLine("element vertex ${map.points3d.size + map.markers.size}")
                    appendLine("property float x")
                    appendLine("property float y")
                    appendLine("property float z")
                    appendLine("property uchar red")
                    appendLine("property uchar green")
                    appendLine("property uchar blue")
                    appendLine("end_header")
                    
                    // Map points
                    map.points3d.forEachIndexed { i, p ->
                        val colorInt = map.colors[i]
                        val r = Color.red(colorInt)
                        val g = Color.green(colorInt)
                        val b = Color.blue(colorInt)
                        // Note: OpenCV coordinate system (X right, Y down, Z forward)
                        appendLine("${p.x} ${p.y} ${p.z} $r $g $b")
                    }
                    
                    // Markers as cyan points
                    for (m in map.markers) {
                        appendLine("${m.position.x} ${m.position.y} ${m.position.z} 0 255 255")
                    }
                }
            }
            
            val mimeType = if (isAutoSave) "text/csv" else "application/octet-stream"
            writeToDownloads(filename, mimeType, content, silent = isAutoSave)
        } catch (e: Exception) {
            logExceptionTelemetry("save_slam_map", "error", e)
            Log.e(TAG, "Failed to save SLAM map", e)
            if (!isAutoSave) {
                runOnUiThread { Toast.makeText(this, R.string.point_cloud_save_error, Toast.LENGTH_SHORT).show() }
            }
        }
    }

    private fun loadSlamMap(uri: Uri) {
        backgroundExecutor.execute {
            try {
                contentResolver.openInputStream(uri)?.bufferedReader()?.use { reader ->
                    val filename = queryFileName(uri)
                    val isPly = filename.endsWith(".ply", ignoreCase = true)
                    val points = mutableListOf<Point3>()
                    val colors = mutableListOf<Int>()
                    val markers = mutableListOf<FullOdometryEngine.MarkerLandmark>()

                    if (isPly) {
                        var line = reader.readLine()
                        while (line != null && line.trim() != "end_header") {
                            line = reader.readLine()
                        }
                        while (true) {
                            line = reader.readLine() ?: break
                            if (line.isBlank()) continue
                            val parts = line.trim().split(Regex("\\s+"))
                            if (parts.size >= 3) {
                                val x = parts[0].toDoubleOrNull() ?: 0.0
                                val y = parts[1].toDoubleOrNull() ?: 0.0
                                val z = parts[2].toDoubleOrNull() ?: 0.0
                                points.add(Point3(x, y, z))
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
                    } else {
                        // CSV
                        reader.readLine() // skip header
                        for (rawLine in reader.lineSequence()) {
                            val trimmed = rawLine.trim()
                            if (trimmed.isEmpty()) continue
                            val parts = trimmed.split(",")
                            if (parts.size >= 3) {
                                val x = parts[0].toDoubleOrNull() ?: continue
                                val y = parts[1].toDoubleOrNull() ?: continue
                                val z = parts[2].toDoubleOrNull() ?: continue
                                val label = parts.getOrNull(6) ?: "landmark"
                                
                                if (label == "landmark") {
                                    points.add(Point3(x, y, z))
                                    if (parts.size >= 6) {
                                        val r = parts[3].toIntOrNull() ?: 255
                                        val g = parts[4].toIntOrNull() ?: 255
                                        val b = parts[5].toIntOrNull() ?: 255
                                        colors.add(Color.rgb(r, g, b))
                                    } else {
                                        colors.add(Color.WHITE)
                                    }
                                } else {
                                    markers.add(FullOdometryEngine.MarkerLandmark(label, Point3(x, y, z), label))
                                }
                            }
                        }
                    }

                    if (points.isNotEmpty() || markers.isNotEmpty()) {
                        val state = FullOdometryEngine.MapState(points, colors, null, null, markers)
                        imageProcessor.fullOdometryEngine.importMap(state)
                        runOnUiThread {
                            Toast.makeText(this, "Imported ${points.size} points and ${markers.size} markers", Toast.LENGTH_LONG).show()
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load SLAM map", e)
                runOnUiThread { Toast.makeText(this, "Load failed: ${e.message}", Toast.LENGTH_SHORT).show() }
            }
        }
    }

    private fun queryFileName(uri: Uri): String {
        contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (cursor.moveToFirst()) return cursor.getString(nameIndex)
        }
        return uri.lastPathSegment ?: "map.ply"
    }

    private fun savePointCloud(cloud: VisualOdometryEngine.PointCloudState, usePly: Boolean) {
        try {
            val timestamp = System.currentTimeMillis()
            if (usePly) {
                val content = buildString {
                    appendLine("ply")
                    appendLine("format ascii 1.0")
                    appendLine("comment MobileCV pseudo-3D point cloud")
                    appendLine("comment mean_parallax=${cloud.meanParallax}")
                    appendLine("element vertex ${cloud.points.size}")
                    appendLine("property float x")
                    appendLine("property float y")
                    appendLine("property float z")
                    appendLine("property uchar red")
                    appendLine("property uchar green")
                    appendLine("property uchar blue")
                    appendLine("end_header")
                    cloud.points.forEachIndexed { i, p ->
                        val color = cloud.colors[i]
                        val r = color.`val`[0].toInt().coerceIn(0, 255)
                        val g = color.`val`[1].toInt().coerceIn(0, 255)
                        val b = color.`val`[2].toInt().coerceIn(0, 255)
                        appendLine("${p.x} ${p.y} ${"%.4f".format(pseudoZ(p.y, cloud.meanParallax))} $r $g $b")
                    }
                }
                writeToDownloads("pointcloud_$timestamp.ply", "application/octet-stream", content)
            } else {
                val content = buildString {
                    appendLine("x,y,z,r,g,b")
                    appendLine("# Pseudo-3D point cloud with colors. mean_parallax=${cloud.meanParallax}")
                    cloud.points.forEachIndexed { i, p ->
                        val color = cloud.colors[i]
                        val r = color.`val`[0].toInt().coerceIn(0, 255)
                        val g = color.`val`[1].toInt().coerceIn(0, 255)
                        val b = color.`val`[2].toInt().coerceIn(0, 255)
                        appendLine("${p.x},${p.y},${"%.4f".format(pseudoZ(p.y, cloud.meanParallax))},$r,$g,$b")
                    }
                }
                writeToDownloads("pointcloud_$timestamp.csv", "text/csv", content)
            }
        } catch (e: IOException) {
            logExceptionTelemetry("save_point_cloud", "io", e)
            Log.e(TAG, "Failed to save point cloud: storage I/O", e)
            Toast.makeText(this, R.string.point_cloud_save_error, Toast.LENGTH_SHORT).show()
        } catch (e: SecurityException) {
            logExceptionTelemetry("save_point_cloud", "permission", e)
            Log.e(TAG, "Failed to save point cloud: missing storage permission", e)
            Toast.makeText(this, R.string.point_cloud_save_error, Toast.LENGTH_SHORT).show()
        } catch (e: IllegalStateException) {
            logExceptionTelemetry("save_point_cloud", "state", e)
            Log.e(TAG, "Failed to save point cloud: invalid storage state", e)
            Toast.makeText(this, R.string.point_cloud_save_error, Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            logExceptionTelemetry("save_point_cloud", "unexpected", e)
            Log.e(
                TAG,
                "Unhandled point cloud save error type=${e::class.java.name} message=${e.message}",
                e,
            )
            Toast.makeText(this, R.string.point_cloud_save_error, Toast.LENGTH_SHORT).show()
        }
    }

    /** Estimates a pseudo-depth z value from screen y position and mean parallax. */
    private fun pseudoZ(y: Double, meanParallax: Double): Double = (meanParallax - y) * 0.1

    private fun writeToDownloads(filename: String, mimeType: String, content: String, silent: Boolean = false) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val values = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
                put(MediaStore.MediaColumns.MIME_TYPE, mimeType)
                put(MediaStore.MediaColumns.RELATIVE_PATH, "Download/MobileCV")
            }
            val uri = contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
            if (uri != null) {
                contentResolver.openOutputStream(uri)?.use { it.write(content.toByteArray()) }
                if (!silent) Toast.makeText(this, getString(R.string.point_cloud_saved, "Download/MobileCV/$filename"), Toast.LENGTH_SHORT).show()
            } else if (!silent) {
                Toast.makeText(this, R.string.point_cloud_save_error, Toast.LENGTH_SHORT).show()
            }
        } else {
            @Suppress("DEPRECATION")
            val dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            val subDir = File(dir, "MobileCV").also { it.mkdirs() }
            File(subDir, filename).writeText(content)
            if (!silent) Toast.makeText(this, getString(R.string.point_cloud_saved, "${subDir.absolutePath}/$filename"), Toast.LENGTH_SHORT).show()
        }
    }

    private fun openResolutionMenu() {
        // Use the Fragment Result API to handle the result from the bottom sheet.
        // This is lifecycle-aware and survives configuration changes like rotation.
        supportFragmentManager.setFragmentResultListener("resolution_request", this) { _, bundle ->
            val resolutionName = bundle.getString("selected_resolution") ?: return@setFragmentResultListener
            val resolution = CameraResolution.valueOf(resolutionName)
            updateAndPersistCameraResolution(resolution)
            startCamera()
        }

        ResolutionBottomSheet.newInstance(currentResolution)
            .show(supportFragmentManager, ResolutionBottomSheet.TAG)
    }

    private fun updateAndPersistCameraResolution(resolution: CameraResolution) {
        currentResolution = resolution
        prefs.edit {
            putString(PREF_CAMERA_RESOLUTION, resolution.name)
        }
    }

    private fun openCalibrationMenu() {
        CalibrationBottomSheet().apply {
            onCollectFrame = {
                val collected = cameraCalibrator.collectLastFrame()
                if (collected) {
                    val count = cameraCalibrator.frameCount
                    runOnUiThread {
                        Toast.makeText(
                            this@MainActivity,
                            getString(R.string.calibration_frame_collected, count, CameraCalibrator.MIN_FRAMES),
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                }
                collected
            }
            onCalibrate = {
                val res = cameraCalibrator.calibrate()
                runOnUiThread {
                    Toast.makeText(
                        this@MainActivity,
                        if (res != null) R.string.calibration_success else R.string.calibration_failed,
                        Toast.LENGTH_SHORT
                    ).show()
                }
                res
            }
            onReset = { cameraCalibrator.reset() }
        }.show(supportFragmentManager, CalibrationBottomSheet.TAG)
    }

    private fun requestPermissionsOrStart() { if (requiredPermissions().all { ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED }) startCamera() else permissionsLauncher.launch(requiredPermissions()) }

    private fun requiredPermissions(): Array<String> = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO
    ) + if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE) else emptyArray()

    private fun startCamera() {
        cameraStartTimeMs = SystemClock.elapsedRealtime()
        firstFrameRenderedLogged.set(false)
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = try {
                cameraProviderFuture.get()
            } catch (e: IllegalStateException) {
                logExceptionTelemetry("camera_provider", "state", e)
                Log.e(TAG, "Camera provider unavailable: invalid lifecycle state", e)
                null
            } catch (e: Exception) {
                logExceptionTelemetry("camera_provider", "unexpected", e)
                Log.e(
                    TAG,
                    "Unhandled camera provider acquisition error type=${e::class.java.name} message=${e.message}",
                    e,
                )
                null
            }
            bindUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindUseCases() {
        val provider = cameraProvider ?: return
        val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
        val resolutionSelector = ResolutionSelector.Builder().setResolutionStrategy(ResolutionStrategy(currentResolution.size, ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)).build()
        val imageAnalysis = ImageAnalysis.Builder().setResolutionSelector(resolutionSelector).setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888).build().also { it.setAnalyzer(cameraAnalysisExecutor) { proxy -> processFrame(proxy) } }
        imageCapture = ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY).build()
        try {
            provider.unbindAll()
            fpsCounter.reset()
            provider.bindToLifecycle(this, cameraSelector, imageAnalysis, imageCapture)
        } catch (e: IllegalArgumentException) {
            logExceptionTelemetry("camera_bind", "invalid_argument", e)
            Log.e(TAG, "Camera bind failed: invalid CameraX configuration", e)
        } catch (e: IllegalStateException) {
            logExceptionTelemetry("camera_bind", "state", e)
            Log.e(TAG, "Camera bind failed: camera lifecycle state mismatch", e)
        } catch (e: Exception) {
            logExceptionTelemetry("camera_bind", "unexpected", e)
            Log.e(
                TAG,
                "Unhandled camera bind error type=${e::class.java.name} message=${e.message}",
                e,
            )
        }
    }

    private fun processFrame(imageProxy: ImageProxy) {
        if (isFinishing || isDestroyed) {
            imageProxy.close()
            return
        }
        try {
            val bitmap = imageProxy.toBitmap()
            val oriented = orientBitmap(bitmap, imageProxy.imageInfo.rotationDegrees, lensFacing)
            
            // Fix memory leak: recycle original bitmap if a new oriented one was created.
            if (oriented !== bitmap) {
                bitmap.recycle()
            }

            val start = System.nanoTime()
            val processed = imageProcessor.processFrame(oriented, currentFilter)
            val time = (System.nanoTime() - start) / 1_000_000L
            
            frameWidth = processed.width; frameHeight = processed.height; fpsCounter.onFrame()

            // Write processed frame (with overlays) to the video recorder if active.
            if (isRecording) processedVideoRecorder.writeFrame(processed)
            
            imageProcessor.consumeBenchmarkSnapshot(currentFilter)?.let { snapshot ->
                Log.i(
                    TAG,
                    "Runtime benchmark ${snapshot.filter.name} " +
                        "samples=${snapshot.samples} " +
                        "before_avg_ms=${"%.2f".format(snapshot.avgBeforeMs)} " +
                        "after_avg_ms=${"%.2f".format(snapshot.avgAfterMs)} " +
                        "before_fps=${"%.2f".format(snapshot.fpsBefore)} " +
                        "after_fps=${"%.2f".format(snapshot.fpsAfter)}"
                )
            }

            val nowNs = System.nanoTime()
            val shouldUpdateUi = nowNs - lastUiUpdateNs >= UI_UPDATE_MIN_INTERVAL_NS
            if (shouldUpdateUi && uiUpdatePending.compareAndSet(false, true)) {
                lastUiUpdateNs = nowNs
                runOnUiThread {
                    pendingRecycleBitmap?.recycle(); pendingRecycleBitmap = lastProcessedBitmap; binding.imageViewPreview.setImageBitmap(processed); lastProcessedBitmap = processed; uiUpdatePending.set(false)
                    if (firstFrameRenderedLogged.compareAndSet(false, true) && cameraStartTimeMs > 0L) {
                        val startupLatencyMs = SystemClock.elapsedRealtime() - cameraStartTimeMs
                        Log.d(TAG, "Camera startup latency to first rendered frame: ${startupLatencyMs}ms")
                    }
                    updateDiagnosticsOverlay(
                        fpsCounter.fps,
                        frameWidth,
                        frameHeight,
                        time,
                        currentFilter,
                        lensFacing == CameraSelector.LENS_FACING_FRONT,
                        isActiveVisionEnabled,
                        isActiveVisionVisualizationEnabled,
                    )
                }
            } else {
                processed.recycle()
            }
            
            // Recycle the oriented bitmap as it's no longer needed (processed is a copy).
            oriented.recycle()
            
        } catch (e: org.opencv.core.CvException) {
            logExceptionTelemetry("frame_processing", "opencv", e)
            currentFilter = OpenCvFilter.ORIGINAL
            Log.e(TAG, "Frame processing failed in OpenCV. Fallback to ORIGINAL filter.", e)
        } catch (e: IllegalArgumentException) {
            logExceptionTelemetry("frame_processing", "invalid_argument", e)
            currentFilter = OpenCvFilter.ORIGINAL
            Log.e(TAG, "Frame processing failed due to invalid arguments. Fallback to ORIGINAL.", e)
        } catch (e: IllegalStateException) {
            logExceptionTelemetry("frame_processing", "ui_state", e)
            isActiveVisionVisualizationEnabled = false
            Log.e(TAG, "Frame processing failed due to UI/state issue. Visualization disabled.", e)
        } catch (e: Exception) {
            logExceptionTelemetry("frame_processing", "unexpected", e)
            Log.e(
                TAG,
                "Unhandled frame processing error type=${e::class.java.name} message=${e.message}",
                e,
            )
        } finally { 
            imageProxy.close() 
        }
    }

    private fun updateDiagnosticsOverlay(
        fps: Double,
        w: Int,
        h: Int,
        t: Long,
        f: OpenCvFilter,
        front: Boolean,
        av: Boolean,
        avVisualization: Boolean,
    ) {
        val label = getString(if (front) R.string.diagnostics_camera_front else R.string.diagnostics_camera_back)
        binding.textViewDiagnostics.text = buildString {
            appendLine(getString(R.string.diagnostics_fps, fps))
            appendLine(getString(R.string.diagnostics_resolution, w, h))
            appendLine(getString(R.string.diagnostics_processing_time, t))
            appendLine(getString(R.string.diagnostics_filter, f.displayName))
            appendLine(getString(R.string.diagnostics_active_vision, av))
            appendLine(getString(R.string.diagnostics_active_vision_visualization, avVisualization))
            append(label)
        }
    }

    private fun orientBitmap(bitmap: Bitmap, rotation: Int, facing: Int): Bitmap {
        val front = facing == CameraSelector.LENS_FACING_FRONT
        if (rotation == 0 && !front) return bitmap
        val matrix = Matrix(); if (rotation != 0) matrix.postRotate(rotation.toFloat()); if (front) matrix.postScale(-1f, 1f)
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun takePhoto() {
        val capture = imageCapture ?: return
        val values = ContentValues().apply { put(MediaStore.MediaColumns.DISPLAY_NAME, "IMG_${System.currentTimeMillis()}"); put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg"); if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/MobileCV") }
        capture.takePicture(ImageCapture.OutputFileOptions.Builder(contentResolver, MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values).build(), ContextCompat.getMainExecutor(this), object : ImageCapture.OnImageSavedCallback {
            override fun onImageSaved(res: ImageCapture.OutputFileResults) { Toast.makeText(this@MainActivity, R.string.photo_saved, Toast.LENGTH_SHORT).show() }
            override fun onError(e: ImageCaptureException) { Toast.makeText(this@MainActivity, R.string.photo_error, Toast.LENGTH_SHORT).show() }
        })
    }

    @SuppressLint("MissingPermission")
    private fun startVideoRecording() {
        val hasAudio = requiredPermissions().contains(Manifest.permission.RECORD_AUDIO) &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED
        val started = processedVideoRecorder.start(withAudio = hasAudio)
        if (!started) {
            Toast.makeText(this, R.string.video_error, Toast.LENGTH_SHORT).show()
            return
        }
        isRecording = true
        updateCaptureButtonState()
    }

    private fun stopVideoRecording() {
        if (!isRecording) return
        isRecording = false
        updateCaptureButtonState()
        backgroundExecutor.submit {
            processedVideoRecorder.finalize { success ->
                runOnUiThread {
                    if (!isDestroyed && !isFinishing) {
                        Toast.makeText(
                            this,
                            if (success) R.string.video_saved else R.string.video_error,
                            Toast.LENGTH_SHORT,
                        ).show()
                    }
                }
            }
        }
    }

    private fun startRecordingTimer() {
        recordingStartTimeMs = System.currentTimeMillis()
        val runnable = object : Runnable {
            override fun run() {
                val elapsed = (System.currentTimeMillis() - recordingStartTimeMs) / 1000L
                binding.textViewRecordingTimer.text = RECORDING_TIMER_FORMAT.format(elapsed / 60, elapsed % 60)
                recordingTimerHandler.postDelayed(this, 1000)
            }
        }
        recordingTimerRunnable = runnable
        recordingTimerHandler.post(runnable)
    }

    private fun stopRecordingTimer() {
        recordingTimerRunnable?.let { recordingTimerHandler.removeCallbacks(it) }
        recordingTimerRunnable = null
        if (!isDestroyed) binding.textViewRecordingTimer.text = RECORDING_TIMER_FORMAT.format(0, 0)
    }

    private fun updateCaptureButtonState() {
        binding.btnCapture.isActivated = isRecording
        binding.btnCapture.contentDescription = getString(if (isRecording) R.string.stop_recording_description else R.string.capture_button_description)
        if (isRecording) {
            binding.layoutRecordingIndicator.visibility = View.VISIBLE
            startRecordingTimer()
        } else {
            binding.layoutRecordingIndicator.visibility = View.GONE
            stopRecordingTimer()
        }
    }

    private fun enableImmersiveFullscreen() { WindowCompat.setDecorFitsSystemWindows(window, false); WindowInsetsControllerCompat(window, window.decorView).let { it.hide(WindowInsetsCompat.Type.systemBars()); it.systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE } }
}
