package pl.edu.mobilecv

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
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Main (and only) activity of the MobileCV application.
 */
class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MobileCV"
        private const val PREFS_NAME = "mobilecv_prefs"
        private const val PREF_ROBOT_HOST = "robot_host"
        private const val PREF_ROBOT_PORT = "robot_port"
        private const val PREF_CAMERA_RESOLUTION = "camera_resolution"
        private const val RECORDING_TIMER_FORMAT = "%02d:%02d"
    }

    private lateinit var binding: ActivityMainBinding

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

    @Volatile private var mediaPipeDownloadInProgress = false
    private val yoloDownloadInProgress = AtomicBoolean(false)
    @Volatile private var currentFilter = OpenCvFilter.ORIGINAL
    @Volatile private var currentMode: AnalysisMode = AnalysisMode.entries.first()
    @Volatile private var isActiveVisionEnabled = false
    @Volatile private var isActiveVisionVisualizationEnabled = false

    // Calibration
    val cameraCalibrator = CameraCalibrator()

    // ROS2 / ROSBridge
    private val rosBridgeClient = RosBridgeClient()
    private var lastRobotHost: String = "192.168.1.100"
    private var lastRobotPort: Int = RosBridgeClient.DEFAULT_PORT

    // FPS and diagnostics
    private val fpsCounter = FpsCounter()
    @Volatile private var frameWidth: Int = 0
    @Volatile private var frameHeight: Int = 0
    @Volatile private var lastProcessingTimeMs: Long = 0

    private var lastProcessedBitmap: Bitmap? = null
    private var pendingRecycleBitmap: Bitmap? = null
    private val uiUpdatePending = AtomicBoolean(false)
    private lateinit var analysisExecutor: ExecutorService

    private val permissionsLauncher = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { results ->
        if (results[Manifest.permission.CAMERA] == true) startCamera()
        else { Toast.makeText(this, getString(R.string.camera_permission_denied), Toast.LENGTH_LONG).show(); finish() }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Initialize OpenCV first before any components that might use it are created.
        initOpenCv()

        analysisExecutor = Executors.newSingleThreadExecutor()

        val prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
        lastRobotHost = prefs.getString(PREF_ROBOT_HOST, lastRobotHost) ?: lastRobotHost
        lastRobotPort = prefs.getInt(PREF_ROBOT_PORT, lastRobotPort)
        currentResolution = CameraResolution.entries.find { it.name == prefs.getString(PREF_CAMERA_RESOLUTION, null) } ?: CameraResolution.DEFAULT

        imageProcessor.calibrator = cameraCalibrator
        imageProcessor.onMarkersDetected = { rosBridgeClient.publishMarkers(it) }
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

        analysisExecutor.execute {
            mediaPipeProcessor.initialize()
            imageProcessor.mediaPipeProcessor = mediaPipeProcessor
            yoloProcessor.initialize()
            imageProcessor.yoloProcessor = yoloProcessor
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
                    Log.e(TAG, "Background YOLO model download failed", e)
                    runOnUiThread {
                        if (!isDestroyed && !isFinishing)
                            Toast.makeText(this, getString(R.string.yolo_models_download_failed), Toast.LENGTH_LONG).show()
                    }
                } finally {
                    yoloDownloadInProgress.set(false)
                }
            }
        }

        rosBridgeClient.onStateChanged = { state ->
            if (!isDestroyed && !isFinishing) runOnUiThread { onRosBridgeStateChanged(state) }
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
        setupRobotConnectionFab()
        setupResolutionFab()
        setupSavePointCloudFab()
        setupBackToMenuButton()
        requestPermissionsOrStart()
    }

    override fun onResume() { super.onResume(); enableImmersiveFullscreen() }
    override fun onWindowFocusChanged(hasFocus: Boolean) { super.onWindowFocusChanged(hasFocus); if (hasFocus) enableImmersiveFullscreen() }

    override fun onDestroy() {
        super.onDestroy()
        if (isRecording) {
            isRecording = false
            analysisExecutor.execute {
                processedVideoRecorder.finalize { success ->
                    Log.d(TAG, "Recording finalized on destroy, success=$success")
                }
            }
        }
        stopRecordingTimer()
        analysisExecutor.execute { mediaPipeProcessor.close(); yoloProcessor.close() }
        analysisExecutor.shutdown()
        pendingRecycleBitmap?.recycle(); lastProcessedBitmap?.recycle()
        rosBridgeClient.shutdown()
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

        val isOdometry = mode == AnalysisMode.ODOMETRY
        binding.layoutVoMaxFeatures.visibility = if (isOdometry) View.VISIBLE else View.GONE
        binding.layoutVoMinParallax.visibility = if (isOdometry) View.VISIBLE else View.GONE
        binding.layoutVoMesh.visibility = if (isOdometry) View.VISIBLE else View.GONE

        binding.fabCalibrationMenu.visibility = if (mode == AnalysisMode.CALIBRATION) View.VISIBLE else View.GONE
        binding.fabRobotConnection.visibility = if (mode == AnalysisMode.CALIBRATION) View.GONE else View.VISIBLE

        if (mode == AnalysisMode.POSE && !ModelDownloadManager.areAllModelsReady(this)) startMediaPipeModelDownload()
        if (mode == AnalysisMode.YOLO && !ModelDownloadManager.areYoloModelsReady(this)) startYoloModelDownload()

        updateContextualControls()
    }

    private fun updateContextualControls() {
        val isFiltersMode = currentMode == AnalysisMode.FILTERS
        binding.switchActiveVision.visibility = if (isFiltersMode) View.VISIBLE else View.GONE
        binding.switchActiveVisionVisualization.visibility =
            if (isFiltersMode && isActiveVisionEnabled) View.VISIBLE else View.GONE
        binding.fabSavePointCloud.visibility =
            if (currentFilter == OpenCvFilter.POINT_CLOUD) View.VISIBLE else View.GONE
    }

    private fun startMediaPipeModelDownload() {
        if (mediaPipeDownloadInProgress) return
        mediaPipeDownloadInProgress = true
        Toast.makeText(this, getString(R.string.mediapipe_models_downloading), Toast.LENGTH_LONG).show()
        analysisExecutor.execute {
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
        analysisExecutor.execute {
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
            } catch (e: Exception) {
                Log.e(TAG, "YOLO model download failed", e)
                runOnUiThread {
                    if (!isDestroyed && !isFinishing)
                        Toast.makeText(this, getString(R.string.yolo_models_download_failed), Toast.LENGTH_LONG).show()
                }
            } finally { yoloDownloadInProgress.set(false) }
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
    private fun setupRobotConnectionFab() = binding.fabRobotConnection.setOnClickListener { openRobotConnectionMenu() }
    private fun setupResolutionFab() = binding.fabResolution.setOnClickListener { openResolutionMenu() }
    private fun setupSavePointCloudFab() = binding.fabSavePointCloud.setOnClickListener { showSavePointCloudDialog() }

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
                    appendLine("end_header")
                    cloud.points.forEach { p ->
                        appendLine("${p.x} ${p.y} ${"%.4f".format(pseudoZ(p.y, cloud.meanParallax))}")
                    }
                }
                writeToDownloads("pointcloud_$timestamp.ply", "application/octet-stream", content)
            } else {
                val content = buildString {
                    appendLine("x,y,z")
                    appendLine("# Pseudo-3D point cloud (screen projections). mean_parallax=${cloud.meanParallax}")
                    cloud.points.forEach { p ->
                        appendLine("${p.x},${p.y},${"%.4f".format(pseudoZ(p.y, cloud.meanParallax))}")
                    }
                }
                writeToDownloads("pointcloud_$timestamp.csv", "text/csv", content)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save point cloud", e)
            Toast.makeText(this, R.string.point_cloud_save_error, Toast.LENGTH_SHORT).show()
        }
    }

    /** Estimates a pseudo-depth z value from screen y position and mean parallax. */
    private fun pseudoZ(y: Double, meanParallax: Double): Double = (meanParallax - y) * 0.1

    private fun writeToDownloads(filename: String, mimeType: String, content: String) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val values = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
                put(MediaStore.MediaColumns.MIME_TYPE, mimeType)
                put(MediaStore.MediaColumns.RELATIVE_PATH, "Download/MobileCV")
            }
            val uri = contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
            if (uri != null) {
                contentResolver.openOutputStream(uri)?.use { it.write(content.toByteArray()) }
                Toast.makeText(this, getString(R.string.point_cloud_saved, "Download/MobileCV/$filename"), Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, R.string.point_cloud_save_error, Toast.LENGTH_SHORT).show()
            }
        } else {
            @Suppress("DEPRECATION")
            val dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            val subDir = File(dir, "MobileCV").also { it.mkdirs() }
            File(subDir, filename).writeText(content)
            Toast.makeText(this, getString(R.string.point_cloud_saved, "${subDir.absolutePath}/$filename"), Toast.LENGTH_SHORT).show()
        }
    }

    private fun openRobotConnectionMenu() {
        RobotConnectionSheet().apply {
            currentState = rosBridgeClient.state; lastHost = this@MainActivity.lastRobotHost; lastPort = this@MainActivity.lastRobotPort
            onConnect = { h, p -> this@MainActivity.lastRobotHost = h; this@MainActivity.lastRobotPort = p; getSharedPreferences(
                PREFS_NAME,
                MODE_PRIVATE
            ).edit { putString(PREF_ROBOT_HOST, h).putInt(PREF_ROBOT_PORT, p)}; rosBridgeClient.connect(h, p) }
            onDisconnect = { rosBridgeClient.disconnect() }
        }.show(supportFragmentManager, RobotConnectionSheet.TAG)
    }

    private fun openResolutionMenu() {
        ResolutionBottomSheet().apply {
            currentResolution = this@MainActivity.currentResolution
            onResolutionSelected = { r -> this@MainActivity.currentResolution = r; getSharedPreferences(PREFS_NAME, MODE_PRIVATE).edit {
                putString(
                    PREF_CAMERA_RESOLUTION,
                    r.name
                )
            }; startCamera() }
        }.show(supportFragmentManager, ResolutionBottomSheet.TAG)
    }

    private fun onRosBridgeStateChanged(state: RosBridgeClient.State) {
        val colorRes = when (state) { RosBridgeClient.State.CONNECTED -> R.color.robot_connected; RosBridgeClient.State.DISCONNECTED -> R.color.button_robot_default; RosBridgeClient.State.ERROR -> R.color.robot_error; RosBridgeClient.State.CONNECTING -> return }
        binding.fabRobotConnection.backgroundTintList = android.content.res.ColorStateList.valueOf(ContextCompat.getColor(this, colorRes))
    }

    private fun openCalibrationMenu() {
        CalibrationBottomSheet().apply {
            onCollectFrame = { val c = cameraCalibrator.collectLastFrame(); if (c) { val count = cameraCalibrator.frameCount; runOnUiThread { Toast.makeText(this@MainActivity, getString(R.string.calibration_frame_collected, count, CameraCalibrator.MIN_FRAMES), Toast.LENGTH_SHORT).show() } }; c }
            onCalibrate = { val res = cameraCalibrator.calibrate(); runOnUiThread { Toast.makeText(this@MainActivity, if (res != null) R.string.calibration_success else R.string.calibration_failed, Toast.LENGTH_SHORT).show() }; res }
            onReset = { cameraCalibrator.reset() }
        }.show(supportFragmentManager, CalibrationBottomSheet.TAG)
    }

    private fun requestPermissionsOrStart() { if (requiredPermissions().all { ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED }) startCamera() else permissionsLauncher.launch(requiredPermissions()) }

    private fun requiredPermissions(): Array<String> = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO
    ) + if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE) else emptyArray()

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = try { cameraProviderFuture.get() } catch (_: Exception) { null }
            bindUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindUseCases() {
        val provider = cameraProvider ?: return
        val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
        val resolutionSelector = ResolutionSelector.Builder().setResolutionStrategy(ResolutionStrategy(currentResolution.size, ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)).build()
        val imageAnalysis = ImageAnalysis.Builder().setResolutionSelector(resolutionSelector).setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888).build().also { it.setAnalyzer(analysisExecutor) { proxy -> processFrame(proxy) } }
        imageCapture = ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY).build()
        try { provider.unbindAll(); fpsCounter.reset(); provider.bindToLifecycle(this, cameraSelector, imageAnalysis, imageCapture) } catch (e: Exception) { Log.e(TAG, "Binding failed", e) }
    }

    private fun processFrame(imageProxy: ImageProxy) {
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
            
            if (uiUpdatePending.compareAndSet(false, true)) {
                runOnUiThread {
                    pendingRecycleBitmap?.recycle(); pendingRecycleBitmap = lastProcessedBitmap; binding.imageViewPreview.setImageBitmap(processed); lastProcessedBitmap = processed; uiUpdatePending.set(false)
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
            
        } catch (e: Exception) { 
            Log.e(TAG, "Process error", e) 
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
        analysisExecutor.submit {
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
