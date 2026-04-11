package pl.edu.mobilecv

import android.Manifest
import android.app.AlertDialog
import android.content.ContentValues
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.SystemClock
import android.provider.MediaStore
import android.provider.OpenableColumns
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageProxy
import androidx.core.content.ContextCompat
import androidx.core.content.edit
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import org.opencv.android.OpenCVLoader
import org.opencv.core.Point3
import pl.edu.mobilecv.databinding.ActivityMainBinding
import pl.edu.mobilecv.odometry.FullOdometryEngine
import pl.edu.mobilecv.odometry.VisualOdometryEngine
import pl.edu.mobilecv.vision.CameraCalibrator
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlin.getValue

import pl.edu.mobilecv.ProcessedVideoRecorder
import pl.edu.mobilecv.MenuActivity
import pl.edu.mobilecv.OpenCvFilter
import pl.edu.mobilecv.isFullOdometry

/**
 * Główna aktywność: tylko bindowanie widoków, spinanie kontrolerów oraz routing lifecycle.
 */
class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MobileCV"
        private const val PREFS_NAME = "mobilecv_prefs"
        private const val PREF_CAMERA_RESOLUTION = "camera_resolution"
        private const val UI_UPDATE_MIN_INTERVAL_NS = 33_000_000L
    }

    private lateinit var binding: ActivityMainBinding
    private val prefs by lazy { getSharedPreferences(PREFS_NAME, MODE_PRIVATE) }

    private val imageProcessor by lazy { ImageProcessor() }
    private val mediaPipeProcessor by lazy { MediaPipeProcessor(this) }
    private val yoloProcessor by lazy { YoloProcessor(this) }
    private val tfliteProcessor by lazy { TfliteProcessor(this) }
    private val processedVideoRecorder: ProcessedVideoRecorder by lazy { ProcessedVideoRecorder(this) }

    private lateinit var cameraAnalysisExecutor: ExecutorService
    private lateinit var backgroundExecutor: ExecutorService

    private lateinit var cameraController: CameraController
    private lateinit var analysisUiController: AnalysisUiController
    private lateinit var moduleLifecycleManager: ModuleLifecycleManager
    private lateinit var recordingController: RecordingController

    internal val cameraCalibrator = CameraCalibrator()
    private val fpsCounter = FpsCounter()
    private var lastProcessedBitmap: Bitmap? = null
    private var pendingRecycleBitmap: Bitmap? = null
    private val uiUpdatePending = AtomicBoolean(false)
    private var lastUiUpdateNs: Long = 0
    private var cameraStartTimeMs: Long = 0
    private val firstFrameRenderedLogged = AtomicBoolean(false)
    private var frameWidth: Int = 0
    private var frameHeight: Int = 0

    private lateinit var dataCollectionCache: DataCollectionCacheDataStore
    private val telemetryScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private val exceptionTelemetry = ConcurrentHashMap<String, AtomicInteger>()

    private val permissionsLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { results ->
        val cameraGranted = results[Manifest.permission.CAMERA] == true ||
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        if (!cameraGranted) {
            Toast.makeText(this, getString(R.string.camera_permission_denied), Toast.LENGTH_LONG).show()
            finish()
            return@registerForActivityResult
        }

        // Audio permission is optional for analysis itself, but needed for video recording with sound.
        val audioGranted = results[Manifest.permission.RECORD_AUDIO] == true ||
            ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED
        if (!audioGranted) {
            Toast.makeText(
                this,
                getString(R.string.audio_permission_denied_warning),
                Toast.LENGTH_LONG,
            ).show()
        }

        val storagePermission = Manifest.permission.WRITE_EXTERNAL_STORAGE
        val storageRequired = Build.VERSION.SDK_INT <= Build.VERSION_CODES.P
        val storageGranted = !storageRequired || results[storagePermission] == true ||
            ContextCompat.checkSelfPermission(this, storagePermission) ==
            PackageManager.PERMISSION_GRANTED
        if (!storageGranted) {
            Toast.makeText(
                this,
                getString(R.string.storage_permission_denied_warning),
                Toast.LENGTH_LONG,
            ).show()
        }

        cameraController.startCamera()
    }

    private val filePickerSlam = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) {
            loadSlamMap(uri)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        dataCollectionCache = DataCollectionCacheDataStore(this)
        AssistantDailyDataSyncWorker.schedule(this)
        initOpenCv()

        cameraAnalysisExecutor = Executors.newSingleThreadExecutor()
        backgroundExecutor = Executors.newSingleThreadExecutor()

        bindControllers()
        bindActionButtons()
        configureImageProcessorLabels()

        analysisUiController.setupAll()
        analysisUiController.applyInitialMode(intent.getStringExtra(MenuActivity.EXTRA_MODE))

        moduleLifecycleManager.initializeModules()
        requestPermissionsOrStart()
    }

    override fun onResume() {
        super.onResume()
        enableImmersiveFullscreen()
        if (requestedPermissions().all {
                ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
            }
        ) {
            cameraController.startCamera()
        }
    }

    override fun onPause() {
        super.onPause()
        cameraController.stopCamera()
    }

    override fun onDestroy() {
        super.onDestroy()
        telemetryScope.cancel()
        recordingController.onDestroy()
        moduleLifecycleManager.closeModules()
        cameraAnalysisExecutor.shutdown()
        backgroundExecutor.shutdown()
        pendingRecycleBitmap?.recycle()
        lastProcessedBitmap?.recycle()
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) {
            enableImmersiveFullscreen()
        }
    }

    private fun bindControllers() {
        val resolution = CameraResolution.entries.find {
            it.name == prefs.getString(PREF_CAMERA_RESOLUTION, null)
        } ?: CameraResolution.DEFAULT

        cameraController = CameraController(
            context = this,
            lifecycleOwner = this,
            analysisExecutor = cameraAnalysisExecutor,
            callbacks = object : CameraController.Callbacks {
                override fun onFrame(imageProxy: ImageProxy) = processFrame(imageProxy)

                override fun onCameraError(error: Throwable) {
                    logExceptionTelemetry("camera_controller", "unexpected", error)
                    Log.e(TAG, "Camera controller error", error)
                }

                override fun onLensFacingChanged(lensFacing: Int) {
                    // No action needed
                }

                override fun onCameraStarted() {
                    cameraStartTimeMs = SystemClock.elapsedRealtime()
                    firstFrameRenderedLogged.set(false)
                    fpsCounter.reset()
                }
            },
        )
        cameraController.updateResolution(resolution)

        moduleLifecycleManager = ModuleLifecycleManager(
            context = this,
            imageProcessor = imageProcessor,
            mediaPipeProcessor = mediaPipeProcessor,
            yoloProcessor = yoloProcessor,
            tfliteProcessor = tfliteProcessor,
            backgroundExecutor = backgroundExecutor,
            callbacks = object : ModuleLifecycleManager.Callbacks {
                override fun onTelemetry(scope: String, category: String, error: Throwable) {
                    logExceptionTelemetry(scope, category, error)
                }

                override fun showToast(messageRes: Int, longDuration: Boolean) {
                    runOnUiThread {
                        if (!isDestroyed && !isFinishing) {
                            Toast.makeText(
                                this@MainActivity,
                                getString(messageRes),
                                if (longDuration) Toast.LENGTH_LONG else Toast.LENGTH_SHORT,
                            ).show()
                        }
                    }
                }
            },
        )

        analysisUiController = AnalysisUiController(
            binding = binding,
            imageProcessor = imageProcessor,
            callbacks = object : AnalysisUiController.Callbacks {
                override fun ensureModelsForMode(mode: AnalysisMode) {
                    moduleLifecycleManager.ensureModelsForMode(mode)
                }

                override fun onFilterChanged(filter: OpenCvFilter) = Unit

                override fun onModeChanged(mode: AnalysisMode) {
                    imageProcessor.resetModule(mode)
                }

                override fun onEyeTrackingCalibrationRequested() {
                    mediaPipeProcessor.startEyeTrackingCalibration()
                    Toast.makeText(
                        this@MainActivity,
                        getString(R.string.eye_tracking_calibration_started),
                        Toast.LENGTH_SHORT,
                    ).show()
                }

                override fun onUnknownInitialMode(modeName: String) {
                    Log.w(TAG, "Unknown analysis mode received from intent: $modeName")
                }
            },
        )

        recordingController = RecordingController(
            context = this,
            binding = binding,
            recorder = processedVideoRecorder,
            backgroundExecutor = backgroundExecutor,
            callbacks = object : RecordingController.Callbacks {
                override fun canUpdateUi(): Boolean = !isDestroyed && !isFinishing

                override fun stringRes(id: Int): String = getString(id)

                override fun showToast(messageRes: Int) {
                    runOnUiThread {
                        if (!isDestroyed && !isFinishing) {
                            Toast.makeText(this@MainActivity, getString(messageRes), Toast.LENGTH_SHORT)
                                .show()
                        }
                    }
                }
            },
        )
        recordingController.bindCaptureButton {
            cameraController.takePhoto(
                onSuccess = { Toast.makeText(this, R.string.photo_saved, Toast.LENGTH_SHORT).show() },
                onError = { Toast.makeText(this, R.string.photo_error, Toast.LENGTH_SHORT).show() },
            )
        }
    }

    private fun bindActionButtons() {
        binding.fabSwitchCamera.setOnClickListener { cameraController.switchCamera() }
        binding.fabCalibrationMenu.setOnClickListener { openCalibrationMenu() }
        binding.fabResolution.setOnClickListener { openResolutionMenu() }
        binding.fabBackToMenu.setOnClickListener {
            val intent = Intent(this, MenuActivity::class.java)
            intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP
            startActivity(intent)
            finish()
        }
        binding.fabSavePointCloud.setOnClickListener {
            // W trybie SLAM przycisk zachowuje istniejącą akcję zapisu mapy.
            if (analysisUiController.currentMode == AnalysisMode.SLAM &&
                analysisUiController.currentFilter.isFullOdometry
            ) {
                showSaveSlamMapDialog()
            } else {
                showSavePointCloudDialog()
            }
        }
        binding.fabSaveSlamMap.setOnClickListener {
            if (analysisUiController.currentMode == AnalysisMode.SLAM) {
                showSaveSlamMapDialog()
            }
        }
        binding.fabLoadSlamMap.setOnClickListener {
            if (analysisUiController.currentMode == AnalysisMode.SLAM) {
                filePickerSlam.launch("*/*")
            }
        }
    }

    private fun processFrame(imageProxy: ImageProxy) {
        if (isFinishing || isDestroyed) {
            imageProxy.close()
            return
        }
        try {
            val bitmap = imageProxy.toBitmap()
            val oriented = orientBitmap(bitmap, imageProxy.imageInfo.rotationDegrees, cameraController.lensFacing)
            if (oriented !== bitmap) {
                bitmap.recycle()
            }

            val start = System.nanoTime()
            val processed = imageProcessor.processFrame(
                oriented,
                analysisUiController.currentFilter,
                ImageProcessor.EmptyState
            )
            val processingTimeMs = (System.nanoTime() - start) / 1_000_000L

            frameWidth = processed.width
            frameHeight = processed.height
            fpsCounter.onFrame()
            recordingController.writeFrame(processed)

            val nowNs = System.nanoTime()
            val shouldUpdateUi = nowNs - lastUiUpdateNs >= UI_UPDATE_MIN_INTERVAL_NS
            if (shouldUpdateUi && uiUpdatePending.compareAndSet(false, true)) {
                lastUiUpdateNs = nowNs
                runOnUiThread {
                    pendingRecycleBitmap?.recycle()
                    pendingRecycleBitmap = lastProcessedBitmap
                    binding.imageViewPreview.setImageBitmap(processed)
                    lastProcessedBitmap = processed
                    uiUpdatePending.set(false)
                    if (firstFrameRenderedLogged.compareAndSet(false, true) && cameraStartTimeMs > 0L) {
                        val startupLatencyMs = SystemClock.elapsedRealtime() - cameraStartTimeMs
                        Log.d(TAG, "Camera startup latency: ${startupLatencyMs}ms")
                    }
                    analysisUiController.updateDiagnosticsOverlay(
                        fps = fpsCounter.fps,
                        width = frameWidth,
                        height = frameHeight,
                        processingTimeMs = processingTimeMs,
                        appVersionName = "1.0", // Hardcoded for now as BuildConfig is missing
                        lensFacingFront = cameraController.lensFacing == CameraSelector.LENS_FACING_FRONT,
                        moduleStatusLine = resolveCurrentModuleStatusLine(),
                    ) { resId, args -> getString(resId, *args) }
                }
            } else {
                processed.recycle()
            }
            oriented.recycle()
        } catch (error: Exception) {
            logExceptionTelemetry("frame_processing", "unexpected", error)
            Log.e(TAG, "Frame processing failed", error)
        } finally {
            imageProxy.close()
        }
    }

    private fun resolveCurrentModuleStatusLine(): String {
        val moduleType = when (analysisUiController.currentMode) {
            AnalysisMode.POSE -> ModuleStatusStore.ModuleType.MEDIAPIPE
            AnalysisMode.YOLO -> ModuleStatusStore.ModuleType.YOLO
            else -> return getString(R.string.module_status_not_applicable)
        }
        val snapshot = ModuleStatusStore.get(moduleType)
        return getString(R.string.module_status_template, moduleType.name, snapshot.status.name)
    }

    private fun configureImageProcessorLabels() {
        imageProcessor.calibrator = cameraCalibrator
        imageProcessor.labelFrameCountSuffix = getString(R.string.calibration_overlay_frames_suffix)
        imageProcessor.labelBoardNotFound = getString(R.string.calibration_overlay_board_not_found)
        imageProcessor.labelNoCalibration = getString(R.string.calibration_overlay_no_calibration)
        imageProcessor.labelOdometryTracks = getString(R.string.vo_overlay_tracks)
        imageProcessor.labelOdometryInliers = getString(R.string.odometry_overlay_inliers)
        imageProcessor.labelOdometryFrames = getString(R.string.odometry_overlay_frames)
        imageProcessor.labelOdometrySteps = getString(R.string.odometry_overlay_steps)
        imageProcessor.labelOdometryPosition = getString(R.string.odometry_overlay_position)
        imageProcessor.labelPointCloud = getString(R.string.vo_overlay_point_cloud)
        imageProcessor.labelTrajectory = getString(R.string.odometry_overlay_trajectory)
        imageProcessor.labelMap3D = getString(R.string.odometry_overlay_map)
        imageProcessor.labelOdometryPoints = getString(R.string.odometry_overlay_points)
        imageProcessor.labelCollectingData = getString(R.string.odometry_overlay_collecting_data)
        imageProcessor.labelCollectingPoints = getString(R.string.odometry_overlay_collecting_points)
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
        imageProcessor.onLargeMapDetected = { map -> backgroundExecutor.execute { saveSlamMap(map, isAutoSave = true) } }
        imageProcessor.onLoopClosed = { message -> runOnUiThread { Toast.makeText(this, message, Toast.LENGTH_LONG).show() } }
    }

    private fun logExceptionTelemetry(scope: String, category: String, error: Throwable) {
        val key = "$scope:$category"
        exceptionTelemetry.getOrPut(key) { AtomicInteger(0) }.incrementAndGet()
        telemetryScope.launch {
            try {
                dataCollectionCache.incrementErrorCount(scope, category)
            } catch (persistError: Exception) {
                Log.e(TAG, "Telemetry persistence failure", persistError)
            }
        }
    }

    private fun initOpenCv() {
        if (!OpenCVLoader.initLocal()) {
            Toast.makeText(this, getString(R.string.opencv_init_error), Toast.LENGTH_LONG).show()
        }
    }

    private fun requestPermissionsOrStart() {
        if (requestedPermissions().all {
                ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
            }
        ) {
            cameraController.startCamera()
        } else {
            permissionsLauncher.launch(requestedPermissions())
        }
    }

    // Lista uprawnień, o które aplikacja aktywnie prosi użytkownika podczas startu.
    private fun requestedPermissions(): Array<String> = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO,
    ) + if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
        arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE)
    } else {
        emptyArray()
    }

    private fun orientBitmap(bitmap: Bitmap, rotation: Int, facing: Int): Bitmap {
        val isFront = facing == CameraSelector.LENS_FACING_FRONT
        if (rotation == 0 && !isFront) return bitmap
        val matrix = Matrix()
        if (rotation != 0) matrix.postRotate(rotation.toFloat())
        if (isFront) matrix.postScale(-1f, 1f)
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun openResolutionMenu() {
        supportFragmentManager.setFragmentResultListener("resolution_request", this) { _, bundle ->
            val resolutionName = bundle.getString("selected_resolution") ?: return@setFragmentResultListener
            val resolution = CameraResolution.valueOf(resolutionName)
            cameraController.updateResolution(resolution)
            prefs.edit { putString(PREF_CAMERA_RESOLUTION, resolution.name) }
            cameraController.startCamera()
        }
        ResolutionBottomSheet.newInstance(cameraController.currentResolution)
            .show(supportFragmentManager, ResolutionBottomSheet.TAG)
    }

    private fun openCalibrationMenu() {
        CalibrationBottomSheet().apply {
            onCollectFrame = {
                val collected = cameraCalibrator.collectLastFrame()
                if (collected) {
                    Toast.makeText(
                        this@MainActivity,
                        getString(
                            R.string.calibration_frame_collected,
                            cameraCalibrator.frameCount,
                            CameraCalibrator.MIN_FRAMES,
                        ),
                        Toast.LENGTH_SHORT,
                    ).show()
                }
                collected
            }
            onCalibrate = {
                val result = cameraCalibrator.calibrate()
                Toast.makeText(
                    this@MainActivity,
                    if (result != null) R.string.calibration_success else R.string.calibration_failed,
                    Toast.LENGTH_SHORT,
                ).show()
                result
            }
            onReset = { cameraCalibrator.reset() }
        }.show(supportFragmentManager, CalibrationBottomSheet.TAG)
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
            .setItems(formats) { _, which -> savePointCloud(cloud, usePly = which == 1) }
            .show()
    }

    private fun showSaveSlamMapDialog() {
        val map = imageProcessor.currentSlamMap
        if (map.points3d.isEmpty() && map.markers.isEmpty()) {
            Toast.makeText(this, getString(R.string.point_cloud_empty), Toast.LENGTH_SHORT).show()
            return
        }
        AlertDialog.Builder(this)
            .setTitle(getString(R.string.point_cloud_format_title))
            .setItems(arrayOf(getString(R.string.point_cloud_format_ply))) { _, _ -> saveSlamMap(map) }
            .show()
    }

    private fun saveSlamMap(map: FullOdometryEngine.MapState, isAutoSave: Boolean = false) {
        val timestamp = System.currentTimeMillis()
        val filename = if (isAutoSave) "autosave_slam_map_$timestamp.csv" else "slam_map_$timestamp.ply"
        val content = if (isAutoSave) {
            buildString {
                appendLine("x,y,z,r,g,b,label")
                map.points3d.forEachIndexed { i, p ->
                    val colorInt = map.colors[i]
                    appendLine("${p.x},${p.y},${p.z},${Color.red(colorInt)},${Color.green(colorInt)},${Color.blue(colorInt)},landmark")
                }
                map.markers.forEach { marker ->
                    appendLine("${marker.position.x},${marker.position.y},${marker.position.z},0,255,255,${marker.label}")
                }
            }
        } else {
            buildString {
                appendLine("ply")
                appendLine("format ascii 1.0")
                appendLine("element vertex ${map.points3d.size + map.markers.size}")
                appendLine("property float x")
                appendLine("property float y")
                appendLine("property float z")
                appendLine("property uchar red")
                appendLine("property uchar green")
                appendLine("property uchar blue")
                appendLine("end_header")
                map.points3d.forEachIndexed { i, p ->
                    val c = map.colors[i]
                    appendLine("${p.x} ${p.y} ${p.z} ${Color.red(c)} ${Color.green(c)} ${Color.blue(c)}")
                }
                map.markers.forEach { marker ->
                    appendLine("${marker.position.x} ${marker.position.y} ${marker.position.z} 0 255 255")
                }
            }
        }
        writeToDownloads(
            filename = filename,
            mimeType = if (isAutoSave) "text/csv" else "application/octet-stream",
            content = content,
            silent = isAutoSave,
        )
    }

    private fun loadSlamMap(uri: Uri) {
        backgroundExecutor.execute {
            try {
                contentResolver.openInputStream(uri)?.bufferedReader()?.use { reader ->
                    val isPly = queryFileName(uri).endsWith(".ply", ignoreCase = true)
                    val points = mutableListOf<Point3>()
                    val colors = mutableListOf<Int>()
                    val markers = mutableListOf<FullOdometryEngine.MarkerLandmark>()
                    if (isPly) {
                        var line = reader.readLine()
                        while (line != null && line.trim() != "end_header") line = reader.readLine()
                        reader.lineSequence().forEach { row ->
                            val parts = row.trim().split(Regex("\\s+"))
                            if (parts.size >= 3) {
                                val x = parts[0].toDoubleOrNull() ?: return@forEach
                                val y = parts[1].toDoubleOrNull() ?: return@forEach
                                val z = parts[2].toDoubleOrNull() ?: return@forEach
                                points.add(Point3(x, y, z))
                                if (parts.size >= 6) {
                                    colors.add(
                                        Color.rgb(
                                            parts[3].toIntOrNull() ?: 255,
                                            parts[4].toIntOrNull() ?: 255,
                                            parts[5].toIntOrNull() ?: 255,
                                        ),
                                    )
                                } else {
                                    colors.add(Color.WHITE)
                                }
                            }
                        }
                    } else {
                        reader.readLine()
                        reader.lineSequence().forEach { row ->
                            val parts = row.trim().split(",")
                            if (parts.size >= 3) {
                                val x = parts[0].toDoubleOrNull() ?: return@forEach
                                val y = parts[1].toDoubleOrNull() ?: return@forEach
                                val z = parts[2].toDoubleOrNull() ?: return@forEach
                                val label = parts.getOrNull(6) ?: "landmark"
                                if (label == "landmark") {
                                    points.add(Point3(x, y, z))
                                    colors.add(
                                        Color.rgb(
                                            parts.getOrNull(3)?.toIntOrNull() ?: 255,
                                            parts.getOrNull(4)?.toIntOrNull() ?: 255,
                                            parts.getOrNull(5)?.toIntOrNull() ?: 255,
                                        ),
                                    )
                                } else {
                                    markers.add(
                                        FullOdometryEngine.MarkerLandmark(label, Point3(x, y, z), label),
                                    )
                                }
                            }
                        }
                    }
                    if (points.isNotEmpty() || markers.isNotEmpty()) {
                        imageProcessor.fullOdometryEngine.importMap(
                            FullOdometryEngine.MapState(points, colors, null, null, markers),
                        )
                    }
                }
            } catch (error: Exception) {
                Log.e(TAG, "Failed to load SLAM map", error)
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
        val timestamp = System.currentTimeMillis()
        val content = if (usePly) {
            buildString {
                appendLine("ply")
                appendLine("format ascii 1.0")
                appendLine("element vertex ${cloud.points.size}")
                appendLine("property float x")
                appendLine("property float y")
                appendLine("property float z")
                appendLine("property uchar red")
                appendLine("property uchar green")
                appendLine("property uchar blue")
                appendLine("end_header")
                cloud.points.forEachIndexed { i, p ->
                    val c = cloud.colors[i]
                    appendLine("${p.x} ${p.y} ${pseudoZ(p.y, cloud.meanParallax)} ${c.`val`[0].toInt()} ${c.`val`[1].toInt()} ${c.`val`[2].toInt()}")
                }
            }
        } else {
            buildString {
                appendLine("x,y,z,r,g,b")
                cloud.points.forEachIndexed { i, p ->
                    val c = cloud.colors[i]
                    appendLine("${p.x},${p.y},${pseudoZ(p.y, cloud.meanParallax)},${c.`val`[0].toInt()},${c.`val`[1].toInt()},${c.`val`[2].toInt()}")
                }
            }
        }
        writeToDownloads(
            filename = "pointcloud_$timestamp.${if (usePly) "ply" else "csv"}",
            mimeType = if (usePly) "application/octet-stream" else "text/csv",
            content = content,
        )
    }

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
                if (!silent) {
                    Toast.makeText(this, getString(R.string.point_cloud_saved, "Download/MobileCV/$filename"), Toast.LENGTH_SHORT).show()
                }
            }
        } else {
            @Suppress("DEPRECATION")
            val dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            val subDir = File(dir, "MobileCV").also { it.mkdirs() }
            File(subDir, filename).writeText(content)
            if (!silent) {
                Toast.makeText(this, getString(R.string.point_cloud_saved, "${subDir.absolutePath}/$filename"), Toast.LENGTH_SHORT).show()
            }
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
}
