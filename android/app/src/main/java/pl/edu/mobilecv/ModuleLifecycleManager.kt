package pl.edu.mobilecv

import android.content.Context
import android.util.Log
import java.util.concurrent.ExecutorService
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Menedżer cyklu życia modułów ML/CV oraz ich pobierania modeli.
 */
class ModuleLifecycleManager(
    private val context: Context,
    private val imageProcessor: ImageProcessor,
    private val mediaPipeProcessor: MediaPipeProcessor,
    private val yoloProcessor: YoloProcessor,
    private val tfliteProcessor: TfliteProcessor,
    private val backgroundExecutor: ExecutorService,
    private val callbacks: Callbacks,
) {
    enum class RepairAction {
        RETRY_DOWNLOAD,
        DISABLE_MODULE,
    }

    interface Callbacks {
        /** Logowanie błędu telemetrycznego po stronie hosta. */
        fun onTelemetry(scope: String, category: String, error: Throwable)

        /** Wyświetla informację dla użytkownika na wątku UI. */
        fun showToast(messageRes: Int, longDuration: Boolean = false)
    }

    companion object {
        private const val TAG = "MobileCV"
    }

    @Volatile
    private var mediaPipeDownloadInProgress = false
    private val yoloDownloadInProgress = AtomicBoolean(false)
    private val tfliteDownloadInProgress = AtomicBoolean(false)

    fun initializeModules() {
        initializeModuleAsync(ModuleStatusStore.ModuleType.MEDIAPIPE) {
            initialize("startup_module_init", "mediapipe") {
                mediaPipeProcessor.initialize()
                imageProcessor.mediaPipeProcessor = mediaPipeProcessor
            }
        }
        initializeModuleAsync(ModuleStatusStore.ModuleType.YOLO) {
            initialize("startup_module_init", "yolo") {
                yoloProcessor.initialize()
                imageProcessor.yoloProcessor = yoloProcessor
            }
        }
        initializeModuleAsync(ModuleStatusStore.ModuleType.TFLITE) {
            initialize("startup_module_init", "tflite") {
                tfliteProcessor.initialize()
                imageProcessor.tfliteProcessor = tfliteProcessor
            }
        }
        backgroundExecutor.execute {
            ensureStartupDownloads()
        }
    }

    fun closeModules() {
        backgroundExecutor.execute {
            mediaPipeProcessor.close()
            yoloProcessor.close()
            tfliteProcessor.close()
            imageProcessor.release()
        }
    }

    fun ensureModelsForMode(mode: AnalysisMode) {
        when (mode) {
            AnalysisMode.POSE -> if (!ModelDownloadManager.areAllModelsReady(context)) {
                startMediaPipeModelDownload()
            }
            AnalysisMode.YOLO -> if (!ModelDownloadManager.areYoloModelsReady(context)) {
                startYoloModelDownload()
            }
            else -> Unit
        }
    }

    /** Obsługuje akcję naprawczą z UI dla wskazanego modułu. */
    fun applyRepairAction(moduleType: ModuleStatusStore.ModuleType, action: RepairAction) {
        when (action) {
            RepairAction.RETRY_DOWNLOAD -> when (moduleType) {
                ModuleStatusStore.ModuleType.MEDIAPIPE -> startMediaPipeModelDownload()
                ModuleStatusStore.ModuleType.YOLO -> startYoloModelDownload()
                ModuleStatusStore.ModuleType.TFLITE -> startTfliteModelDownload()
            }
            RepairAction.DISABLE_MODULE -> {
                disableModule(moduleType)
                ModuleStatusStore.setStatus(moduleType, ModuleStatusStore.ModuleStatus.DISABLED)
            }
        }
    }

    private fun ensureStartupDownloads() {
        if (!ModelDownloadManager.areYoloModelsReady(context)) {
            startYoloModelDownload()
        }
    }

    private fun startMediaPipeModelDownload() {
        if (mediaPipeDownloadInProgress) return
        mediaPipeDownloadInProgress = true
        ModuleStatusStore.setStatus(ModuleStatusStore.ModuleType.MEDIAPIPE, ModuleStatusStore.ModuleStatus.DOWNLOADING)
        callbacks.showToast(R.string.mediapipe_models_downloading, longDuration = true)
        backgroundExecutor.execute {
            try {
                if (ModelDownloadManager.downloadMissingModels(context)) {
                    mediaPipeProcessor.close()
                    mediaPipeProcessor.initialize()
                    imageProcessor.mediaPipeProcessor = mediaPipeProcessor
                    ModuleStatusStore.setStatus(ModuleStatusStore.ModuleType.MEDIAPIPE, ModuleStatusStore.ModuleStatus.READY)
                    callbacks.showToast(R.string.mediapipe_models_ready)
                } else {
                    ModuleStatusStore.setStatus(
                        ModuleStatusStore.ModuleType.MEDIAPIPE,
                        ModuleStatusStore.ModuleStatus.ERROR,
                        R.string.module_error_mediapipe,
                    )
                }
            } catch (error: Exception) {
                callbacks.onTelemetry("mediapipe_download", "unexpected", error)
                ModuleStatusStore.setStatus(
                    ModuleStatusStore.ModuleType.MEDIAPIPE,
                    ModuleStatusStore.ModuleStatus.ERROR,
                    R.string.module_error_mediapipe,
                )
            } finally {
                mediaPipeDownloadInProgress = false
            }
        }
    }

    private fun startYoloModelDownload() {
        if (!yoloDownloadInProgress.compareAndSet(false, true)) return
        ModuleStatusStore.setStatus(ModuleStatusStore.ModuleType.YOLO, ModuleStatusStore.ModuleStatus.DOWNLOADING)
        callbacks.showToast(R.string.yolo_models_downloading, longDuration = true)
        backgroundExecutor.execute {
            try {
                val downloaded = ModelDownloadManager.downloadMissingYoloModels(context)
                if (downloaded) {
                    yoloProcessor.close()
                    yoloProcessor.initialize()
                    imageProcessor.yoloProcessor = yoloProcessor
                    ModuleStatusStore.setStatus(ModuleStatusStore.ModuleType.YOLO, ModuleStatusStore.ModuleStatus.READY)
                    callbacks.showToast(R.string.yolo_models_ready)
                } else {
                    ModuleStatusStore.setStatus(
                        ModuleStatusStore.ModuleType.YOLO,
                        ModuleStatusStore.ModuleStatus.ERROR,
                        R.string.module_error_yolo,
                    )
                    callbacks.showToast(R.string.yolo_models_download_failed, longDuration = true)
                }
            } catch (error: Exception) {
                callbacks.onTelemetry("yolo_download", "unexpected", error)
                Log.e(TAG, "YOLO model download failed", error)
                ModuleStatusStore.setStatus(
                    ModuleStatusStore.ModuleType.YOLO,
                    ModuleStatusStore.ModuleStatus.ERROR,
                    R.string.module_error_yolo,
                )
                callbacks.showToast(R.string.yolo_models_download_failed, longDuration = true)
            } finally {
                yoloDownloadInProgress.set(false)
            }
        }
    }

    private fun startTfliteModelDownload() {
        if (!tfliteDownloadInProgress.compareAndSet(false, true)) return
        ModuleStatusStore.setStatus(ModuleStatusStore.ModuleType.TFLITE, ModuleStatusStore.ModuleStatus.DOWNLOADING)
        callbacks.showToast(R.string.tflite_models_downloading, longDuration = true)
        backgroundExecutor.execute {
            try {
                if (ModelDownloadManager.downloadMissingTfliteModels(context)) {
                    tfliteProcessor.close()
                    tfliteProcessor.initialize()
                    imageProcessor.tfliteProcessor = tfliteProcessor
                    ModuleStatusStore.setStatus(ModuleStatusStore.ModuleType.TFLITE, ModuleStatusStore.ModuleStatus.READY)
                    callbacks.showToast(R.string.tflite_models_ready)
                } else {
                    ModuleStatusStore.setStatus(
                        ModuleStatusStore.ModuleType.TFLITE,
                        ModuleStatusStore.ModuleStatus.ERROR,
                        R.string.module_error_tflite,
                    )
                    callbacks.showToast(R.string.tflite_models_download_failed, longDuration = true)
                }
            } catch (error: Exception) {
                callbacks.onTelemetry("tflite_download", "unexpected", error)
                Log.e(TAG, "TFLite model download failed", error)
                ModuleStatusStore.setStatus(
                    ModuleStatusStore.ModuleType.TFLITE,
                    ModuleStatusStore.ModuleStatus.ERROR,
                    R.string.module_error_tflite,
                )
                callbacks.showToast(R.string.tflite_models_download_failed, longDuration = true)
            } finally {
                tfliteDownloadInProgress.set(false)
            }
        }
    }

    private fun initializeModuleAsync(moduleType: ModuleStatusStore.ModuleType, block: () -> Unit) {
        backgroundExecutor.execute {
            ModuleStatusStore.setStatus(moduleType, ModuleStatusStore.ModuleStatus.DOWNLOADING)
            try {
                block()
                ModuleStatusStore.setStatus(moduleType, ModuleStatusStore.ModuleStatus.READY)
            } catch (error: Throwable) {
                val errorMessageResId = when (moduleType) {
                    ModuleStatusStore.ModuleType.MEDIAPIPE -> R.string.module_error_mediapipe
                    ModuleStatusStore.ModuleType.YOLO -> R.string.module_error_yolo
                    ModuleStatusStore.ModuleType.TFLITE -> R.string.module_error_tflite
                }
                ModuleStatusStore.setStatus(
                    moduleType = moduleType,
                    status = ModuleStatusStore.ModuleStatus.ERROR,
                    errorMessageResId = errorMessageResId,
                )
                throw error
            }
        }
    }

    private fun disableModule(moduleType: ModuleStatusStore.ModuleType) {
        when (moduleType) {
            ModuleStatusStore.ModuleType.MEDIAPIPE -> {
                mediaPipeProcessor.close()
                imageProcessor.mediaPipeProcessor = null
            }
            ModuleStatusStore.ModuleType.YOLO -> {
                yoloProcessor.close()
                imageProcessor.yoloProcessor = null
            }
            ModuleStatusStore.ModuleType.TFLITE -> {
                tfliteProcessor.close()
                imageProcessor.tfliteProcessor = null
            }
        }
    }

    private fun initialize(scope: String, category: String, block: () -> Unit) {
        try {
            block()
        } catch (error: Throwable) {
            callbacks.onTelemetry(scope, category, error)
            Log.e(TAG, "Initialization failed for $category", error)
        }
    }
}
