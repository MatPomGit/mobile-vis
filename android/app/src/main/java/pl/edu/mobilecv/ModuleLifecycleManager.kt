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
    private val rtmDetProcessor: RtmDetProcessor,
    private val tfliteProcessor: TfliteProcessor,
    private val backgroundExecutor: ExecutorService,
    private val callbacks: Callbacks,
) {
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
    private val rtmDetDownloadInProgress = AtomicBoolean(false)
    private val mobilintDownloadInProgress = AtomicBoolean(false)
    private val tfliteDownloadInProgress = AtomicBoolean(false)

    fun initializeModules() {
        backgroundExecutor.execute {
            initialize("startup_module_init", "mediapipe") {
                mediaPipeProcessor.initialize()
                imageProcessor.mediaPipeProcessor = mediaPipeProcessor
            }
            initialize("startup_module_init", "yolo") {
                yoloProcessor.initialize()
                imageProcessor.yoloProcessor = yoloProcessor
            }
            initialize("startup_module_init", "rtmdet") {
                rtmDetProcessor.initialize()
                imageProcessor.rtmDetProcessor = rtmDetProcessor
            }
            initialize("startup_module_init", "tflite") {
                tfliteProcessor.initialize()
                imageProcessor.tfliteProcessor = tfliteProcessor
            }
            ensureStartupDownloads()
        }
    }

    fun closeModules() {
        backgroundExecutor.execute {
            mediaPipeProcessor.close()
            yoloProcessor.close()
            rtmDetProcessor.close()
            tfliteProcessor.close()
            imageProcessor.release()
        }
    }

    fun ensureModelsForMode(mode: AnalysisMode) {
        when (mode) {
            AnalysisMode.POSE -> if (!ModelDownloadManager.areAllModelsReady(context)) {
                startMediaPipeModelDownload()
            }
            AnalysisMode.YOLO, AnalysisMode.TRACKING -> if (!ModelDownloadManager.areYoloModelsReady(context)) {
                startYoloModelDownload()
            }
            AnalysisMode.RTMDET -> if (!ModelDownloadManager.areRtmDetModelsReady(context)) {
                startRtmDetModelDownload()
            }
            AnalysisMode.MOBILINT -> if (!ModelDownloadManager.areMobilintModelsReady(context)) {
                startMobilintModelDownload()
            }
            AnalysisMode.TFLITE -> if (!ModelDownloadManager.areTfliteModelsReady(context)) {
                startTfliteModelDownload()
            }
            else -> Unit
        }
    }

    private fun ensureStartupDownloads() {
        if (!ModelDownloadManager.areYoloModelsReady(context)) {
            startYoloModelDownload()
        }
        if (!ModelDownloadManager.areRtmDetModelsReady(context)) {
            startRtmDetModelDownload()
        }
    }

    private fun startMediaPipeModelDownload() {
        if (mediaPipeDownloadInProgress) return
        mediaPipeDownloadInProgress = true
        callbacks.showToast(R.string.mediapipe_models_downloading, longDuration = true)
        backgroundExecutor.execute {
            try {
                if (ModelDownloadManager.downloadMissingModels(context)) {
                    mediaPipeProcessor.close()
                    mediaPipeProcessor.initialize()
                    imageProcessor.mediaPipeProcessor = mediaPipeProcessor
                    callbacks.showToast(R.string.mediapipe_models_ready)
                }
            } finally {
                mediaPipeDownloadInProgress = false
            }
        }
    }

    private fun startYoloModelDownload() {
        if (!yoloDownloadInProgress.compareAndSet(false, true)) return
        callbacks.showToast(R.string.yolo_models_downloading, longDuration = true)
        backgroundExecutor.execute {
            try {
                val downloaded = ModelDownloadManager.downloadMissingYoloModels(context)
                if (downloaded) {
                    yoloProcessor.close()
                    yoloProcessor.initialize()
                    imageProcessor.yoloProcessor = yoloProcessor
                    callbacks.showToast(R.string.yolo_models_ready)
                } else {
                    callbacks.showToast(R.string.yolo_models_download_failed, longDuration = true)
                }
            } catch (error: Exception) {
                callbacks.onTelemetry("yolo_download", "unexpected", error)
                Log.e(TAG, "YOLO model download failed", error)
                callbacks.showToast(R.string.yolo_models_download_failed, longDuration = true)
            } finally {
                yoloDownloadInProgress.set(false)
            }
        }
    }

    private fun startRtmDetModelDownload() {
        if (!rtmDetDownloadInProgress.compareAndSet(false, true)) return
        callbacks.showToast(R.string.rtmdet_models_downloading, longDuration = true)
        backgroundExecutor.execute {
            try {
                val downloaded = ModelDownloadManager.downloadMissingRtmDetModels(context)
                if (downloaded) {
                    rtmDetProcessor.close()
                    rtmDetProcessor.initialize()
                    imageProcessor.rtmDetProcessor = rtmDetProcessor
                    callbacks.showToast(R.string.rtmdet_models_ready)
                } else {
                    callbacks.showToast(R.string.rtmdet_models_download_failed, longDuration = true)
                }
            } catch (error: Exception) {
                callbacks.onTelemetry("rtmdet_download", "unexpected", error)
                Log.e(TAG, "RTMDet model download failed", error)
                callbacks.showToast(R.string.rtmdet_models_download_failed, longDuration = true)
            } finally {
                rtmDetDownloadInProgress.set(false)
            }
        }
    }

    private fun startMobilintModelDownload() {
        if (!mobilintDownloadInProgress.compareAndSet(false, true)) return
        callbacks.showToast(R.string.mobilint_models_downloading, longDuration = true)
        backgroundExecutor.execute {
            try {
                if (ModelDownloadManager.downloadMissingMobilintModels(context)) {
                    callbacks.showToast(R.string.mobilint_models_ready)
                } else {
                    callbacks.showToast(R.string.mobilint_models_download_failed, longDuration = true)
                }
            } catch (error: Exception) {
                callbacks.onTelemetry("mobilint_download", "unexpected", error)
                Log.e(TAG, "Mobilint model download failed", error)
                callbacks.showToast(R.string.mobilint_models_download_failed, longDuration = true)
            } finally {
                mobilintDownloadInProgress.set(false)
            }
        }
    }

    private fun startTfliteModelDownload() {
        if (!tfliteDownloadInProgress.compareAndSet(false, true)) return
        callbacks.showToast(R.string.tflite_models_downloading, longDuration = true)
        backgroundExecutor.execute {
            try {
                if (ModelDownloadManager.downloadMissingTfliteModels(context)) {
                    tfliteProcessor.close()
                    tfliteProcessor.initialize()
                    imageProcessor.tfliteProcessor = tfliteProcessor
                    callbacks.showToast(R.string.tflite_models_ready)
                } else {
                    callbacks.showToast(R.string.tflite_models_download_failed, longDuration = true)
                }
            } catch (error: Exception) {
                callbacks.onTelemetry("tflite_download", "unexpected", error)
                Log.e(TAG, "TFLite model download failed", error)
                callbacks.showToast(R.string.tflite_models_download_failed, longDuration = true)
            } finally {
                tfliteDownloadInProgress.set(false)
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
