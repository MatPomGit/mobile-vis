package pl.edu.mobilecv

import android.content.ContentValues
import android.content.Context
import android.os.Build
import android.provider.MediaStore
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.atomic.AtomicReference

/**
 * Kontroler odpowiedzialny za zarządzanie CameraX.
 */
class CameraController(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val analysisExecutor: ExecutorService,
    private val callbacks: Callbacks,
) {
    interface Callbacks {
        /** Callback dla klatki wejściowej z analizatora. */
        fun onFrame(imageProxy: ImageProxy)

        /** Powiadomienie o błędzie konfiguracji kamery. */
        fun onCameraError(error: Throwable)

        /** Powiadomienie o zmianie aktywnej kamery. */
        fun onLensFacingChanged(lensFacing: Int)

        /** Powiadomienie o wystartowaniu kamery (dla telemetry/debug). */
        fun onCameraStarted()
    }

    @Volatile
    private var cameraProvider: ProcessCameraProvider? = null

    @Volatile
    var lensFacing: Int = CameraSelector.LENS_FACING_BACK
        private set

    @Volatile
    var currentResolution: CameraResolution = CameraResolution.DEFAULT
        private set

    private val imageCaptureRef = AtomicReference<ImageCapture?>(null)

    fun updateResolution(resolution: CameraResolution) {
        currentResolution = resolution
    }

    fun switchCamera() {
        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
            CameraSelector.LENS_FACING_FRONT
        } else {
            CameraSelector.LENS_FACING_BACK
        }
        callbacks.onLensFacingChanged(lensFacing)
        startCamera()
    }

    fun startCamera() {
        callbacks.onCameraStarted()
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()
                bindUseCases()
            } catch (error: Exception) {
                callbacks.onCameraError(error)
            }
        }, ContextCompat.getMainExecutor(context))
    }

    fun stopCamera() {
        cameraProvider?.unbindAll()
    }

    fun takePhoto(onSuccess: () -> Unit, onError: (ImageCaptureException) -> Unit) {
        val capture = imageCaptureRef.get() ?: return
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, "IMG_${System.currentTimeMillis()}")
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/MobileCV")
            }
        }
        val outputOptions = ImageCapture.OutputFileOptions.Builder(
            context.contentResolver,
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            values,
        ).build()
        capture.takePicture(outputOptions, ContextCompat.getMainExecutor(context),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                    onSuccess()
                }

                override fun onError(exception: ImageCaptureException) {
                    onError(exception)
                }
            })
    }

    private fun bindUseCases() {
        val provider = cameraProvider ?: return
        val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
        val resolutionSelector = ResolutionSelector.Builder()
            .setResolutionStrategy(
                ResolutionStrategy(
                    currentResolution.size,
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
                ),
            )
            .build()
        val imageAnalysis = ImageAnalysis.Builder()
            .setResolutionSelector(resolutionSelector)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            .also { analysis ->
                analysis.setAnalyzer(analysisExecutor) { proxy ->
                    callbacks.onFrame(proxy)
                }
            }
        val imageCapture = ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .build()
        imageCaptureRef.set(imageCapture)

        provider.unbindAll()
        provider.bindToLifecycle(lifecycleOwner, cameraSelector, imageAnalysis, imageCapture)
    }
}
