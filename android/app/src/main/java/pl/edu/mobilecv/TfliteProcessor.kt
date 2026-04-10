package pl.edu.mobilecv

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.File

/**
 * Handles TensorFlow Lite model inference for object detection.
 * Optimized for mobile using GPU acceleration when available.
 */
class TfliteProcessor(private val context: Context) {

    companion object {
        private const val TAG = "TfliteProcessor"
        const val MODEL_SSD_MOBILENET = "ssd_mobilenet_v2.tflite"
        
        private const val CONFIDENCE_THRESHOLD = 0.5f
        private const val NUM_DETECTIONS = 10
    }

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var inputImageSize = 300 // Default for SSD MobileNet V2

    // Bufory współdzielone między klatkami ograniczają alokacje i GC podczas inferencji.
    private val inputTensorImage = TensorImage(DataType.UINT8)
    private var resizeProcessor: ImageProcessor = buildResizeProcessor(inputImageSize)
    private val inferenceOutputs: MutableMap<Int, Any> = mutableMapOf()
    private val inferenceInputs: Array<Any> = arrayOf(ByteArray(0))

    // Output buffers for SSD MobileNet
    private lateinit var outputLocations: Array<Array<FloatArray>>
    private lateinit var outputClasses: Array<FloatArray>
    private lateinit var outputScores: Array<FloatArray>
    private lateinit var numDetections: FloatArray

    // Reużywane obiekty rysujące minimalizują narzut CPU poza samą inferencją.
    private val boxPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 4f
        textSize = 40f
    }
    private val textPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.FILL
        textSize = 36f
    }

    fun initialize() {
        val modelPath = ModelDownloadManager.getTfliteModelPath(context, MODEL_SSD_MOBILENET)
        if (modelPath != null) {
            loadModel(File(modelPath))
        } else {
            Log.e(TAG, "TFLite model not found: $MODEL_SSD_MOBILENET")
        }
    }

    private fun loadModel(modelFile: File) {
        try {
            val options = Interpreter.Options()
            val compatList = CompatibilityList()

            if (compatList.isDelegateSupportedOnThisDevice) {
                val delegateOptions = compatList.bestOptionsForThisDevice
                val gDelegate = GpuDelegate(delegateOptions)
                gpuDelegate = gDelegate
                options.addDelegate(gDelegate)
                Log.i(TAG, "TFLite using GPU acceleration")
            } else {
                options.setNumThreads(4)
                Log.i(TAG, "TFLite using CPU (4 threads)")
            }

            interpreter = Interpreter(modelFile, options)
            
            // Inspect input shape to adjust resizing
            val inputShape = interpreter?.getInputTensor(0)?.shape()
            if (inputShape != null && inputShape.size >= 3) {
                inputImageSize = inputShape[1]
                resizeProcessor = buildResizeProcessor(inputImageSize)
                Log.d(TAG, "Model input size: $inputImageSize")
            }

            // Prepare output buffers (SSD MobileNet V2 format)
            outputLocations = arrayOf(Array(NUM_DETECTIONS) { FloatArray(4) })
            outputClasses = arrayOf(FloatArray(NUM_DETECTIONS))
            outputScores = arrayOf(FloatArray(NUM_DETECTIONS))
            numDetections = FloatArray(1)
            inferenceOutputs[0] = outputLocations
            inferenceOutputs[1] = outputClasses
            inferenceOutputs[2] = outputScores
            inferenceOutputs[3] = numDetections

        } catch (e: Exception) {
            Log.e(TAG, "Error loading TFLite model", e)
        }
    }

    fun close() {
        interpreter?.close()
        interpreter = null
        gpuDelegate?.close()
        gpuDelegate = null
    }

    fun processFrame(bitmap: Bitmap, filter: OpenCvFilter): Bitmap {
        val interp = interpreter ?: return drawModelMissing(bitmap, MODEL_SSD_MOBILENET)

        try {
            val startTime = SystemClock.elapsedRealtimeNanos()
            
            // Reużycie obiektów preprocessingu poprawia stabilność FPS przy delegacie GPU.
            inputTensorImage.load(bitmap)
            val processedImage = resizeProcessor.process(inputTensorImage)
            inferenceInputs[0] = processedImage.buffer

            // Run inference
            interp.runForMultipleInputsOutputs(inferenceInputs, inferenceOutputs)

            // Visualize results
            val result = bitmap.copy(Bitmap.Config.ARGB_8888, true)
            val canvas = Canvas(result)

            val count = numDetections[0].toInt().coerceAtMost(NUM_DETECTIONS)
            for (i in 0 until count) {
                val score = outputScores[0][i]
                if (score < CONFIDENCE_THRESHOLD) continue

                val classId = outputClasses[0][i].toInt()
                val box = outputLocations[0][i] // [top, left, bottom, right]
                
                val rect = RectF(
                    box[1] * bitmap.width,
                    box[0] * bitmap.height,
                    box[3] * bitmap.width,
                    box[2] * bitmap.height
                )

                canvas.drawRect(rect, boxPaint)
                canvas.drawText("Class $classId: ${"%.2f".format(score)}", rect.left, rect.top - 10f, textPaint)
            }

            val processTimeMs = (SystemClock.elapsedRealtimeNanos() - startTime) / 1_000_000.0
            Log.v(TAG, "Inference time: %.2fms".format(processTimeMs))
            
            return result

        } catch (e: Exception) {
            Log.e(TAG, "TFLite processing failed", e)
            return drawModuleError(bitmap, "TFLite error: ${e.message}")
        }
    }

    private fun drawModelMissing(bitmap: Bitmap, model: String): Bitmap {
        val result = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(result)
        val paint = Paint().apply { color = Color.RED; textSize = 40f; isFakeBoldText = true }
        canvas.drawText("Model missing: $model", 50f, 100f, paint)
        return result
    }

    private fun drawModuleError(bitmap: Bitmap, message: String): Bitmap {
        val result = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(result)
        val paint = Paint().apply { color = Color.RED; textSize = 40f; isFakeBoldText = true }
        canvas.drawText(message, 50f, 100f, paint)
        return result
    }

    // Oddzielna funkcja ułatwia utrzymanie preprocessingu spójnego z wejściem modelu.
    private fun buildResizeProcessor(targetSize: Int): ImageProcessor {
        return ImageProcessor.Builder()
            .add(ResizeOp(targetSize, targetSize, ResizeOp.ResizeMethod.BILINEAR))
            .build()
    }
}
