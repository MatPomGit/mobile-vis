package pl.edu.mobilecv

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.File
import java.nio.MappedByteBuffer

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

    // Output buffers for SSD MobileNet
    private lateinit var outputLocations: Array<Array<FloatArray>>
    private lateinit var outputClasses: Array<FloatArray>
    private lateinit var outputScores: Array<FloatArray>
    private lateinit var numDetections: FloatArray

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
                gpuDelegate = GpuDelegate(delegateOptions)
                options.addDelegate(gpuDelegate)
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
                Log.d(TAG, "Model input size: $inputImageSize")
            }

            // Prepare output buffers (SSD MobileNet V2 format)
            outputLocations = arrayOf(Array(NUM_DETECTIONS) { FloatArray(4) })
            outputClasses = arrayOf(FloatArray(NUM_DETECTIONS))
            outputScores = arrayOf(FloatArray(NUM_DETECTIONS))
            numDetections = FloatArray(1)

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
            val startTime = System.currentTimeMillis()
            
            // Pre-process image
            val tensorImage = TensorImage(org.tensorflow.lite.DataType.UINT8)
            tensorImage.load(bitmap)
            
            val imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(inputImageSize, inputImageSize, ResizeOp.Method.BILINEAR))
                .build()
            
            val processedImage = imageProcessor.process(tensorImage)

            // Run inference
            val outputs = mutableMapOf<Int, Any>()
            outputs[0] = outputLocations
            outputs[1] = outputClasses
            outputs[2] = outputScores
            outputs[3] = numDetections
            
            interp.runForMultipleInputsOutputs(arrayOf(processedImage.buffer), outputs)

            // Visualize results
            val result = bitmap.copy(Bitmap.Config.ARGB_8888, true)
            val canvas = Canvas(result)
            val paint = Paint().apply {
                color = Color.GREEN
                style = Paint.Style.STROKE
                strokeWidth = 4f
                textSize = 40f
            }
            val textPaint = Paint().apply {
                color = Color.GREEN
                style = Paint.Style.FILL
                textSize = 36f
            }

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

                canvas.drawRect(rect, paint)
                canvas.drawText("Class $classId: ${"%.2f".format(score)}", rect.left, rect.top - 10f, textPaint)
            }

            val processTime = System.currentTimeMillis() - startTime
            Log.v(TAG, "Inference time: ${processTime}ms")
            
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
}
