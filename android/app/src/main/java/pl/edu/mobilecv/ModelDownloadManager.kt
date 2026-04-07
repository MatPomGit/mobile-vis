package pl.edu.mobilecv

import android.content.Context
import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * Downloads and manages MediaPipe task model files.
 *
 * Models are stored in [Context.getFilesDir]/mediapipe/ so they survive app updates
 * but are excluded from backups.  Each model is downloaded on demand the first time
 * a MediaPipe filter is selected.
 *
 * All network I/O is performed on the calling thread; always invoke download methods
 * from a background thread.
 */
object ModelDownloadManager {

    private const val TAG = "ModelDownloadManager"
    private const val MEDIAPIPE_DIR = "mediapipe"

    /** Connection timeout in seconds for model download requests. */
    private const val CONNECT_TIMEOUT_SECONDS = 30L

    /** Read timeout in seconds for model download responses. */
    private const val READ_TIMEOUT_SECONDS = 120L

    /** Write timeout in seconds for model download requests. */
    private const val WRITE_TIMEOUT_SECONDS = 60L

    /** Maximum number of download attempts before giving up (includes the initial attempt). */
    private const val MAX_DOWNLOAD_RETRIES = 3

    /** Milliseconds to wait before the first retry.  Doubles on each subsequent retry. */
    private const val RETRY_DELAY_MS = 2_000L

    /**
     * Remote base URL for MediaPipe model bundles hosted on Google Cloud Storage.
     *
     * Files are downloaded from:
     * ``${BASE_URL}/<model_path>/<model_name>/float16/1/<model_name>``
     */
    private const val BASE_URL =
        "https://storage.googleapis.com/mediapipe-models"

    /** Remote download URLs keyed by local model filename. */
    val MODEL_URLS: Map<String, String> = mapOf(
        MediaPipeProcessor.MODEL_POSE to
            "$BASE_URL/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        MediaPipeProcessor.MODEL_HAND to
            "$BASE_URL/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MediaPipeProcessor.MODEL_FACE to
            "$BASE_URL/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        MediaPipeProcessor.MODEL_FACE_DETECTOR to
            "$BASE_URL/face_detector/face_detector/float16/1/face_detector.task",
    )

    /**
     * Remote download URLs for YOLOv8-nano models exported to the ``.pt`` format.
     *
     * These ``*.pt`` files are hosted in the project's GitHub Releases. Each model
     * has been pre-exported from the base Ultralytics weights to be compatible with
     * the mobile runtime.
     *
     * The resulting ``*.pt`` files (see [YoloProcessor.MODEL_DETECT] etc.) are what
     * the Android app loads via ``Module.load()``.  Place the converted files in the YOLO
     * model directory ([getYoloModelPath]) before starting inference.
     */
    val YOLO_MODEL_URLS: Map<String, String> = mapOf(
        "yolov8n.pt" to
            "https://github.com/MatPomGit/mobile-vis/releases/download/models/yolov8n.pt",
        "yolov8n-seg.pt" to
            "https://github.com/MatPomGit/mobile-vis/releases/download/models/yolov8n-seg.pt",
        "yolov8n-pose.pt" to
            "https://github.com/MatPomGit/mobile-vis/releases/download/models/yolov8n-pose.pt",
        "yolov8n-cls.pt" to
            "https://github.com/MatPomGit/mobile-vis/releases/download/models/yolov8n-cls.pt",
        "yolov8n-obb.pt" to
            "https://github.com/MatPomGit/mobile-vis/releases/download/models/yolov8n-obb.pt",
    )

    private const val YOLO_DIR = "yolo"

    /**
     * Remote download URLs for RTMDet-nano TorchScript models hosted on GitHub Releases.
     *
     * The models are exported from the official OpenMMLab RTMDet weights via
     * mmdeploy and stored in the project's GitHub Releases under the ``models`` tag.
     * Replace these URLs with your own CDN or GitHub Releases links if you host
     * the models elsewhere.
     */
    val RTMDET_MODEL_URLS: Map<String, String> = mapOf(
        RtmDetProcessor.MODEL_DETECT to
            "https://github.com/MatPomGit/mobile-vis/releases/download/models/rtmdet_nano_det.pt",
        RtmDetProcessor.MODEL_ROTATED to
            "https://github.com/MatPomGit/mobile-vis/releases/download/models/rtmdet_nano_rotated.pt",
    )

    private const val RTMDET_DIR = "rtmdet"

    /**
     * Remote download URLs for Mobilint Ares NPU optimized models.
     *
     * These models are specially compiled for the Mobilint hardware and
     * are hosted in the project's GitHub Releases.
     */
    val MOBILINT_MODEL_URLS: Map<String, String> = mapOf(
        "mobilint_detect.mbl" to
            "https://github.com/MatPomGit/mobile-vis/releases/download/models/mobilint_detect.mbl",
    )

    private const val MOBILINT_DIR = "mobilint"

    private val httpClient: OkHttpClient by lazy {
        OkHttpClient.Builder()
            .connectTimeout(CONNECT_TIMEOUT_SECONDS, TimeUnit.SECONDS)
            .readTimeout(READ_TIMEOUT_SECONDS, TimeUnit.SECONDS)
            .writeTimeout(WRITE_TIMEOUT_SECONDS, TimeUnit.SECONDS)
            .build()
    }

    /**
     * Return the absolute path to [modelFilename] if the file exists in internal storage,
     * or ``null`` if it has not been downloaded yet.
     *
     * @param context Application or activity context.
     * @param modelFilename Filename such as [MediaPipeProcessor.MODEL_POSE].
     */
    fun getModelPath(context: Context, modelFilename: String): String? {
        val file = modelFile(context, modelFilename)
        return if (file.exists() && file.length() > 0) file.absolutePath else null
    }

    /**
     * Return ``true`` if **all** required model files are present and non-empty.
     *
     * @param context Application or activity context.
     */
    fun areAllModelsReady(context: Context): Boolean =
        MODEL_URLS.keys.all { getModelPath(context, it) != null }

    /**
     * Download all missing model files.
     *
     * This is a blocking call; run it on a background thread (e.g. inside
     * [MainActivity]'s analysis executor or a coroutine).
     *
     * @param context Application or activity context.
     * @param onProgress Optional callback invoked with current and total file count.
     * @return ``true`` if all models are now available; ``false`` if any download failed.
     */
    fun downloadMissingModels(
        context: Context,
        onProgress: ((downloaded: Int, total: Int) -> Unit)? = null,
    ): Boolean {
        val missing = MODEL_URLS.filter { (filename, _) ->
            getModelPath(context, filename) == null
        }
        if (missing.isEmpty()) return true

        var downloaded = 0
        val total = missing.size

        for ((filename, url) in missing) {
            val success = downloadModel(context, filename, url)
            if (success) {
                downloaded++
                onProgress?.invoke(downloaded, total)
            } else {
                Log.e(TAG, "Failed to download $filename")
                return false
            }
        }
        return true
    }

    /**
     * Download a single model file from [url] to internal storage.
     *
     * @param context Application or activity context.
     * @param filename Target filename.
     * @param url Source URL.
     * @return ``true`` on success.
     */
    fun downloadModel(context: Context, filename: String, url: String): Boolean =
        downloadModel(context, filename, url, modelFile(context, filename))

    /**
     * Download a single YOLO source model file from [url] to the YOLO directory.
     *
     * @param context Application or activity context.
     * @param filename Target filename (e.g. ``"yolov8n.pt"``).
     * @param url Source URL.
     * @return ``true`` on success.
     */
    fun downloadYoloModel(context: Context, filename: String, url: String): Boolean =
        downloadModel(context, filename, url, yoloModelFile(context, filename))

    /**
     * Download a single RTMDet model file from [url] to the RTMDet directory.
     *
     * @param context Application or activity context.
     * @param filename Target filename (e.g. [RtmDetProcessor.MODEL_DETECT]).
     * @param url Source URL.
     * @return ``true`` on success.
     */
    fun downloadRtmDetModel(context: Context, filename: String, url: String): Boolean =
        downloadModel(context, filename, url, rtmdetModelFile(context, filename))

    /**
     * Download a single model file from [url] to [dest].
     *
     * Each attempt writes to a temporary file alongside [dest] and atomically renames
     * it on success.  Failed attempts are retried up to [MAX_DOWNLOAD_RETRIES] times
     * with exponential back-off (starting at [RETRY_DELAY_MS] ms, doubling each time).
     *
     * @param context Application or activity context.
     * @param filename Logical filename used for logging.
     * @param url Source URL.
     * @param dest Destination [File] on internal storage.
     * @return ``true`` on success.
     */
    fun downloadMobilintModel(context: Context, modelName: String, url: String): Boolean {
        return downloadModel(context, modelName, url, mobilintModelFile(context, modelName))
    }

    private fun downloadModel(context: Context, filename: String, url: String, dest: File): Boolean {
        dest.parentFile?.mkdirs()

        Log.i(TAG, "Downloading $filename from $url")

        var delay = RETRY_DELAY_MS
        for (attempt in 1..MAX_DOWNLOAD_RETRIES) {
            val tempFile = File(dest.parent, "$filename.tmp")
            var attemptSucceeded = false
            try {
                val request = Request.Builder().url(url).build()
                httpClient.newCall(request).execute().use { response ->
                    if (!response.isSuccessful) {
                        Log.w(TAG, "HTTP ${response.code} downloading $filename (attempt $attempt/$MAX_DOWNLOAD_RETRIES)")
                        return@use
                    }
                    response.body.byteStream().use { input ->
                        tempFile.outputStream().use { output ->
                            input.copyTo(output)
                        }
                    }
                    if (tempFile.renameTo(dest)) {
                        Log.i(TAG, "Downloaded $filename (${dest.length()} bytes)")
                        attemptSucceeded = true
                    } else {
                        Log.w(TAG, "Failed to rename temp file for $filename (attempt $attempt/$MAX_DOWNLOAD_RETRIES)")
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "Exception downloading $filename (attempt $attempt/$MAX_DOWNLOAD_RETRIES): ${e.message}")
            } finally {
                if (!attemptSucceeded) {
                    tempFile.delete()
                }
            }

            if (attemptSucceeded) return true

            if (attempt < MAX_DOWNLOAD_RETRIES) {
                Log.i(TAG, "Retrying $filename in ${delay}ms")
                Thread.sleep(delay)
                delay *= 2
            }
        }

        Log.e(TAG, "Failed to download $filename after $MAX_DOWNLOAD_RETRIES attempts")
        return false
    }

    /**
     * Delete all downloaded model files from internal storage.
     *
     * @param context Application or activity context.
     */
    fun deleteAllModels(context: Context) {
        val dir = mediapipeDir(context)
        dir.listFiles()?.forEach { it.delete() }
        Log.i(TAG, "All MediaPipe models deleted")
    }

    // ------------------------------------------------------------------
    // YOLO model management
    // ------------------------------------------------------------------

    /**
     * Return the absolute path to a YOLO model file if it exists in internal
     * storage, or ``null`` if it has not been downloaded yet.
     *
     * @param context Application or activity context.
     * @param modelFilename Filename such as [YoloProcessor.MODEL_DETECT].
     */
    fun getYoloModelPath(context: Context, modelFilename: String): String? {
        val file = yoloModelFile(context, modelFilename)
        return if (file.exists() && file.length() > 0) file.absolutePath else null
    }

    /**
     * Return ``true`` if **all** YOLO model files are present and non-empty.
     *
     * @param context Application or activity context.
     */
    fun areYoloModelsReady(context: Context): Boolean {
        val models = listOf(
            YoloProcessor.MODEL_DETECT,
            YoloProcessor.MODEL_SEGMENT,
            YoloProcessor.MODEL_POSE,
            YoloProcessor.MODEL_CLASSIFY,
            YoloProcessor.MODEL_OBB,
        )
        return models.all { getYoloModelPath(context, it) != null }
    }

    /**
     * Download all missing YOLO model files from Ultralytics.
     *
     * This is a blocking call; run it on a background thread.
     *
     * @param context Application or activity context.
     * @param onProgress Optional callback invoked with current and total file count.
     * @return ``true`` if all files are now available; ``false`` if any
     *         download failed.
     */
    fun downloadMissingYoloModels(
        context: Context,
        onProgress: ((downloaded: Int, total: Int) -> Unit)? = null,
    ): Boolean {
        val missing = YOLO_MODEL_URLS.filter { (filename, _) ->
            getYoloModelPath(context, filename) == null
        }
        if (missing.isEmpty()) return true

        var downloaded = 0
        val total = missing.size

        for ((filename, url) in missing) {
            val dest = yoloModelFile(context, filename)
            val success = downloadModel(context, filename, url, dest)

            if (success) {
                downloaded++
                onProgress?.invoke(downloaded, total)
            } else {
                Log.e(TAG, "Failed to download YOLO model: $filename")
                return false
            }
        }
        return true
    }

    /**
     * Delete all downloaded YOLO model files from internal storage.
     *
     * @param context Application or activity context.
     */
    fun deleteAllYoloModels(context: Context) {
        val dir = yoloDir(context)
        dir.listFiles()?.forEach { it.delete() }
        Log.i(TAG, "All YOLO models deleted")
    }

    // ------------------------------------------------------------------
    // RTMDet model management
    // ------------------------------------------------------------------

    /**
     * Return the absolute path to an RTMDet model file if it exists in
     * internal storage, or ``null`` if it has not been downloaded yet.
     *
     * @param context Application or activity context.
     * @param modelFilename Filename such as [RtmDetProcessor.MODEL_DETECT].
     */
    fun getRtmDetModelPath(context: Context, modelFilename: String): String? {
        val file = rtmdetModelFile(context, modelFilename)
        return if (file.exists() && file.length() > 0) file.absolutePath else null
    }

    /**
     * Return ``true`` if **all** RTMDet model files are present and non-empty.
     *
     * @param context Application or activity context.
     */
    fun areRtmDetModelsReady(context: Context): Boolean =
        RTMDET_MODEL_URLS.keys.all { getRtmDetModelPath(context, it) != null }

    /**
     * Download all missing RTMDet model files.
     *
     * This is a blocking call; run it on a background thread.
     *
     * @param context Application or activity context.
     * @param onProgress Optional callback invoked with current and total file count.
     * @return ``true`` if all RTMDet models are now available; ``false`` if any download failed.
     */
    fun downloadMissingRtmDetModels(
        context: Context,
        onProgress: ((downloaded: Int, total: Int) -> Unit)? = null,
    ): Boolean {
        val missing = RTMDET_MODEL_URLS.filter { (filename, _) ->
            getRtmDetModelPath(context, filename) == null
        }
        if (missing.isEmpty()) return true

        var downloaded = 0
        val total = missing.size

        for ((filename, url) in missing) {
            val dest = rtmdetModelFile(context, filename)
            val success = downloadModel(context, filename, url, dest)
            if (success) {
                downloaded++
                onProgress?.invoke(downloaded, total)
            } else {
                Log.e(TAG, "Failed to download RTMDet model: $filename")
                return false
            }
        }
        return true
    }

    /**
     * Delete all downloaded RTMDet model files from internal storage.
     *
     * @param context Application or activity context.
     */
    fun deleteAllRtmDetModels(context: Context) {
        val dir = rtmdetDir(context)
        dir.listFiles()?.forEach { it.delete() }
        Log.i(TAG, "All RTMDet models deleted")
    }

    // ------------------------------------------------------------------
    // Mobilint Model Management
    // ------------------------------------------------------------------

    fun getMobilintModelPath(context: Context, modelName: String): String? {
        val file = mobilintModelFile(context, modelName)
        return if (file.exists()) file.absolutePath else null
    }

    fun areMobilintModelsReady(context: Context): Boolean {
        return MOBILINT_MODEL_URLS.keys.all { mobilintModelFile(context, it).exists() }
    }

    fun downloadMissingMobilintModels(context: Context, onProgress: ((downloaded: Int, total: Int) -> Unit)? = null): Boolean {
        val missing = MOBILINT_MODEL_URLS.keys.filter { !mobilintModelFile(context, it).exists() }
        if (missing.isEmpty()) return true

        var successCount = 0
        missing.forEach { modelName ->
            val url = MOBILINT_MODEL_URLS[modelName]!!
            val file = mobilintModelFile(context, modelName)
            if (downloadModel(context, modelName, url, file)) {
                successCount++
                onProgress?.invoke(successCount, missing.size)
            }
        }
        return successCount == missing.size
    }

    fun deleteAllMobilintModels(context: Context) {
        val dir = mobilintDir(context)
        dir.listFiles()?.forEach { it.delete() }
        Log.i(TAG, "All Mobilint models deleted")
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    private fun mediapipeDir(context: Context): File =
        File(context.filesDir, MEDIAPIPE_DIR).also { it.mkdirs() }

    private fun modelFile(context: Context, filename: String): File =
        File(mediapipeDir(context), filename)

    private fun yoloDir(context: Context): File =
        File(context.filesDir, YOLO_DIR).also { it.mkdirs() }

    private fun yoloModelFile(context: Context, filename: String): File =
        File(yoloDir(context), filename)

    private fun rtmdetDir(context: Context): File =
        File(context.filesDir, RTMDET_DIR).also { it.mkdirs() }

    private fun rtmdetModelFile(context: Context, filename: String): File =
        File(rtmdetDir(context), filename)

    private fun mobilintDir(context: Context): File =
        File(context.filesDir, MOBILINT_DIR).also { it.mkdirs() }

    private fun mobilintModelFile(context: Context, filename: String): File =
        File(mobilintDir(context), filename)
}
