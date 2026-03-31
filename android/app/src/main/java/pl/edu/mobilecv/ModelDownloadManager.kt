package pl.edu.mobilecv

import android.content.Context
import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.IOException
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
    )

    private val httpClient: OkHttpClient by lazy {
        OkHttpClient.Builder()
            .connectTimeout(CONNECT_TIMEOUT_SECONDS, TimeUnit.SECONDS)
            .readTimeout(READ_TIMEOUT_SECONDS, TimeUnit.SECONDS)
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
    fun downloadModel(context: Context, filename: String, url: String): Boolean {
        val dest = modelFile(context, filename)
        val tempFile = File(dest.parent, "$filename.tmp")
        dest.parentFile?.mkdirs()

        Log.i(TAG, "Downloading $filename from $url")

        return try {
            val request = Request.Builder().url(url).build()
            httpClient.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    Log.e(TAG, "HTTP ${response.code} downloading $filename")
                    return false
                }
                response.body?.byteStream()?.use { input ->
                    tempFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                } ?: return false
            }
            tempFile.renameTo(dest)
            Log.i(TAG, "Downloaded $filename (${dest.length()} bytes)")
            true
        } catch (e: IOException) {
            Log.e(TAG, "IOException downloading $filename", e)
            tempFile.delete()
            false
        }
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
    // Internal helpers
    // ------------------------------------------------------------------

    private fun mediapipeDir(context: Context): File =
        File(context.filesDir, MEDIAPIPE_DIR).also { it.mkdirs() }

    private fun modelFile(context: Context, filename: String): File =
        File(mediapipeDir(context), filename)
}
