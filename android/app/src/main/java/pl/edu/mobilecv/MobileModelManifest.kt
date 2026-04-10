package pl.edu.mobilecv

import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import java.util.Locale

/**
 * Pomocnicze struktury i parser manifestu artefaktów modeli mobilnych.
 */
data class MobileModelManifestEntry(
    val modelName: String,
    val format: String,
    val inputShape: List<Int>,
    val dtype: String,
    val classMap: Map<String, String>,
    val preprocess: JSONObject,
    val postprocess: JSONObject,
    val version: String,
    val sha256: String,
)

/**
 * Utility do wczytywania i walidacji manifestu oraz sum kontrolnych.
 */
object MobileModelManifest {
    private const val TAG = "MobileModelManifest"

    /**
     * Wczytuje manifest z pliku JSON.
     */
    fun loadFromFile(file: File): List<MobileModelManifestEntry> {
        if (!file.exists()) {
            return emptyList()
        }
        return parse(file.readText())
    }

    /**
     * Parsuje manifest JSON.
     *
     * Wspierane formaty:
     * - root object z polem "models" (tablica),
     * - root jako tablica modeli.
     */
    fun parse(jsonText: String): List<MobileModelManifestEntry> {
        val trimmed = jsonText.trim()
        if (trimmed.isEmpty()) {
            return emptyList()
        }

        val modelsArray = if (trimmed.startsWith("{")) {
            JSONObject(trimmed).optJSONArray("models") ?: JSONArray()
        } else {
            JSONArray(trimmed)
        }

        val models = mutableListOf<MobileModelManifestEntry>()
        for (idx in 0 until modelsArray.length()) {
            val obj = modelsArray.optJSONObject(idx) ?: continue
            val inputShape = mutableListOf<Int>()
            val shapeJson = obj.optJSONArray("input_shape") ?: JSONArray()
            for (shapeIdx in 0 until shapeJson.length()) {
                inputShape.add(shapeJson.optInt(shapeIdx))
            }

            val classMap = mutableMapOf<String, String>()
            val classMapObj = obj.optJSONObject("class_map") ?: JSONObject()
            classMapObj.keys().forEach { key ->
                classMap[key] = classMapObj.optString(key)
            }

            models.add(
                MobileModelManifestEntry(
                    modelName = obj.optString("model_name"),
                    format = obj.optString("format"),
                    inputShape = inputShape,
                    dtype = obj.optString("dtype"),
                    classMap = classMap,
                    preprocess = obj.optJSONObject("preprocess") ?: JSONObject(),
                    postprocess = obj.optJSONObject("postprocess") ?: JSONObject(),
                    version = obj.optString("version"),
                    sha256 = obj.optString("sha256").lowercase(Locale.US),
                ),
            )
        }
        return models
    }

    /**
     * Zwraca rekord manifestu dla konkretnego pliku artefaktu.
     */
    fun findByModelName(
        manifest: List<MobileModelManifestEntry>,
        modelFilename: String,
    ): MobileModelManifestEntry? = manifest.firstOrNull { it.modelName == modelFilename }

    /**
     * Oblicza SHA-256 pliku jako lowercase hex.
     */
    fun computeSha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val read = input.read(buffer)
                if (read <= 0) break
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    /**
     * Waliduje podstawową zgodność wpisu manifestu.
     */
    fun isEntryCompatible(entry: MobileModelManifestEntry, artifactFile: File): Boolean {
        if (entry.modelName != artifactFile.name) {
            Log.w(TAG, "Manifest model_name mismatch: ${entry.modelName} != ${artifactFile.name}")
            return false
        }
        if (entry.format.isBlank() || entry.inputShape.isEmpty() || entry.dtype.isBlank()) {
            Log.w(TAG, "Manifest entry incomplete for ${entry.modelName}")
            return false
        }
        val ext = artifactFile.extension.lowercase(Locale.US)
        if (ext != entry.format.lowercase(Locale.US)) {
            Log.w(TAG, "Manifest format mismatch for ${entry.modelName}: .$ext != ${entry.format}")
            return false
        }
        return true
    }
}
