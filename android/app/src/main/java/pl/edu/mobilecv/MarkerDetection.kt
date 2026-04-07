package pl.edu.mobilecv

import org.json.JSONArray
import org.json.JSONObject

/**
 * Unified marker result shared across processing, UI overlay, and diagnostics.
 *
 * Corner order convention: top-left, top-right, bottom-right, bottom-left.
 */
sealed class MarkerDetection {
    abstract val type: String
    abstract val id: String
    abstract val corners: List<Pair<Float, Float>>
    abstract val rvec: List<Double>?
    abstract val tvec: List<Double>?
    abstract val quality: Quality
    abstract val timestamp: Long

    data class Quality(
        val confidence: Double? = null,
        val reprojectionErrorPx: Double? = null,
    ) {
        fun toJson(): JSONObject = JSONObject().apply {
            put("confidence", confidence)
            put("reprojection_error_px", reprojectionErrorPx)
        }

        fun toOverlayString(): String {
            val conf = confidence?.let { "%.2f".format(it) } ?: "n/a"
            val err = reprojectionErrorPx?.let { "%.2fpx".format(it) } ?: "n/a"
            return "q=$conf err=$err"
        }
    }

    open fun toCommonJson(): JSONObject = JSONObject().apply {
        put("type", this@MarkerDetection.type)
        put("id", this@MarkerDetection.id)
        put("corners", cornersToJsonArray(this@MarkerDetection.corners))
        put("rvec", vectorToJsonArray(this@MarkerDetection.rvec))
        put("tvec", vectorToJsonArray(this@MarkerDetection.tvec))
        put("quality", this@MarkerDetection.quality.toJson())
        put("timestamp", this@MarkerDetection.timestamp)
    }

    fun toDiagnosticSummary(): String = buildString {
        append("type=${this@MarkerDetection.type} id=${this@MarkerDetection.id} timestamp=${this@MarkerDetection.timestamp}")
        append(" corners=${this@MarkerDetection.corners.size}")
        append(" ${this@MarkerDetection.quality.toOverlayString()}")
        this@MarkerDetection.rvec?.let { append(" rvec=${formatVector(it)}") }
        this@MarkerDetection.tvec?.let { append(" tvec=${formatVector(it)}") }
    }

    data class AprilTag(
        val markerId: Int,
        override val corners: List<Pair<Float, Float>>,
        override val rvec: List<Double>? = null,
        override val tvec: List<Double>? = null,
        override val quality: Quality = Quality(),
        override val timestamp: Long = System.currentTimeMillis(),
    ) : MarkerDetection() {
        override val type: String = "apriltag"
        override val id: String = markerId.toString()
    }

    data class Aruco(
        val markerId: Int,
        override val corners: List<Pair<Float, Float>>,
        override val rvec: List<Double>? = null,
        override val tvec: List<Double>? = null,
        override val quality: Quality = Quality(),
        override val timestamp: Long = System.currentTimeMillis(),
    ) : MarkerDetection() {
        override val type: String = "aruco"
        override val id: String = markerId.toString()
    }

    data class QrCode(
        val text: String,
        override val corners: List<Pair<Float, Float>>,
        override val rvec: List<Double>? = null,
        override val tvec: List<Double>? = null,
        override val quality: Quality = Quality(),
        override val timestamp: Long = System.currentTimeMillis(),
    ) : MarkerDetection() {
        override val type: String = "qr"
        override val id: String = text
    }

    data class CCTag(
        val markerId: Int,
        val center: Pair<Float, Float>,
        val radius: Float,
        override val corners: List<Pair<Float, Float>>,
        override val rvec: List<Double>? = null,
        override val tvec: List<Double>? = null,
        override val quality: Quality = Quality(),
        override val timestamp: Long = System.currentTimeMillis(),
    ) : MarkerDetection() {
        override val type: String = "cctag"
        override val id: String = markerId.toString()

        override fun toCommonJson(): JSONObject = super.toCommonJson().apply {
            put("center", JSONObject().apply {
                put("x", center.first.toDouble())
                put("y", center.second.toDouble())
            })
            put("radius", radius.toDouble())
        }
    }

    data class YoloObject(
        val label: String,
        val classId: Int,
        val confidence: Float,
        val bbox: android.graphics.RectF,
        override val corners: List<Pair<Float, Float>> = listOf(
            Pair(bbox.left, bbox.top),
            Pair(bbox.right, bbox.top),
            Pair(bbox.right, bbox.bottom),
            Pair(bbox.left, bbox.bottom),
        ),
        override val rvec: List<Double>? = null,
        override val tvec: List<Double>? = null,
        override val quality: Quality = Quality(confidence = confidence.toDouble()),
        override val timestamp: Long = System.currentTimeMillis(),
    ) : MarkerDetection() {
        override val type: String = "yolo"
        override val id: String = "$classId:$label"

        override fun toCommonJson(): JSONObject = super.toCommonJson().apply {
            put("label", label)
            put("class_id", classId)
            put("bbox", JSONObject().apply {
                put("x1", bbox.left.toDouble())
                put("y1", bbox.top.toDouble())
                put("x2", bbox.right.toDouble())
                put("y2", bbox.bottom.toDouble())
            })
        }
    }

    /**
     * A single RTMDet object detection result.
     *
     * @param label Predicted class name from the COCO vocabulary.
     * @param classId Numeric class index (0-based).
     * @param confidence Detection confidence in [0, 1].
     * @param bbox Axis-aligned bounding box in screen coordinates.
     * @param angleDeg Rotation angle in degrees; ``null`` for axis-aligned detections.
     */
    data class RtmDetObject(
        val label: String,
        val classId: Int,
        val confidence: Float,
        val bbox: android.graphics.RectF,
        val angleDeg: Float? = null,
        override val corners: List<Pair<Float, Float>> = listOf(
            Pair(bbox.left, bbox.top),
            Pair(bbox.right, bbox.top),
            Pair(bbox.right, bbox.bottom),
            Pair(bbox.left, bbox.bottom),
        ),
        override val rvec: List<Double>? = null,
        override val tvec: List<Double>? = null,
        override val quality: Quality = Quality(confidence = confidence.toDouble()),
        override val timestamp: Long = System.currentTimeMillis(),
    ) : MarkerDetection() {
        override val type: String = "rtmdet"
        override val id: String = "$classId:$label"

        override fun toCommonJson(): JSONObject = super.toCommonJson().apply {
            put("label", label)
            put("class_id", classId)
            put("bbox", JSONObject().apply {
                put("x1", bbox.left.toDouble())
                put("y1", bbox.top.toDouble())
                put("x2", bbox.right.toDouble())
                put("y2", bbox.bottom.toDouble())
            })
            angleDeg?.let { put("angle_deg", it.toDouble()) }
        }
    }

    companion object {
        fun cornersToJsonArray(corners: List<Pair<Float, Float>>): JSONArray {
            val array = JSONArray()
            for ((x, y) in corners) {
                array.put(JSONObject().apply {
                    put("x", x.toDouble())
                    put("y", y.toDouble())
                })
            }
            return array
        }

        fun vectorToJsonArray(vector: List<Double>?): JSONArray {
            val array = JSONArray()
            vector?.forEach { array.put(it) }
            return array
        }

        fun formatVector(vector: List<Double>): String {
            return vector.joinToString(prefix = "[", postfix = "]") { "%.3f".format(it) }
        }
    }
}
