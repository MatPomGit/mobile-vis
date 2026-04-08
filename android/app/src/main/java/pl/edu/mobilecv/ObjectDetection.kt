package pl.edu.mobilecv

import android.graphics.RectF
import org.json.JSONObject

/**
 * Result model for general object detectors (YOLO, RTMDet).
 * Separated from fiducial markers (AprilTag, ArUco) as they serve different purposes.
 */
sealed class ObjectDetection {
    abstract val label: String
    abstract val classId: Int
    abstract val confidence: Float
    abstract val bbox: RectF
    abstract val timestamp: Long

    data class Yolo(
        override val label: String,
        override val classId: Int,
        override val confidence: Float,
        override val bbox: RectF,
        override val timestamp: Long = System.currentTimeMillis()
    ) : ObjectDetection()

    data class RtmDet(
        override val label: String,
        override val classId: Int,
        override val confidence: Float,
        override val bbox: RectF,
        val angleDeg: Float? = null,
        override val timestamp: Long = System.currentTimeMillis()
    ) : ObjectDetection()

    fun toJson(): JSONObject = JSONObject().apply {
        put("type", when (this@ObjectDetection) {
            is Yolo -> "yolo"
            is RtmDet -> "rtmdet"
        })
        put("label", label)
        put("class_id", classId)
        put("confidence", confidence.toDouble())
        put("bbox", JSONObject().apply {
            put("x1", bbox.left.toDouble())
            put("y1", bbox.top.toDouble())
            put("x2", bbox.right.toDouble())
            put("y2", bbox.bottom.toDouble())
        })
        if (this@ObjectDetection is RtmDet && angleDeg != null) {
            put("angle_deg", angleDeg.toDouble())
        }
        put("timestamp", timestamp)
    }

    fun toDiagnosticSummary(): String = buildString {
        val type = if (this@ObjectDetection is Yolo) "YOLO" else "RTMDet"
        append("[$type] label=$label conf=${"%.2f".format(confidence)} bbox=[${bbox.left.toInt()}, ${bbox.top.toInt()}, ${bbox.right.toInt()}, ${bbox.bottom.toInt()}]")
        if (this@ObjectDetection is RtmDet && angleDeg != null) {
            append(" angle=${"%.1f".format(angleDeg)}°")
        }
    }
}
