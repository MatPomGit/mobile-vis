package pl.edu.mobilecv

import android.util.Size

/**
 * Predefined camera resolution presets available in the resolution picker.
 *
 * Each entry maps a human-readable [displayName] to a target [size] used by
 * [androidx.camera.core.resolutionselector.ResolutionStrategy].  CameraX will
 * pick the closest supported resolution when the device does not support the
 * requested size exactly.
 *
 * Lower resolutions result in faster per-frame processing times and are
 * preferred for real-time analysis; higher resolutions produce sharper output
 * and are better suited for photo/video capture.
 */
enum class CameraResolution(val size: Size, val displayName: String) {

    /** 640 × 480 – lowest latency, recommended for real-time processing. */
    RES_480P(Size(640, 480), "480p (640×480)"),

    /** 1280 × 720 – balanced quality/performance. */
    RES_720P(Size(1280, 720), "720p (1280×720)"),

    /** 1920 × 1080 – highest detail, may reduce frame rate on slower devices. */
    RES_1080P(Size(1920, 1080), "1080p (1920×1080)");

    companion object {
        /** Default resolution used on first launch. */
        val DEFAULT: CameraResolution = RES_480P
    }
}
