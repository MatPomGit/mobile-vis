package pl.edu.mobilecv.util

/**
 * Computes a rolling-window frames-per-second metric.
 */
class FpsCounter {

    companion object {
        private const val UPDATE_INTERVAL_NS = 500_000_000L
    }

    @Volatile
    var fps: Double = 0.0
        private set

    private var frameCount = 0
    private var windowStart = System.nanoTime()

    fun onFrame() {
        frameCount++
        val now = System.nanoTime()
        val elapsed = now - windowStart
        if (elapsed >= UPDATE_INTERVAL_NS) {
            fps = frameCount * 1_000_000_000.0 / elapsed
            frameCount = 0
            windowStart = now
        }
    }

    fun reset() {
        fps = 0.0
        frameCount = 0
        windowStart = System.nanoTime()
    }
}
