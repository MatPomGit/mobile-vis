package pl.edu.mobilecv

/**
 * Computes a rolling-window frames-per-second metric.
 *
 * Call [onFrame] every time a frame is processed.  Read [fps] to get the
 * current smoothed FPS value (updated at most once per [UPDATE_INTERVAL_MS]).
 *
 * Thread-safe: [onFrame] may be called from any thread; [fps] is volatile.
 */
class FpsCounter {

    companion object {
        /** Minimum time between FPS value updates, in nanoseconds. */
        private const val UPDATE_INTERVAL_NS = 500_000_000L
    }

    /** Current smoothed FPS. Updated every [UPDATE_INTERVAL_NS] nanoseconds. */
    @Volatile
    var fps: Double = 0.0
        private set

    private var frameCount = 0
    private var windowStart = System.nanoTime()

    /**
     * Record one processed frame and update [fps] if the update interval has elapsed.
     *
     * Must not be called concurrently from multiple threads; use a single dedicated thread
     * (e.g. the CameraX analysis executor).
     */
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

    /** Reset all counters. Useful when the camera is re-bound. */
    fun reset() {
        fps = 0.0
        frameCount = 0
        windowStart = System.nanoTime()
    }
}
