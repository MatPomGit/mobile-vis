package pl.edu.mobilecv

import kotlin.math.sin
import kotlin.random.Random
import org.junit.Assert.assertTrue
import org.junit.Test

class PoseTemporalFilterTest {
    @Test
    fun `smoothed pose reduces jitter for static scene and stays under threshold`() {
        val filter = PoseTemporalFilter(
            PoseTemporalConfig(
                enabled = true,
                filterType = PoseTemporalFilterType.EMA,
                emaAlpha = 0.2,
                jitterWindowSize = 40,
                staticJitterThresholdTvecMm = 2.5,
                staticJitterThresholdRvecDeg = 0.35,
            ),
        )
        val rng = Random(42)
        var last: PoseTemporalResult? = null
        repeat(240) { i ->
            val t = i * 33_333_333L
            val noisyTvec = doubleArrayOf(
                0.04 + (rng.nextDouble() - 0.5) * 0.004,
                -0.02 + (rng.nextDouble() - 0.5) * 0.004,
                0.80 + (rng.nextDouble() - 0.5) * 0.004,
            )
            val noisyRvec = doubleArrayOf(
                0.02 + sin(i * 0.01) * 0.002 + (rng.nextDouble() - 0.5) * 0.02,
                -0.01 + (rng.nextDouble() - 0.5) * 0.02,
                0.03 + (rng.nextDouble() - 0.5) * 0.02,
            )
            last = filter.process("marker-1", noisyTvec, noisyRvec, t)
        }
        val result = requireNotNull(last)
        val raw = requireNotNull(result.rawJitter)
        val smooth = requireNotNull(result.smoothedJitter)
        assertTrue("smoothed tvec jitter should be lower", smooth.tvecMm < raw.tvecMm)
        assertTrue("smoothed rvec jitter should be lower", smooth.rvecDeg < raw.rvecDeg)
        assertTrue("static-scene jitter should be accepted", filter.isStaticSceneStable(smooth))
    }
}
