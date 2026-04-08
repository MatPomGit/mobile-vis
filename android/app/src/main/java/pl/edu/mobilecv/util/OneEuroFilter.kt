package pl.edu.mobilecv.util

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.exp

/**
 * One Euro Filter for smoothing noisy signals in real-time.
 * See: http://cristal.univ-lille.fr/~casiez/1euro/
 */
class OneEuroFilter(
    var minCutoff: Double = 1.0,
    var beta: Double = 0.0,
    var dCutoff: Double = 1.0
) {
    private var firstTime = true
    private var lastValue = 0.0
    private var lastDerivative = 0.0

    fun filter(value: Double, timestampNs: Long): Double {
        val dt = if (firstTime) 0.0 else (timestampNs - lastTimestamp) / 1e9
        lastTimestamp = timestampNs

        if (firstTime) {
            firstTime = false
            lastValue = value
            lastDerivative = 0.0
            return value
        }

        val dValue = (value - lastValue) / dt
        val alphaD = alpha(dt, dCutoff)
        val dx = lastDerivative + alphaD * (dValue - lastDerivative)
        
        val cutoff = minCutoff + beta * abs(dx)
        val alpha = alpha(dt, cutoff)
        val result = lastValue + alpha * (value - lastValue)
        
        lastValue = result
        lastDerivative = dx
        return result
    }

    private var lastTimestamp: Long = 0

    private fun alpha(dt: Double, cutoff: Double): Double {
        val tau = 1.0 / (2.0 * PI * cutoff)
        return 1.0 / (1.0 + tau / dt)
    }

    fun reset() {
        firstTime = true
        lastValue = 0.0
        lastDerivative = 0.0
    }
}
