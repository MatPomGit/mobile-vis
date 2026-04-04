package pl.edu.mobilecv

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.sqrt

enum class PoseTemporalFilterType {
    EMA,
    ONE_EURO,
}

enum class PoseOutputMode {
    RAW,
    SMOOTHED,
    RAW_VS_SMOOTHED,
}

data class PoseTemporalConfig(
    val enabled: Boolean = true,
    val filterType: PoseTemporalFilterType = PoseTemporalFilterType.EMA,
    val emaAlpha: Double = 0.35,
    val oneEuroMinCutoff: Double = 1.0,
    val oneEuroBeta: Double = 0.02,
    val oneEuroDerivativeCutoff: Double = 1.0,
    val jitterWindowSize: Int = 30,
    val staticJitterThresholdTvecMm: Double = 2.5,
    val staticJitterThresholdRvecDeg: Double = 0.35,
)

data class PoseTemporalJitter(
    val tvecMm: Double,
    val rvecDeg: Double,
)

data class PoseTemporalResult(
    val rawTvec: DoubleArray,
    val rawRvec: DoubleArray,
    val smoothedTvec: DoubleArray,
    val smoothedRvec: DoubleArray,
    val rawJitter: PoseTemporalJitter?,
    val smoothedJitter: PoseTemporalJitter?,
)

/**
 * Temporal filtering helper for marker pose vectors (`tvec` and `rvec`).
 */
class PoseTemporalFilter(private var config: PoseTemporalConfig = PoseTemporalConfig()) {
    private data class MarkerState(
        var lastTsNs: Long = 0L,
        var emaTvec: DoubleArray? = null,
        var emaRvec: DoubleArray? = null,
        val tvecFilters: Array<OneEuroScalarFilter> = Array(3) { OneEuroScalarFilter() },
        val rvecFilters: Array<OneEuroScalarFilter> = Array(3) { OneEuroScalarFilter() },
        val rawTvecHistory: ArrayDeque<DoubleArray> = ArrayDeque(),
        val rawRvecHistory: ArrayDeque<DoubleArray> = ArrayDeque(),
        val smoothedTvecHistory: ArrayDeque<DoubleArray> = ArrayDeque(),
        val smoothedRvecHistory: ArrayDeque<DoubleArray> = ArrayDeque(),
    )

    private val states = HashMap<String, MarkerState>()

    fun updateConfig(newConfig: PoseTemporalConfig) {
        config = newConfig
    }

    fun reset(markerKey: String? = null) {
        if (markerKey == null) states.clear() else states.remove(markerKey)
    }

    fun process(markerKey: String, tvec: DoubleArray, rvec: DoubleArray, timestampNs: Long): PoseTemporalResult {
        val state = states.getOrPut(markerKey) { MarkerState() }
        val safeTvec = tvec.copyOf(3)
        val safeRvec = rvec.copyOf(3)
        val smoothed = if (!config.enabled) {
            Pair(safeTvec, safeRvec)
        } else {
            when (config.filterType) {
                PoseTemporalFilterType.EMA -> applyEma(state, safeTvec, safeRvec)
                PoseTemporalFilterType.ONE_EURO -> applyOneEuro(state, safeTvec, safeRvec, timestampNs)
            }
        }
        state.lastTsNs = timestampNs

        push(state.rawTvecHistory, safeTvec)
        push(state.rawRvecHistory, safeRvec)
        push(state.smoothedTvecHistory, smoothed.first)
        push(state.smoothedRvecHistory, smoothed.second)

        val rawJitter = computeJitter(state.rawTvecHistory, state.rawRvecHistory)
        val smoothedJitter = computeJitter(state.smoothedTvecHistory, state.smoothedRvecHistory)
        return PoseTemporalResult(
            rawTvec = safeTvec,
            rawRvec = safeRvec,
            smoothedTvec = smoothed.first,
            smoothedRvec = smoothed.second,
            rawJitter = rawJitter,
            smoothedJitter = smoothedJitter,
        )
    }

    fun isStaticSceneStable(jitter: PoseTemporalJitter?): Boolean {
        if (jitter == null) return false
        return jitter.tvecMm <= config.staticJitterThresholdTvecMm &&
            jitter.rvecDeg <= config.staticJitterThresholdRvecDeg
    }

    private fun applyEma(state: MarkerState, tvec: DoubleArray, rvec: DoubleArray): Pair<DoubleArray, DoubleArray> {
        val alpha = config.emaAlpha.coerceIn(0.01, 0.99)
        val prevT = state.emaTvec
        val prevR = state.emaRvec
        val outT = DoubleArray(3) { i -> if (prevT == null) tvec[i] else prevT[i] + alpha * (tvec[i] - prevT[i]) }
        val outR = DoubleArray(3) { i ->
            // Rodrigues rotation-vector components are not independent periodic angles,
            // so EMA must not wrap them with angleDeltaRad() per component.
            if (prevR == null) rvec[i] else prevR[i] + alpha * (rvec[i] - prevR[i])
        }
        state.emaTvec = outT
        state.emaRvec = outR
        return Pair(outT, outR)
    }

    private fun applyOneEuro(
        state: MarkerState,
        tvec: DoubleArray,
        rvec: DoubleArray,
        timestampNs: Long,
    ): Pair<DoubleArray, DoubleArray> {
        val dt = if (state.lastTsNs > 0L && timestampNs > state.lastTsNs) {
            ((timestampNs - state.lastTsNs).toDouble() / 1_000_000_000.0).coerceAtLeast(1e-4)
        } else {
            1.0 / 30.0
        }
        val outT = DoubleArray(3) { i ->
            state.tvecFilters[i].filter(
                value = tvec[i],
                dtSeconds = dt,
                minCutoff = config.oneEuroMinCutoff,
                beta = config.oneEuroBeta,
                derivativeCutoff = config.oneEuroDerivativeCutoff,
            )
        }
        val outR = DoubleArray(3) { i ->
            state.rvecFilters[i].filter(
                value = rvec[i],
                dtSeconds = dt,
                minCutoff = config.oneEuroMinCutoff,
                beta = config.oneEuroBeta,
                derivativeCutoff = config.oneEuroDerivativeCutoff,
            )
        }
        return Pair(outT, outR)
    }

    private fun computeJitter(tHistory: ArrayDeque<DoubleArray>, rHistory: ArrayDeque<DoubleArray>): PoseTemporalJitter? {
        if (tHistory.size < 2 || rHistory.size < 2) return null
        val tSteps = pairwiseStepNorms(tHistory)
        val rSteps = pairwiseStepNorms(rHistory)
        val tMm = rms(tSteps) * 1000.0
        val rDeg = Math.toDegrees(rms(rSteps))
        return PoseTemporalJitter(tvecMm = tMm, rvecDeg = rDeg)
    }

    private fun pairwiseStepNorms(history: ArrayDeque<DoubleArray>): List<Double> {
        if (history.size < 2) return emptyList()
        val out = ArrayList<Double>(history.size - 1)
        var prev = history.first()
        history.drop(1).forEach { cur ->
            var sum = 0.0
            for (i in 0 until 3) {
                val d = cur[i] - prev[i]
                sum += d * d
            }
            out.add(sqrt(sum))
            prev = cur
        }
        return out
    }

    private fun rms(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        val meanSq = values.sumOf { it * it } / values.size
        return sqrt(meanSq)
    }

    private fun push(history: ArrayDeque<DoubleArray>, sample: DoubleArray) {
        history.addLast(sample.copyOf(3))
        while (history.size > config.jitterWindowSize) history.removeFirst()
    }

    private fun angleDeltaRad(current: Double, previous: Double): Double {
        var delta = current - previous
        while (delta > PI) delta -= 2.0 * PI
        while (delta < -PI) delta += 2.0 * PI
        return delta
    }
}

private class OneEuroScalarFilter {
    private var initialized = false
    private var prevValue = 0.0
    private var prevDerivative = 0.0

    fun filter(
        value: Double,
        dtSeconds: Double,
        minCutoff: Double,
        beta: Double,
        derivativeCutoff: Double,
    ): Double {
        if (!initialized) {
            initialized = true
            prevValue = value
            prevDerivative = 0.0
            return value
        }
        val dValue = (value - prevValue) / dtSeconds
        val alphaD = alpha(derivativeCutoff, dtSeconds)
        val smoothedDerivative = lowPass(prevDerivative, dValue, alphaD)
        val cutoff = minCutoff + beta * abs(smoothedDerivative)
        val alpha = alpha(cutoff, dtSeconds)
        val smoothedValue = lowPass(prevValue, value, alpha)
        prevValue = smoothedValue
        prevDerivative = smoothedDerivative
        return smoothedValue
    }

    private fun alpha(cutoff: Double, dtSeconds: Double): Double {
        val tau = 1.0 / (2.0 * PI * cutoff.coerceAtLeast(1e-6))
        return 1.0 / (1.0 + exp(-dtSeconds / tau))
    }

    private fun lowPass(previous: Double, current: Double, alpha: Double): Double {
        return previous + alpha * (current - previous)
    }
}
