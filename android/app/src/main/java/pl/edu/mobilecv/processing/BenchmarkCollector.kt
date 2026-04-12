package pl.edu.mobilecv.processing

import pl.edu.mobilecv.OpenCvFilter

/**
 * Neutralny snapshot metryk runtime dla pojedynczego filtra.
 */
data class RuntimeBenchmarkSnapshot(
    val filter: OpenCvFilter,
    val samples: Int,
    val avgBeforeMs: Double,
    val avgAfterMs: Double,
    val fpsBefore: Double,
    val fpsAfter: Double,
)

/**
 * Kolekcjonuje metryki wydajności uruchomień filtra przed i po optymalizacji.
 */
class BenchmarkCollector(
    private val benchmarkFilters: Set<OpenCvFilter>,
    var sampleLimit: Int = 30,
) {
    private data class Accumulator(
        var samples: Int = 0,
        var beforeNs: Long = 0,
        var afterNs: Long = 0,
    )

    private val accumulators = mutableMapOf<OpenCvFilter, Accumulator>()

    /**
     * Informuje czy dla danego filtra należy zebrać benchmark.
     */
    fun shouldCollect(filter: OpenCvFilter): Boolean = filter in benchmarkFilters

    /**
     * Dodaje nową próbkę czasową dla filtra.
     */
    fun addSample(filter: OpenCvFilter, beforeNs: Long, afterNs: Long) {
        val acc = accumulators.getOrPut(filter) { Accumulator() }
        if (acc.samples >= sampleLimit) return
        acc.samples += 1
        acc.beforeNs += beforeNs
        acc.afterNs += afterNs
    }

    /**
     * Zwraca gotowy snapshot i czyści stan po osiągnięciu limitu próbek.
     */
    fun consumeSnapshot(filter: OpenCvFilter): RuntimeBenchmarkSnapshot? {
        val acc = accumulators[filter] ?: return null
        if (acc.samples == 0 || acc.samples < sampleLimit) return null
        accumulators.remove(filter)

        val avgBeforeMs = acc.beforeNs.toDouble() / acc.samples / 1_000_000.0
        val avgAfterMs = acc.afterNs.toDouble() / acc.samples / 1_000_000.0
        val fpsBefore = if (avgBeforeMs > 0.0) 1000.0 / avgBeforeMs else 0.0
        val fpsAfter = if (avgAfterMs > 0.0) 1000.0 / avgAfterMs else 0.0

        return RuntimeBenchmarkSnapshot(
            filter = filter,
            samples = acc.samples,
            avgBeforeMs = avgBeforeMs,
            avgAfterMs = avgAfterMs,
            fpsBefore = fpsBefore,
            fpsAfter = fpsAfter,
        )
    }
}
