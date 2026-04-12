package pl.edu.mobilecv.processing

import pl.edu.mobilecv.AnalysisMode
import pl.edu.mobilecv.OpenCvFilter
import pl.edu.mobilecv.processing.modes.ModeStrategy
import pl.edu.mobilecv.processing.modes.StaticModeStrategy

/**
 * Router wybierający strategię logiki zależnie od [AnalysisMode].
 */
class ModeRouter(
    private val strategies: List<ModeStrategy> = AnalysisMode.entries.map { StaticModeStrategy(it) },
) {
    /**
     * Zwraca strategię odpowiadającą aktywnemu trybowi UI.
     */
    fun strategyForMode(mode: AnalysisMode): ModeStrategy =
        strategies.firstOrNull { it.mode == mode } ?: error("Missing strategy for mode=$mode")

    /**
     * Wyszukuje strategię dla filtra; null oznacza nieobsługiwany filtr.
     */
    fun strategyForFilter(filter: OpenCvFilter): ModeStrategy? =
        strategies.firstOrNull { it.supports(filter) }
}
