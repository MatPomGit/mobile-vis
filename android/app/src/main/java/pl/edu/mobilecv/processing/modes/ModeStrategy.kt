package pl.edu.mobilecv.processing.modes

import pl.edu.mobilecv.AnalysisMode
import pl.edu.mobilecv.OpenCvFilter

/**
 * Kontrakt strategii odpowiedzialnej za pojedynczy tryb analizy.
 */
interface ModeStrategy {
    val mode: AnalysisMode

    /**
     * Sprawdza czy filtr należy do danego trybu.
     */
    fun supports(filter: OpenCvFilter): Boolean = mode.filters.contains(filter)
}

/**
 * Prosta implementacja strategii oparta na definicji filtrów z [AnalysisMode].
 */
class StaticModeStrategy(
    override val mode: AnalysisMode,
) : ModeStrategy
