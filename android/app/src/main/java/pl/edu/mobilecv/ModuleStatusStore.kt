package pl.edu.mobilecv

import java.util.concurrent.ConcurrentHashMap

/**
 * Wspólne kontrakty statusów modułów wykorzystywane przez lifecycle i warstwę UI.
 */
sealed class ModuleStatusState {
    /** Moduł gotowy do pracy. */
    data object Ready : ModuleStatusState()

    /** Moduł jest w trakcie inicjalizacji albo pobierania artefaktów. */
    data object Downloading : ModuleStatusState()

    /** Moduł jest wyłączony przez użytkownika lub politykę awaryjną. */
    data object Disabled : ModuleStatusState()

    /** Moduł zakończył inicjalizację błędem. */
    data class Error(val errorMessageResId: Int? = null) : ModuleStatusState()
}

/**
 * Centralny store statusu modułów ML/CV.
 *
 * Przechowuje bieżący status modułu oraz znacznik czasu ostatniej próby inicjalizacji.
 * Dzięki temu UI korzysta z jednego źródła prawdy zamiast lokalnych flag.
 */
object ModuleStatusStore {

    /** Typy modułów zarządzane przez aplikację. */
    enum class ModuleType {
        MEDIAPIPE,
        YOLO,
        TFLITE,
    }

    /** Migawka stanu pojedynczego modułu. */
    data class ModuleSnapshot(
        val status: ModuleStatusState,
        val lastInitAttemptEpochMs: Long? = null,
    )

    private val state = ConcurrentHashMap<ModuleType, ModuleSnapshot>()

    init {
        ModuleType.entries.forEach { module ->
            state[module] = ModuleSnapshot(status = ModuleStatusState.Disabled)
        }
    }

    /** Zwraca aktualny stan wskazanego modułu. */
    fun get(moduleType: ModuleType): ModuleSnapshot =
        state[moduleType] ?: ModuleSnapshot(status = ModuleStatusState.Disabled)

    /** Zwraca pełną migawkę wszystkich modułów. */
    fun snapshot(): Map<ModuleType, ModuleSnapshot> = ModuleType.entries.associateWith { get(it) }

    /** Aktualizuje status modułu i zapisuje timestamp ostatniej próby inicjalizacji. */
    @Synchronized
    fun setStatus(moduleType: ModuleType, status: ModuleStatusState) {
        val existing = get(moduleType)
        state[moduleType] = existing.copy(
            status = status,
            lastInitAttemptEpochMs = System.currentTimeMillis(),
        )
    }
}
